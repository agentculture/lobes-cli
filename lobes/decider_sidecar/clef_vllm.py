"""``DECIDER_BACKEND=clef-vllm`` — Clef's backbone in vLLM's pooling runner, Cloudflare's head.

Runs ONLY in the ``Dockerfile.decider-vllm`` image (pinned vllm/vllm-openai
v0.31.0). The backbone is served by vLLM in-process (``LLM(runner="pooling")``)
so a compressed-tensors W4A16 checkpoint loads through vLLM's own kernels; the
joint schema head is Cloudflare's own ``JointSchemaHead`` class, imported from
the ``joint_schema_model.py`` the served repo ships, run in bf16 on the
backbone's final hidden states. The answer is assembled exactly as Cloudflare's
``systemone`` assembles it, so a client cannot tell this backend from ``clef``.

Written against Cloudflare's code only (Cloudflare/clef @
2f3de3dd85f379784083b0814d997ab627200f0c, ``joint_schema_model.py``, read
2026-10-14). No third-party wrapper or engine patch is copied or executed.

What the head needs (read from ``JointSchemaHead.forward`` / ``ClefModel``)
--------------------------------------------------------------------------
* ``hidden_states`` — the text model's ``last_hidden_state``: every position,
  AFTER the final norm (``ClefModel.forward`` calls ``base_model.model`` /
  ``.language_model``, not the LM head). vLLM's ``Qwen3NextModel.forward``
  (which ``Qwen3_5Model`` inherits) also returns ``self.norm(...)``, and the
  ``token_embed`` task with ``ALL`` pooling returns one row per prompt token.
  ``use_activation=False`` is required: the token-embed pooler head otherwise
  L2-normalises each row (``TokenEmbeddingPoolerHead`` -> ``PoolerNormalize``).
* ``input_ids`` / ``attention_mask`` — from Cloudflare's ``encode_record``
  (the schema, spans and chat framing are its, never re-derived here).
* ``output_embedding_weight`` — the UNTIED ``lm_head.weight`` (Clef's
  ``tie_word_embeddings`` is false), indexed ONLY at the option-span tokens
  (``output_embedding_weight[input_ids[start:end]]``). The pooling runner
  exposes no lm_head, so the rows are read lazily from the checkpoint's own
  safetensors and handed to the head as a compact table with remapped ids —
  numerically identical, without a second 2.5 GB copy of the matrix.

Assumptions (DECLARED, not yet measured — t13 is the live check)
-----------------------------------------------------------------
* The served repo (``DECIDER_MODEL`` @ ``DECIDER_REVISION``) ships Cloudflare's
  ``joint_schema_model.py``, ``joint_head.safetensors`` and
  ``joint_head_config.json`` beside the backbone, and keeps ``lm_head.weight``
  as a plain bf16 tensor (our W4A16 build's keep-list does). Both the original
  Cloudflare/clef and our quant satisfy this.
* Text-only: ``images`` / ``videos`` are refused (422) by this backend; the
  ``clef`` backend serves them. Matching vLLM's multimodal placeholder expansion
  to Cloudflare's span arithmetic is not done here.
* vLLM 0.31 runs this hybrid (GatedDeltaNet + attention) checkpoint under the
  pooling runner on the Orin's sm_87. Prefix caching is turned off explicitly
  (a cache hit would drop hidden-state rows); a row-count check guards it anyway.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence

NAME = "clef-vllm"
#: Cloudflare's ``encode_record`` default ``max_length`` (and ``systemone``'s).
DEFAULT_MAX_MODEL_LEN = 16384
#: DECLARED (#108): a share of the GPU for the 27B W4A16 backbone; t13 measures it.
DEFAULT_GPU_MEM_UTIL = 0.5
DEFAULT_HF_HOME = "/root/.cache/huggingface"
BUNDLE_SUBDIR = "lobes-decider-bundles"
_QUESTION_TYPES = ("noul", "choice", "score")
_UNSAFE = re.compile(r"[^A-Za-z0-9._@-]")


class Engine(Protocol):
    """The loaded model: encoding, scoring and answer formatting."""

    def encode(self, record: dict) -> Any: ...

    def score(self, encoded: Any) -> list[list[float]]: ...

    def systemone_answer(self, question: dict, probabilities: dict[str, float]) -> dict: ...


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def validate_request(record: Any) -> None:
    """Cloudflare ``systemone``'s own checks, plus this backend's text-only rule.

    The messages are Cloudflare's verbatim so a caller sees the same 422 detail
    from either Clef backend.
    """
    if not isinstance(record, dict):
        raise ValueError("request body must be a JSON object")
    questions = record.get("questions")
    if not isinstance(record.get("model"), str) or "state" not in record:
        raise ValueError("model and state are required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for question_id, question in questions.items():
        if not isinstance(question, dict) or question.get("type") not in _QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria must not be empty")
    if record.get("images") or record.get("videos"):
        raise ValueError(
            "images and videos are not supported by the clef-vllm backend "
            "(text-only); serve this domain with DECIDER_BACKEND=clef for media"
        )


def lexical_rows(encoded: Any) -> list[int]:
    """The sorted unique token ids the head looks up in ``lm_head`` (option spans only)."""
    ids = encoded.input_ids
    rows = {
        ids[pos]
        for question in encoded.questions
        for start, end in question.option_spans
        for pos in range(start, end)
    }
    return sorted(rows)


def compact_input_ids(encoded: Any, rows: Sequence[int]) -> list[int]:
    """``input_ids`` remapped to row indices of the compact ``rows`` table.

    Positions outside every option span are never read by the head's lexical
    lookup; they map to row 0 so the tensor stays a valid index everywhere.
    """
    index = {token: row for row, token in enumerate(rows)}
    return [index.get(token, 0) for token in encoded.input_ids]


def assemble_response(
    request: dict,
    encoded: Any,
    probabilities: Sequence[Sequence[float]],
    answer: Callable[[dict, dict[str, float]], dict],
) -> dict:
    """Cloudflare ``systemone``'s return value (joint_schema_model.py:565-576)."""
    questions = request["questions"]
    answers = {
        question.question_id: answer(
            questions[question.question_id],
            dict(zip(question.option_ids, probs)),
        )
        for question, probs in zip(encoded.questions, probabilities)
    }
    return {
        "model": request["model"],
        "answers": answers,
        "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
    }


def bundle_dir(model_id: str, revision: str | None, hf_home: str) -> str:
    """A REAL directory (``snapshot_download(local_dir=...)``) for one repo@revision.

    Not the HF cache's ``snapshots/`` tree, whose files are symlinks into
    ``blobs/`` — bundle loaders that refuse symlinked files would reject it.
    Lives under ``HF_HOME`` so the compose volume keeps it across restarts.
    """
    name = _UNSAFE.sub("_", model_id.replace("/", "--"))
    rev = _UNSAFE.sub("_", revision or "main")
    return os.path.join(hf_home, BUNDLE_SUBDIR, f"{name}@{rev}")


def parse_knobs(env: Mapping[str, str]) -> tuple[int, float]:
    """``(max_model_len, gpu_mem_util)`` from env; ``ValueError`` names the bad key."""
    raw_len = env.get("DECIDER_MAX_MODEL_LEN", "").strip() or str(DEFAULT_MAX_MODEL_LEN)
    raw_util = env.get("DECIDER_GPU_MEM_UTIL", "").strip() or str(DEFAULT_GPU_MEM_UTIL)
    try:
        max_len = int(raw_len)
    except ValueError:
        raise ValueError(f"DECIDER_MAX_MODEL_LEN must be an integer; got {raw_len!r}") from None
    try:
        util = float(raw_util)
    except ValueError:
        raise ValueError(f"DECIDER_GPU_MEM_UTIL must be a number; got {raw_util!r}") from None
    if max_len <= 0:
        raise ValueError(f"DECIDER_MAX_MODEL_LEN must be positive; got {max_len}")
    if not 0 < util < 1:
        raise ValueError(f"DECIDER_GPU_MEM_UTIL must be in (0, 1); got {util}")
    return max_len, util


def parse_kv_bytes(env: Mapping[str, str]) -> int | None:
    """``DECIDER_KV_CACHE_MEMORY_BYTES`` (a positive integer) or None; ``ValueError`` if bad."""
    raw = env.get("DECIDER_KV_CACHE_MEMORY_BYTES", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"DECIDER_KV_CACHE_MEMORY_BYTES must be an integer number of bytes; got {raw!r}"
        ) from None
    if value <= 0:
        raise ValueError(f"DECIDER_KV_CACHE_MEMORY_BYTES must be positive; got {value}")
    return value


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------


class ClefVllmBackend:
    """``DECIDER_BACKEND=clef-vllm``. ``engine_factory`` is the test seam."""

    name = NAME

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        device: str,
        *,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_mem_util: float = DEFAULT_GPU_MEM_UTIL,
        hf_home: str = DEFAULT_HF_HOME,
        attention_backend: str | None = None,
        kv_cache_memory_bytes: int | None = None,
        engine_factory: Callable[["ClefVllmBackend"], Engine] | None = None,
    ):
        self.model_id, self.revision, self.device = model_id, revision, device
        self.max_model_len, self.gpu_mem_util, self.hf_home = max_model_len, gpu_mem_util, hf_home
        # None = vLLM picks; the Thor (sm_110) needs TRITON_ATTN for pooling (#105).
        self.attention_backend = attention_backend
        # None = vLLM profiles free memory; a fixed size skips that profile, which a
        # unified-memory box can trip by releasing memory mid-profile (Thor, 2026-10-14).
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self._engine_factory = engine_factory or _VllmClefEngine
        self._engine: Engine | None = None
        # One forward at a time, like the clef/strands backends.
        self._lock = threading.Lock()

    def load(self) -> None:
        self._engine = self._engine_factory(self)

    def decide(self, body: bytes) -> dict:
        try:
            record = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"request body is not JSON: {exc}") from exc
        validate_request(record)
        engine = self._engine
        if engine is None:  # pragma: no cover — Sidecar answers 503 before load
            raise RuntimeError("engine not loaded")
        encoded = engine.encode(record)  # ValueError (schema too long) -> 422
        with self._lock:
            probabilities = engine.score(encoded)
        return assemble_response(record, encoded, probabilities, engine.systemone_answer)


# ---------------------------------------------------------------------------
# the real engine (vllm + torch; sidecar image only)
# ---------------------------------------------------------------------------


class _VllmClefEngine:  # pragma: no cover — needs vllm, torch and a GPU
    def __init__(self, cfg: ClefVllmBackend):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.config import PoolerConfig

        self._torch = torch
        self.device = torch.device(cfg.device)
        self.max_length = cfg.max_model_len
        path = self._bundle(cfg)
        self._jsm = _import_joint_schema_model(path)
        # A local directory, already fetched at the pinned DECIDER_REVISION.
        self.tokenizer = AutoTokenizer.from_pretrained(path)  # nosec B615

        # The backbone first. task=token_embed + ALL pooling returns one hidden-state
        # row per prompt token; use_activation=False (per request, below) keeps
        # them un-normalised. language_model_only skips the ViT (text-only).
        self.llm = LLM(
            model=path,
            runner="pooling",
            convert="embed",
            pooler_config=PoolerConfig(task="token_embed", tok_pooling_type="ALL"),
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_mem_util,
            enable_prefix_caching=False,
            language_model_only=True,
            **({"attention_backend": cfg.attention_backend} if cfg.attention_backend else {}),
            **(
                {"kv_cache_memory_bytes": cfg.kv_cache_memory_bytes}
                if cfg.kv_cache_memory_bytes
                else {}
            ),
        )

        # The head goes on the device only AFTER the engine is up, so this
        # process has not initialised CUDA before vLLM starts its engine core.
        head_config = json.loads(_read(os.path.join(path, "joint_head_config.json")))
        head = self._jsm.JointSchemaHead(**head_config)
        head.load_state_dict(load_file(os.path.join(path, "joint_head.safetensors")), strict=True)
        self.head = head.to(device=self.device, dtype=torch.bfloat16).eval()
        self._lm_head = _LmHeadRows(path)

    @staticmethod
    def _bundle(cfg: ClefVllmBackend) -> str:
        if os.path.isdir(cfg.model_id):
            return cfg.model_id
        from huggingface_hub import snapshot_download

        target = bundle_dir(cfg.model_id, cfg.revision, os.environ.get("HF_HOME", cfg.hf_home))
        return snapshot_download(cfg.model_id, revision=cfg.revision, local_dir=target)

    def encode(self, record: dict) -> Any:
        return self._jsm.encode_record(self.tokenizer, record, max_length=self.max_length)

    def systemone_answer(self, question: dict, probabilities: dict[str, float]) -> dict:
        return self._jsm.systemone_answer(question, probabilities)

    def score(self, encoded: Any) -> list[list[float]]:
        from vllm import PoolingParams
        from vllm.inputs import TokensPrompt

        torch = self._torch
        ids = list(encoded.input_ids)
        out = self.llm.encode(
            [TokensPrompt(prompt_token_ids=ids)],
            PoolingParams(use_activation=False),
            pooling_task="token_embed",
            use_tqdm=False,
        )[0]
        hidden = out.outputs.data
        if hidden.shape[0] != len(ids):
            raise RuntimeError(
                f"pooling runner returned {hidden.shape[0]} hidden rows for {len(ids)} tokens"
            )
        rows = lexical_rows(encoded)
        with torch.inference_mode():
            weight = self._lm_head.rows(rows).to(device=self.device, dtype=torch.bfloat16)
            logits = self.head(
                hidden.to(device=self.device, dtype=torch.bfloat16).unsqueeze(0),
                torch.tensor([compact_input_ids(encoded, rows)], device=self.device),
                torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
                [encoded],
                weight,
            )[0]
            return [q.float().softmax(-1).tolist() for q in logits]


class _LmHeadRows:  # pragma: no cover — needs torch + safetensors
    """Lazy, cached rows of the checkpoint's untied ``lm_head.weight``."""

    KEY = "lm_head.weight"
    #: Bound on cached rows (~10 KB each at hidden 5120 bf16): ~640 MB worst case.
    MAX_CACHED = 65536

    def __init__(self, path: str):
        from safetensors import safe_open

        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            shard = json.loads(_read(index))["weight_map"][self.KEY]
        else:
            shard = "model.safetensors"
        self._file = safe_open(os.path.join(path, shard), framework="pt", device="cpu")
        self._slice = self._file.get_slice(self.KEY)
        self._cache: dict[int, Any] = {}

    def rows(self, token_ids: Sequence[int]) -> Any:
        import torch

        if len(self._cache) > self.MAX_CACHED:
            self._cache.clear()
        for token in token_ids:
            if token not in self._cache:
                self._cache[token] = self._slice[token : token + 1][0]
        return torch.stack([self._cache[token] for token in token_ids])


def _read(path: str) -> str:  # pragma: no cover
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _import_joint_schema_model(path: str) -> Any:  # pragma: no cover — imports torch
    """Cloudflare's ``joint_schema_model.py`` from the served bundle, by file path."""
    spec = importlib.util.spec_from_file_location(
        "joint_schema_model", os.path.join(path, "joint_schema_model.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{path} ships no joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: its frozen dataclasses (with postponed annotations)
    # look themselves up in sys.modules — measured: AttributeError otherwise.
    sys.modules["joint_schema_model"] = module
    spec.loader.exec_module(module)
    return module
