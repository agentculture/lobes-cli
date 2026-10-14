"""Decider sidecar, ``clef-vllm`` backend (t21): pure decisions tested with fakes.

No vllm, torch or transformers import anywhere here — the engine (vLLM pooling
runner + Cloudflare's joint head) is replaced by a fake that returns fixed
per-option probabilities, and the response assembly is checked against the
exact shape of Cloudflare's own ``systemone`` (``joint_schema_model.py`` at
Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c, lines 565-576).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from importlib.resources import files

import pytest

from lobes.decider_sidecar import clef_vllm as cv
from lobes.decider_sidecar import server as ds

# --- fakes -------------------------------------------------------------------


@dataclass(frozen=True)
class FakeQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class FakeEncoded:
    input_ids: tuple[int, ...]
    questions: tuple[FakeQuestion, ...]


def fake_systemone_answer(question, probabilities):
    """Stand-in for Cloudflare's ``systemone_answer`` (the engine supplies the real one)."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    choice = max(probabilities, key=probabilities.__getitem__)
    return {"type": question["type"], "choice": choice, "probabilities": probabilities}


class FakeEngine:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.encoded = []
        self.scored = []

    def encode(self, record):
        questions = tuple(
            FakeQuestion(
                question_id=qid,
                question_type=0,
                question_span=(1, 2),
                option_spans=((3, 4),) * len(self.probabilities[qid]),
                option_ids=tuple(self.probabilities[qid]),
            )
            for qid in record["questions"]
        )
        enc = FakeEncoded(input_ids=tuple(range(37)), questions=questions)
        self.encoded.append(record)
        return enc

    def score(self, encoded):
        self.scored.append(encoded)
        return [list(self.probabilities[q.question_id].values()) for q in encoded.questions]

    def systemone_answer(self, question, probabilities):
        return fake_systemone_answer(question, probabilities)


PROBS = {
    "department": {"billing": 0.2, "technical": 0.8},
    "outage": {"true": 0.9, "false": 0.1},
}
REQUEST = {
    "model": "clef",
    "state": "Checkout returns errors.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payments", "technical": "Bugs"},
        },
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}


def _loaded(probabilities=PROBS):
    engine = FakeEngine(probabilities)
    backend = cv.ClefVllmBackend("org/clef-w4a16", "rev1", "cuda", engine_factory=lambda b: engine)
    sidecar = ds.Sidecar(backend)
    return sidecar, backend, engine


# --- backend selection ---------------------------------------------------------


def test_select_backend_picks_clef_vllm_with_model_and_revision():
    backend = ds.select_backend(
        "clef-vllm", {"DECIDER_MODEL": "org/clef-w4a16", "DECIDER_REVISION": "abc"}
    )
    assert isinstance(backend, cv.ClefVllmBackend)
    assert backend.name == "clef-vllm"
    assert backend.model_id == "org/clef-w4a16"
    assert backend.revision == "abc"
    assert "clef-vllm" in ds.BACKENDS


def test_select_backend_clef_vllm_still_requires_model():
    with pytest.raises(ds.DeciderConfigError, match="DECIDER_MODEL"):
        ds.select_backend("clef-vllm", {})


def test_constructing_clef_vllm_imports_no_vllm_or_torch():
    ds.select_backend("clef-vllm", {"DECIDER_MODEL": "a/b"})
    assert "vllm" not in sys.modules
    assert "torch" not in sys.modules


def test_engine_knobs_come_from_env_with_declared_defaults():
    backend = ds.select_backend(
        "clef-vllm",
        {
            "DECIDER_MODEL": "a/b",
            "DECIDER_MAX_MODEL_LEN": "8192",
            "DECIDER_GPU_MEM_UTIL": "0.4",
        },
    )
    assert backend.max_model_len == 8192
    assert backend.gpu_mem_util == 0.4
    default = ds.select_backend("clef-vllm", {"DECIDER_MODEL": "a/b"})
    assert default.max_model_len == cv.DEFAULT_MAX_MODEL_LEN == 16384


def test_attention_backend_is_an_optional_env_knob():
    # sm_110 (Jetson AGX Thor) pooling needs TRITON_ATTN (#105); unset = vLLM's choice.
    forced = ds.select_backend(
        "clef-vllm", {"DECIDER_MODEL": "a/b", "DECIDER_ATTENTION_BACKEND": "TRITON_ATTN"}
    )
    assert forced.attention_backend == "TRITON_ATTN"
    assert ds.select_backend("clef-vllm", {"DECIDER_MODEL": "a/b"}).attention_backend is None


def test_kv_cache_bytes_is_an_optional_env_knob():
    # Unified-memory boxes can release memory mid-profiling and trip vLLM's
    # free-memory assert (Thor, 2026-10-14); a fixed KV size skips that profile.
    fixed = ds.select_backend(
        "clef-vllm", {"DECIDER_MODEL": "a/b", "DECIDER_KV_CACHE_MEMORY_BYTES": "8589934592"}
    )
    assert fixed.kv_cache_memory_bytes == 8589934592
    assert ds.select_backend("clef-vllm", {"DECIDER_MODEL": "a/b"}).kv_cache_memory_bytes is None
    with pytest.raises(ds.DeciderConfigError):
        ds.select_backend(
            "clef-vllm", {"DECIDER_MODEL": "a/b", "DECIDER_KV_CACHE_MEMORY_BYTES": "8G"}
        )


@pytest.mark.parametrize(
    "env",
    [{"DECIDER_MAX_MODEL_LEN": "lots"}, {"DECIDER_GPU_MEM_UTIL": "1.5"}],
)
def test_bad_engine_knobs_are_config_errors(env):
    with pytest.raises(ds.DeciderConfigError):
        ds.select_backend("clef-vllm", {"DECIDER_MODEL": "a/b", **env})


# --- HTTP behaviour through the shared Sidecar --------------------------------------


def test_health_503_until_the_engine_loads():
    sidecar, _, _ = _loaded()
    assert sidecar.handle("GET", "/health", b"")[0] == 503
    sidecar.load()
    status, body, _ = sidecar.handle("GET", "/health", b"")
    assert status == 200
    assert json.loads(body) == {"status": "ok", "backend": "clef-vllm", "model": "org/clef-w4a16"}


def test_response_shape_matches_cloudflare_systemone():
    sidecar, _, engine = _loaded()
    sidecar.load()
    status, body, ctype = sidecar.handle("POST", "/v1/systemone", json.dumps(REQUEST).encode())
    assert status == 200
    assert ctype == "application/json"
    # Cloudflare's systemone (joint_schema_model.py:565-576) returns exactly
    # {model, answers{qid: systemone_answer(question, dict(zip(option_ids, probs)))},
    #  usage{input_tokens: len(input_ids), output_tokens: 0}}.
    expected = {
        "model": "clef",
        "answers": {
            qid: fake_systemone_answer(REQUEST["questions"][qid], PROBS[qid]) for qid in PROBS
        },
        "usage": {"input_tokens": 37, "output_tokens": 0},
    }
    assert json.loads(body) == expected
    assert list(json.loads(body)) == ["model", "answers", "usage"]
    assert len(engine.scored) == 1


def test_answers_follow_encoded_question_order_and_option_ids():
    probs = {"outage": {"true": 0.3, "false": 0.7}, "department": {"technical": 1.0}}
    sidecar, _, _ = _loaded(probs)
    sidecar.load()
    _, body, _ = sidecar.handle("POST", "/v1/systemone", json.dumps(REQUEST).encode())
    answers = json.loads(body)["answers"]
    assert answers["outage"] == {"type": "noul", "noul": 0.3}
    assert answers["department"]["probabilities"] == {"technical": 1.0}


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda r: r.pop("model"), "model and state"),
        (lambda r: r.pop("state"), "model and state"),
        (lambda r: r.update(questions={}), "at least one question"),
        (lambda r: r["questions"]["outage"].update(type="maybe"), "noul, choice, or score"),
        (lambda r: r["questions"]["department"].update(criteria={}), "criteria"),
        (lambda r: r.update(images=["x"]), "clef-vllm"),
        (lambda r: r.update(videos=[[0]]), "clef-vllm"),
    ],
)
def test_invalid_requests_are_422_and_never_reach_the_engine(mutate, fragment):
    record = json.loads(json.dumps(REQUEST))
    mutate(record)
    sidecar, _, engine = _loaded()
    sidecar.load()
    status, body, _ = sidecar.handle("POST", "/v1/systemone", json.dumps(record).encode())
    assert status == 422
    assert fragment in json.loads(body)["detail"]
    assert engine.scored == []


def test_non_json_body_is_422():
    sidecar, _, _ = _loaded()
    sidecar.load()
    assert sidecar.handle("POST", "/v1/systemone", b"not json")[0] == 422


def test_engine_value_error_from_encoding_is_422():
    sidecar, _, engine = _loaded()

    def too_long(record):
        raise ValueError("schema requires 99999 tokens before state; maximum is 16384")

    engine.encode = too_long
    sidecar.load()
    status, body, _ = sidecar.handle("POST", "/v1/systemone", json.dumps(REQUEST).encode())
    assert status == 422
    assert "maximum is 16384" in json.loads(body)["detail"]


# --- pure helpers for the joint head's lexical rows -----------------------------------


def test_lexical_rows_are_the_sorted_unique_option_token_ids():
    enc = FakeEncoded(
        input_ids=(5, 9, 7, 9, 2, 7, 11),
        questions=(
            FakeQuestion("a", 1, (0, 1), ((1, 3), (3, 4)), ("x", "y")),
            FakeQuestion("b", 0, (0, 1), ((5, 7),), ("true",)),
        ),
    )
    assert cv.lexical_rows(enc) == [7, 9, 11]


def test_compact_input_ids_index_the_compact_table_inside_option_spans():
    enc = FakeEncoded(
        input_ids=(5, 9, 7, 9, 2, 7, 11),
        questions=(FakeQuestion("a", 1, (0, 1), ((1, 3), (5, 7)), ("x", "y")),),
    )
    rows = cv.lexical_rows(enc)
    compact = cv.compact_input_ids(enc, rows)
    assert len(compact) == len(enc.input_ids)
    for start, end in enc.questions[0].option_spans:
        for pos in range(start, end):
            # compact[pos] picks the same token id out of the compact table as
            # input_ids[pos] picks out of the full lm_head.
            assert rows[compact[pos]] == enc.input_ids[pos]
    assert all(0 <= c < len(rows) for c in compact)


def test_bundle_dir_is_a_real_directory_not_the_symlinked_hf_cache():
    path = cv.bundle_dir("Cloudflare/clef", "2f3de3dd", "/root/.cache/huggingface")
    assert path.startswith("/root/.cache/huggingface/lobes-decider-bundles/")
    assert "Cloudflare--clef" in path
    assert path.endswith("2f3de3dd")
    assert "snapshots" not in path
    assert cv.bundle_dir("a/b", None, "/h").endswith("a--b@main")


# --- the image variant ------------------------------------------------------------


def _dockerfile() -> str:
    return (
        files("lobes.templates.fleet")
        .joinpath("Dockerfile.decider-vllm")
        .read_text(encoding="utf-8")
    )


def test_dockerfile_pins_vllm_031_by_digest():
    text = _dockerfile()
    froms = [ln for ln in text.splitlines() if ln.startswith(("FROM", "ARG DECIDER_VLLM_BASE"))]
    assert any(
        re.search(r"vllm/vllm-openai@sha256:[0-9a-f]{64}", ln) for ln in froms
    ), "base image must be pinned by digest"
    assert "v0.31.0" in text
    assert "arm64" in text


def test_dockerfile_uses_uv_pip_only_and_publishes_no_port():
    text = _dockerfile()
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    body = "\n".join(code)
    assert "uv pip install --system" in body
    assert not re.search(r"(?<!uv )\bpip3? install", body)
    assert "EXPOSE" not in body
    assert "-p " not in body
    assert 'ENTRYPOINT ["python3", "-m", "lobes.decider_sidecar.server"]' in body
    assert "ENV DECIDER_BACKEND=clef-vllm" in body


def test_materialize_bundle_hard_links_the_cache_blobs_never_copies(tmp_path):
    # t11 (2026-10-14): snapshot_download(local_dir=...) COPIED an 18 GB
    # checkpoint inside the container, and the copy's page cache pushed the
    # engine past its 32g memcg (OOM-killed). The bundle must be real files
    # (not the cache's symlinks) that share the cache blobs' inodes.
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "aaa").write_bytes(b"weights")
    (blobs / "bbb").write_bytes(b"{}")
    snap = tmp_path / "snapshots" / "rev"
    (snap / "sub").mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blobs / "aaa")
    (snap / "sub" / "config.json").symlink_to(blobs / "bbb")
    target = tmp_path / "bundle"

    out = cv.materialize_bundle(str(snap), str(target))

    assert out == str(target)
    for rel, blob in (("model.safetensors", "aaa"), ("sub/config.json", "bbb")):
        made = target / rel
        assert made.is_file()
        assert not made.is_symlink()
        assert made.stat().st_ino == (blobs / blob).stat().st_ino
    # Idempotent: a second call keeps the same links.
    cv.materialize_bundle(str(snap), str(target))
    assert (target / "model.safetensors").stat().st_ino == (blobs / "aaa").stat().st_ino
