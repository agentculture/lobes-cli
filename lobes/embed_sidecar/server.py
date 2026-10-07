"""EmbeddingGemma 2 sidecar — an OpenAI-shaped ``/v1/embeddings`` HTTP server.

Runs ONLY in the embed sidecar container (the ``[embed-sidecar]`` extra:
fastapi, uvicorn; torch + sentence-transformers >= 6.1.0 are installed by that
container's Dockerfile, never by the base wheel). The offline dev/CI env has
none of those and no GPU, so the app shell and the real model load are thin
``# pragma: no cover`` wrappers. Every decision — request parsing, prompt
resolution, MRL truncation + re-normalization, the vector guard, the response
body — is a pure function below, tested with an injected fake encoder
(``tests/test_embed_sidecar.py``).

HTTP contract
-------------
GET  /health          → 503 ``{"status":"loading"}`` until the model is loaded;
                        200 ``{"status":"ok", "model", "dtype",
                        "modalities_loaded"}`` once ready.
POST /v1/embeddings   → the vLLM ``/v1/embeddings`` extension shape. Exactly one of:

  * ``input`` — a str or a non-empty list of str (one embedding per string);
  * ``messages`` — a chat-style list whose ``content`` is a str or a list of
    parts ``{"type":"text","text":…}``, ``{"type":"image_url","image_url":{"url":…}}``,
    ``{"type":"video_url","video_url":{"url":…}}``,
    ``{"type":"audio_url","audio_url":{"url":…}}`` or
    ``{"type":"input_audio","input_audio":{"data":<b64>,"format":"wav"}}``.
    All parts across all messages form ONE input → ONE embedding.

  Optional: ``prompt_name`` (one of the checkpoint's named prompts, e.g.
  ``SearchQuery`` / ``Document``; the default is none — the card's
  ``default_prompt_name`` is null), ``dimensions`` (MRL: 128/256/512/768,
  default 768 — the vector is truncated, then re-normalized to unit length),
  ``encoding_format`` (``float`` default, or ``base64`` little-endian float32).

  The response is the OpenAI embeddings shape plus top-level ``model``,
  ``prompt_name``, ``prompt``, ``dimensions``, ``modalities_loaded`` and
  ``modalities`` (the modalities this request actually used). ``usage`` token
  counts are reported as 0: the sidecar does not count tokens.

Design decisions
----------------
* **Media URLs: ``data:`` only.** An ``http(s)://`` URL is refused with 400
  ``remote_url_unsupported`` — the sidecar never fetches. Fetching would make a
  GPU sidecar an SSRF primitive inside the fleet network and add a timeout /
  size policy nobody asked for; the client inlines the bytes instead.
* **Never caption.** A non-text part is decoded to bytes and handed to the
  encoder tagged with its own modality; no code path turns it into text. A part
  whose modality is not loaded (``EMBED_MODALITIES``) is refused with 400
  ``modality_not_loaded`` — never dropped, never encoded as its text siblings.
* **One part per modality per input.** sentence-transformers keys a
  multimodal input by modality, so a second image in one input would lose its
  position in the sequence; such an input is refused rather than reordered.
* **float16 is refused at load.** The checkpoint returns NaN or silently
  degraded embeddings in float16 without raising; bf16 (default) or fp32 only.
* **A degenerate vector is an error.** NaN, ±inf or an all-zero vector (before
  or after MRL truncation) answers 500 ``degenerate_embedding``, never a 200.

Environment
-----------
``EMBED_MODEL_ID`` (default ``google/embeddinggemma-2``; a fine-tune instance
points it at a local checkpoint path), ``EMBED_SERVED_NAME`` (the identity
reported in ``model`` / ``/health``, e.g. ``local:my-tune``; default
``EMBED_MODEL_ID`` — one process serves one checkpoint), ``EMBED_MODALITIES``
(comma list, default ``text``; ``text`` is mandatory), ``EMBED_DTYPE``
(``bf16`` default, ``fp32``), ``EMBED_HOST`` / ``EMBED_PORT`` (0.0.0.0:8000).
"""

from __future__ import annotations

import base64
import logging
import math
import os
import struct
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/embeddinggemma-2"
NATIVE_DIM = 768
MRL_DIMENSIONS = (128, 256, 512, 768)
KNOWN_MODALITIES = frozenset({"text", "image", "video", "audio"})

_DTYPES = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp32": "float32", "float32": "float32"}
_FLOAT16_SPELLINGS = frozenset({"fp16", "float16", "half", "torch.float16"})
_AUDIO_FORMATS = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
    "m4a": "audio/mp4",
}
# part type → (modality, key holding {"url": …}); input_audio is handled apart.
_URL_PARTS = {
    "image_url": ("image", "image_url"),
    "video_url": ("video", "video_url"),
    "audio_url": ("audio", "audio_url"),
}


class EmbedError(Exception):
    """A refusal with an HTTP status and a stable machine-readable code."""

    def __init__(self, message: str, code: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status

    def body(self) -> dict:
        kind = "invalid_request_error" if self.status < 500 else "server_error"
        return {"error": {"message": self.message, "type": kind, "code": self.code}}


@dataclass(frozen=True)
class Settings:
    model_id: str
    modalities: frozenset
    dtype: str
    served_name: str = ""  # reported identity; empty means model_id

    @property
    def reported_model(self) -> str:
        return self.served_name or self.model_id


@dataclass(frozen=True)
class Part:
    """One input part. ``value`` is ``str`` for text and raw ``bytes`` for media."""

    modality: str
    value: Any
    mime: str | None = None


@dataclass(frozen=True)
class EmbedInput:
    parts: tuple


@dataclass(frozen=True)
class EmbedRequest:
    inputs: tuple
    prompt_name: str | None
    dimensions: int
    encoding_format: str


class Encoder(Protocol):
    """What the server needs from a model: its named prompts and ``encode``."""

    prompts: Mapping[str, str]

    def encode(
        self, inputs: Sequence[EmbedInput], prompt_name: str | None
    ) -> Sequence[Sequence[float]]: ...


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def resolve_dtype(value: str | None) -> str:
    """Map an ``EMBED_DTYPE`` spelling to a torch dtype name; refuse float16."""
    key = (value or "bf16").strip().lower()
    if key in _FLOAT16_SPELLINGS:
        raise EmbedError(
            "EMBED_DTYPE=float16 is refused: EmbeddingGemma 2 returns NaN or silently "
            "degraded embeddings in float16; use bf16 (default) or fp32",
            "unsupported_dtype",
        )
    if key not in _DTYPES:
        raise EmbedError(f"unknown EMBED_DTYPE {value!r} (bf16 or fp32)", "unsupported_dtype")
    return _DTYPES[key]


def parse_modalities(value: str | None) -> frozenset:
    names = frozenset(n.strip().lower() for n in (value or "text").split(",") if n.strip())
    unknown = sorted(names - KNOWN_MODALITIES)
    if unknown:
        raise EmbedError(f"unknown EMBED_MODALITIES {unknown}", "unsupported_modality")
    if "text" not in names:
        raise EmbedError("EMBED_MODALITIES must include text", "unsupported_modality")
    return names


def encoder_config_kwargs(modalities: frozenset) -> dict:
    """The model card's selective-encoder load for a modality set.

    EmbeddingGemma 2's card ("Selective Encoder Loading") disables unused
    towers through ``SentenceTransformer(config_kwargs=...)``: text only ->
    ``{"vision_config": None, "audio_config": None}`` (270M), text+image ->
    ``{"audio_config": None}`` (440M), text+audio -> ``{"vision_config":
    None}`` (570M), full -> ``{}`` (740M). Image and video share the vision
    tower; code is text.
    """
    kwargs: dict = {}
    if not modalities & {"image", "video"}:
        kwargs["vision_config"] = None
    if "audio" not in modalities:
        kwargs["audio_config"] = None
    return kwargs


def load_settings(env: Mapping[str, str]) -> Settings:
    """Read and validate the sidecar's env. Raises :class:`EmbedError` at load."""
    return Settings(
        model_id=env.get("EMBED_MODEL_ID") or DEFAULT_MODEL_ID,
        modalities=parse_modalities(env.get("EMBED_MODALITIES")),
        dtype=resolve_dtype(env.get("EMBED_DTYPE")),
        served_name=(env.get("EMBED_SERVED_NAME") or "").strip(),
    )


def health_status(loaded: bool, settings: Settings, error: str = "") -> tuple[int, dict]:
    """``/health``: 200 ok, 503 while loading, 500 once the load has FAILED.

    A failed load is terminal and distinct from "loading" -- the process also
    exits non-zero (see ``_load_encoder``) so the container's restart policy
    retries it visibly instead of sitting at "loading" forever.
    """
    if error:
        return 500, {"status": "failed", "error": error}
    if not loaded:
        return 503, {"status": "loading"}
    return 200, {
        "status": "ok",
        "model": settings.reported_model,
        "dtype": settings.dtype,
        "modalities_loaded": sorted(settings.modalities),
    }


# --------------------------------------------------------------------------
# request parsing
# --------------------------------------------------------------------------


def _bad(message: str, code: str = "invalid_request") -> EmbedError:
    return EmbedError(message, code, 400)


def parse_dimensions(value: Any) -> int:
    if value is None:
        return NATIVE_DIM
    if isinstance(value, bool) or not isinstance(value, int) or value not in MRL_DIMENSIONS:
        raise _bad(f"dimensions must be one of {list(MRL_DIMENSIONS)}", "unsupported_dimensions")
    return value


def parse_encoding_format(value: Any) -> str:
    fmt = "float" if value is None else value
    if fmt not in ("float", "base64"):
        raise _bad("encoding_format must be 'float' or 'base64'", "unsupported_encoding_format")
    return fmt


def resolve_prompt(name: Any, prompts: Mapping[str, str]) -> tuple[str | None, str | None]:
    """Return ``(prompt_name, prompt_text)``; ``(None, None)`` when no prompt."""
    if name is None:
        return None, None
    if not isinstance(name, str) or name not in prompts:
        raise _bad(f"unknown prompt_name {name!r}; known: {sorted(prompts)}", "unknown_prompt_name")
    return name, prompts[name]


def _decode_b64(data: Any, what: str) -> bytes:
    if not isinstance(data, str) or not data:
        raise _bad(f"{what}: empty or non-string base64 payload", "invalid_media")
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as exc:  # binascii.Error is a ValueError subclass
        raise _bad(f"{what}: invalid base64 ({exc})", "invalid_media") from exc


def decode_data_url(url: Any, modality: str) -> tuple[bytes, str]:
    """Decode a ``data:<modality>/*;base64,…`` URL. Remote URLs are refused."""
    if not isinstance(url, str) or not url:
        raise _bad(f"{modality} part needs a url", "invalid_media")
    if url.startswith(("http://", "https://")):
        raise _bad(
            "remote media URLs are not fetched; inline the bytes as a data: URL",
            "remote_url_unsupported",
        )
    header, sep, payload = url.partition(",")
    if not (header.startswith("data:") and header.endswith(";base64") and sep):
        raise _bad(f"{modality} url must be a base64 data: URL", "invalid_media")
    mime = header[len("data:") : -len(";base64")].lower()
    if not mime.startswith(f"{modality}/"):
        raise _bad(f"{modality} part carries mime {mime!r}", "invalid_media")
    return _decode_b64(payload, f"{modality} data URL"), mime


def _url_part(part: Mapping, modality: str, key: str) -> Part:
    holder = part.get(key)
    if not isinstance(holder, Mapping):
        raise _bad(f"{key} must be an object with a url", "invalid_media")
    data, mime = decode_data_url(holder.get("url"), modality)
    return Part(modality, data, mime)


def _input_audio_part(part: Mapping) -> Part:
    holder = part.get("input_audio")
    if not isinstance(holder, Mapping):
        raise _bad("input_audio must be an object", "invalid_media")
    fmt = holder.get("format")
    if fmt not in _AUDIO_FORMATS:
        raise _bad(f"input_audio.format must be one of {sorted(_AUDIO_FORMATS)}", "invalid_media")
    return Part("audio", _decode_b64(holder.get("data"), "input_audio"), _AUDIO_FORMATS[fmt])


def _text_part(part: Mapping) -> Part:
    text = part.get("text")
    if not isinstance(text, str) or not text:
        raise _bad("text part needs a non-empty text", "invalid_request")
    return Part("text", text)


def parse_part(part: Any) -> Part:
    """Parse one content part, keeping its modality. Never converts media to text."""
    if not isinstance(part, Mapping):
        raise _bad("each content part must be an object")
    kind = part.get("type")
    if kind == "text":
        return _text_part(part)
    if kind == "input_audio":
        return _input_audio_part(part)
    if kind in _URL_PARTS:
        modality, key = _URL_PARTS[kind]
        return _url_part(part, modality, key)
    raise _bad(f"unsupported content part type {kind!r}", "unsupported_part_type")


def _message_parts(message: Any) -> list[Part]:
    if not isinstance(message, Mapping):
        raise _bad("each message must be an object")
    content = message.get("content")
    if isinstance(content, str) and content:
        return [Part("text", content)]
    if not isinstance(content, list) or not content:
        raise _bad("message content must be a non-empty string or list of parts")
    return [parse_part(p) for p in content]


def _check_parts(parts: list[Part], loaded: frozenset) -> None:
    seen: set[str] = set()
    for part in parts:
        if part.modality not in loaded:
            raise _bad(
                f"modality {part.modality!r} is not loaded on this sidecar "
                f"(EMBED_MODALITIES={','.join(sorted(loaded))})",
                "modality_not_loaded",
            )
        if part.modality in seen:
            raise _bad(f"more than one {part.modality} part in one input", "duplicate_modality")
        seen.add(part.modality)


def _merge_text_parts(parts: list[Part]) -> list[Part]:
    """Join every text part into ONE, in order, at the first text part's position.

    A chat-style body routinely carries several text parts (a system instruction
    plus the user's text, or a caption after an image). sentence-transformers
    keys a multimodal input by modality, so text is concatenated (newline-
    joined) rather than refused; a SECOND media part of one modality is still
    refused, because its order could not be preserved.
    """
    texts = [p.value for p in parts if p.modality == "text"]
    if len(texts) < 2:
        return parts
    merged = Part("text", "\n".join(texts))
    out: list[Part] = []
    for part in parts:
        if part.modality != "text":
            out.append(part)
        elif merged not in out:
            out.append(merged)
    return out


def parse_messages(messages: Any, loaded: frozenset) -> tuple[EmbedInput, ...]:
    if not isinstance(messages, list) or not messages:
        raise _bad("messages must be a non-empty list")
    parts = _merge_text_parts([p for message in messages for p in _message_parts(message)])
    _check_parts(parts, loaded)
    return (EmbedInput(parts=tuple(parts)),)


def parse_text_input(value: Any) -> tuple[EmbedInput, ...]:
    texts = [value] if isinstance(value, str) else value
    if not isinstance(texts, list) or not texts:
        raise _bad("input must be a non-empty string or list of strings")
    if not all(isinstance(t, str) and t for t in texts):
        raise _bad("every input item must be a non-empty string")
    return tuple(EmbedInput(parts=(Part("text", t),)) for t in texts)


def _parse_inputs(body: Mapping, loaded: frozenset) -> tuple[EmbedInput, ...]:
    has_input, has_messages = "input" in body, "messages" in body
    if has_input == has_messages:
        raise _bad("send exactly one of 'input' or 'messages'")
    if has_input:
        return parse_text_input(body["input"])
    return parse_messages(body["messages"], loaded)


def parse_request(body: Any, loaded: frozenset) -> EmbedRequest:
    if not isinstance(body, Mapping):
        raise _bad("request body must be a JSON object")
    prompt_name = body.get("prompt_name")
    if prompt_name is not None and not isinstance(prompt_name, str):
        raise _bad("prompt_name must be a string", "unknown_prompt_name")
    return EmbedRequest(
        inputs=_parse_inputs(body, loaded),
        prompt_name=prompt_name,
        dimensions=parse_dimensions(body.get("dimensions")),
        encoding_format=parse_encoding_format(body.get("encoding_format")),
    )


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------


def _degenerate(message: str) -> EmbedError:
    return EmbedError(message, "degenerate_embedding", 500)


def guard_vector(vec: Sequence[float]) -> float:
    """Return the vector's L2 norm; raise on NaN, ±inf or an all-zero vector."""
    if not all(math.isfinite(x) for x in vec):
        raise _degenerate("encoder returned a non-finite (NaN/inf) embedding")
    norm = math.sqrt(math.fsum(x * x for x in vec))
    if norm <= 0.0 or not math.isfinite(norm):
        raise _degenerate("encoder returned an all-zero embedding")
    return norm


def truncate_and_normalize(vec: Sequence[float], dims: int) -> list[float]:
    """MRL: keep the leading ``dims`` components, then re-normalize to unit length."""
    head = [float(x) for x in vec[:dims]]
    norm = guard_vector(head)
    return [x / norm for x in head]


def _finish_vector(raw: Any, dims: int) -> list[float]:
    vec = [float(x) for x in raw]
    if len(vec) != NATIVE_DIM:
        raise EmbedError(
            f"encoder returned a {len(vec)}-d vector, expected {NATIVE_DIM}",
            "bad_encoder_output",
            500,
        )
    guard_vector(vec)
    return truncate_and_normalize(vec, dims)


def _encode(encoder: Encoder, request: EmbedRequest) -> list[list[float]]:
    try:
        raw = list(encoder.encode(list(request.inputs), request.prompt_name))
    # Any model failure is a 500, never a 200.
    except Exception as exc:  # noqa: BLE001
        log.exception("embed encode failed")
        raise EmbedError(f"encode failed: {type(exc).__name__}", "encode_failed", 500) from exc
    if len(raw) != len(request.inputs):
        raise EmbedError("encoder returned the wrong number of vectors", "bad_encoder_output", 500)
    return [_finish_vector(v, request.dimensions) for v in raw]


# --------------------------------------------------------------------------
# response
# --------------------------------------------------------------------------


def _format_embedding(vec: list[float], encoding_format: str) -> Any:
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")
    return vec


def _used_modalities(inputs: Sequence[EmbedInput]) -> list[str]:
    used: list[str] = []
    for item in inputs:
        used.extend(p.modality for p in item.parts if p.modality not in used)
    return used


def build_response(
    request: EmbedRequest,
    vectors: list[list[float]],
    settings: Settings,
    prompt: tuple[str | None, str | None],
) -> dict:
    data = [
        {
            "object": "embedding",
            "index": i,
            "embedding": _format_embedding(v, request.encoding_format),
        }
        for i, v in enumerate(vectors)
    ]
    return {
        "object": "list",
        "data": data,
        "model": settings.reported_model,
        "prompt_name": prompt[0],
        "prompt": prompt[1],
        "dimensions": request.dimensions,
        "modalities_loaded": sorted(settings.modalities),
        "modalities": _used_modalities(request.inputs),
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def handle_embeddings(body: Any, encoder: Encoder, settings: Settings) -> tuple[int, dict]:
    """The whole ``POST /v1/embeddings`` decision: ``(http_status, json_body)``."""
    try:
        request = parse_request(body, settings.modalities)
        prompt = resolve_prompt(request.prompt_name, encoder.prompts)
        vectors = _encode(encoder, request)
    except EmbedError as err:
        return err.status, err.body()
    return 200, build_response(request, vectors, settings, prompt)


# --------------------------------------------------------------------------
# real encoder + FastAPI shell (sidecar container only)
# --------------------------------------------------------------------------


class SentenceTransformerEncoder:  # pragma: no cover — needs torch + a GPU
    """Adapter from :class:`EmbedInput` to sentence-transformers >= 6.1.0.

    Text-only inputs go in as plain strings; a multimodal input becomes one
    ``{"text":…, "image":…, "audio":…, "video":…}`` dict in part order (the
    order is why >= 6.1.0 is required). Image bytes are opened with PIL; audio
    and video bytes go to torchcodec decoders. Nothing is captioned.

    The modality gate is enforced per request by :func:`parse_request`; the
    weight-level modular load skips the unused towers via the card's
    ``config_kwargs`` (:func:`encoder_config_kwargs`).
    """

    def __init__(self, settings: Settings) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        dtype = getattr(torch, settings.dtype)
        self.model = SentenceTransformer(
            settings.model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"torch_dtype": dtype},
            config_kwargs=encoder_config_kwargs(settings.modalities),
        )
        loaded = next(self.model.parameters()).dtype
        if loaded == torch.float16:
            raise EmbedError("model loaded in float16; refusing to serve", "unsupported_dtype")
        self.prompts = dict(self.model.prompts or {})

    @staticmethod
    def _media(part: Part) -> Any:
        if part.modality == "text":
            return part.value
        if part.modality == "image":
            import io

            from PIL import Image

            return Image.open(io.BytesIO(part.value)).convert("RGB")
        from torchcodec.decoders import AudioDecoder, VideoDecoder

        decoder = AudioDecoder if part.modality == "audio" else VideoDecoder
        return decoder(part.value)

    def _to_st(self, item: EmbedInput) -> Any:
        if len(item.parts) == 1 and item.parts[0].modality == "text":
            return item.parts[0].value
        return {p.modality: self._media(p) for p in item.parts}

    def encode(self, inputs, prompt_name):
        batch = [self._to_st(i) for i in inputs]
        out = self.model.encode(batch, prompt_name=prompt_name, convert_to_numpy=True)
        return [row.tolist() for row in out]


_encoder: Any = None
_load_error = ""
_encoder_lock = threading.Lock()


def _load_encoder(settings: Settings) -> None:  # pragma: no cover
    global _encoder, _load_error
    with _encoder_lock:
        if _encoder is not None:
            return
        log.info(
            "[embed] loading %s (%s, %s)",
            settings.model_id,
            settings.dtype,
            ",".join(sorted(settings.modalities)),
        )
        try:
            _encoder = SentenceTransformerEncoder(settings)
        except Exception as exc:  # noqa: BLE001
            # OOM, a refused dtype, a bad checkpoint, a missing codec: never sit at
            # "loading" forever -- report it, then exit so restart: policies apply.
            _load_error = f"{type(exc).__name__}: {exc}"
            log.exception("[embed] model load FAILED")
            os._exit(1)
        log.info("[embed] ready")


def build_app(settings: Settings):  # pragma: no cover — needs the [embed-sidecar] extra
    import anyio
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="lobes embed-sidecar", version="1")

    @app.on_event("startup")
    async def _warm() -> None:
        threading.Thread(target=_load_encoder, args=(settings,), daemon=True).start()

    @app.get("/health")
    async def health() -> JSONResponse:
        code, body = health_status(_encoder is not None, settings, _load_error)
        return JSONResponse(status_code=code, content=body)

    # ``from __future__ import annotations`` turns ``Request`` into a string that
    # FastAPI resolves against MODULE globals, where this lazily imported class is
    # absent; it then reads ``request`` as a required QUERY parameter (422 on every
    # call, found on the live Orin 2026-10-07). Bind the real classes explicitly.
    async def embeddings(request):
        if _encoder is None:
            return JSONResponse(status_code=503, content={"error": {"message": "loading"}})
        try:
            body = await request.json()
        except ValueError:
            body = None
        code, payload = await anyio.to_thread.run_sync(handle_embeddings, body, _encoder, settings)
        return JSONResponse(status_code=code, content=payload)

    embeddings.__annotations__ = {"request": Request, "return": JSONResponse}
    app.post("/v1/embeddings")(embeddings)

    return app


def main() -> None:  # pragma: no cover
    """Process entrypoint — ``python -m lobes.embed_sidecar.server``."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    settings = load_settings(os.environ)  # float16 refused here, before any weight loads
    host = os.environ.get("EMBED_HOST", "0.0.0.0")  # nosec B104
    port = int(os.environ.get("EMBED_PORT", "8000"))
    uvicorn.run(build_app(settings), host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
