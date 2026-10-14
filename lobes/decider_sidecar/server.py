"""Decider sidecar — one decision checkpoint behind ``POST /v1/systemone``.

Runs ONLY in the decider container (templates/fleet/Dockerfile.decider). The
offline dev/CI env has no torch/transformers/GPU, so the real model loaders and
the FastAPI shell are ``# pragma: no cover``; the HTTP decisions are the pure
:class:`Sidecar` below, tested with a fake backend
(``tests/test_decider_sidecar.py``).

HTTP contract
-------------
GET  /health          503 ``{"status":"loading"}`` until the backend has loaded,
                      then 200 ``{"status":"ok","backend":...,"model":...}``.
POST /v1/systemone    the request body bytes go to the backend UNCHANGED and the
                      backend's native response comes back UNCHANGED (Clef's and
                      strands-decider's own shapes — lobes never reshapes it).
                      A backend ``ValueError`` is a caller error: 422.
anything else         404 (known path, wrong method: 405).

No auth here by design: the container publishes no host port and the gateway is
the only way in. The server binds 0.0.0.0 INSIDE the container; exposure is
the compose file's decision.

Configuration (env)
-------------------
DECIDER_BACKEND   ``clef`` | ``strands`` | ``clef-vllm`` (anything else is a startup error)
DECIDER_MODEL     checkpoint: HF repo id or a local path (never hardcoded here)
DECIDER_REVISION  optional HF revision, every backend (snapshot_download; ``strands``
                  resolves it itself, since its loader would fetch the repo HEAD)
DECIDER_DEVICE    torch device, default ``cuda``
DECIDER_PORT      listen port, default 8000
DECIDER_MAX_MODEL_LEN / DECIDER_GPU_MEM_UTIL
                  ``clef-vllm`` only: the vLLM engine's context and GPU share
                  (defaults 16384 — Cloudflare's ``max_length`` — and a DECLARED
                  0.5); see :mod:`lobes.decider_sidecar.clef_vllm`.
DECIDER_ATTENTION_BACKEND
                  ``clef-vllm`` only, optional: vLLM's ``attention_backend``
                  (e.g. ``TRITON_ATTN``); unset lets vLLM choose.
DECIDER_KV_CACHE_MEMORY_BYTES
                  ``clef-vllm`` only, optional: a fixed KV-cache size in bytes.
                  vLLM then skips its free-memory profile, whose assert fails
                  when the page cache is reclaimed mid-boot (Jetson AGX Thor,
                  2026-10-14). Unset keeps the GPU_MEM_UTIL-derived size.

``clef-vllm`` (t21) serves the same Clef response as ``clef`` but runs the
backbone in vLLM's pooling runner (``Dockerfile.decider-vllm``, pinned
vllm/vllm-openai v0.31.0) and Cloudflare's joint head in bf16 beside it;
text-only. Its design notes live in :mod:`lobes.decider_sidecar.clef_vllm`.

One checkpoint per process (answer to c39/h17)
----------------------------------------------
strands-decider CANNOT serve more than one checkpoint per process. Read from
https://github.com/strands-labs/strands-decider, ``src/strands_decider/server.py``
(branch main, read 2026-10-14): the engine is a module-global ``_engine`` (line
21); ``create_app(checkpoint, ...)`` takes a single checkpoint (line 35) and
assigns that one global (lines 47, 72-73); ``serve`` runs ``uvicorn.run(...,
workers=1)`` with the comment "the model owns the GPU, and forking more would
just duplicate it" (lines 134-135). A second ``create_app`` would overwrite the
global. Clef's loader (``load_release_model(path, device)`` in the checkpoint's
``joint_schema_model.py``) likewise returns one ``(model, processor)`` pair, and
this sidecar holds exactly one. Serving N checkpoints means N sidecars.

The strands backend does NOT run strands-decider's ``serve`` (its HTTP app has
no auth); it calls the in-process loader the same way ``create_app`` does
(lines 72-73: ``StrandsDeciderModel.load`` + ``SystemOneEngine(model,
EngineConfig(...))``) and ``eng.evaluate(SystemOneRequest)`` (lines 97-104),
adding ``latency_ms`` exactly as the upstream handler does. Like upstream, one
engine call runs at a time (a lock), since overlapping calls crash the device.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Mapping, Protocol

from lobes.decider_sidecar import clef_vllm

SYSTEMONE_PATH = "/v1/systemone"
HEALTH_PATH = "/health"
BACKENDS = ("clef", "strands", "clef-vllm")
DEFAULT_PORT = 8000
DEFAULT_DEVICE = "cuda"

Response = tuple[int, bytes, str]
_JSON = "application/json"


class DeciderConfigError(ValueError):
    """The sidecar's environment names no usable backend/checkpoint."""


class Backend(Protocol):
    """One loaded decision checkpoint."""

    name: str
    model_id: str

    def load(self) -> None: ...

    def decide(self, body: bytes) -> bytes | dict: ...


def _json(status: int, payload: Mapping) -> Response:
    return status, json.dumps(payload).encode(), _JSON


class Sidecar:
    """Pure HTTP decisions; the FastAPI shell only moves bytes in and out."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.ready = False

    def load(self) -> None:
        self.backend.load()
        self.ready = True

    def _health(self) -> Response:
        if not self.ready:
            return _json(503, {"status": "loading"})
        return _json(
            200,
            {
                "status": "ok",
                "backend": getattr(self.backend, "name", "?"),
                "model": getattr(self.backend, "model_id", "?"),
            },
        )

    def _systemone(self, body: bytes) -> Response:
        if not self.ready:
            return _json(503, {"status": "loading"})
        try:
            out = self.backend.decide(body)
        except ValueError as exc:
            return _json(422, {"detail": str(exc)})
        if isinstance(out, (bytes, bytearray)):
            return 200, bytes(out), _JSON
        return 200, json.dumps(out).encode(), _JSON

    def handle(self, method: str, path: str, body: bytes) -> Response:
        routes = {HEALTH_PATH: "GET", SYSTEMONE_PATH: "POST"}
        if path not in routes:
            return _json(404, {"detail": "not found"})
        if method != routes[path]:
            return _json(405, {"detail": "method not allowed"})
        return self._health() if path == HEALTH_PATH else self._systemone(body)


# ---------------------------------------------------------------------------
# backends (heavy imports are lazy and live inside load())
# ---------------------------------------------------------------------------


class ClefBackend:  # pragma: no cover — needs torch + a GPU
    name = "clef"

    def __init__(self, model_id: str, revision: str | None, device: str):
        self.model_id, self.revision, self.device = model_id, revision, device
        self._lock = threading.Lock()
        self._model = self._processor = self._systemone = None

    def load(self) -> None:
        import sys

        from huggingface_hub import snapshot_download

        path = (
            self.model_id
            if os.path.isdir(self.model_id)
            else snapshot_download(self.model_id, revision=self.revision)
        )
        sys.path.insert(0, path)  # the checkpoint ships joint_schema_model.py
        from joint_schema_model import load_release_model, systemone

        self._model, self._processor = load_release_model(path, device=self.device)
        self._systemone = systemone

    def decide(self, body: bytes) -> dict:
        try:
            record = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"request body is not JSON: {exc}") from exc
        with self._lock:
            return self._systemone(self._model, self._processor, record)


class StrandsBackend:  # pragma: no cover — needs torch + strands-decider
    name = "strands"

    def __init__(self, model_id: str, revision: str | None, device: str):
        self.model_id, self.revision, self.device = model_id, revision, device
        self._lock = threading.Lock()
        self._engine = self._request_cls = None

    def load(self) -> None:
        from strands_decider.infer import EngineConfig, SystemOneEngine
        from strands_decider.modeling import StrandsDeciderModel, checkpoint_dir
        from strands_decider.schema import SystemOneRequest

        name = os.path.basename(self.model_id.rstrip("/")) or self.model_id
        config = EngineConfig(device=self.device, use_prefix_cache=True, model_name=name)
        # StrandsDeciderModel.load resolves a hub id with no revision (the repo
        # HEAD), so resolve the pinned snapshot here and hand it a local dir. The
        # base model's revision comes from the checkpoint's own provenance.json.
        path = checkpoint_dir(self.model_id, self.revision)
        self._engine = SystemOneEngine(StrandsDeciderModel.load(path), config)
        self._request_cls = SystemOneRequest

    def decide(self, body: bytes) -> dict:
        try:
            request = self._request_cls.model_validate_json(body)
        except ValueError as exc:  # pydantic ValidationError is a ValueError
            raise ValueError(str(exc)) from exc
        started = time.perf_counter()
        with self._lock:
            payload = self._engine.evaluate(request).model_dump()
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return payload


def select_backend(name: str, env: Mapping[str, str]) -> Backend:
    """Build (not load) the backend ``DECIDER_BACKEND`` names."""
    if name not in BACKENDS:
        raise DeciderConfigError(
            f"DECIDER_BACKEND must be one of {', '.join(BACKENDS)}; got {name!r}"
        )
    model = env.get("DECIDER_MODEL", "").strip()
    if not model:
        raise DeciderConfigError("DECIDER_MODEL is required (HF repo id or local path)")
    device = env.get("DECIDER_DEVICE", DEFAULT_DEVICE)
    revision = env.get("DECIDER_REVISION", "").strip() or None
    if name == "clef":
        return ClefBackend(model, revision, device)
    if name == clef_vllm.NAME:
        try:
            max_model_len, gpu_mem_util = clef_vllm.parse_knobs(env)
            kv_bytes = clef_vllm.parse_kv_bytes(env)
        except ValueError as exc:
            raise DeciderConfigError(str(exc)) from exc
        return clef_vllm.ClefVllmBackend(
            model,
            revision,
            device,
            max_model_len=max_model_len,
            gpu_mem_util=gpu_mem_util,
            hf_home=env.get("HF_HOME", "").strip() or clef_vllm.DEFAULT_HF_HOME,
            attention_backend=env.get("DECIDER_ATTENTION_BACKEND", "").strip() or None,
            kv_cache_memory_bytes=kv_bytes,
        )
    return StrandsBackend(model, revision, device)


# ---------------------------------------------------------------------------
# FastAPI shell (sidecar container only)
# ---------------------------------------------------------------------------


def build_app(sidecar: Sidecar):  # pragma: no cover — needs the [decider-sidecar] extra
    import threading as _t

    from fastapi import FastAPI, Request
    from fastapi import Response as HttpResponse

    app = FastAPI(title="lobes decider-sidecar", version="1")

    @app.on_event("startup")
    def _start_load() -> None:
        # Load off the event loop so /health answers 503 while weights load.
        _t.Thread(target=sidecar.load, daemon=True).start()

    async def _dispatch(request: Request) -> HttpResponse:
        status, body, ctype = sidecar.handle(request.method, request.url.path, await request.body())
        return HttpResponse(content=body, status_code=status, media_type=ctype)

    app.add_route(HEALTH_PATH, _dispatch, methods=["GET", "POST"])
    app.add_route(SYSTEMONE_PATH, _dispatch, methods=["GET", "POST"])
    return app


def main() -> None:  # pragma: no cover
    import uvicorn

    env: Mapping[str, Any] = os.environ
    sidecar = Sidecar(select_backend(env.get("DECIDER_BACKEND", ""), env))
    port = int(env.get("DECIDER_PORT", DEFAULT_PORT))
    # 0.0.0.0 is container-internal; the compose file decides exposure.
    uvicorn.run(build_app(sidecar), host="0.0.0.0", port=port, log_level="info")  # nosec B104


if __name__ == "__main__":  # pragma: no cover
    main()
