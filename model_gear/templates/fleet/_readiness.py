"""Readiness decision logic for the Parakeet ``/v1/health/ready`` probe.

Backs the cheap health probe described in issue #39 (decision c16): report
``200 ready`` only when the ASR model is loaded AND the CUDA context is live —
never unconditionally, as the old liveness-only handler did.

"Cheap" means: model loaded flag + a trivial CUDA tensor op.  No
``model.transcribe`` call, no real audio.  This keeps probe overhead ~1 ms
instead of seconds.

This is a VENDORED COPY of ``model_gear/realtime/_readiness.py`` placed in the
Parakeet container build context (``model_gear/templates/fleet/``).  The
Dockerfile COPYs it next to ``listen_server.py`` so the container can import it
as a top-level module without needing the ``model_gear`` wheel.

IMPORTANT: keep this file in sync with the canonical copy at
``model_gear/realtime/_readiness.py`` — cite-don't-import convention.
"""

from __future__ import annotations


def evaluate_readiness(model_loaded: bool, cuda_ok: bool) -> tuple[int, dict]:
    """Return an ``(http_status, body)`` pair reflecting real ASR readiness.

    Parameters
    ----------
    model_loaded:
        ``True`` when the NeMo ASR model object has been loaded and is not
        ``None``; ``False`` during startup or after a load failure.
    cuda_ok:
        ``True`` when a trivial CUDA tensor op (e.g. ``torch.zeros(1,
        device="cuda"); torch.cuda.synchronize()``) completes without
        exception; ``False`` on any CUDA error (unknown error, OOM, …).

    Returns
    -------
    tuple[int, dict]
        ``(200, {"status": "ready"})`` when both checks pass.
        ``(503, {"status": "not_ready", "reason": <str>})`` otherwise —
        Docker healthcheck treats non-2xx as failing, so the container will
        not be reported healthy until the model AND CUDA are actually live.
    """
    if not model_loaded:
        return 503, {"status": "not_ready", "reason": "model not loaded"}
    if not cuda_ok:
        return 503, {"status": "not_ready", "reason": "CUDA not available"}
    return 200, {"status": "ready"}
