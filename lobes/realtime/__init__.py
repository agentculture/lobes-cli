"""lobes realtime — the audio + speech surface of the fleet.

Vendored from the ``realtime-api`` sibling (cite-don't-import: lobes owns
these copies). Runs ONLY in the ``realtime`` fleet container, which installs the
``[realtime]`` extra (fastapi, uvicorn, httpx, numpy, scipy, silero-vad, torch);
the base wheel and the stdlib gateway never import this package.

Public surface:

- :mod:`lobes.realtime._settings` — env → :class:`Settings` (stdlib-only).
- :mod:`lobes.realtime.audio_facade` — pure codec / request-parse helpers
  for the OpenAI ``/v1/audio/*`` endpoints (stdlib-only; unit-testable offline).
- :mod:`lobes.realtime.app` — the FastAPI app (needs the extra).

Keep this ``__init__`` import-light: ``import lobes.realtime._settings`` must
succeed with the standard library alone (the import-isolation guard depends on it).
"""

from __future__ import annotations

__all__ = ["_settings", "audio_facade"]
