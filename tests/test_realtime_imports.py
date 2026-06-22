"""Import-isolation guard: the base wheel + gateway must stay stdlib-only.

The gateway container (and the base ``lobes-cli`` install) never pull the
``[realtime]`` or ``[chatterbox]`` extras, so importing
:mod:`lobes.realtime.app` / ``tts_client`` / ``chatterbox_server``
(fastapi / httpx / torch) would crash them. These tests fail fast if that
boundary is ever crossed.

``chatterbox_server`` guards its fastapi/uvicorn/anyio imports with a try/except
and sets ``_FASTAPI_AVAILABLE = False`` when they are absent — so the module
itself imports fine in the offline CI env (the routes are just not registered).
The stdlib PCM16 helper in the same file is still importable and testable.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import lobes.gateway


def test_stdlib_realtime_helpers_import_without_the_extra() -> None:
    # The offline test env has no fastapi/httpx/torch — these must import anyway.
    for name in (
        "lobes.realtime._settings",
        "lobes.realtime.protocol",
        "lobes.realtime.audio_facade",
    ):
        importlib.import_module(name)


def test_chatterbox_server_imports_without_fastapi() -> None:
    # chatterbox_server guards [chatterbox] extras behind try/except.
    # It must be importable offline (only the routes won't be registered).
    mod = importlib.import_module("lobes.realtime.chatterbox_server")
    # The stdlib PCM16 helper is always available.
    assert callable(mod.float_tensor_to_pcm16)


def test_gateway_source_never_imports_realtime() -> None:
    gw_dir = Path(lobes.gateway.__file__).parent
    offenders = [
        py.name for py in gw_dir.rglob("*.py") if "lobes.realtime" in py.read_text(encoding="utf-8")
    ]
    assert not offenders, f"gateway must not import lobes.realtime: {offenders}"


def test_base_package_imports_without_the_extra() -> None:
    # `import lobes` must not transitively pull fastapi/torch.
    mod = importlib.import_module("lobes")
    assert hasattr(mod, "__version__")
