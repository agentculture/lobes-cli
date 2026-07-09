"""Shared fixtures: make external probes deterministic and offline.

Several verbs shell out to docker / nvidia-smi and probe ``/health`` /
``/capabilities``. The autouse fixture below neutralises those so the suite
never depends on a running container, a GPU, the host's ``~/.lobes``, or
whatever happens to already be listening on a guessed port on the machine
running the tests — every probe degrades to its "nothing there" answer. This
is not hypothetical: the reference dev rig has an unrelated daemon bound to
host port 8000 (see ``lobes.roles._gateway_base_url``'s docstring), so a test
that skipped this neutralisation could observe a real, but *wrong*, answer.
Tests that need a deployment scaffold one into a tmp dir and pass
``--compose-dir``; tests that need a genuinely live gateway spin up their own
loopback server on an ephemeral port and explicitly restore the real probe
function (see ``tests/test_cli_capabilities.py``'s fake-gateway test).
"""

from __future__ import annotations

import pytest

from lobes.cli._commands import capabilities as _capabilities
from lobes.runtime import _compose, _health


@pytest.fixture(autouse=True)
def offline_runtime(monkeypatch, tmp_path):
    # docker / nvidia-smi best-effort probes → "not available".
    monkeypatch.setattr(_compose, "_probe", lambda *a, **k: None)
    # /health never responds.
    monkeypatch.setattr(_health, "is_healthy", lambda *a, **k: False)
    # /health's parsed-JSON sibling (issue #99 — `lobes doctor`'s version-skew
    # check) also never responds by default, for the same determinism reason.
    monkeypatch.setattr(_health, "fetch_health", lambda *a, **k: None)
    # `lobes capabilities` / `lobes endpoint` (issue #96, t7) try a live GET
    # /capabilities against the resolved port before falling back to the
    # offline .env-derived registry — neutralise that probe too, for the same
    # reason /health is neutralised above, so the whole suite is deterministic
    # regardless of what is (or isn't) actually listening on the guessed port.
    monkeypatch.setattr(_capabilities, "_fetch_gateway_capabilities", lambda *a, **k: None)
    # No deployment scaffolded by default: point the home at an empty tmp dir.
    monkeypatch.delenv("LOBES_DIR", raising=False)
    monkeypatch.delenv("MODEL_GEAR_DIR", raising=False)  # also clear legacy back-compat var
    empty = tmp_path / "home-lobes"
    monkeypatch.setattr(_compose, "default_deployment_dir", lambda: empty)
