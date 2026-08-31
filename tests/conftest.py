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
from lobes.runtime import _compose, _detect, _health


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


@pytest.fixture(autouse=True)
def _pin_unknown_card(monkeypatch) -> None:
    """Pin detection to an UNKNOWN card so no test inherits the host's real GPU.

    ``detect_card`` probes the live host (nvidia-smi, /proc/meminfo, ...), so a
    suite whose result depends on which physical machine runs it is broken —
    the same reason ``offline_runtime`` above neutralises the other probes.
    An unrecognised card is the honest ``resolved="unknown"`` result
    (``DetectedCard.is_known`` is False); the raw facts are left ``None``
    since they are incidental here. A test that wants a specific card
    overrides this by re-monkeypatching ``_detect.detect_card`` itself, same
    as ``tests/test_init.py``'s ``_pin_spark_detection`` does.

    Only the bare no-arg call is neutralised — that is the one path that
    touches the real host. Calls that inject their own probe functions
    (``tests/test_detect.py``, ``tests/test_variation.py``) are already
    deterministic and are delegated to the real implementation.
    """
    card = _detect.DetectedCard(
        resolved=_detect.UNKNOWN,
        device_name=None,
        compute_capability=None,
        total_memory_gb=None,
        hostname=None,
        device_tree_model=None,
        sources={},
    )
    real_detect_card = _detect.detect_card

    def _detect_card(*args, **kwargs):
        if not args and not kwargs:
            return card
        return real_detect_card(*args, **kwargs)

    monkeypatch.setattr(_detect, "detect_card", _detect_card)
