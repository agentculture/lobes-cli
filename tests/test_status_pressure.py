"""Tests for ``lobes status --pressure`` and the no-train boundary guard (#68/#69/#85).

Acceptance contract (three groups):
1. ``status --pressure --json`` emits the busy/serve decision a full-tier
   (``main``) request would receive right now — box-level ``mode``
   (``warm``/``busy``), whether it is ``shed`` (HTTP 429), the ``servable_tier``
   that still answers, the ``model`` serving it, ``reason``, ``retry_after`` and
   the nested ``pressure`` sample. Values follow from ``decide()`` for the fixed
   monkeypatched sample.
2. The command is strictly read-only: ``compose_down`` and
   ``compose_up_detached`` must never be called.
3. No ``train`` / ``finetune`` / ``fine-tune`` verb is registered in the CLI
   parser (LoRA-training is an explicit non-goal — the boundary is enforced
   here at the parser level).

The #85 change: under swap/iowait pressure the gateway no longer degrades a
``main``/``senses`` request onto a different model — it **sheds** it with 429
(busy). ``minor`` is the floor and is always served (never shed). ``status
--pressure`` reports that decision without issuing a live request, and because
it calls the same ``decide()`` handle_post consults, the report matches what a
live request would receive.
"""

from __future__ import annotations

import argparse
import json

from lobes.cli import _build_parser, main
from lobes.gateway._pressure_policy import BUSY_RETRY_AFTER_SECONDS
from lobes.runtime import _compose
from lobes.runtime import _pressure as _pressure_mod

# Tier model IDs — mirrors test_catalog_tiers.py constants so that any catalog
# change that renames an ID also breaks *this* test (intentional coupling).
# The cheap tier's gear — the servable floor under pressure. The `hand` lobe
# replaced Qwen/Qwen3.5-4B here; `minor`/`cheap` remain accepted request
# spellings but the tier RESOLVES to this model and REPORTS itself as "hand".
_HAND_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
_PRIMARY_ID = "unsloth/Qwen3.8-27B-NVFP4"  # main tier (full)

_KEYS = {
    "mode",
    "shed",
    "servable_tier",
    "model",
    "reason",
    "retry_after",
    "pressure",
    # lane-safety t14: which signal sheds, and the thermal reading vs its band.
    "shed_signal",
    "thermal",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _set_pressure(monkeypatch, *, swap: float, iowait: float) -> None:
    """Monkeypatch sample_pressure to return fixed values (no /proc reads)."""
    monkeypatch.setattr(
        _pressure_mod,
        "sample_pressure",
        lambda: {"swap_used_percent": swap, "iowait_percent": iowait},
    )


# ---------------------------------------------------------------------------
# JSON shape — swap=80 → busy: full-tier request shed, minor is servable
# ---------------------------------------------------------------------------


def test_status_pressure_json_high_swap_busy(capsys, monkeypatch) -> None:
    """swap=80 > 75 → busy → a main request is shed; minor is the servable floor."""
    _set_pressure(monkeypatch, swap=80.0, iowait=5.0)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) == _KEYS
    assert payload["mode"] == "busy"
    assert payload["shed"] is True
    assert payload["servable_tier"] == "hand"
    assert payload["model"] == _HAND_ID
    assert payload["reason"] == "pressure"
    assert payload["retry_after"] == BUSY_RETRY_AFTER_SECONDS
    assert payload["pressure"] == {"swap_used_percent": 80.0, "iowait_percent": 5.0}


# ---------------------------------------------------------------------------
# JSON shape — swap=10, iowait=5 → warm: nothing shed, main served
# ---------------------------------------------------------------------------


def test_status_pressure_json_low_pressure_warm(capsys, monkeypatch) -> None:
    """swap=10, iowait=5 — no thresholds fire → warm, nothing shed, main served."""
    _set_pressure(monkeypatch, swap=10.0, iowait=5.0)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) == _KEYS
    assert payload["mode"] == "warm"
    assert payload["shed"] is False
    assert payload["servable_tier"] == "main"
    assert payload["model"] == _PRIMARY_ID
    assert payload["reason"] == "default"
    assert payload["retry_after"] is None
    assert payload["pressure"] == {"swap_used_percent": 10.0, "iowait_percent": 5.0}


# ---------------------------------------------------------------------------
# JSON shape — swap=60 → warm (strictly-greater threshold; no intermediate rung)
# ---------------------------------------------------------------------------


def test_status_pressure_json_mid_swap_still_warm(capsys, monkeypatch) -> None:
    """swap=60 > 50 but <= 75 → NOT busy → warm, main served, nothing shed.

    There is no intermediate band: only the degraded floor (swap > 75) triggers
    busy. Below it, ``main`` is served unconstrained.
    """
    _set_pressure(monkeypatch, swap=60.0, iowait=5.0)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) == _KEYS
    assert payload["mode"] == "warm"
    assert payload["shed"] is False
    assert payload["servable_tier"] == "main"
    assert payload["model"] == _PRIMARY_ID
    assert payload["reason"] == "default"
    assert payload["retry_after"] is None


# ---------------------------------------------------------------------------
# JSON shape — iowait=60 → busy
# ---------------------------------------------------------------------------


def test_status_pressure_json_high_iowait_busy(capsys, monkeypatch) -> None:
    """iowait=60 > 50 → busy → main request shed, minor servable."""
    _set_pressure(monkeypatch, swap=10.0, iowait=60.0)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "busy"
    assert payload["shed"] is True
    assert payload["servable_tier"] == "hand"
    assert payload["model"] == _HAND_ID
    assert payload["reason"] == "pressure"
    assert payload["retry_after"] == BUSY_RETRY_AFTER_SECONDS


# ---------------------------------------------------------------------------
# Human-readable (non-JSON) form
# ---------------------------------------------------------------------------


def test_status_pressure_text_output_warm(capsys, monkeypatch) -> None:
    """Non-JSON output prints mode / servable / model / reason legibly to stdout."""
    _set_pressure(monkeypatch, swap=60.0, iowait=5.0)
    rc = main(["status", "--pressure"])
    assert rc == 0
    out = capsys.readouterr().out
    # Key labels must appear.
    assert "mode" in out
    assert "servable" in out
    assert "model" in out
    assert "reason" in out
    # Expected values for swap=60 (warm): warm / main / default.
    assert "warm" in out
    assert "main" in out
    assert "default" in out
    # Pressure numbers must appear.
    assert "60" in out  # swap
    assert "5" in out  # iowait


def test_status_pressure_text_output_busy_shows_shed(capsys, monkeypatch) -> None:
    """Under pressure the text form reports busy + shed + the minor floor."""
    _set_pressure(monkeypatch, swap=80.0, iowait=5.0)
    rc = main(["status", "--pressure"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "busy" in out
    assert "429" in out  # the shed line names the busy status
    assert "hand" in out
    assert "pressure" in out


# ---------------------------------------------------------------------------
# Read-only contract — no compose_down / compose_up_detached calls
# ---------------------------------------------------------------------------


def test_status_pressure_does_not_call_compose_down(capsys, monkeypatch) -> None:
    """status --pressure must NEVER invoke compose_down (it is read-only)."""
    _set_pressure(monkeypatch, swap=10.0, iowait=5.0)

    def _boom(d: object) -> None:
        raise AssertionError("compose_down was called — status --pressure must be read-only")

    monkeypatch.setattr(_compose, "compose_down", _boom)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0  # _boom was not triggered


def test_status_pressure_does_not_call_compose_up(capsys, monkeypatch) -> None:
    """status --pressure must NEVER invoke compose_up_detached (it is read-only)."""
    _set_pressure(monkeypatch, swap=10.0, iowait=5.0)

    def _boom(d: object) -> None:
        raise AssertionError("compose_up_detached was called — status --pressure must be read-only")

    monkeypatch.setattr(_compose, "compose_up_detached", _boom)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0  # _boom was not triggered


# ---------------------------------------------------------------------------
# No-train boundary guard
# ---------------------------------------------------------------------------


def test_no_train_verb_registered() -> None:
    """The CLI must NOT register any train / finetune / fine-tune verb.

    lobes is a serve-only tool — LoRA / fine-tuning tooling belongs in a
    separate package (issue #68 non-goal boundary).  This test inspects the
    argparse subparser choices directly so that any future accidental
    registration of a training verb causes an immediate, descriptive failure.
    """
    parser = _build_parser()
    # Locate the _SubParsersAction that owns all registered verb names.
    sub_action = next(
        (a for a in parser._actions if isinstance(a, argparse._SubParsersAction)),
        None,
    )
    assert sub_action is not None, "No _SubParsersAction found in the CLI parser"
    registered: set[str] = set(sub_action.choices.keys())

    forbidden = {"train", "finetune", "fine-tune", "fine_tune"}
    violations = forbidden & registered
    assert not violations, (
        f"Boundary violation: training verbs {sorted(violations)!r} are registered "
        f"in the CLI parser — lobes is serve-only (issue #68); LoRA / fine-tuning "
        f"tooling belongs in a separate package."
    )


# ---------------------------------------------------------------------------
# Thermal (lane-safety t14) — the reading, its band, and the signal
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from lobes.gateway import _pressure_policy as _P  # noqa: E402


@pytest.fixture
def spark_band(monkeypatch):
    monkeypatch.setattr(_P, "THERMAL_SHED_C", 91.0)
    monkeypatch.setattr(_P, "THERMAL_RESUME_C", 86.0)


def _set_thermal(monkeypatch, t) -> None:
    monkeypatch.setattr(
        _pressure_mod,
        "sample_pressure",
        lambda: {"swap_used_percent": 1.0, "iowait_percent": 1.0, "thermal_c": t},
    )


def test_status_pressure_json_hot_sheds_with_thermal_signal(capsys, monkeypatch, spark_band):
    _set_thermal(monkeypatch, 93.5)
    assert main(["status", "--pressure", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == _KEYS
    assert payload["shed"] is True
    assert payload["mode"] == "busy"
    assert payload["shed_signal"] == "thermal"
    assert payload["servable_tier"] == "hand"
    assert payload["thermal"] == {
        "c": 93.5,
        "shed_c": 91.0,
        "resume_c": 86.0,
        "band": "above_shed",
        "shedding": True,
    }


def test_status_pressure_json_hysteresis_band_is_reported_honestly(capsys, monkeypatch, spark_band):
    # A one-shot CLI has no latch history: inside the band it cannot know
    # whether the gateway is still shedding, so it says so (shedding: null).
    _set_thermal(monkeypatch, 88.0)
    assert main(["status", "--pressure", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shed"] is False
    assert payload["shed_signal"] == ""
    assert payload["thermal"]["band"] == "hysteresis"
    assert payload["thermal"]["shedding"] is None


def test_status_pressure_json_unknown_reading_never_sheds(capsys, monkeypatch, spark_band):
    _set_thermal(monkeypatch, None)
    assert main(["status", "--pressure", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shed"] is False
    assert payload["thermal"]["c"] is None
    assert payload["thermal"]["band"] == "unknown"
    assert payload["thermal"]["shedding"] is False


def test_status_pressure_json_band_off(capsys, monkeypatch):
    monkeypatch.setattr(_P, "THERMAL_SHED_C", None)
    monkeypatch.setattr(_P, "THERMAL_RESUME_C", None)
    _set_thermal(monkeypatch, 99.0)
    assert main(["status", "--pressure", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shed"] is False
    assert payload["thermal"]["band"] == "off"
    assert payload["thermal"]["shed_c"] is None


def test_status_pressure_text_shows_thermal(capsys, monkeypatch, spark_band):
    _set_thermal(monkeypatch, 93.5)
    assert main(["status", "--pressure"]) == 0
    out = capsys.readouterr().out
    assert "thermal: 93.5 C" in out
    assert "above shed 91.0 C" in out
    assert "signal:  thermal" in out


def test_status_pressure_text_thermal_na(capsys, monkeypatch):
    _set_pressure(monkeypatch, swap=10.0, iowait=5.0)  # a sampler with no thermal key
    assert main(["status", "--pressure"]) == 0
    assert "thermal: n/a" in capsys.readouterr().out
