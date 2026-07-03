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
_MINOR_ID = "Qwen/Qwen3.5-4B"  # minor tier (the servable floor under pressure)
_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"  # main tier (full)

_KEYS = {"mode", "shed", "servable_tier", "model", "reason", "retry_after", "pressure"}


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
    assert payload["servable_tier"] == "minor"
    assert payload["model"] == _MINOR_ID
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
    assert payload["servable_tier"] == "minor"
    assert payload["model"] == _MINOR_ID
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
    assert "minor" in out
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
