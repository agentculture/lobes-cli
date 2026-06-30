"""Tests for ``lobes status --pressure`` and the no-train boundary guard (t6/t7, #68/#69).

Acceptance contract (three groups):
1. ``status --pressure --json`` emits exactly 5 top-level keys with a nested
   ``pressure`` dict; the tier/model/mode/reason values follow from
   ``decide()`` for the fixed monkeypatched sample, expressed in the new
   **main/minor/multimodal** vocabulary (issue #69 / t6 seam).
2. The command is strictly read-only: ``compose_down`` and
   ``compose_up_detached`` must never be called.
3. No ``train`` / ``finetune`` / ``fine-tune`` verb is registered in the CLI
   parser (LoRA-training is an explicit non-goal — the boundary is enforced
   here at the parser level).

The seam (t6): under degraded pressure the only cheaper target is ``minor``; a
``main`` (or ``multimodal``) request downgrades to ``minor``. Below the degraded
floor nothing is downgraded — there is no intermediate rung, because
``multimodal`` is a *different capability*, not a cheaper version of ``main``.
"""

from __future__ import annotations

import argparse
import json

from lobes.cli import _build_parser, main
from lobes.runtime import _compose
from lobes.runtime import _pressure as _pressure_mod

# Tier model IDs — mirrors test_catalog_tiers.py constants so that any catalog
# change that renames an ID also breaks *this* test (intentional coupling).
_MINOR_ID = "Qwen/Qwen3.5-4B"  # minor tier (degraded floor)
_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"  # main tier (full ceiling)


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
# JSON shape — swap=80 → minor / degraded / pressure
# ---------------------------------------------------------------------------


def test_status_pressure_json_high_swap_degraded(capsys, monkeypatch) -> None:
    """swap=80 > 75 → degraded → tier=minor, mode=degraded, reason=pressure."""
    _set_pressure(monkeypatch, swap=80.0, iowait=5.0)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) == {"tier", "model", "mode", "reason", "pressure"}
    assert payload["tier"] == "minor"
    assert payload["model"] == _MINOR_ID
    assert payload["mode"] == "degraded"
    assert payload["reason"] == "pressure"
    assert payload["pressure"] == {"swap_used_percent": 80.0, "iowait_percent": 5.0}


# ---------------------------------------------------------------------------
# JSON shape — swap=10, iowait=5 → main / warm / default
# ---------------------------------------------------------------------------


def test_status_pressure_json_low_pressure_warm(capsys, monkeypatch) -> None:
    """swap=10, iowait=5 — no thresholds fire → tier=main, warm, default."""
    _set_pressure(monkeypatch, swap=10.0, iowait=5.0)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) == {"tier", "model", "mode", "reason", "pressure"}
    assert payload["tier"] == "main"
    assert payload["model"] == _PRIMARY_ID
    assert payload["mode"] == "warm"
    assert payload["reason"] == "default"
    assert payload["pressure"] == {"swap_used_percent": 10.0, "iowait_percent": 5.0}


# ---------------------------------------------------------------------------
# JSON shape — swap=60 → main / warm / default (seam: no intermediate rung)
# ---------------------------------------------------------------------------


def test_status_pressure_json_mid_swap_not_downgraded(capsys, monkeypatch) -> None:
    """swap=60 > 50 but <= 75 → NOT degraded → tier=main (full), warm, default.

    Migrated from the OLD linear behaviour (swap=60 → normal/constrained). The
    seam resolution collapses the intermediate band: ``multimodal`` is not a
    cheaper rung, so the only downgrade is to ``minor`` and only under the
    degraded floor (swap > 75). Below it, ``main`` is served unconstrained.
    """
    _set_pressure(monkeypatch, swap=60.0, iowait=5.0)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) == {"tier", "model", "mode", "reason", "pressure"}
    assert payload["tier"] == "main"
    assert payload["model"] == _PRIMARY_ID
    assert payload["mode"] == "warm"
    assert payload["reason"] == "default"
    assert payload["pressure"] == {"swap_used_percent": 60.0, "iowait_percent": 5.0}


# ---------------------------------------------------------------------------
# JSON shape — iowait=60 → minor / degraded / pressure
# ---------------------------------------------------------------------------


def test_status_pressure_json_high_iowait_degraded(capsys, monkeypatch) -> None:
    """iowait=60 > 50 → degraded → tier=minor, mode=degraded, reason=pressure."""
    _set_pressure(monkeypatch, swap=10.0, iowait=60.0)
    rc = main(["status", "--pressure", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier"] == "minor"
    assert payload["model"] == _MINOR_ID
    assert payload["mode"] == "degraded"
    assert payload["reason"] == "pressure"


# ---------------------------------------------------------------------------
# Human-readable (non-JSON) form
# ---------------------------------------------------------------------------


def test_status_pressure_text_output(capsys, monkeypatch) -> None:
    """Non-JSON output must print tier / model / mode / reason legibly to stdout."""
    _set_pressure(monkeypatch, swap=60.0, iowait=5.0)
    rc = main(["status", "--pressure"])
    assert rc == 0
    out = capsys.readouterr().out
    # All key labels must appear.
    assert "tier" in out
    assert "model" in out
    assert "mode" in out
    assert "reason" in out
    # Expected values for swap=60 (not degraded): main / warm / default.
    assert "main" in out
    assert "warm" in out
    assert "default" in out
    # Pressure numbers must appear.
    assert "60" in out  # swap
    assert "5" in out  # iowait


def test_status_pressure_text_output_degraded_shows_minor(capsys, monkeypatch) -> None:
    """Under degraded pressure the text form reports the minor floor."""
    _set_pressure(monkeypatch, swap=80.0, iowait=5.0)
    rc = main(["status", "--pressure"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "minor" in out
    assert "degraded" in out
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
