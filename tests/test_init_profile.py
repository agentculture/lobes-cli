"""Tests for ``lobes init``'s per-machine-profile resolution (t4; the UNKNOWN
branch is replaced by t14 — warn + serve the conservative 'base' profile
instead of refusing).

Detection is injected via ``monkeypatch.setattr(_detect, "detect_card", ...)``
(the offline-probe idiom this repo already uses, see ``tests/conftest.py``) so
these never touch real hardware — the real-detection path is exercised
separately by a live smoke run on an actual box (not a pytest test).
"""

from __future__ import annotations

import json

from lobes.cli import _runtime_ops, main
from lobes.runtime import _detect, _env


def _fake_card(
    resolved: str,
    *,
    device_name: str | None = "NVIDIA Thor",
    compute_capability: str | None = "sm_110",
    total_memory_gb: float | None = 125.9,
) -> _detect.DetectedCard:
    return _detect.DetectedCard(
        resolved=resolved,
        device_name=device_name,
        compute_capability=compute_capability,
        total_memory_gb=total_memory_gb,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )


def _patch_detect(monkeypatch, card: _detect.DetectedCard) -> None:
    monkeypatch.setattr(_detect, "detect_card", lambda: card)


# --- acceptance 1: bare init picks the right profile with no flags ---------


def test_bare_init_picks_spark_profile_from_injected_facts(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(
        monkeypatch, _fake_card("spark", device_name="NVIDIA GB10", compute_capability="sm_121")
    )
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply"])
    assert rc == 0
    env = (target / ".env").read_text()
    assert "PRIMARY_KV_CACHE_DTYPE=fp8" in env  # spark's own default, unchanged
    assert "EMBED_ATTENTION_BACKEND=auto" in env
    out = capsys.readouterr().out
    assert ">> profile: spark" in out


def test_bare_init_picks_thor_profile_from_injected_facts(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply"])
    assert rc == 0
    env = (target / ".env").read_text()
    # thor's 4 machine-derived divergences from spark's template defaults.
    assert "PRIMARY_KV_CACHE_DTYPE=auto" in env
    assert "EMBED_ATTENTION_BACKEND=TRITON_ATTN" in env
    assert "RERANK_ATTENTION_BACKEND=TRITON_ATTN" in env
    assert "RERANK_ENFORCE_EAGER=--enforce-eager" in env
    out = capsys.readouterr().out
    assert ">> profile: thor" in out


def test_bare_init_dry_run_names_chosen_profile_and_facts(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target)])
    assert rc == 0
    assert not target.exists()
    out = capsys.readouterr().out
    assert "Profile: thor (auto-detected: device_name='NVIDIA Thor'" in out
    assert "compute_capability='sm_110'" in out
    assert "total_memory_gb=125.9" in out


def test_bare_init_apply_json_reports_profile(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "thor"
    assert payload["profile_forced"] is False
    assert payload["detected_card"] == "thor"


# --- acceptance 2: --profile overrides detection, warns on a mismatch ------


def test_explicit_profile_overrides_detection(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--profile", "spark", "--apply"])
    assert rc == 0
    env = (target / ".env").read_text()
    # Spark's own defaults win — none of thor's divergences are applied.
    assert "PRIMARY_KV_CACHE_DTYPE=fp8" in env
    assert "EMBED_ATTENTION_BACKEND=auto" in env
    err = capsys.readouterr().err
    assert "--profile 'spark'" in err
    assert "detected card 'thor'" in err


def test_explicit_profile_matching_detection_warns_not(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--profile", "thor", "--apply"])
    assert rc == 0
    err = capsys.readouterr().err
    assert err == ""  # no mismatch, no warning


def test_explicit_profile_onto_unknown_card_warns_and_proceeds(
    tmp_path, monkeypatch, capsys
) -> None:
    _patch_detect(
        monkeypatch,
        _fake_card(_detect.UNKNOWN, device_name="NVIDIA H100", compute_capability="sm_90"),
    )
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--profile", "spark", "--apply"])
    assert rc == 0
    assert (target / ".env").is_file()
    err = capsys.readouterr().err
    assert "--profile 'spark'" in err
    assert "undetected card" in err
    assert "NVIDIA H100" in err


def test_explicit_unknown_profile_name_still_raises_unknown_profile_error(
    tmp_path, monkeypatch, capsys
) -> None:
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--profile", "not-a-real-profile", "--apply"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown profile" in err


# --- acceptance 3 (t14): UNKNOWN card + no --profile WARNS and serves the
# conservative 'base' profile, instead of refusing (replaces t4's refusal). ---


def test_unknown_card_with_no_profile_warns_and_serves_base(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(
        monkeypatch,
        _fake_card(
            _detect.UNKNOWN,
            device_name="NVIDIA H100",
            compute_capability="sm_90",
            total_memory_gb=80.0,
        ),
    )
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply"])
    assert rc == 0
    assert target.is_dir()  # proceeds — no longer half-scaffolded-and-refused
    captured = capsys.readouterr()
    err = captured.err
    assert "device_name='NVIDIA H100'" in err
    assert "compute_capability='sm_90'" in err
    assert "total_memory_gb=80.0" in err
    assert "--profile" in err
    assert "base" in err  # names the assumption made (the profile it fell back to)
    env = _env.read_env_file(target / ".env")
    assert env["PRIMARY_MODEL"] == "Qwen/Qwen3.5-4B"  # the small generate model, not the 27B
    assert "Qwen3.6-27B" not in env["PRIMARY_MODEL"]
    assert "Qwen3.6-27B" not in env["PRIMARY_SERVED_NAME"]
    assert ">> profile: base" in captured.out


def test_unknown_card_dry_run_also_warns_and_proceeds(tmp_path, monkeypatch, capsys) -> None:
    # The dry-run plan must be honest about what --apply would do.
    _patch_detect(monkeypatch, _fake_card(_detect.UNKNOWN))
    target = tmp_path / "deploy"
    rc = main(["init", str(target)])
    assert rc == 0
    assert not target.exists()  # dry run still writes nothing
    captured = capsys.readouterr()
    assert "--profile" in captured.err
    assert "Profile: base (auto-detected:" in captured.out


def test_unknown_card_never_silently_falls_back_to_spark(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card(_detect.UNKNOWN))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply"])
    assert rc == 0
    env = _env.read_env_file(target / ".env")
    # base's own model, not spark's 27B primary.
    assert env["PRIMARY_MODEL"] == "Qwen/Qwen3.5-4B"
    assert "Qwen3.6-27B" not in env["PRIMARY_MODEL"]
    out = capsys.readouterr().out
    assert ">> profile: base" in out
    assert ">> profile: spark" not in out


def test_unknown_card_base_profile_has_no_27b_anywhere(tmp_path) -> None:
    """Acceptance 1: the base profile's rendered env never names the 27B —
    checked directly against profile_env, independent of the CLI plumbing."""
    from lobes.profiles.loader import resolve_profile
    from lobes.profiles.render import profile_env

    env = profile_env(resolve_profile("base"))
    for value in env.values():
        assert "Qwen3.6-27B" not in value


def test_unknown_card_base_profile_marks_senses_infeasible(tmp_path, monkeypatch) -> None:
    """The 12B multimodal gear is NOT assumed on an unrecognised card."""
    _patch_detect(monkeypatch, _fake_card(_detect.UNKNOWN))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply"])
    assert rc == 0
    env = (target / ".env").read_text()
    assert "MULTIMODAL_FEASIBLE=false" in env


# --- acceptance 4: dry-run-by-default preserved ------------------------------


def test_fleet_dry_run_writes_nothing_even_with_a_resolvable_profile(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target)])
    assert rc == 0
    assert not target.exists()


# --- --single is untouched by profile resolution ----------------------------


def test_single_topology_never_calls_detection(tmp_path, monkeypatch) -> None:
    # --single is the legacy, non-fleet scaffold; profile resolution (and its
    # UNKNOWN-card refusal) must not apply to it at all.
    calls = []
    monkeypatch.setattr(_detect, "detect_card", lambda: calls.append(1) or _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--single", "--apply"])
    assert rc == 0
    assert calls == []


# --- resolve_init_profile (the _runtime_ops glue) directly -----------------


def test_resolve_init_profile_resolves_base_and_warns_on_unknown(tmp_path) -> None:
    # t14: the UNKNOWN branch no longer raises — it resolves the conservative
    # 'base' built-in and returns a warning naming the detected facts and the
    # assumption made, so 'lobes init' can proceed instead of refusing.
    profile, card, warning = _runtime_ops.resolve_init_profile(
        None,
        tmp_path,
        detect_fn=lambda: _fake_card(
            _detect.UNKNOWN,
            device_name="NVIDIA H100",
            compute_capability="sm_90",
            total_memory_gb=80.0,
        ),
    )
    assert profile.name == "base"
    assert card.resolved == _detect.UNKNOWN
    assert warning is not None
    assert "device_name='NVIDIA H100'" in warning
    assert "compute_capability='sm_90'" in warning
    assert "total_memory_gb=80.0" in warning
    assert "base" in warning
    assert "--profile" in warning


def test_resolve_init_profile_returns_no_warning_on_clean_match(tmp_path) -> None:
    profile, card, warning = _runtime_ops.resolve_init_profile(
        None, tmp_path, detect_fn=lambda: _fake_card("spark")
    )
    assert profile.name == "spark"
    assert card.resolved == "spark"
    assert warning is None
