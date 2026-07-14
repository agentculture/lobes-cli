"""Tests for ``lobes init --shape`` (brain-shapes t4).

``--shape`` selects the deployment-shape axis (brain-shapes t1-t3: which roles
a box HOSTS) at scaffold time, composed over whichever per-machine
:class:`~lobes.profiles.schema.Profile` detection/``--profile`` resolves
(issue #110), via :func:`lobes.profiles.shape_render.render_shape` — never a
re-implementation of that composition. Bare ``lobes init`` (no ``--shape``)
resolves the ``machine-as-brain`` identity shape — a strict no-op over the
profile (the invariant ``tests/test_shape_goldens.py`` pins) — so the default
path renders exactly as it did before this flag existed: zero new required
decisions.

Card detection is injected via ``monkeypatch.setattr(_detect, "detect_card",
...)`` — this repo's offline-probe idiom (see ``tests/conftest.py`` and
``tests/test_init_profile.py``) — so none of these touch real hardware.
"""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.profiles.loader import resolve_profile
from lobes.profiles.render import profile_env
from lobes.profiles.shape_render import render_shape
from lobes.profiles.shapes import builtin_shape_names, resolve_shape
from lobes.runtime import _detect, _env


def _fake_card(
    resolved: str,
    *,
    device_name: str | None = "NVIDIA GB10",
    compute_capability: str | None = "sm_121",
    total_memory_gb: float | None = 119.7,
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


# --- acceptance 2: bare init == explicit machine-as-brain, byte-identical ---


def test_bare_init_env_equals_explicit_machine_as_brain_shape(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    bare = tmp_path / "bare"
    explicit = tmp_path / "explicit"
    assert main(["init", str(bare), "--apply"]) == 0
    assert main(["init", "--shape", "machine-as-brain", str(explicit), "--apply"]) == 0
    assert (bare / ".env").read_text() == (explicit / ".env").read_text()
    assert (bare / "docker-compose.yml").read_text() == (
        explicit / "docker-compose.yml"
    ).read_text()


def _numerically_equal(a: str | None, b: str) -> bool:
    """Same tolerant comparison lobes.cli._commands.init._values_equal applies:
    a resolved float restating a template default in fewer digits (0.3 vs the
    shipped 0.30) is the SAME value, not a mismatch."""
    if a == b:
        return True
    try:
        return float(a) == float(b)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def test_bare_init_env_matches_pre_shape_profile_env(tmp_path, monkeypatch) -> None:
    """The rendered .env carries every value plain profile_env(profile) would —
    the pre-change rendering, from before --shape existed at all — modulo the
    pre-existing numeric-formatting tolerance (see _values_equal in init.py)."""
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    env = _env.read_env_file(target / ".env")
    expected = profile_env(resolve_profile("thor"))
    for key, value in expected.items():
        assert _numerically_equal(env.get(key), value), f"{key}: {env.get(key)!r} != {value!r}"


def test_bare_init_dry_run_names_default_shape(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Shape: machine-as-brain" in out
    assert not target.exists()


def test_bare_init_apply_json_reports_default_shape(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape"] == "machine-as-brain"


def test_single_topology_is_untouched_by_shape_machinery(tmp_path, monkeypatch) -> None:
    """--single (no --shape given at all) never resolves a shape or calls
    detection — mirrors test_single_topology_never_calls_detection."""
    calls = []
    monkeypatch.setattr(_detect, "detect_card", lambda: calls.append(1) or _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", "--single", str(target), "--apply"])
    assert rc == 0
    assert calls == []


# --- shape selection changes the rendered .env ------------------------------


def test_shape_spark_lobe_drops_senses_and_reclaims_cortex_budget(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target), "--apply"])
    assert rc == 0
    env = _env.read_env_file(target / ".env")
    assert env["MULTIMODAL_FEASIBLE"] == "false"
    assert env["PRIMARY_GPU_MEM_UTIL"] == "0.44"


def test_shape_thor_lobe_drops_cortex_and_reclaims_senses_budget(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "thor-lobe", str(target), "--apply"])
    assert rc == 0
    env = _env.read_env_file(target / ".env")
    assert env["PRIMARY_FEASIBLE"] == "false"
    assert env["MULTIMODAL_GPU_MEM_UTIL"] == "0.44"


def test_scaffolded_env_matches_render_shape_directly(tmp_path, monkeypatch) -> None:
    """The scaffolded .env carries every key render_shape() itself would produce
    — init never reimplements the shape x profile composition."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    env = _env.read_env_file(target / ".env")
    expected = render_shape(resolve_shape("spark-lobe"), resolve_profile("spark")).env
    for key, value in expected.items():
        assert env.get(key) == value, f"{key}: {env.get(key)!r} != {value!r}"


def test_shape_composes_with_explicit_profile_override(tmp_path, monkeypatch) -> None:
    """--shape composes over WHICHEVER profile is resolved, including a forced
    --profile — the two axes are independent."""
    _patch_detect(monkeypatch, _fake_card("thor"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", "--profile", "spark", str(target), "--apply"])
    assert rc == 0
    env = _env.read_env_file(target / ".env")
    assert env["MULTIMODAL_FEASIBLE"] == "false"
    assert env["PRIMARY_GPU_MEM_UTIL"] == "0.44"
    assert env["PRIMARY_KV_CACHE_DTYPE"] == "fp8"  # spark's own default, not thor's


# --- unknown --shape is a user error naming the valid (sorted) shapes ------


def test_unknown_shape_is_user_error_naming_valid_shapes(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "not-a-real-shape", str(target), "--apply"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown shape" in err
    assert "machine-as-brain, spark-lobe, thor-lobe" in err
    assert not target.exists()  # aborts before any scaffolding


def test_unknown_shape_dry_run_also_errors_before_writing(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "not-a-real-shape", str(target)])
    assert rc == 1
    assert not target.exists()


def test_builtin_shape_names_are_sorted() -> None:
    assert list(builtin_shape_names()) == sorted(builtin_shape_names())
    assert builtin_shape_names() == ("machine-as-brain", "spark-lobe", "thor-lobe")


# --- --shape x --single: user error (shapes are a fleet-scaffold axis) -----


def test_shape_with_single_is_a_user_error(tmp_path, capsys) -> None:
    target = tmp_path / "x"
    rc = main(["init", "--single", "--shape", "spark-lobe", str(target), "--apply"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--shape" in err and "--single" in err
    assert not target.exists()


def test_shape_with_single_errors_even_for_the_default_shape_name(tmp_path, capsys) -> None:
    # The recommendation this encodes: shapes are a fleet-scaffold axis, so
    # ANY explicit --shape (even spelling out the default) conflicts with
    # --single, which never runs profile/shape resolution at all.
    target = tmp_path / "x"
    rc = main(["init", "--single", "--shape", "machine-as-brain", str(target), "--apply"])
    assert rc == 1
    assert not target.exists()


# --- --shape x --audio: harmless / idempotent -------------------------------


def test_shape_with_audio_scaffolds_both_normally(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", "--audio", str(target), "--apply"])
    assert rc == 0
    assert (target / "docker-compose.audio.yml").is_file()
    env = _env.read_env_file(target / ".env")
    assert env["MULTIMODAL_FEASIBLE"] == "false"
    assert env["PRIMARY_GPU_MEM_UTIL"] == "0.44"
    assert "CHATTERBOX_PORT" in env  # audio keys still appended, unaffected by shape


def test_default_shape_with_audio_is_unaffected(tmp_path, monkeypatch) -> None:
    """machine-as-brain (the default) hosts stt/tts too, exactly like every
    built-in shape — --audio remains the sole switch that scaffolds the
    overlay; passing both is harmless and changes nothing about the audio
    scaffold compared to --audio alone."""
    _patch_detect(monkeypatch, _fake_card("spark"))
    with_shape = tmp_path / "with-shape"
    without_shape = tmp_path / "without-shape"
    assert main(["init", "--audio", str(without_shape), "--apply"]) == 0
    assert main(["init", "--shape", "machine-as-brain", "--audio", str(with_shape), "--apply"]) == 0
    assert (with_shape / ".env").read_text() == (without_shape / ".env").read_text()
    assert (with_shape / "docker-compose.audio.yml").read_text() == (
        without_shape / "docker-compose.audio.yml"
    ).read_text()


# --- dry-run: zero bytes on disk, --apply required -------------------------


def test_shape_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target)])
    assert rc == 0
    assert not target.exists()


def test_shape_dry_run_json_reports_shape_and_hosts(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape"] == "spark-lobe"
    assert set(payload["shape_hosts"]) == {"cortex", "embedder", "reranker", "stt", "tts"}
    assert not target.exists()


def test_shape_dry_run_text_names_the_shape(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    rc = main(["init", "--shape", "spark-lobe", str(target)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Shape: spark-lobe" in out
    assert "Re-run with --apply to write." in out


# --- apply requires the flag; re-running with the previous shape restores
# the previous rendering byte-for-byte -------------------------------------


def test_apply_is_required_to_commit_a_shape(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target)]) == 0
    assert not target.exists()
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    assert target.exists()


def test_reapplying_the_same_shape_is_byte_for_byte_idempotent(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    first = (target / ".env").read_text()
    # Switch away to a different shape...
    assert main(["init", "--shape", "machine-as-brain", str(target), "--apply", "--force"]) == 0
    assert (target / ".env").read_text() != first  # sanity: it actually changed
    # ...then switch back to the original shape — byte-for-byte restore.
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply", "--force"]) == 0
    assert (target / ".env").read_text() == first
