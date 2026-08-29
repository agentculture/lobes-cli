"""``lobes init --from-lock`` — restore a committed deployment variation (t7).

The deployment-lock plan's t7. ``--from-lock`` is a distinct **SOURCE**, not a
fourth input to the renderer: init's three existing axes (topology
``--single``/``--audio``, card ``--profile``, shape ``--shape``) all feed
``lobes/profiles/render.py``, and ``--from-lock`` bypasses that path entirely.
That bypass is exactly what makes a restore byte-identical to what the box ran
— and it is also what makes the machine-type guard load-bearing, because
bypassing resolution also bypasses ``_sync_gpu_overrides``' card-driven
correction.

The five acceptance criteria, one test group each:

1. a restored box has a byte-identical compose/override/Dockerfile set, proven
   by diff, with ``.env`` untouched;
2. restoring on a box whose DETECTED card differs from the lock refuses by
   default, and any override is explicit;
3. a lock captured on a csv-mode card restores BOTH GPU overlays; one from a
   devices card restores NEITHER (and scrubs stale ones — the remove-on-mismatch
   behaviour of ``_sync_gpu_overrides`` surviving a lock round-trip);
4. profile/shape resolution is provably not consulted — every entry point is
   patched to raise and the restore still succeeds;
5. dry-run by default, ``--apply`` to write.
"""

from __future__ import annotations

import filecmp
import json
import shutil
from pathlib import Path

import pytest

from lobes.cli import main
from lobes.cli._commands import init as init_cmd
from lobes.cli._errors import EXIT_USER_ERROR
from lobes.runtime import _compose, _detect, _lock

from .test_init_shape import _fake_card, _patch_detect

# --- helpers ----------------------------------------------------------------

#: Card facts that make `_fake_card` plausible for each built-in we exercise.
_CARD_FACTS = {
    "spark": {"device_name": "NVIDIA GB10", "compute_capability": "sm_121"},
    "orin": {"device_name": "Orin", "compute_capability": "sm_87", "total_memory_gb": 61.3},
}


def _card(resolved: str) -> _detect.DetectedCard:
    return _fake_card(resolved, **_CARD_FACTS.get(resolved, {}))


def _committed_names(box: Path) -> list[str]:
    """The deployment files a variation commits verbatim: compose + Dockerfiles.

    Derived from what is actually on disk rather than hand-listed, so a card
    that generates an extra overlay (the csv-mode GPU pair) contributes it
    without this helper knowing which card it was.
    """
    return sorted(
        path.name
        for path in box.iterdir()
        if path.is_file()
        and (path.name.startswith("docker-compose") or path.name.startswith("Dockerfile"))
    )


def _make_variation(
    tmp_path: Path,
    monkeypatch,
    *,
    card: str = "spark",
    shape: str | None = None,
    name: str = "variation",
    lobes_version: str | None = "0.67.0",
) -> tuple[Path, Path]:
    """Scaffold a real deployment, then commit it as a variation folder + lock.

    Returns ``(variation_dir, box)`` — the committed source a restore reads, and
    the deployment it was captured from (the byte-for-byte comparison target).
    """
    _patch_detect(monkeypatch, _card(card))
    box = tmp_path / f"box-{name}"
    argv = ["init", str(box), "--apply"]
    if shape is not None:
        argv[1:1] = ["--shape", shape]
    assert main(argv) == 0

    source = tmp_path / name
    source.mkdir()
    files: dict[str, str] = {}
    for fname in _committed_names(box):
        shutil.copyfile(box / fname, source / fname)
        files[fname] = _lock.file_digest(box / fname)
    lock = _lock.capture_lock(
        box,
        variation=card,
        profile=card,
        shape=shape or init_cmd.DEFAULT_SHAPE,
        # A real capture always records the version it was taken at; the
        # buildability preflight reads it, so the fixture records one too.
        lobes_version=lobes_version,
        files=files,
    )
    _lock.write_lock(source, lock)
    return source, box


def _boom(*args, **kwargs):
    raise AssertionError("profile/shape resolution must not be consulted by --from-lock")


# --- criterion 1: byte-identical restore, .env untouched --------------------


def test_restore_is_byte_identical_to_the_captured_box(tmp_path, monkeypatch) -> None:
    source, box = _make_variation(tmp_path, monkeypatch)
    target = tmp_path / "restored"
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == 0

    names = _committed_names(box)
    assert names, "the captured variation committed no compose/Dockerfile at all"
    match, mismatch, errors = filecmp.cmpfiles(box, target, names, shallow=False)
    assert (sorted(match), mismatch, errors) == (sorted(names), [], [])


def test_restore_accepts_the_lock_file_itself_as_the_source(tmp_path, monkeypatch) -> None:
    source, box = _make_variation(tmp_path, monkeypatch)
    target = tmp_path / "restored"
    lock_file = source / _lock.LOCK_FILENAME
    assert main(["init", "--from-lock", str(lock_file), str(target), "--apply"]) == 0
    assert (target / _compose.COMPOSE_FILE).read_bytes() == (
        box / _compose.COMPOSE_FILE
    ).read_bytes()


def test_restore_leaves_an_existing_env_byte_identical(tmp_path, monkeypatch) -> None:
    """The merge-only guarantee: a live ``.env`` survives a restore verbatim.

    The box the lock was captured FROM already carries every key the lock
    records, so a restore back onto it appends nothing at all — the file's
    bytes, not merely its values, are unchanged.
    """
    source, box = _make_variation(tmp_path, monkeypatch)
    before = (box / _compose.ENV_FILE).read_bytes()
    assert main(["init", "--from-lock", str(source), str(box), "--apply"]) == 0
    assert (box / _compose.ENV_FILE).read_bytes() == before


def test_restore_never_rewrites_an_existing_env_line(tmp_path, monkeypatch) -> None:
    """A key the lock sets and the ``.env`` already sets DIFFERENTLY is left alone."""
    source, box = _make_variation(tmp_path, monkeypatch)
    lock = _lock.load_lock(source / _lock.LOCK_FILENAME)
    key = "PRIMARY_GPU_MEM_UTIL"
    assert key in lock.env, "fixture drifted: the lock no longer records this knob"
    env_path = box / _compose.ENV_FILE
    hand_edited = env_path.read_text(encoding="utf-8").replace(
        f"{key}={lock.env[key]}", f"{key}=0.99"
    )
    env_path.write_text(hand_edited, encoding="utf-8")

    assert main(["init", "--from-lock", str(source), str(box), "--apply"]) == 0
    assert env_path.read_text(encoding="utf-8") == hand_edited


def test_restore_appends_only_the_keys_an_env_is_missing(tmp_path, monkeypatch) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch)
    lock = _lock.load_lock(source / _lock.LOCK_FILENAME)
    target = tmp_path / "restored"
    target.mkdir()
    kept = "# hand written\nPRIMARY_MAX_MODEL_LEN=4096\n"
    (target / _compose.ENV_FILE).write_text(kept, encoding="utf-8")

    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == 0
    text = (target / _compose.ENV_FILE).read_text(encoding="utf-8")
    assert text.startswith(kept)
    assert "PRIMARY_MAX_MODEL_LEN=4096" in text
    assert f"PRIMARY_MODEL={lock.env['PRIMARY_MODEL']}" in text


def test_restore_creates_an_env_holding_only_the_locks_knobs(tmp_path, monkeypatch) -> None:
    """A fresh target gets the lock's rendered knobs — and NO secret, by construction."""
    source, _box = _make_variation(tmp_path, monkeypatch)
    lock = _lock.load_lock(source / _lock.LOCK_FILENAME)
    target = tmp_path / "fresh"
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == 0
    written = _compose.env_keys((target / _compose.ENV_FILE).read_text(encoding="utf-8"))
    assert written == set(lock.env)
    assert not {"GATEWAY_API_KEY", "HF_TOKEN", "PRIMARY_PEER_ORIGIN"} & written


# --- criterion 2: machine-type guard ----------------------------------------


def test_restore_refuses_when_the_detected_card_differs(tmp_path, monkeypatch, capsys) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch, card="spark")
    _patch_detect(monkeypatch, _card("thor"))
    target = tmp_path / "restored"
    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "spark" in err
    assert "thor" in err
    assert "--allow-variation-mismatch" in err
    assert not target.exists()


def test_variation_guard_refuses_on_the_dry_run_too(tmp_path, monkeypatch) -> None:
    """A plan that describes a deployment whose GPU access would be wrong is the
    same bug one step earlier — so the guard runs before ``--apply``, exactly
    like ``_guard_coresidency``."""
    source, _box = _make_variation(tmp_path, monkeypatch, card="spark")
    _patch_detect(monkeypatch, _card("thor"))
    assert main(["init", "--from-lock", str(source), str(tmp_path / "restored")]) == EXIT_USER_ERROR


def test_an_unknown_card_is_a_mismatch_not_a_pass(tmp_path, monkeypatch) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch, card="spark")
    _patch_detect(monkeypatch, _card(_detect.UNKNOWN))
    assert (
        main(["init", "--from-lock", str(source), str(tmp_path / "restored"), "--apply"])
        == EXIT_USER_ERROR
    )


def test_the_override_is_explicit_and_warns(tmp_path, monkeypatch, capsys) -> None:
    source, box = _make_variation(tmp_path, monkeypatch, card="spark")
    _patch_detect(monkeypatch, _card("thor"))
    target = tmp_path / "restored"
    assert (
        main(
            [
                "init",
                "--from-lock",
                str(source),
                str(target),
                "--apply",
                "--allow-variation-mismatch",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "warning" in captured.err
    assert "spark" in captured.err
    assert "thor" in captured.err
    assert (target / _compose.COMPOSE_FILE).read_bytes() == (
        box / _compose.COMPOSE_FILE
    ).read_bytes()


def test_matching_variation_restores_without_a_warning(tmp_path, monkeypatch, capsys) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch, card="spark")
    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(tmp_path / "restored"), "--apply"]) == 0
    assert "warning" not in capsys.readouterr().err


# --- criterion 3: the GPU overlays survive the round trip -------------------


def test_a_csv_mode_lock_restores_both_gpu_overlays(tmp_path, monkeypatch) -> None:
    source, box = _make_variation(tmp_path, monkeypatch, card="orin", shape="orin-lobe")
    overlays = [_compose.GPU_OVERLAY, _compose.GPU_AUDIO_OVERLAY]
    assert all((box / name).is_file() for name in overlays), "fixture: orin is the csv-mode card"

    target = tmp_path / "restored"
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == 0
    for name in overlays:
        assert (target / name).read_bytes() == (box / name).read_bytes()


def test_a_devices_mode_lock_restores_neither_gpu_overlay(tmp_path, monkeypatch) -> None:
    source, box = _make_variation(tmp_path, monkeypatch, card="spark")
    assert not (box / _compose.GPU_OVERLAY).exists(), "fixture: spark takes the default access"
    target = tmp_path / "restored"
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == 0
    assert not (target / _compose.GPU_OVERLAY).exists()
    assert not (target / _compose.GPU_AUDIO_OVERLAY).exists()


def test_a_devices_mode_lock_scrubs_stale_generated_overlays(tmp_path, monkeypatch) -> None:
    """``_sync_gpu_overrides``' remove-on-mismatch behaviour, through the lock."""
    source, _box = _make_variation(tmp_path, monkeypatch, card="spark")
    target = tmp_path / "restored"
    target.mkdir()
    stale = [_compose.GPU_OVERLAY, _compose.GPU_AUDIO_OVERLAY, _compose.SHAPE_OVERLAY]
    for name in stale:
        (target / name).write_text("# stale\n", encoding="utf-8")

    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == 0
    assert [name for name in stale if (target / name).exists()] == []


def test_a_shape_dropping_lock_restores_its_shape_overlay(tmp_path, monkeypatch) -> None:
    source, box = _make_variation(tmp_path, monkeypatch, card="spark", shape="spark-lobe")
    assert (box / _compose.SHAPE_OVERLAY).is_file(), "fixture: spark-lobe drops senses"
    target = tmp_path / "restored"
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == 0
    assert (target / _compose.SHAPE_OVERLAY).read_bytes() == (
        box / _compose.SHAPE_OVERLAY
    ).read_bytes()


# --- criterion 4: resolution is provably not consulted ----------------------


def _break_resolution(monkeypatch) -> None:
    """Make every profile/shape resolution entry point explode.

    This is the only honest proof of the bypass: if ``--from-lock`` reached the
    renderer at all — for the card profile, the shape, the shape overlay, the
    GPU-access pair or the packaged scaffold — one of these would fire. Called
    AFTER the variation fixture, which legitimately renders a deployment to
    capture.
    """
    for name in (
        "resolve_init_profile",
        "resolve_shape",
        "render_shape",
        "render_shape_override",
        "_sync_shape_override",
        "render_gpu_overrides",
        "_sync_gpu_overrides",
    ):
        monkeypatch.setattr(init_cmd, name, _boom)
    monkeypatch.setattr(_compose, "write_scaffold", _boom)


def test_restore_never_consults_resolution(tmp_path, monkeypatch) -> None:
    source, box = _make_variation(tmp_path, monkeypatch)
    _break_resolution(monkeypatch)
    target = tmp_path / "restored"
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == 0
    assert (target / _compose.COMPOSE_FILE).read_bytes() == (
        box / _compose.COMPOSE_FILE
    ).read_bytes()


def test_dry_run_never_consults_resolution(tmp_path, monkeypatch) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch)
    _break_resolution(monkeypatch)
    assert main(["init", "--from-lock", str(source), str(tmp_path / "restored")]) == 0


def test_restore_never_consults_resolution_across_machine_types(tmp_path, monkeypatch) -> None:
    """The spec's own wording: a restore on a DIFFERENT card still produces the
    lock's files verbatim once the operator has said so explicitly."""
    source, box = _make_variation(tmp_path, monkeypatch, card="orin", shape="orin-lobe")
    _patch_detect(monkeypatch, _card("spark"))
    _break_resolution(monkeypatch)
    target = tmp_path / "restored"
    assert (
        main(
            [
                "init",
                "--from-lock",
                str(source),
                str(target),
                "--apply",
                "--allow-variation-mismatch",
            ]
        )
        == 0
    )
    # The csv-mode overlays come back even though the restoring card is a
    # devices-mode one: the lock, not the renderer, decided.
    assert (target / _compose.GPU_OVERLAY).read_bytes() == (box / _compose.GPU_OVERLAY).read_bytes()


# --- criterion 5: mutation safety -------------------------------------------


def test_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch)
    target = tmp_path / "restored"
    assert main(["init", "--from-lock", str(source), str(target)]) == 0
    assert not target.exists()


def test_dry_run_names_every_file_apply_would_write(tmp_path, monkeypatch, capsys) -> None:
    source, box = _make_variation(tmp_path, monkeypatch, card="orin", shape="orin-lobe")
    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(tmp_path / "restored")]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "--apply" in out
    for name in _committed_names(box):
        assert name in out


def test_dry_run_json_payload(tmp_path, monkeypatch, capsys) -> None:
    source, box = _make_variation(tmp_path, monkeypatch)
    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(tmp_path / "restored"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["from_lock"] is True
    assert payload["variation"] == "spark"
    assert payload["detected_variation"] == "spark"
    assert payload["variation_mismatch"] is False
    assert [entry["name"] for entry in payload["files"]] == _committed_names(box)


def test_apply_json_payload(tmp_path, monkeypatch, capsys) -> None:
    source, box = _make_variation(tmp_path, monkeypatch)
    target = tmp_path / "restored"
    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(target), "--apply", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["restored"] == str(target)
    assert payload["from_lock"] is True
    assert payload["files"] == _committed_names(box)


def test_restore_overwrites_a_hand_edited_compose_without_force(tmp_path, monkeypatch) -> None:
    """The motivating incident: a box whose compose was hand-edited is restorable.

    Compose files and Dockerfiles are replaced wholesale (the spec's split);
    only ``.env`` is merge-only, and ``--apply`` is the whole mutation gate.
    """
    source, box = _make_variation(tmp_path, monkeypatch)
    (box / _compose.COMPOSE_FILE).write_text("# hand edited\n", encoding="utf-8")
    assert main(["init", "--from-lock", str(source), str(box), "--apply"]) == 0
    assert (box / _compose.COMPOSE_FILE).read_bytes() == (
        source / _compose.COMPOSE_FILE
    ).read_bytes()


# --- source validation ------------------------------------------------------


def test_missing_lock_is_a_user_error(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _card("spark"))
    empty = tmp_path / "empty"
    empty.mkdir()
    capsys.readouterr()
    assert main(["init", "--from-lock", str(empty), str(tmp_path / "restored")]) == EXIT_USER_ERROR
    assert _lock.LOCK_FILENAME in capsys.readouterr().err


def test_a_lock_naming_no_files_is_refused(tmp_path, monkeypatch) -> None:
    _patch_detect(monkeypatch, _card("spark"))
    source = tmp_path / "variation"
    source.mkdir()
    _lock.write_lock(source, _lock.build_lock(variation="spark", env={}))
    assert main(["init", "--from-lock", str(source), str(tmp_path / "restored")]) == EXIT_USER_ERROR


def test_a_missing_committed_file_is_refused_before_any_write(
    tmp_path, monkeypatch, capsys
) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch)
    (source / _compose.COMPOSE_FILE).unlink()
    target = tmp_path / "restored"
    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == EXIT_USER_ERROR
    assert _compose.COMPOSE_FILE in capsys.readouterr().err
    assert not target.exists()


def test_a_digest_mismatch_is_refused_before_any_write(tmp_path, monkeypatch, capsys) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch)
    (source / _compose.COMPOSE_FILE).write_text("# tampered\n", encoding="utf-8")
    target = tmp_path / "restored"
    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert _compose.COMPOSE_FILE in err
    assert "digest" in err
    assert not target.exists()


@pytest.mark.parametrize("name", ["../escape.yml", "sub/dir.yml", ".env", "secret.env"])
def test_a_lock_naming_an_unsafe_file_is_refused(tmp_path, monkeypatch, name: str) -> None:
    """Two separate hazards, one gate: a path that escapes the variation folder,
    and a name in the ``.env`` SECRET family the repo's positional gitignore
    rule ignores by construction."""
    _patch_detect(monkeypatch, _card("spark"))
    source = tmp_path / "variation"
    source.mkdir()
    _lock.write_lock(
        source,
        _lock.build_lock(variation="spark", env={}, files={name: "sha256:" + "0" * 64}),
    )
    assert main(["init", "--from-lock", str(source), str(tmp_path / "restored")]) == EXIT_USER_ERROR


# --- the source is not a renderer input -------------------------------------


@pytest.mark.parametrize(
    "extra",
    [["--profile", "spark"], ["--shape", "spark-lobe"], ["--single"], ["--audio"]],
)
def test_from_lock_refuses_the_renderer_axes(tmp_path, monkeypatch, capsys, extra) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch)
    capsys.readouterr()
    assert (
        main(["init", "--from-lock", str(source), str(tmp_path / "restored"), *extra])
        == EXIT_USER_ERROR
    )
    assert extra[0] in capsys.readouterr().err


def test_from_lock_is_absent_from_a_plain_init(tmp_path, monkeypatch, capsys) -> None:
    """A bare ``lobes init`` is untouched by this flag existing."""
    _patch_detect(monkeypatch, _card("spark"))
    capsys.readouterr()
    assert main(["init", str(tmp_path / "plain"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "from_lock" not in payload


# --- buildability preflight (t10's guard, wired here — deviation d3) --------


def test_a_dev_pinned_variation_warns_that_its_wheel_may_be_gone(
    tmp_path, monkeypatch, capsys
) -> None:
    """A `.devN` pin is published to TestPyPI by a PR and may not outlive it."""
    source, _box = _make_variation(tmp_path, monkeypatch, lobes_version="0.67.0.dev428")
    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(tmp_path / "restored"), "--apply"]) == 0
    assert "development wheel" in capsys.readouterr().err


def test_an_unversioned_variation_warns_but_still_restores(tmp_path, monkeypatch, capsys) -> None:
    """Absent is "not recorded", never "broken" — it warns, it does not refuse."""
    source, _box = _make_variation(tmp_path, monkeypatch, lobes_version=None)
    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(tmp_path / "restored"), "--apply"]) == 0
    assert "no MODEL_GEAR_VERSION" in capsys.readouterr().err


# --- PR #223 review: a destination symlink must never be written through -----


def test_restore_refuses_to_write_through_a_symlink_in_the_target(
    tmp_path, monkeypatch, capsys
) -> None:
    """``_check_restorable_name`` gates the LOCK's names; this gates the TARGET.

    A plain-name entry cannot escape the deployment directory, but a symlink
    already sitting at the destination would carry the write anywhere the
    process can reach. Both the lock and the target deserve suspicion, so the
    restore refuses rather than following (or silently replacing) the link.
    """
    source, _box = _make_variation(tmp_path, monkeypatch)
    target = tmp_path / "restored"
    target.mkdir()
    outside = tmp_path / "outside.yml"
    outside.write_text("# untouched\n", encoding="utf-8")
    (target / _compose.COMPOSE_FILE).symlink_to(outside)

    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert _compose.COMPOSE_FILE in err
    assert "symlink" in err
    assert outside.read_text(encoding="utf-8") == "# untouched\n"
    assert (target / _compose.COMPOSE_FILE).is_symlink()


def test_the_symlink_guard_refuses_on_the_dry_run_too(tmp_path, monkeypatch) -> None:
    source, _box = _make_variation(tmp_path, monkeypatch)
    target = tmp_path / "restored"
    target.mkdir()
    (target / _compose.COMPOSE_FILE).symlink_to(tmp_path / "outside.yml")
    assert main(["init", "--from-lock", str(source), str(target)]) == EXIT_USER_ERROR


def test_an_env_symlink_is_refused_before_the_merge(tmp_path, monkeypatch, capsys) -> None:
    """``.env`` is merged, not overwritten — but an appending open() follows a
    symlink just as happily."""
    source, _box = _make_variation(tmp_path, monkeypatch)
    target = tmp_path / "restored"
    target.mkdir()
    outside = tmp_path / "outside.env"
    outside.write_text("KEEP=1\n", encoding="utf-8")
    (target / _compose.ENV_FILE).symlink_to(outside)

    capsys.readouterr()
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == EXIT_USER_ERROR
    assert "symlink" in capsys.readouterr().err
    assert outside.read_text(encoding="utf-8") == "KEEP=1\n"


# --- PR #223 review: the write phase is staged, never half-applied ----------


def _seed_previous_deployment(target: Path, names: list[str]) -> None:
    target.mkdir(exist_ok=True)
    for name in names:
        (target / name).write_text("# previous\n", encoding="utf-8")
    (target / _compose.ENV_FILE).write_text("# previous env\n", encoding="utf-8")


def _assert_previous_deployment_intact(target: Path, names: list[str]) -> None:
    for name in names:
        assert (target / name).read_text(encoding="utf-8") == "# previous\n"
    assert (target / _compose.ENV_FILE).read_text(encoding="utf-8") == "# previous env\n"
    leftovers = sorted(p.name for p in target.iterdir() if init_cmd._RESTORE_TEMP_TAG in p.name)
    assert leftovers == []


def test_a_staging_failure_leaves_the_deployment_untouched(tmp_path, monkeypatch) -> None:
    """An I/O failure while staging mutates nothing at all."""
    source, box = _make_variation(tmp_path, monkeypatch)
    names = _committed_names(box)
    target = tmp_path / "restored"
    _seed_previous_deployment(target, names)

    real = init_cmd._write_new_file
    calls = {"n": 0}

    def flaky(path: Path, data: bytes) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        real(path, data)

    monkeypatch.setattr(init_cmd, "_write_new_file", flaky)
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) != 0
    _assert_previous_deployment_intact(target, names)


def test_a_commit_failure_is_rolled_back(tmp_path, monkeypatch) -> None:
    """Every destination is parked before the first replace, so a failure part
    way through the commit puts the deployment back the way it was."""
    source, box = _make_variation(tmp_path, monkeypatch)
    names = _committed_names(box)
    target = tmp_path / "restored"
    _seed_previous_deployment(target, names)

    real = init_cmd._replace_file
    calls = {"n": 0}

    def flaky(src: Path, dst: Path) -> None:
        # Fail on the SECOND file placed (a staged temp moving onto its final
        # name), i.e. after the commit has already mutated the deployment.
        if init_cmd._RESTORE_TEMP_TAG in src.name and not src.name.endswith(".bak.tmp"):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError(5, "Input/output error")
        real(src, dst)

    monkeypatch.setattr(init_cmd, "_replace_file", flaky)
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) != 0
    assert calls["n"] >= 2, "the injected failure never reached the placement phase"
    _assert_previous_deployment_intact(target, names)


def test_a_stale_overlay_removal_is_rolled_back_too(tmp_path, monkeypatch) -> None:
    """The removals are part of the same commit, not a separate pass that has
    already happened by the time a write fails."""
    source, box = _make_variation(tmp_path, monkeypatch)
    names = _committed_names(box)
    stale = next(name for name in init_cmd.RESTORE_SYNCED_FILES if name not in names)
    target = tmp_path / "restored"
    _seed_previous_deployment(target, names)
    (target / stale).write_text("# stale\n", encoding="utf-8")

    real = init_cmd._replace_file
    calls = {"n": 0}

    def flaky(src: Path, dst: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(5, "Input/output error")
        real(src, dst)

    monkeypatch.setattr(init_cmd, "_replace_file", flaky)
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) != 0
    _assert_previous_deployment_intact(target, names)
    assert (target / stale).read_text(encoding="utf-8") == "# stale\n"


def test_a_successful_restore_leaves_no_staging_files_behind(tmp_path, monkeypatch) -> None:
    source, box = _make_variation(tmp_path, monkeypatch)
    target = tmp_path / "restored"
    assert main(["init", "--from-lock", str(source), str(target), "--apply"]) == 0
    assert [p.name for p in target.iterdir() if init_cmd._RESTORE_TEMP_TAG in p.name] == []
    names = _committed_names(box)
    match, mismatch, errors = filecmp.cmpfiles(box, target, names, shallow=False)
    assert sorted(match) == sorted(names)
    assert (mismatch, errors) == ([], [])


# --- PR #223 review: a malformed lock is a user error, not a traceback ------


def test_a_malformed_lock_is_a_clean_user_error(tmp_path, monkeypatch, capsys) -> None:
    _patch_detect(monkeypatch, _card("spark"))
    source = tmp_path / "variation"
    source.mkdir()
    (source / _lock.LOCK_FILENAME).write_text("this is not toml ][\n", encoding="utf-8")
    capsys.readouterr()
    assert (
        main(["init", "--from-lock", str(source), str(tmp_path / "restored"), "--apply"])
        == EXIT_USER_ERROR
    )
    err = capsys.readouterr().err
    assert _lock.LOCK_FILENAME in err
    assert "Traceback" not in err
