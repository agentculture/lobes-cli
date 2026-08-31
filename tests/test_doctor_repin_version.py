"""``lobes doctor --repin-version`` — the one key doctor may rewrite (issue #99).

The asymmetry under test is the point. ``lobes init`` writes
``MODEL_GEAR_VERSION`` once and no verb ever re-pins it, so a merged gateway
fix never reaches a deployment (#99). But doctor's heal lane is append-only by
contract, and #174/#191 are what breaking that costs. So the re-pin is its own
named flag, and ``--fix --apply`` must still leave every existing line alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lobes import __version__
from lobes.cli._commands import doctor
from lobes.runtime import _compose, _env


def _env_path(deploy: Path) -> Path:
    return deploy / _compose.ENV_FILE


@pytest.fixture()
def deploy(tmp_path: Path) -> Path:
    d = tmp_path / "deploy"
    d.mkdir()
    _env_path(d).write_text(
        "# operator-typed header\n"
        "MODEL_GEAR_VERSION=0.1.0\n"
        "PRIMARY_PEER_ORIGIN=http://peer:8000\n",
        encoding="utf-8",
    )
    return d


def test_repin_rewrites_the_stale_pin(deploy: Path) -> None:
    actions = doctor._repin_version(deploy)
    assert _env.read_env(_env_path(deploy), "MODEL_GEAR_VERSION") == __version__
    assert actions and "0.1.0" in actions[0]


def test_repin_preserves_every_other_operator_line(deploy: Path) -> None:
    doctor._repin_version(deploy)
    text = _env_path(deploy).read_text(encoding="utf-8")
    assert "# operator-typed header" in text
    assert "PRIMARY_PEER_ORIGIN=http://peer:8000" in text


def test_repin_is_a_noop_when_already_current(deploy: Path) -> None:
    _env.set_env(_env_path(deploy), "MODEL_GEAR_VERSION", __version__)
    before = _env_path(deploy).read_text(encoding="utf-8")
    assert doctor._repin_version(deploy) == []
    assert _env_path(deploy).read_text(encoding="utf-8") == before


def test_repin_appends_when_the_key_is_absent(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    _env_path(d).write_text("OTHER=1\n", encoding="utf-8")
    actions = doctor._repin_version(d)
    assert _env.read_env(_env_path(d), "MODEL_GEAR_VERSION") == __version__
    assert actions and "appended" in actions[0]


def test_the_heal_lane_still_never_rewrites_an_existing_line(deploy: Path) -> None:
    """The invariant #174/#191 bought: ``--fix --apply`` touches no existing key.

    ``_apply_fix`` is the whole heal lane. Pointed at a deployment whose pin is
    stale, it must leave that line exactly as it found it — only
    ``--repin-version`` may change it.
    """
    before = _env_path(deploy).read_text(encoding="utf-8")
    doctor._apply_fix(deploy)
    after = _env_path(deploy).read_text(encoding="utf-8")
    assert "MODEL_GEAR_VERSION=0.1.0" in after
    assert before.splitlines()[:3] == after.splitlines()[:3]


def test_apply_without_fix_or_repin_is_a_user_error() -> None:
    import argparse

    from lobes.cli._errors import ModelGearError

    args = argparse.Namespace(
        role=None, fix=False, apply=True, repin_version=False, compose_dir=None, json=False
    )
    with pytest.raises(ModelGearError) as err:
        doctor.cmd_doctor(args)
    assert "--fix or --repin-version" in err.value.message


def test_repin_and_role_are_refused_together() -> None:
    import argparse

    from lobes.cli._errors import ModelGearError

    args = argparse.Namespace(
        role="cortex", fix=False, apply=False, repin_version=True, compose_dir=None, json=False
    )
    with pytest.raises(ModelGearError) as err:
        doctor.cmd_doctor(args)
    assert "--repin-version" in err.value.message
