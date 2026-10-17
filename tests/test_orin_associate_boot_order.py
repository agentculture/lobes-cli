"""Associate-first boot order for shapes hosting `associate` (approved deviation d4, issue #260).

The shipped orin-associate budget (gpu_mem_util 0.70 at max_model_len 1048576) was
measured and accepted with vllm-associate healthy BEFORE the pooling gears started
(docs/evidence/2026-09-13-accept-orin-associate-1m.txt); the base fleet template
orders it the other way (vllm-associate depends_on both gears), and that gears-first
order refused util 0.70 at 128K on 2026-08-26. The generated shape override must
reverse the edge so a plain `lobes fleet up --apply` boots associate first.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from lobes.cli._commands.init import render_shape_override
from lobes.profiles import resolve_profile
from lobes.profiles.shapes import builtin_shape_names, resolve_shape
from lobes.runtime import _compose

_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet" / "docker-compose.yml"
)


class _TagTolerantLoader(yaml.SafeLoader):
    """SafeLoader that reads compose merge tags (``!reset``/``!override``) as plain values."""


def _tagged(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node) -> object:
    if isinstance(node, yaml.MappingNode):
        return {"__tag__": tag_suffix, "value": loader.construct_mapping(node)}
    if isinstance(node, yaml.SequenceNode):
        return {"__tag__": tag_suffix, "value": loader.construct_sequence(node)}
    return {"__tag__": tag_suffix, "value": loader.construct_scalar(node)}


_TagTolerantLoader.add_multi_constructor("!", _tagged)


def _override(shape_name: str, card: str = "orin") -> dict:
    text = render_shape_override(resolve_shape(shape_name), resolve_profile(card))
    assert text is not None
    return yaml.load(text, Loader=_TagTolerantLoader)  # noqa: S506 - SafeLoader subclass


def test_orin_associate_override_reverses_the_gears_first_edge() -> None:
    services = _override("orin-associate")["services"]
    assert services["vllm-associate"]["depends_on"] == {"__tag__": "reset", "value": "null"}
    assert "profiles" not in services["vllm-associate"]  # hosted lane is never parked
    for gear in ("vllm-embed", "vllm-rerank"):
        assert services[gear]["depends_on"] == {"vllm-associate": {"condition": "service_healthy"}}


def test_the_override_still_parks_every_dropped_core_service() -> None:
    services = _override("orin-associate")["services"]
    for dropped in ("vllm-hand", "vllm-multimodal", "vllm-primary"):
        assert services[dropped]["profiles"] == ["shape-dropped"]


@pytest.mark.parametrize("shape_name", [s for s in builtin_shape_names() if s != "orin-associate"])
def test_no_other_builtin_shape_gains_a_start_order_edge(shape_name: str) -> None:
    text = render_shape_override(resolve_shape(shape_name), resolve_profile("orin"))
    if text is None:
        return
    assert "depends_on" not in text
    assert "vllm-associate" not in text


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker compose not available")
def test_docker_compose_merges_the_reversed_order(tmp_path: Path) -> None:
    """The real compose merge: associate waits on nothing, both gears wait on associate."""
    (tmp_path / "docker-compose.yml").write_text(
        _TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    text = render_shape_override(resolve_shape("orin-associate"), resolve_profile("orin"))
    (tmp_path / "docker-compose.shape.yml").write_text(text, encoding="utf-8")
    (tmp_path / ".env").write_text("COMPOSE_PROFILES=associate\n", encoding="utf-8")
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            "docker-compose.shape.yml",
            "config",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and "unknown flag: --format" in result.stderr:
        pytest.skip("docker compose too old for --format json")
    assert result.returncode == 0, result.stderr
    import json

    services = json.loads(result.stdout)["services"]
    assert not services["vllm-associate"].get("depends_on")
    for gear in ("vllm-embed", "vllm-rerank"):
        assert services[gear]["depends_on"]["vllm-associate"]["condition"] == "service_healthy"


def test_boot_order_blocks_are_not_read_as_dropped_lobes() -> None:
    """A start-order reversal must never be mistaken for a shape DROP.

    ``docker-compose.shape.yml`` holds two unrelated kinds of block: services
    PARKED in the inert ``shape-dropped`` profile, and services merely re-ordered
    (this file's whole subject). Reading "any service key present" as "dropped" —
    which lobes did until the ``INNEREYE_UI_PORT`` work introduced a third kind —
    reported ``vllm-associate`` and both pooling gears as dropped on the one shape
    that hosts them, so ``lobes up associate`` would have been refused on the Orin
    and ``doctor`` would have stopped demanding their knobs.
    """
    text = render_shape_override(resolve_shape("orin-associate"), resolve_profile("orin"))
    parked = _compose.shape_parked_service_keys(text)
    assert "vllm-associate" not in parked
    assert "vllm-embed" not in parked
    assert "vllm-rerank" not in parked
    # The shape genuinely drops the two heavy default lobes; those ARE parked.
    assert {"vllm-primary", "vllm-multimodal"} <= parked


# --- The wrapper's release order (lane-safety LOBES_BOOT_ORDER) ---------------
# After a host reboot dockerd restarts every lane at once and compose
# `depends_on` is ignored, so the override above cannot order the boot; the
# wrapper's LOBES_BOOT_ORDER does. The Orin card default releases senses and the
# gears first, which on this shape would hold associate LAST — the gears-first
# order its util 0.70 budget was never measured at. The shape's own [host_env]
# must put associate first.


def _shape_env(shape_name: str, card: str = "orin") -> dict[str, str]:
    from lobes.profiles.shape_render import render_shape

    return dict(render_shape(resolve_shape(shape_name), resolve_profile(card)).env)


def test_orin_associate_renders_an_associate_first_release_order() -> None:
    env = _shape_env("orin-associate")
    # Gears UNLISTED: released together, one full stagger after associate.
    assert env["LOBES_BOOT_ORDER"] == "associate"
    assert env["LOBES_BOOT_STAGGER_S"] == "400"


def test_orin_associate_golden_carries_the_associate_first_order() -> None:
    golden = Path(__file__).resolve().parent / "goldens" / "shapes" / "orin-associate__orin.env"
    text = golden.read_text(encoding="utf-8")
    assert "LOBES_BOOT_ORDER=associate\n" in text
    assert "LOBES_BOOT_STAGGER_S=400\n" in text


def test_the_shape_order_wins_on_every_card() -> None:
    # The associate-first order is a property of the shape's budget, not the
    # card's (the same reason the depends_on override above is card-independent).
    for card in ("orin", "spark", "thor"):
        env = _shape_env("orin-associate", card)
        assert env["LOBES_BOOT_ORDER"] == "associate"
        assert env["LOBES_BOOT_STAGGER_S"] == "400"


# One measured boot: vllm-associate went healthy 360 s after its container
# started on the rendered 1M shape (docs/evidence/2026-09-13-accept-orin-
# associate-1m.txt, "healthy after 360s"). A different engine or context would
# need re-measuring (#108); this pins the margin, not a law.
_ASSOCIATE_MEASURED_HEALTHY_S = 360
_WRAPPER = Path(__file__).resolve().parents[1] / "lobes" / "templates" / "mg-logwrap.sh"


def _wrapper_hold(tmp_path: Path, name: str, env: dict[str, str]) -> int:
    """Run the real wrapper at host uptime 0 and return the hold it sleeps."""
    import os

    lane = tmp_path / name
    proc = lane / "proc"
    proc.mkdir(parents=True)
    (proc / "uptime").write_text("0.00 0.00\n")
    bindir = lane / "bin"
    bindir.mkdir()
    rec = lane / "sleeps"
    shim = bindir / "sleep"
    shim.write_text(f'#!/bin/sh\necho "$1" >> "{rec}"\n')
    shim.chmod(0o755)
    result = subprocess.run(
        ["bash", str(_WRAPPER), "true"],
        env={
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "LOBES_PROC_ROOT": str(proc),
            "MG_LOG_DIR": str(lane / "logs"),
            "MG_LOG_NAME": name,
            "LOBES_BOOT_ORDER": env["LOBES_BOOT_ORDER"],
            "LOBES_BOOT_STAGGER_S": env["LOBES_BOOT_STAGGER_S"],
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return int(rec.read_text().split()[0])


def _start_period_s(service: str) -> int:
    compose = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    period = compose["services"][service]["healthcheck"]["start_period"]
    assert period.endswith("s"), period
    return int(period[:-1])


def test_gears_are_released_after_associate_is_resident(tmp_path: Path) -> None:
    env = _shape_env("orin-associate")
    associate = _wrapper_hold(tmp_path, "associate", env)
    embed = _wrapper_hold(tmp_path, "embed", env)
    rerank = _wrapper_hold(tmp_path, "rerank", env)
    assert associate == 120  # the default LOBES_BOOT_DELAY_S
    assert embed == rerank == 520  # 120 + len(order)=1 x 400
    # Associate's measured healthy point (120 + 360 = 480) lands before the
    # gears are released: memory-wise associate-first, as measured.
    assert associate + _ASSOCIATE_MEASURED_HEALTHY_S < embed


def test_the_gear_hold_fits_inside_their_healthcheck_start_period(tmp_path: Path) -> None:
    env = _shape_env("orin-associate")
    hold = max(_wrapper_hold(tmp_path, name, env) for name in ("embed", "rerank"))
    for service in ("vllm-embed", "vllm-rerank"):
        assert _start_period_s(service) == 780
        assert hold < _start_period_s(service)


def test_the_senses_shape_keeps_the_card_default() -> None:
    # The card default IS the measured senses-first order: orin-lobe keeps it.
    assert _shape_env("orin-lobe")["LOBES_BOOT_ORDER"] == "multimodal,embed,rerank"
    assert "host_env" not in resolve_shape("orin-lobe").to_dict()


def test_shape_host_env_rejects_a_bad_key() -> None:
    from lobes.cli._errors import ModelGearError
    from lobes.profiles.shapes import Shape

    with pytest.raises(ModelGearError, match="shape 'x'.*not a valid env-var name"):
        Shape.from_dict("x", {"hosts": ["embedder"], "host_env": {"1BAD": "v"}})
    with pytest.raises(ModelGearError, match="shape 'x'.*must be str"):
        Shape.from_dict("x", {"hosts": ["embedder"], "host_env": {"OK": 3}})
