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
