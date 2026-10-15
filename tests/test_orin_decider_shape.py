"""The ``orin-decider`` shape (orin-decision-models plan t9).

orin-embed plus ``decider:general``: our own W4A16 Clef build served through
the decider sidecar's clef-vllm backend, at the values the t13 run measured on
the Orin. The render itself is DECLARED until t11 boots it (#108).
"""

from __future__ import annotations

import yaml

from lobes.catalog import SUPPORTED_MODELS
from lobes.cli._commands import init as init_cmd
from lobes.profiles.loader import resolve_profile
from lobes.profiles.shape_render import overcommitted_groups, shape_env
from lobes.profiles.shapes import Shape, builtin_shape_names, resolve_shape

_SHAPE = "orin-decider"


def _shape() -> Shape:
    return resolve_shape(_SHAPE)


def test_the_shape_is_built_in() -> None:
    assert _SHAPE in builtin_shape_names()


def test_it_hosts_orin_embed_plus_the_decider() -> None:
    shape = _shape()
    embed = resolve_shape("orin-embed")
    assert set(shape.hosts) == set(embed.hosts) | {"decider"}
    assert shape.lanes == embed.lanes
    assert shape.lane_knobs == embed.lane_knobs


def test_the_general_domain_serves_the_catalog_default_decider_pinned() -> None:
    default = next(m for m in SUPPORTED_MODELS if m.role_hint == "decider")
    domain = _shape().decider_domains["general"]
    assert list(_shape().decider_domains) == ["general"]
    assert domain["backend"] == "clef-vllm"
    assert domain["model"] == default.id
    assert domain["revision"] == default.hf_revision


def test_the_measured_engine_budget_is_rendered() -> None:
    svc = yaml.safe_load(init_cmd.render_shape_override(_shape(), resolve_profile("orin")))[
        "services"
    ]["decider-general"]
    assert svc["build"]["dockerfile"] == "Dockerfile.decider-vllm"
    assert svc["mem_limit"] == "32g"
    env = svc["environment"]
    # t13 (docs/evidence/2026-10-14-ab-clef-vllm-vs-sidecar-orin.txt): util 0.40,
    # max_model_len 16384, beside the orin-embed lanes.
    assert "DECIDER_GPU_MEM_UTIL=0.40" in env
    assert "DECIDER_MAX_MODEL_LEN=16384" in env
    # t11: the KV pool is pinned (4 GiB). Unpinned, vLLM sizes it from the
    # board's free memory, and the render's first boot OOM-killed its engine at
    # the 32g cap during CUDA-graph capture.
    assert "DECIDER_KV_CACHE_MEMORY_BYTES=4294967296" in env


def test_the_gateway_is_wired_to_the_domain() -> None:
    env = shape_env(_shape(), resolve_profile("orin"))
    assert env["DECIDER_DOMAINS"] == "general=http://decider-general:8000"
    assert "decider" in env["COMPOSE_PROFILES"].split(",")


def test_the_orin_card_lists_it_in_the_exclusive_heavy_lobe_group() -> None:
    (group,) = resolve_profile("orin").exclusive_roles
    assert "decider" in group.roles
    assert _SHAPE in group.shapes


def test_it_over_hosts_nothing_on_the_orin() -> None:
    assert overcommitted_groups(_shape(), resolve_profile("orin")) == ()


def test_a_shape_hosting_associate_and_the_decider_is_refused() -> None:
    both = Shape.from_dict(
        "x",
        {
            "hosts": ["embedder", "reranker", "associate", "decider"],
            "decider_domains": dict(_shape().decider_domains),
        },
    )
    assert overcommitted_groups(both, resolve_profile("orin"))


def test_the_gateway_learns_each_domains_served_window() -> None:
    # Code review (finding 4): the sidecar's DECIDER_MAX_MODEL_LEN never reaches
    # the gateway, so /capabilities advertised the catalog's native 262144.
    env = shape_env(_shape(), resolve_profile("orin"))
    assert env["DECIDER_DOMAIN_CONTEXTS"] == "general=16384"
