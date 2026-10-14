"""Rendering the decider role's per-domain lanes (orin-decision-models plan, t8).

Covers c14 / c37 / h20: ``decider`` is an opt-in core role on every render
table; machine-as-brain renders nothing for it; a hosted-but-unwired decider
renders ``DECIDER_FEASIBLE=false``; each declared domain renders a
``decider-<domain>`` service built from ``Dockerfile.decider`` into the
GENERATED ``docker-compose.shape.yml`` with no ``ports:`` mapping and no
``depends_on`` either way; ``DECIDER_DOMAINS`` carries the pinned
``name=http://decider-<name>:8000`` contract shared with the gateway (t5).
"""

from __future__ import annotations

import json
from importlib.resources import files

import pytest
import yaml

from lobes.cli import main
from lobes.cli._commands import init as init_cmd
from lobes.cli._commands import up as up_cmd
from lobes.cli._errors import ModelGearError
from lobes.decider_sidecar import server as decider_server
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import ROLE_ENV_PREFIX
from lobes.profiles.schema import ROLES as PROFILE_ROLES
from lobes.profiles.schema import Profile, RoleProfile
from lobes.profiles.shape_render import (
    OPT_IN_CORE_COMPOSE_PROFILE,
    ROLE_SERVICE,
    decider_compose_lines,
    render_shape,
    shape_env,
    shape_services,
)
from lobes.profiles.shapes import (
    DECIDER_BACKENDS,
    DEFAULT_HOSTED_ROLES,
    OPT_IN_CORE_ROLES,
    Shape,
    builtin_shape_names,
    resolve_shape,
)
from lobes.runtime import _compose
from tests.test_cli_up import _scaffold_fleet

CLEF = "Cloudflare/clef"
STRANDS = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"


def _decider_shape(domains: dict | None = None, **extra) -> Shape:
    data = {
        "hosts": ["embedder", "reranker", "decider"],
        **extra,
    }
    if domains is not None:
        data["decider_domains"] = domains
    return Shape.from_dict("test-decider", data)


def _two_domains() -> dict:
    return {
        "general": {
            "backend": "clef",
            "model": CLEF,
            "revision": "abc123",
            "mem_limit": "32g",
        },
        "triage": {"backend": "strands", "model": STRANDS, "mem_limit": "8g"},
    }


def _card() -> Profile:
    return resolve_profile("orin")


def _override_doc(shape: Shape, profile: Profile) -> dict:
    text = init_cmd.render_shape_override(shape, profile)
    assert text is not None
    return yaml.safe_load(text)


# --- every render table knows the role ----------------------------------------


def test_decider_is_an_opt_in_core_role_on_every_render_table() -> None:
    assert "decider" in OPT_IN_CORE_ROLES
    assert "decider" not in DEFAULT_HOSTED_ROLES
    assert "decider" in PROFILE_ROLES
    assert ROLE_ENV_PREFIX["decider"] == "DECIDER"
    assert ROLE_SERVICE["decider"] == "decider-general"
    assert up_cmd.ROLE_SERVICE["decider"] == "decider-general"
    assert OPT_IN_CORE_COMPOSE_PROFILE["decider"] == "decider"


def test_decider_backends_mirror_the_sidecar() -> None:
    assert DECIDER_BACKENDS == decider_server.BACKENDS


# --- machine-as-brain / non-hosting shapes render nothing ---------------------


#: The built-in shapes that host the decider (t9). Every other one renders nothing.
DECIDER_HOSTING_SHAPES = ("orin-decider",)


@pytest.mark.parametrize("card", builtin_names())
def test_no_builtin_shape_renders_any_decider_key_or_service(card: str) -> None:
    profile = resolve_profile(card)
    for name in builtin_shape_names():
        if name in DECIDER_HOSTING_SHAPES:
            continue
        rendered = render_shape(resolve_shape(name), profile)
        assert not [k for k in rendered.env if k.startswith("DECIDER_")], name
        assert not [s for s in rendered.services if s.startswith("decider")], name
        assert "decider" not in rendered.env.get("COMPOSE_PROFILES", "").split(",")


def test_machine_as_brain_writes_no_decider_override() -> None:
    shape = resolve_shape("machine-as-brain")
    for card in builtin_names():
        assert decider_compose_lines(shape, resolve_profile(card)) == []


# --- hosted but unwired -> DECIDER_FEASIBLE=false ------------------------------


def test_hosted_decider_with_no_domains_renders_feasible_false() -> None:
    shape = _decider_shape()
    env = shape_env(shape, _card())
    assert env["DECIDER_FEASIBLE"] == "false"
    assert "DECIDER_DOMAINS" not in env
    assert "decider" not in env.get("COMPOSE_PROFILES", "").split(",")
    assert not [s for s in shape_services(shape, _card()) if s.startswith("decider")]
    assert decider_compose_lines(shape, _card()) == []
    override = init_cmd.render_shape_override(shape, _card()) or ""
    assert not [k for k in _compose._override_service_keys(override) if k.startswith("decider")]


def test_card_veto_wins_over_declared_domains() -> None:
    vetoing = Profile(name="veto", roles={"decider": RoleProfile(feasible=False)})
    shape = _decider_shape(_two_domains())
    env = shape_env(shape, vetoing)
    assert env["DECIDER_FEASIBLE"] == "false"
    assert "DECIDER_DOMAINS" not in env
    assert decider_compose_lines(shape, vetoing) == []


# --- hosted and wired -----------------------------------------------------------


def test_declared_domains_render_the_pinned_decider_domains_contract() -> None:
    env = shape_env(_decider_shape(_two_domains()), _card())
    assert env["DECIDER_DOMAINS"] == (
        "general=http://decider-general:8000,triage=http://decider-triage:8000"
    )
    assert "DECIDER_FEASIBLE" not in env
    assert "decider" in env["COMPOSE_PROFILES"].split(",")


def test_declared_domains_render_each_domains_checkpoint() -> None:
    # odm t7: the gateway's /capabilities `decider.domains` advert and the
    # per-domain mesh fingerprint read each domain's checkpoint from here.
    env = shape_env(_decider_shape(_two_domains()), _card())
    assert env["DECIDER_DOMAIN_MODELS"] == f"general={CLEF}@abc123,triage={STRANDS}"


def test_declared_domains_are_the_services_brought_up() -> None:
    services = shape_services(_decider_shape(_two_domains()), _card())
    assert "decider-general" in services
    assert "decider-triage" in services
    assert "decider" not in services


def test_each_domain_renders_a_port_less_decider_service() -> None:
    doc = _override_doc(_decider_shape(_two_domains()), _card())
    services = doc["services"]
    for domain, backend, model, limit in (
        ("general", "clef", CLEF, "32g"),
        ("triage", "strands", STRANDS, "8g"),
    ):
        svc = services[f"decider-{domain}"]
        assert svc["build"]["dockerfile"] == "Dockerfile.decider"
        assert "ports" not in svc
        assert svc["expose"] == ["8000"]
        assert "depends_on" not in svc
        assert svc["mem_limit"] == limit
        assert svc["profiles"] == ["decider"]
        assert svc["container_name"] == f"model-gear-decider-{domain}"
        env = svc["environment"]
        assert f"DECIDER_BACKEND={backend}" in env
        assert f"DECIDER_MODEL={model}" in env
        assert "DECIDER_PORT=8000" in env
        assert "/health" in " ".join(svc["healthcheck"]["test"])
    assert "DECIDER_REVISION=abc123" in services["decider-general"]["environment"]
    assert not any(
        e.startswith("DECIDER_REVISION") for e in services["decider-triage"]["environment"]
    )


def test_no_service_anywhere_depends_on_a_decider() -> None:
    doc = _override_doc(_decider_shape(_two_domains()), _card())
    template = yaml.safe_load(
        files("lobes.templates.fleet").joinpath("docker-compose.yml").read_text(encoding="utf-8")
    )
    for source in (doc, template):
        for name, svc in source["services"].items():
            depends = (svc or {}).get("depends_on") or {}
            named = set(depends if isinstance(depends, dict) else depends)
            assert not [d for d in named if d.startswith("decider")], name


def test_decider_services_read_as_hosted_not_parked() -> None:
    text = init_cmd.render_shape_override(_decider_shape(_two_domains()), _card())
    assert {"decider-general", "decider-triage"} <= _compose._override_service_keys(text)
    assert not _compose.shape_parked_service_keys(text) & {"decider-general", "decider-triage"}


def test_decider_hosting_shape_scaffolds_the_dockerfile() -> None:
    templates = init_cmd._templates(True, False, shape=_decider_shape(_two_domains()))
    assert templates["fleet/Dockerfile.decider"] == "Dockerfile.decider"
    plain = init_cmd._templates(True, False, shape=resolve_shape("machine-as-brain"))
    assert "fleet/Dockerfile.decider" not in plain


def test_override_plan_names_the_decider_services(tmp_path) -> None:
    plan = init_cmd._shape_override_plan(tmp_path, _decider_shape(_two_domains()), _card())
    assert plan["action"] == "write"
    assert plan["deciders"] == ["decider-general", "decider-triage"]
    assert "decider-general" in init_cmd._shape_override_plan_line(plan)


# --- shape-schema validation ------------------------------------------------------


@pytest.mark.parametrize(
    "domains, extra, fragment",
    [
        ({"General": {"backend": "clef", "model": CLEF, "mem_limit": "1g"}}, {}, "domain"),
        ({"-x": {"backend": "clef", "model": CLEF, "mem_limit": "1g"}}, {}, "domain"),
        ({"general": {"backend": "sglang", "model": CLEF, "mem_limit": "1g"}}, {}, "backend"),
        ({"general": {"backend": "clef", "model": CLEF}}, {}, "mem_limit"),
        ({"general": {"backend": "clef", "model": CLEF, "mem_limit": "1g", "x": "y"}}, {}, "x"),
        ({"general": {"backend": "clef", "model": "a b", "mem_limit": "1g"}}, {}, "model"),
        ({"general": {"backend": "clef", "model": "${X}", "mem_limit": "1g"}}, {}, "model"),
        ({"general": {"backend": "clef", "model": CLEF, "mem_limit": "lots"}}, {}, "mem_limit"),
        ({}, {"overrides": {"decider": {"gpu_mem_util": 0.3}}}, "decider"),
        (
            {
                "general": {
                    "backend": "clef-vllm",
                    "model": CLEF,
                    "mem_limit": "1g",
                    "gpu_mem_util": "lots",
                }
            },
            {},
            "gpu_mem_util",
        ),
        (
            {
                "general": {
                    "backend": "clef-vllm",
                    "model": CLEF,
                    "mem_limit": "1g",
                    "max_model_len": "16k",
                }
            },
            {},
            "max_model_len",
        ),
        (
            {
                "general": {
                    "backend": "clef-vllm",
                    "model": CLEF,
                    "mem_limit": "1g",
                    "kv_cache_memory_bytes": "4GiB",
                }
            },
            {},
            "kv_cache_memory_bytes",
        ),
    ],
)
def test_malformed_decider_declarations_are_load_errors(domains, extra, fragment) -> None:
    with pytest.raises(ModelGearError) as info:
        _decider_shape(domains, **extra)
    assert fragment in info.value.message


def test_domains_without_hosting_decider_is_a_load_error() -> None:
    with pytest.raises(ModelGearError) as info:
        Shape.from_dict(
            "x",
            {
                "hosts": ["embedder"],
                "decider_domains": {
                    "general": {"backend": "clef", "model": CLEF, "mem_limit": "1g"}
                },
            },
        )
    assert "does not host 'decider'" in info.value.message


def test_decider_domains_round_trip_through_to_dict() -> None:
    shape = _decider_shape(_two_domains())
    assert list(shape.decider_domains) == ["general", "triage"]
    assert shape.to_dict()["decider_domains"]["triage"]["backend"] == "strands"
    assert "decider_domains" not in resolve_shape("machine-as-brain").to_dict()


# --- lobes up decider ---------------------------------------------------------------


def _scaffold_decider(tmp_path, shape: Shape | None = None):
    _scaffold_fleet(tmp_path)
    text = init_cmd.render_shape_override(shape or _decider_shape(_two_domains()), _card())
    (tmp_path / _compose.SHAPE_OVERLAY).write_text(text, encoding="utf-8")
    (tmp_path / ".env").write_text("COMPOSE_PROFILES=decider\n", encoding="utf-8")
    return tmp_path


def test_up_decider_targets_every_rendered_domain_with_no_deps(tmp_path, capsys) -> None:
    _scaffold_decider(tmp_path)
    assert main(["up", "decider", "--compose-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["services"] == ["decider-general", "decider-triage"]
    assert payload["command"].endswith("up -d --no-deps decider-general decider-triage")
    assert f"-f {_compose.SHAPE_OVERLAY}" in payload["command"]


def test_up_decider_down_stops_every_domain(tmp_path, capsys) -> None:
    _scaffold_decider(tmp_path)
    assert main(["up", "decider", "--down", "--compose-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"].endswith("stop decider-general decider-triage")


def test_up_decider_with_no_rendered_domain_is_a_user_error(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    (tmp_path / ".env").write_text("COMPOSE_PROFILES=decider\n", encoding="utf-8")
    assert main(["up", "decider", "--compose-dir", str(tmp_path)]) != 0
    assert "decider-<domain>" in capsys.readouterr().err


def test_a_clef_vllm_domain_builds_from_the_vllm_image() -> None:
    domains = {"general": {"backend": "clef-vllm", "model": "org/clef-w4a16", "mem_limit": "40g"}}
    text = init_cmd.render_shape_override(_decider_shape(domains), _card())
    svc = _override_doc(_decider_shape(domains), _card())["services"]["decider-general"]
    assert svc["build"]["dockerfile"] == "Dockerfile.decider-vllm"
    assert "DECIDER_BACKEND=clef-vllm" in svc["environment"]
    assert "ports" not in svc
    assert "depends_on" not in svc
    assert "Dockerfile.decider-vllm" in text
    # A clef/strands-only shape renders byte-identically to before t21.
    plain = init_cmd.render_shape_override(_decider_shape(_two_domains()), _card())
    assert "decider-vllm" not in plain


def test_decider_hosting_shape_scaffolds_the_vllm_dockerfile_too() -> None:
    templates = init_cmd._templates(True, False, shape=_decider_shape(_two_domains()))
    assert templates["fleet/Dockerfile.decider-vllm"] == "Dockerfile.decider-vllm"


def test_engine_budget_keys_render_into_the_sidecar_env() -> None:
    # t9: the clef-vllm engine's share and window are per-domain shape data;
    # a domain that omits them keeps the sidecar's own defaults.
    domains = {
        "general": {
            "backend": "clef-vllm",
            "model": "org/clef-w4a16",
            "mem_limit": "32g",
            "gpu_mem_util": "0.40",
            "max_model_len": "16384",
            "kv_cache_memory_bytes": "4294967296",
        }
    }
    env = _override_doc(_decider_shape(domains), _card())["services"]["decider-general"][
        "environment"
    ]
    assert "DECIDER_GPU_MEM_UTIL=0.40" in env
    assert "DECIDER_MAX_MODEL_LEN=16384" in env
    assert "DECIDER_KV_CACHE_MEMORY_BYTES=4294967296" in env
    plain = _override_doc(_decider_shape(_two_domains()), _card())["services"]
    for name, svc in plain.items():
        if not name.startswith("decider-"):
            continue
        assert not any(
            e.startswith(
                ("DECIDER_GPU_MEM_UTIL", "DECIDER_MAX_MODEL_LEN", "DECIDER_KV_CACHE_MEMORY_BYTES")
            )
            for e in svc["environment"]
        )
