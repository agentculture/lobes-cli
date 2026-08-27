"""The TENTH Colleague role, ``associate`` (lightning-on-orin plan, t6).

``associate`` is ``worker`` MINUS ``repo_action`` — "they do, but not act". It
executes, drafts, inspects and calls tools, and hands the result BACK rather
than enacting it. It is a SEPARATE PUBLIC ADDRESS rather than a
responsibilities token on ``worker`` by operator decision: the ``worker`` seat
is being kept free for a possible future worker/cortex switch, and only a
distinct name can carry that.

Adding a role is effectively irreversible (see
``docs/colleague-stack.md``'s "Adding a role is effectively irreversible"),
so this module pins the whole contract in one place:

1. the exact responsibilities / forbidden vocabulary;
2. the capability rung — ``hand < multimodal < worker < muse < associate <
   main``, the HIGHEST non-cortex rung;
3. every public address (``/capabilities``, ``/v1/models``, ``model=``,
   ``lobes up``) plus the honesty rule for an UNHOSTED associate — 404
   ``role_infeasible``, never a silent downgrade;
4. the pressure policy — associate SHEDS like cortex/senses/worker/muse, and
   ``hand`` stays the ONLY servable floor;
5. the no-regression guarantee — the nine pre-existing role prefixes route
   identically, and a deployment declaring no associate config is unchanged;
6. the ``<PREFIX>_*`` env vocabulary the replica-pool work (#199) made generic
   across role prefixes, extended to a tenth exactly as the nine have it.
"""

from __future__ import annotations

import json

import pytest

from lobes import roles as roles_mod
from lobes.catalog import BACKEND_ROLE_CATALOG_HINT, TIER_ROLE, resolve_tier
from lobes.cli._commands import up as up_cmd
from lobes.gateway import server as S
from lobes.gateway._config import (
    FEASIBLE_ENV,
    NEVER_PROXIED_BACKENDS,
    OPT_IN_BACKENDS,
    PEER_API_KEY_ENV,
    PEER_API_KEYS_ENV,
    PEER_ORIGIN_ENV,
    PEER_ORIGINS_ENV,
    PEER_PROXY_ENV,
    build_config,
)
from lobes.gateway._pressure_policy import decide
from lobes.gateway._routing import list_models_payload, resolve_model, tier_aliases
from lobes.profiles.render import ROLE_ENV_PREFIX
from lobes.profiles.schema import ROLES as PROFILE_ROLES
from lobes.profiles.shape_render import (
    OPT_IN_CORE_ACTIVATION_ENV,
    OPT_IN_CORE_COMPOSE_PROFILE,
    ROLE_SERVICE,
)
from lobes.profiles.shapes import DEFAULT_HOSTED_ROLES, OPT_IN_CORE_ROLES

_LIGHTNING_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"

#: The nine role names that existed BEFORE associate — the no-regression set.
_PRE_EXISTING_ROLES: tuple[str, ...] = (
    "cortex",
    "senses",
    "muse",
    "worker",
    "hand",
    "embedder",
    "reranker",
    "stt",
    "tts",
)


def _base_env(**over: str) -> dict[str, str]:
    """A realistic deployment env that declares NOTHING about associate."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "unsloth/Qwen3.8-27B-NVFP4",
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": "coolthor/gemma-4-12B-it-NVFP4A16",
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": "Qwen/Qwen3-Embedding-0.6B",
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": "Qwen/Qwen3-Reranker-0.6B",
    }
    env.update(over)
    return env


# ---------------------------------------------------------------------------
# Criterion 1 — the responsibilities / forbidden vocabulary
# ---------------------------------------------------------------------------


def test_associate_is_a_first_class_role_with_its_own_backend() -> None:
    assert "associate" in roles_mod.ROLES
    assert roles_mod.ROLE_BACKEND["associate"] == "associate"
    assert roles_mod.ROLE_PATH["associate"] == "/v1/chat/completions"
    assert roles_mod.ROLE_MAX_MODEL_LEN_ENV["associate"] == "ASSOCIATE_MAX_MODEL_LEN"


def test_associate_responsibilities_are_exactly_the_declared_set() -> None:
    assert roles_mod.ROLE_RESPONSIBILITIES["associate"] == (
        "execution",
        "ground_work",
        "bulk_transform",
        "drafting",
        "repo_inspection",
        "run_authorized_commands",
        "tool_use",
    )


def test_associate_forbidden_is_workers_plus_repo_action() -> None:
    assert roles_mod.ROLE_FORBIDDEN["associate"] == (
        "final_decision",
        "security_decision",
        "code_authoring",
        "repo_action",
    )
    # The load-bearing difference, stated as a relation and not just two
    # literals: associate forbids everything worker forbids, PLUS repo_action.
    worker_forbidden = set(roles_mod.ROLE_FORBIDDEN["worker"])
    associate_forbidden = set(roles_mod.ROLE_FORBIDDEN["associate"])
    assert worker_forbidden < associate_forbidden
    assert associate_forbidden - worker_forbidden == {"repo_action"}


def test_associate_never_claims_a_responsibility_it_forbids() -> None:
    responsibilities = set(roles_mod.ROLE_RESPONSIBILITIES["associate"])
    forbidden = set(roles_mod.ROLE_FORBIDDEN["associate"])
    assert not responsibilities & forbidden
    # "does, but does not act": repo_action appears ONLY on the forbidden side.
    assert "repo_action" not in responsibilities
    # ... while worker, the role it is derived from, DOES carry it.
    assert "repo_action" in roles_mod.ROLE_RESPONSIBILITIES["worker"]


def test_associate_responsibilities_are_a_subset_of_workers() -> None:
    # The `hand` precedent (docs/colleague-stack.md): ship the CONSERVATIVE
    # list — adding a responsibility later is contract-compatible, removing one
    # is a break. associate must never claim something worker does not.
    assert set(roles_mod.ROLE_RESPONSIBILITIES["associate"]) < set(
        roles_mod.ROLE_RESPONSIBILITIES["worker"]
    )


def test_associate_is_text_only_like_the_checkpoint_it_serves() -> None:
    for token in ("image_understanding", "video_understanding"):
        assert token not in roles_mod.ROLE_RESPONSIBILITIES["associate"]


# ---------------------------------------------------------------------------
# Criterion 2 — the capability rung
# ---------------------------------------------------------------------------


def test_tier_role_places_associate_at_the_highest_non_cortex_rung() -> None:
    assert TIER_ROLE["associate"] == "associate"
    last_pos: dict[str, int] = {}
    for index, role in enumerate(TIER_ROLE.values()):
        last_pos[role] = index
    ascending = sorted(last_pos, key=last_pos.__getitem__)
    assert ascending == ["hand", "multimodal", "worker", "muse", "associate", "primary"]


def test_associate_resolves_to_the_lightning_gear_it_shares_with_worker() -> None:
    # One checkpoint, two public addresses with different authority. The
    # catalog holds ONE entry per id, so the tier layer resolves associate
    # through the declared hint alias rather than a duplicated entry.
    assert BACKEND_ROLE_CATALOG_HINT["associate"] == "worker"
    assert resolve_tier("associate").id == _LIGHTNING_ID
    assert resolve_tier("associate").id == resolve_tier("worker").id
    # The role registry's own model naming agrees with the tier layer's.
    assert roles_mod.ROLE_ROLE_HINT["associate"] == BACKEND_ROLE_CATALOG_HINT["associate"]


# ---------------------------------------------------------------------------
# Criterion 3 — every public address, and the unhosted-role honesty rule
# ---------------------------------------------------------------------------


def test_capabilities_names_associate_with_its_full_metadata_block() -> None:
    registry = roles_mod.role_registry_from_env(_base_env(), gateway_url="http://gw:8000")
    assert "associate" in registry
    info = registry["associate"]
    assert info.role == "associate"
    assert info.model == _LIGHTNING_ID
    assert info.path == "/v1/chat/completions"
    assert info.responsibilities == roles_mod.ROLE_RESPONSIBILITIES["associate"]
    assert info.forbidden_responsibilities == roles_mod.ROLE_FORBIDDEN["associate"]
    # Unwired opt-in role => honestly infeasible, never advertised as usable.
    assert info.feasible is False
    assert info.loaded is False
    assert info.ready is False


def test_lobes_up_addresses_associate_and_the_bundle_excludes_it() -> None:
    assert up_cmd.ROLE_SERVICE["associate"] == "vllm-associate"
    assert "associate" in up_cmd.TARGETS
    # Opt-in-hosted, exactly like muse/worker: never in the default bundle.
    assert "associate" not in DEFAULT_HOSTED_ROLES


def test_associate_is_an_opt_in_core_role_on_every_layer() -> None:
    assert OPT_IN_CORE_ROLES == ("muse", "worker", "associate")
    assert "associate" in OPT_IN_BACKENDS
    assert "associate" in PROFILE_ROLES
    assert ROLE_ENV_PREFIX["associate"] == "ASSOCIATE"
    assert ROLE_SERVICE["associate"] == "vllm-associate"
    assert OPT_IN_CORE_COMPOSE_PROFILE["associate"] == "associate"
    assert OPT_IN_CORE_ACTIVATION_ENV["associate"] == {
        "ASSOCIATE_BASE_URL": "http://vllm-associate:8000",
    }


def test_unwired_associate_is_infeasible_by_default() -> None:
    table, _cfg = build_config(_base_env())
    assert "associate" in table.infeasible
    # ... and an explicit truthy declaration still wins over the default.
    table, _cfg = build_config(_base_env(ASSOCIATE_BASE_URL="http://vllm-associate:8000"))
    assert "associate" not in table.infeasible


def _post(table, cfg, model: str):
    """POST ``model`` at the generate lane with EVERY outbound dial a tripwire."""
    calls: list[str] = []

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append(backend.name)
        raise AssertionError(f"gateway dialed {backend.name} for an infeasible role")

    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], json.dumps({"model": model}).encode(), opener
    )
    # The whole point: no silent substitution by cortex or any other gear.
    assert calls == []
    return resp


def test_unhosted_associate_404s_role_infeasible_and_is_never_downgraded() -> None:
    table, cfg = build_config(_base_env())
    resp = _post(table, cfg, "associate")
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "role_infeasible"


def test_unhosted_associates_concrete_model_id_behaves_exactly_like_workers() -> None:
    """A raw model id nobody serves is ``model_not_found`` — for BOTH roles.

    associate and worker share a checkpoint, so this also pins that the tenth
    role introduced no new behaviour on the concrete-id path: whatever the
    incumbent does, associate does.
    """
    table, cfg = build_config(_base_env())
    resp = _post(table, cfg, _LIGHTNING_ID)
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "model_not_found"
    # worker's own alias still 404s role_infeasible, unchanged by the tenth role.
    worker_resp = _post(table, cfg, "worker")
    assert worker_resp.status == 404
    assert json.loads(worker_resp.body)["error"]["type"] == "role_infeasible"


def test_v1_models_omits_an_unhosted_associate() -> None:
    table, _cfg = build_config(_base_env())
    ids = {row["id"] for row in list_models_payload(table)["data"]}
    assert _LIGHTNING_ID not in ids
    assert "associate" not in ids


# ---------------------------------------------------------------------------
# Criterion 4 — pressure policy: associate sheds, hand is the only floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["cortex", "senses", "worker", "muse", "associate"])
def test_full_tiers_including_associate_shed_under_pressure(tier: str) -> None:
    verdict = decide(90.0, 10.0, tier)
    assert verdict["shed"] is True
    assert verdict["reason"] == "pressure"


@pytest.mark.parametrize("tier", ["hand", "minor", "cheap"])
def test_hand_remains_the_only_servable_floor(tier: str) -> None:
    verdict = decide(90.0, 90.0, tier)
    assert verdict["shed"] is False
    assert verdict["servable_tier"] == "hand"


def test_associate_is_not_a_floor_for_any_shed_tier() -> None:
    # Whatever is shed, the tier offered back is `hand` — never associate.
    for tier in TIER_ROLE:
        verdict = decide(90.0, 90.0, tier)
        assert verdict["servable_tier"] == "hand"


# ---------------------------------------------------------------------------
# Criterion 5 — no existing role's behaviour changes
# ---------------------------------------------------------------------------


def test_the_nine_pre_existing_prefixes_route_identically_without_associate() -> None:
    """Removing associate from the tier ladder must change NOTHING else.

    The tier layer derives its upward-fallback ladder from TIER_ROLE's value
    ORDER, so a new rung is exactly the kind of change that can silently
    re-point another alias. This recomputes every pre-existing alias against a
    TIER_ROLE with associate deleted and asserts byte-equality.
    """
    table, _cfg = build_config(_base_env())
    without = {tier: role for tier, role in TIER_ROLE.items() if tier != "associate"}
    with_associate = tier_aliases(table.backends, TIER_ROLE)
    without_associate = tier_aliases(table.backends, without)
    assert {k: v for k, v in with_associate.items() if k != "associate"} == without_associate


def test_every_pre_existing_role_still_resolves_to_its_own_backend() -> None:
    for role in _PRE_EXISTING_ROLES:
        assert (
            roles_mod.ROLE_BACKEND[role]
            == {
                "cortex": "primary",
                "senses": "multimodal",
                "muse": "muse",
                "worker": "worker",
                "hand": "hand",
                "embedder": "embed",
                "reranker": "rerank",
                "stt": "stt",
                "tts": "tts",
            }[role]
        )


def test_a_deployment_declaring_no_associate_config_is_otherwise_unchanged() -> None:
    """The only delta on an untouched deployment is associate's own honesty.

    Everything else about the routing table — backends, default model, aliases
    (minus the new one), peer channels — is byte-identical to the pre-associate
    contract, and the ONLY new fact is that associate joins muse/worker in the
    unwired-opt-in infeasible set.
    """
    env = _base_env()
    table, _cfg = build_config(env)
    assert table.infeasible == frozenset({"muse", "worker", "associate"})
    # No associate knob declared anywhere => no peer channel armed for it.
    assert "associate" not in table.peer_origins
    assert "associate" not in table.peer_proxied
    assert "associate" not in table.peer_api_keys
    # Wired backends are untouched: associate adds no backend without its URL.
    assert "associate" not in {b.name for b in table.backends}
    # And the pre-existing aliases still resolve to the pre-existing gears.
    assert resolve_model(table, "cortex") == env["PRIMARY_SERVED_NAME"]
    assert resolve_model(table, "main") == env["PRIMARY_SERVED_NAME"]
    assert resolve_model(table, "senses") == env["MULTIMODAL_SERVED_NAME"]


# ---------------------------------------------------------------------------
# Criterion 6 — the <PREFIX>_* env vocabulary (#199), extended to a tenth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channel,suffix",
    [
        (FEASIBLE_ENV, "FEASIBLE"),
        (PEER_ORIGIN_ENV, "PEER_ORIGIN"),
        (PEER_PROXY_ENV, "PEER_PROXY"),
        (PEER_API_KEY_ENV, "PEER_API_KEY"),
        (PEER_ORIGINS_ENV, "PEER_ORIGINS"),
        (PEER_API_KEYS_ENV, "PEER_API_KEYS"),
    ],
)
def test_associate_carries_every_channel_the_other_roles_carry(channel: dict, suffix: str) -> None:
    assert channel["associate"] == f"ASSOCIATE_{suffix}"


def test_associate_is_not_exempted_from_proxying() -> None:
    # `hand`'s d1 reversal emptied this set; associate must not re-open it.
    assert "associate" not in NEVER_PROXIED_BACKENDS
    assert set(PEER_ORIGIN_ENV) == set(FEASIBLE_ENV) - NEVER_PROXIED_BACKENDS


def test_associate_resolves_a_peer_served_name_so_its_proxy_knob_is_not_inert() -> None:
    # The 0.54.6 worker lesson: a role wired through _config's peer dicts but
    # missing from server.py's two peer tables proxies SILENTLY NOTHING.
    assert S._PEER_SERVED_NAME_ENV["associate"] == "ASSOCIATE_SERVED_NAME"
    assert S._PEER_ROLE_HINT["associate"] == "worker"


def test_associate_replica_pool_channels_are_declared_positionally() -> None:
    env = _base_env(
        ASSOCIATE_PEER_ORIGINS="http://a.local:8001,http://b.local:8001",
        ASSOCIATE_PEER_API_KEYS="key-a,key-b",
    )
    table, _cfg = build_config(env)
    assert table.replica_origins["associate"] == (
        "http://a.local:8001",
        "http://b.local:8001",
    )
    assert table.replica_api_keys["associate"] == ("key-a", "key-b")
