"""Render a resolved :class:`~lobes.profiles.schema.Profile` to fleet ``.env`` vars.

The single profile -> env mapping ``lobes init`` (t4) uses to turn "which
:class:`Profile` did detection/``--profile`` resolve?" into the concrete
``KEY=VALUE`` lines the fleet compose template
(``lobes/templates/fleet/docker-compose.yml`` / ``env.example``) already reads
via ``${PREFIX_SUFFIX:-default}``. Nothing here writes a file — see
:func:`lobes.runtime._env.set_env` for that; this module is pure.

**Role -> env prefix.** The five :data:`~lobes.profiles.schema.ROLES` map onto
the compose template's five service prefixes:

=========  ============
role       env prefix
=========  ============
cortex     PRIMARY
senses     MULTIMODAL
muse       MUSE
embedder   EMBED
reranker   RERANK
=========  ============

**Knob -> env suffix.** Every :class:`~lobes.profiles.schema.RoleProfile` field
other than ``feasible``/``model`` maps to ``<PREFIX>_<SUFFIX>`` by uppercasing
the field name (``gpu_mem_util`` -> ``GPU_MEM_UTIL``, etc.) — this happens to
match every knob name the compose template already spells out
(``PRIMARY_GPU_MEM_UTIL``, ``EMBED_ATTENTION_BACKEND``, ...), so no
translation table beyond "uppercase the field name" is needed. ``model`` is the
one field rendered to TWO keys (``<PREFIX>_MODEL`` and ``<PREFIX>_SERVED_NAME``,
both set to the same value) — the compose template passes the served name to
vLLM's ``--served-model-name`` separately from the model id it downloads/serves,
and the two must agree for the gateway to route to what actually got served.

**``enforce_eager`` is not a plain value knob.** vLLM's ``enforce_eager: bool``
field is exposed as ``argparse.BooleanOptionalAction`` in the compose command
list, i.e. the env var must hold a WHOLE CLI TOKEN (``--enforce-eager`` or
``--no-enforce-eager``), never a bare ``true``/``false`` string (see
``RERANK_ENFORCE_EAGER`` in ``env.example``/``docker-compose.yml``). This module
translates ``True``/``False`` to that token pair.

**``feasible=False`` and the "not yet loaded" convention.** As of t4, neither
:mod:`lobes.roles` nor the gateway defines an env-level "this role's backend is
absent" marker for the *compose/render* layer — ``lobes.roles`` derives
``loaded`` from whether a role's ``*_BASE_URL``/``*_URL`` is set (a wiring
fact), not from a profile's ``feasible`` bit. Rather than invent an env var the
gateway silently ignores forever, this module emits a narrow, clearly-named
placeholder — ``<PREFIX>_FEASIBLE=false`` — and nothing else for an infeasible
role (in particular it does NOT emit ``<PREFIX>_MODEL`` etc. even if the
:class:`~lobes.profiles.schema.RoleProfile` also carries knob opinions, since a
role the box cannot serve has no model to name). Wiring the gateway/CLI to
actually honor this marker (e.g. by omitting the role from ``lobes
capabilities`` or refusing to bring its compose service up) is left to a later
task (t6) — this module only guarantees the marker exists in ``.env`` for that
task to read.

A ``feasible=True`` role (the default) never gets a ``<PREFIX>_FEASIBLE`` key
at all — "feasible" is the assumed default the compose template already
encodes, so spelling out ``=true`` for every role would just be noise.

**The ENGINE axis.** A role's ``model`` also decides WHICH inference server
serves it: the catalog declares an ``engine`` per gear, and a non-vLLM gear's
lane is parked behind a Docker Compose profile and reached at its own origin.
Both of those are ``.env`` values, so they are rendered here — derived from the
same ``model`` declaration that names the gear, never from a second knob that
could drift from it. A card declaring only vLLM gears (every card but ``orin``
today) renders exactly what it rendered before this axis existed. See
:func:`role_engine` / :data:`LLAMA_CPP_ACTIVATION_ENV`.

**Card-level (non-role) keys — ``host_env``.** A few ``.env`` keys the compose
template reads belong to no lane at all (the gateway's pressure-policy
thresholds are the whole of it today). A card whose HOST accounting makes the
shipped default wrong declares them in its profile's ``host_env`` table (see
:attr:`lobes.profiles.schema.Profile.host_env`) and this module renders them
verbatim — *before* the role keys, so a role knob always wins a name
collision and ``host_env`` can never shadow ``PRIMARY_GPU_MEM_UTIL`` & co. A
profile that declares no ``host_env`` (every built-in but ``orin``) renders
exactly what it rendered before the field existed.
"""

from __future__ import annotations

from lobes.catalog import ENGINE_LLAMA_CPP, ENGINE_VLLM, SUPPORTED_MODELS
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.profiles.schema import ROLES, Profile, RoleProfile

# role -> the fleet compose template's env-var prefix for that role's service.
ROLE_ENV_PREFIX: dict[str, str] = {
    "cortex": "PRIMARY",
    "senses": "MULTIMODAL",
    "muse": "MUSE",
    "worker": "WORKER",
    "hand": "HAND",
    "embedder": "EMBED",
    "reranker": "RERANK",
}

# RoleProfile knob field name -> env-var suffix, appended to the role prefix
# with "_" (e.g. "cortex" + "gpu_mem_util" -> "PRIMARY_GPU_MEM_UTIL"). Matches
# the fleet compose template's own ${PREFIX_SUFFIX:-default} spellings exactly
# (lobes/templates/fleet/docker-compose.yml / env.example) — verified against
# both files as part of this module's design (see the module docstring).
_KNOB_ENV_SUFFIX: dict[str, str] = {
    "gpu_mem_util": "GPU_MEM_UTIL",
    "max_model_len": "MAX_MODEL_LEN",
    "quantization": "QUANTIZATION",
    "kv_cache_dtype": "KV_CACHE_DTYPE",
    "attention_backend": "ATTENTION_BACKEND",
    "enforce_eager": "ENFORCE_EAGER",
    "max_num_seqs": "MAX_NUM_SEQS",
    # hf_overrides -> PRIMARY_HF_OVERRIDES (t5): threaded to the compose
    # command's --hf-overrides flag. allow_long_max_model_len ->
    # PRIMARY_ALLOW_LONG_MAX_MODEL_LEN: threaded to the vllm-primary
    # container's VLLM_ALLOW_LONG_MAX_MODEL_LEN environment entry. Both are
    # plain str knobs, so no special-case branch is needed below (only
    # "model" and "enforce_eager" get one).
    "hf_overrides": "HF_OVERRIDES",
    "allow_long_max_model_len": "ALLOW_LONG_MAX_MODEL_LEN",
    # speculative_config -> PRIMARY_SPECULATIVE_CONFIG: the compose command's
    # ${PREFIX_SPECULATIVE_CONFIG-'--speculative-config={...}'} slot. Rendered
    # VERBATIM (a plain str knob, no re-quoting here) because the value already
    # carries both quoting layers the slot needs -- see RoleProfile's field
    # comment in lobes.profiles.schema for why that is the author's job and not
    # this module's. An empty string is meaningful (spec-decode OFF), which is
    # why the loop below skips only None.
    #
    # This mapping is deliberately uniform across role prefixes even though
    # only three lanes expand the variable today. The gate lives at LOAD time
    # instead (schema.SPECULATIVE_CONFIG_ROLES), so a role whose lane has no
    # slot is refused when it is DECLARED -- with a message naming why --
    # rather than silently rendering a key nothing reads. Keeping the render
    # side uniform means wiring a new lane's slot is a one-line change there,
    # not two changes that can drift apart.
    "speculative_config": "SPECULATIVE_CONFIG",
}

# The two argparse.BooleanOptionalAction tokens vLLM's --enforce-eager /
# --no-enforce-eager flag accepts — see RERANK_ENFORCE_EAGER in
# lobes/templates/fleet/docker-compose.yml for the idiom this mirrors.
_ENFORCE_EAGER_TOKEN = {True: "--enforce-eager", False: "--no-enforce-eager"}


# --- the ENGINE axis (qwen3-8-gguf-llamacpp t5) ------------------------------
#
# Every role's model was, until 2026-08-23, served by vLLM, so a role's model id
# was the only thing this module needed from it. The catalog now DECLARES an
# engine per gear (lobes/catalog.py's `engine` field / `serves_with_vllm`), and a
# llama.cpp GGUF gear is served by a DIFFERENT compose lane — one that is parked
# behind a Docker Compose profile and reached at its own origin.
#
# Those two facts are ``.env`` values, so they are rendered HERE, from the same
# `model` declaration that names the gear — never as a second knob a profile
# could set out of step with the model it describes. The engine is read straight
# off the catalog entry for the model id; a model the catalog does not list (an
# operator profile naming an arbitrary checkpoint) takes the vLLM default, which
# is exactly what every card rendered before this axis existed. So every
# pre-existing profile renders byte-identically, and the identity-shape
# invariant (machine-as-brain == the bare card profile) is preserved on a card
# that DOES declare a non-vLLM gear, because the derivation lives on this side
# of the composition rather than in the shape layer.
_ENGINE_BY_MODEL_ID: dict[str, str] = {model.id: model.engine for model in SUPPORTED_MODELS}

# role -> the .env keys that ACTIVATE its llama.cpp lane. Today only `cortex` has
# an alternative-engine lane at all (`llamacpp-primary` in the fleet template).
# The role ALIAS callers use does not move with the engine — only the origin
# behind it does. Mirrors the shipped template exactly (same design as
# ROLE_ENV_PREFIX); tests/test_shape_goldens.py verifies the mirror against the
# real compose file.
# NOTE on the http:// scheme (SonarCloud python:S5332 flags it): this is a
# CONTAINER-INTERNAL address on the private compose network, resolved by the
# Docker DNS alias `llamacpp-primary` and never reachable off-box — the lane
# publishes no host port at all. Every other in-fleet backend URL is the same
# shape (`http://vllm-primary:8000`, `http://vllm-embed:8000`, …), and
# CLAUDE.md states the fleet assumption explicitly: peer origins ride a
# private/tailnet transport and "no TLS termination happens at this layer".
# Serving TLS between two containers in one compose project would add a
# certificate lifecycle for no threat it mitigates. Not a finding here.
LLAMA_CPP_ACTIVATION_ENV: dict[str, dict[str, str]] = {
    "cortex": {"PRIMARY_URL": "http://llamacpp-primary:8000"},  # NOSONAR python:S5332
}

#: The Docker Compose profile gating every llama.cpp lane in the fleet template.
LLAMA_CPP_COMPOSE_PROFILE = "llamacpp"


def role_engine(rp: RoleProfile) -> str:
    """The engine serving ``rp``'s model — a catalog fact, defaulting to vLLM.

    :param rp: A role's (possibly shape-composed)
        :class:`~lobes.profiles.schema.RoleProfile`.
    :returns: One of :data:`lobes.catalog.ENGINES`. A role with no model, or a
        model the catalog does not list, answers
        :data:`lobes.catalog.ENGINE_VLLM` — the pre-engine-axis behaviour.
    """
    return _ENGINE_BY_MODEL_ID.get(rp.model or "", ENGINE_VLLM)


def _engine_activation_env(role: str, rp: RoleProfile) -> dict[str, str]:
    """The ``.env`` keys activating ``role``'s non-vLLM lane; ``{}`` for a vLLM gear.

    A non-vLLM model on a role with NO alternative lane is a LOAD ERROR rather
    than a silently ignored declaration: the render would otherwise name a GGUF
    as the model while pointing the gateway at a ``vllm serve`` lane that cannot
    load it.
    """
    engine = role_engine(rp)
    if engine == ENGINE_VLLM:
        return {}
    # Gate the lookup on the engine, not merely on "not vLLM". LLAMA_CPP_ACTIVATION_ENV
    # is keyed by ROLE, so an ungated `.get(role)` hands a THIRD engine (sglang, since
    # 0.60.0) the llama.cpp lane's activation env — a cortex gear declaring
    # engine="sglang" would silently render PRIMARY_URL=http://llamacpp-primary:8000
    # and COMPOSE_PROFILES=llamacpp, pointing the gateway at a `llama-server` lane that
    # cannot load it. shape_render._role_service already gates this way; this is the
    # matching half, so both paths refuse an engine with no lane instead of disagreeing.
    activation = LLAMA_CPP_ACTIVATION_ENV.get(role) if engine == ENGINE_LLAMA_CPP else None
    if activation is None:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=(
                f"role {role!r} declares model {rp.model!r}, which the catalog serves with "
                f"{engine!r} — but no {engine!r} lane exists for that role"
            ),
            remediation=(
                f"serve {role!r} with a vLLM gear, or add a {engine!r} lane for it to "
                "lobes/templates/fleet/docker-compose.yml, LLAMA_CPP_ACTIVATION_ENV "
                "and shape_render.LLAMA_CPP_ROLE_SERVICE"
            ),
        )
    return dict(activation)


def _role_env(role: str, rp: RoleProfile) -> dict[str, str]:
    prefix = ROLE_ENV_PREFIX[role]
    env: dict[str, str] = {}
    if not rp.feasible:
        # An infeasible role has nothing to serve — no model/knob opinions are
        # rendered for it, just the marker a later task (t6) will honor.
        env[f"{prefix}_FEASIBLE"] = "false"
        return env
    if rp.model is not None:
        env[f"{prefix}_MODEL"] = rp.model
        env[f"{prefix}_SERVED_NAME"] = rp.model
    for field_name, suffix in _KNOB_ENV_SUFFIX.items():
        value = getattr(rp, field_name)
        if value is None:
            continue
        env_name = f"{prefix}_{suffix}"
        if field_name == "enforce_eager":
            env[env_name] = _ENFORCE_EAGER_TOKEN[bool(value)]
        elif isinstance(value, bool):
            env[env_name] = "true" if value else "false"
        else:
            env[env_name] = str(value)
    return env


def profile_env(profile: Profile) -> dict[str, str]:
    """The ``.env`` entries a resolved :class:`Profile` renders to.

    Pure — takes no filesystem/network action, just a plain ``dict`` a caller
    (``lobes init``, t4) merges into the deployment's ``.env`` (see
    :func:`lobes.runtime._env.set_env`). Only knobs the profile actually has an
    opinion on (non-``None`` fields, or an explicit ``feasible=False``) produce
    entries — a role/knob the profile is silent on is simply absent from the
    returned dict, leaving the compose template's own ``${VAR:-default}`` in
    effect. See the module docstring for the role->prefix table, the
    knob->env-suffix mapping, the ``model``->two-keys special case, the
    ``enforce_eager`` bool->flag-token translation, the
    ``<PREFIX>_FEASIBLE=false`` marker convention for ``feasible=False`` roles,
    and the card-level ``host_env`` passthrough.
    """
    # host_env FIRST: a role knob rendered below always wins a name collision,
    # so a card's non-role declaration can never shadow a lane's own budget.
    env: dict[str, str] = dict(profile.host_env)
    compose_profiles: list[str] = []
    for role in ROLES:
        role_profile = profile.role(role)
        env.update(_role_env(role, role_profile))
        # A feasible role whose model is a non-vLLM catalog gear ALSO needs its
        # lane started and wired, in the same render that names the model — the
        # engine's lane is compose-profile-gated and the gateway dials the origin
        # it is given. An INFEASIBLE role activates nothing (it renders only the
        # #110 flagged-off marker), which is what makes a shape that drops the
        # role leave the alternative lane parked too.
        if not role_profile.feasible:
            continue
        activation = _engine_activation_env(role, role_profile)
        if activation:
            env.update(activation)
            compose_profiles.append(LLAMA_CPP_COMPOSE_PROFILE)
    if compose_profiles:
        env["COMPOSE_PROFILES"] = ",".join(compose_profiles)
    return env
