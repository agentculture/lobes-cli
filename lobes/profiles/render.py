"""Render a resolved :class:`~lobes.profiles.schema.Profile` to fleet ``.env`` vars.

The single profile -> env mapping ``lobes init`` (t4) uses to turn "which
:class:`Profile` did detection/``--profile`` resolve?" into the concrete
``KEY=VALUE`` lines the fleet compose template
(``lobes/templates/fleet/docker-compose.yml`` / ``env.example``) already reads
via ``${PREFIX_SUFFIX:-default}``. Nothing here writes a file — see
:func:`lobes.runtime._env.set_env` for that; this module is pure.

**Role -> env prefix.** The four :data:`~lobes.profiles.schema.ROLES` map onto
the compose template's four service prefixes:

=========  ============
role       env prefix
=========  ============
cortex     PRIMARY
senses     MULTIMODAL
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
"""

from __future__ import annotations

from lobes.profiles.schema import ROLES, Profile, RoleProfile

# role -> the fleet compose template's env-var prefix for that role's service.
ROLE_ENV_PREFIX: dict[str, str] = {
    "cortex": "PRIMARY",
    "senses": "MULTIMODAL",
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
}

# The two argparse.BooleanOptionalAction tokens vLLM's --enforce-eager /
# --no-enforce-eager flag accepts — see RERANK_ENFORCE_EAGER in
# lobes/templates/fleet/docker-compose.yml for the idiom this mirrors.
_ENFORCE_EAGER_TOKEN = {True: "--enforce-eager", False: "--no-enforce-eager"}


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
    ``enforce_eager`` bool->flag-token translation, and the
    ``<PREFIX>_FEASIBLE=false`` marker convention for ``feasible=False`` roles.
    """
    env: dict[str, str] = {}
    for role in ROLES:
        env.update(_role_env(role, profile.role(role)))
    return env
