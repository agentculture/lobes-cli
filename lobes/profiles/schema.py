"""The per-machine :class:`Profile` schema — the third profiling axis.

Where the legacy :class:`~lobes.profiles.MachineProfile` (still exported from
:mod:`lobes.profiles`, see the package ``__init__``) is one flat row of
single-model knobs, a :class:`Profile` is the FLEET-shaped declaration this
package resolves: per :data:`ROLES` entry (``cortex`` / ``senses`` / ``muse`` /
``worker`` / ``embedder`` / ``reranker``), whether that role is even feasible on
the target box, which model serves it, and every machine knob the compose templates
substitute (``gpu_mem_util``, ``max_model_len``, ``quantization``,
``kv_cache_dtype``, ``attention_backend``, ``enforce_eager``,
``max_num_seqs``).

Both dataclasses are frozen — a :class:`Profile` loaded by
:mod:`lobes.profiles.loader` is never mutated in place; a caller that wants a
variant builds a new one with :func:`dataclasses.replace`.

Every knob is optional (``None`` = "use the compose template's own default"):
a profile only needs to *say* something about the knobs it actually diverges
on — that is how :mod:`lobes.profiles.builtin` keeps the Thor profile down to
its four validated divergences instead of restating Spark's whole table.

Alongside the per-role tables a profile may also declare a small ``host_env``
table: **card-level** ``.env`` keys that belong to no role at all (see
:attr:`Profile.host_env`). Facts about the BOARD that the compose template
reads box-wide — today only the gateway's pressure-policy thresholds — have
nowhere else to live: they are not a vLLM knob on any lane, so they cannot be
a :class:`RoleProfile` field, and they are not a HOSTING decision, so they do
not belong on a :class:`~lobes.profiles.shapes.Shape` either (a shape-scoped
fix would leave the default ``machine-as-brain`` shape on the same card
broken).

A profile may also declare :attr:`Profile.gpu_access` — the one card fact that
is not an ``.env`` value at all, but a choice of which compose SYNTAX the box's
NVIDIA container runtime accepts. See that attribute's docstring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping

from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError

# The per-machine-profile roles — the six gateway-fronted, generate/pooling
# lanes a compose template actually parameterises per machine. Deliberately a
# SUBSET of lobes.roles.ROLES: stt/tts are fixed audio sidecars (Parakeet /
# Chatterbox) with no machine-dependent vLLM knobs of their own — they are out
# of scope for this schema, matching lobes/roles.py's own
# ROLE_MAX_MODEL_LEN_ENV, which likewise carries no stt/tts entry. ``muse``
# (the opt-in Gemma 4 31B creative lobe) and ``worker`` (the opt-in
# Qwen3.6-35B-A3B ground-work lobe — the fast DOER) are BOTH in scope — each
# carries the full per-machine knob set — but are hosted only by an explicit
# muse-/worker-hosting deployment shape, never by machine-as-brain (see
# lobes.profiles.shapes.OPT_IN_CORE_ROLES).
ROLES: tuple[str, ...] = (
    "cortex",
    "senses",
    "muse",
    "worker",
    # `hand` (LiquidAI LFM2.5-1.2B, the fine-tuning base) is in scope and
    # carries the full per-machine knob set like every other generate lane —
    # but unlike muse/worker it is DEFAULT-HOSTED: ~2.4 GiB of bf16 weights fit
    # beside any other lane on any supported card, which is the entire point of
    # the role. Its per-card gpu_mem_util is declared PER CARD, never once
    # globally: 0.06 is 7.7 GiB on a 128 GB Spark but 3.84 GiB on a 64 GB Orin.
    "hand",
    "embedder",
    "reranker",
)

# The machine knobs a compose template substitutes per role/gear. Order here
# is the canonical field order on RoleProfile below (minus feasible/model).
KNOB_NAMES: tuple[str, ...] = (
    "gpu_mem_util",
    "max_model_len",
    "quantization",
    "kv_cache_dtype",
    "attention_backend",
    "enforce_eager",
    "max_num_seqs",
    # hf_overrides / allow_long_max_model_len (t5 of devague plan
    # lobes-adopts-qwen3.8-27b-nvfp4-as-cortex-p): the two knobs the
    # spark-lobe shape's 1M-context YaRN hypothesis needs to reach the
    # vllm-primary compose command/environment. Both are plain strings
    # (never a bool/JSON dataclass) so the existing str-knob render path in
    # lobes.profiles.render._role_env handles them with no special case, the
    # same way kv_cache_dtype/attention_backend already do.
    "hf_overrides",
    "allow_long_max_model_len",
)


# A ``host_env`` key must be a plain env-var name (a POSIX shell identifier) —
# the .env file is a KEY=VALUE surface, so anything else could not be written
# there at all. Values are required to be STRINGS for the same reason: the
# author writes the exact bytes that land in .env, with no int/float
# formatting surprise (``100`` vs ``100.0``) between the TOML and the file.
# NOTE for future readers (and for SonarCloud's S6353, which suggests `\w`):
# the explicit ASCII class is DELIBERATE and `\w` is NOT equivalent here.
# Python's `\w` is Unicode-aware by default, so `[A-Za-z_]\w*` also accepts
# names like "CAFE\u0301_VAR" or "A\u03c0" — verified. Env-var names written into
# .env must stay ASCII, so widening this would let a profile smuggle a
# non-ASCII key into the file. Keep the class explicit rather than pairing
# `\w` with a re.ASCII flag: the constraint belongs where it is read.
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# How a card's container runtime is asked for the GPU. The compose templates
# spell the modern form; a card whose NVIDIA container toolkit resolves to the
# legacy CSV mode declares the other one and `lobes init` generates the
# override that rewrites it (see Profile.gpu_access).
GPU_ACCESS_DEVICES = "devices"
GPU_ACCESS_RUNTIME = "runtime"
GPU_ACCESS_MODES: tuple[str, ...] = (GPU_ACCESS_DEVICES, GPU_ACCESS_RUNTIME)


def _profile_error(message: str, remediation: str) -> ModelGearError:
    return ModelGearError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _gpu_access_from_value(name: str, raw: Any) -> str:
    """Validate a profile's ``gpu_access`` declaration (loud, never silent)."""
    if not isinstance(raw, str) or raw not in GPU_ACCESS_MODES:
        raise _profile_error(
            message=f"profile {name!r}: gpu_access must be one of {list(GPU_ACCESS_MODES)}, "
            f"got {raw!r}",
            remediation=(
                f'declare gpu_access = "{GPU_ACCESS_DEVICES}" (the default: '
                f'deploy.resources GPU request) or "{GPU_ACCESS_RUNTIME}" '
                "(csv-mode boards: runtime: nvidia)"
            ),
        )
    return raw


def _host_env_from_dict(name: str, raw: Any) -> dict[str, str]:
    """Validate and build a profile's card-level ``host_env`` table.

    Rejects a non-mapping, a key that is not a valid env-var name, and a
    non-string value — the same loud-not-silent contract
    :meth:`RoleProfile.from_dict` applies to knobs.
    """
    if not isinstance(raw, Mapping):
        raise _profile_error(
            message=f"profile {name!r}: 'host_env' must be a table/mapping",
            remediation='declare it as a [host_env] table of KEY = "value" pairs',
        )
    host_env: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key):
            raise _profile_error(
                message=f"profile {name!r}: host_env key {key!r} is not a valid env-var name",
                remediation="use a plain env-var name (letters, digits, underscore; "
                "never starting with a digit)",
            )
        if not isinstance(value, str):
            got = type(value).__name__
            raise _profile_error(
                message=(
                    f"profile {name!r}: host_env value for {key!r} must be str, "
                    f"got {got} ({value!r})"
                ),
                remediation=f'quote it — host_env values are written to .env verbatim: {key} = "…"',
            )
        # `.env` is line-oriented KEY=VALUE and `_env.set_env` writes the value
        # verbatim, so an embedded newline would not "escape" — it would SPLIT
        # the entry into two physical lines, silently turning the tail into a
        # bogus key or a parse error. Reject it here, where the operator still
        # has the profile file in front of them, rather than letting a corrupt
        # .env surface as an unrelated failure at boot.
        if any(ch in value for ch in ("\n", "\r", "\x00")):
            raise _profile_error(
                message=(
                    f"profile {name!r}: host_env value for {key!r} contains a newline "
                    f"or NUL — it would corrupt .env, which is line-oriented KEY=VALUE"
                ),
                remediation="keep host_env values on a single line; move anything "
                "multi-line into a file the service reads instead",
            )
        host_env[key] = value
    return host_env


def _is_strict_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_optional_bool(value: Any) -> bool:
    return value is None or isinstance(value, bool)


def _is_optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _is_optional_number(value: Any) -> bool:
    # bool is a subclass of int in Python — reject it explicitly BEFORE the
    # isinstance(value, (int, float)) check, or `feasible = "false"`-style
    # TOML mistakes (here, a stray `true`/`false` on a numeric knob) would
    # silently pass as a number.
    if isinstance(value, bool):
        return False
    return value is None or isinstance(value, (int, float))


def _is_optional_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return value is None or isinstance(value, int)


# Shared "expected" description for every Optional[str] knob below — defined
# once so the literal isn't duplicated across the validator table (S1192).
_STR_OR_NONE = "str or None"

# Per-field type validator + human-readable "expected" description, used by
# RoleProfile.from_dict to reject a value of the wrong TYPE (not just an
# unknown key) — e.g. `feasible = "false"` (a truthy STRING) must fail loudly
# rather than silently flip a role to feasible via Python truthiness.
_FIELD_VALIDATORS: dict[str, tuple[Any, str]] = {
    "feasible": (_is_strict_bool, "bool"),
    "model": (_is_optional_str, _STR_OR_NONE),
    "gpu_mem_util": (_is_optional_number, "int/float or None"),
    "max_model_len": (_is_optional_int, "int or None"),
    "quantization": (_is_optional_str, _STR_OR_NONE),
    "kv_cache_dtype": (_is_optional_str, _STR_OR_NONE),
    "attention_backend": (_is_optional_str, _STR_OR_NONE),
    "enforce_eager": (_is_optional_bool, "bool or None"),
    "max_num_seqs": (_is_optional_int, "int or None"),
    "hf_overrides": (_is_optional_str, _STR_OR_NONE),
    "allow_long_max_model_len": (_is_optional_str, _STR_OR_NONE),
}


@dataclass(frozen=True)
class RoleProfile:
    """One role's serving declaration within a :class:`Profile`.

    ``feasible=False`` means the target box cannot serve this role at all
    (t6 wires that into ``lobes capabilities`` / the gateway); ``model`` is
    the served model id. Every other field is a machine knob, ``None`` when
    the profile takes no position (the compose template's own ``${VAR:-...}``
    default applies).
    """

    feasible: bool = True
    model: str | None = None
    gpu_mem_util: float | None = None
    max_model_len: int | None = None
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    attention_backend: str | None = None
    enforce_eager: bool | None = None
    max_num_seqs: int | None = None
    # A raw vLLM --hf-overrides JSON string (matching lobes.catalog.SupportedModel's
    # own hf_overrides field's shape/purpose) — a role/machine's opinion on the
    # checkpoint config override, e.g. a YaRN rope_parameters patch for a
    # long-context shape. None means "no opinion" like every other knob.
    hf_overrides: str | None = None
    # The VLLM_ALLOW_LONG_MAX_MODEL_LEN passthrough env value ("1" to allow a
    # max_model_len beyond the checkpoint's declared ceiling, e.g. serving a
    # YaRN-scaled 1M window). A plain string (not bool) so it renders through
    # the same str-knob path as kv_cache_dtype/attention_backend — vLLM reads
    # the env var as a raw string, not a "true"/"false" token.
    allow_long_max_model_len: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view — every declared field, ``None`` included.

        Deliberately keeps ``None`` entries (rather than dropping them) so
        ``from_dict(to_dict())`` round-trips exactly: a caller that reads back
        the dict sees the same "this profile is silent on this knob" shape it
        started with.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @staticmethod
    def from_dict(role: str, data: Mapping[str, Any]) -> "RoleProfile":
        """Build one role's declaration, rejecting any unrecognised key.

        An unknown knob name is a LOAD ERROR, never a silently dropped key —
        a typo'd knob in an operator-authored profile must fail loudly rather
        than pretend the operator's intended override was applied.
        """
        known = {f.name for f in fields(RoleProfile)}
        unknown = set(data.keys()) - known
        if unknown:
            raise _profile_error(
                message=f"unknown knob(s) {sorted(unknown)!r} for role {role!r}",
                remediation=f"known knobs: feasible, model, {', '.join(KNOB_NAMES)}",
            )
        for key, value in data.items():
            validator, expected = _FIELD_VALIDATORS[key]
            if not validator(value):
                got = type(value).__name__
                raise _profile_error(
                    message=(
                        f"role {role!r}: knob {key!r} must be {expected}, got {got} ({value!r})"
                    ),
                    remediation=(
                        f"fix the value's type for {key!r} in role {role!r} "
                        f"(expected {expected})"
                    ),
                )
        return RoleProfile(**dict(data))


@dataclass(frozen=True)
class Profile:
    """A named, per-role machine tuning declaration — the fleet profile axis.

    ``roles`` is read-only (a :class:`~types.MappingProxyType`) so a loaded
    profile can be shared freely without a caller accidentally mutating the
    built-in/operator source of truth.

    ``host_env`` is the card's **non-role** ``.env`` declaration: box-wide keys
    the compose template reads that belong to no lane — today only the
    gateway's pressure-policy thresholds, where a card's own ``/proc/stat``
    accounting quirk makes the shipped default wrong for that board (see
    ``lobes/profiles/builtin/orin.toml``). Rendered verbatim by
    :func:`lobes.profiles.render.profile_env`, BEFORE the role keys, so a role
    knob can never be shadowed by a host_env entry; a card that declares none
    (every profile but ``orin`` today) renders byte-identically to before this
    field existed. Read-only, like ``roles``.

    ``gpu_access`` is the card's **container-runtime** declaration — how this
    board's NVIDIA container toolkit must be asked for the GPU:

    * ``"devices"`` (the DEFAULT, every profile but ``orin``) — the compose
      templates' own ``deploy.resources.reservations.devices`` request, i.e.
      today's behaviour. Nothing extra is generated and the deployment is
      byte-identical to before this field existed.
    * ``"runtime"`` — the board's toolkit resolves to the legacy **csv** mode,
      where that request fails at container create ("invoking the NVIDIA
      Container Runtime Hook directly … is not supported"). ``lobes init``
      GENERATES a compose override that ``!reset``s each GPU service's
      ``deploy:`` stanza and sets ``runtime: nvidia`` instead (see
      ``lobes.cli._commands.init.render_gpu_overrides``).

    It is deliberately NOT a ``host_env`` key: docker-compose has no
    conditional-block syntax, so no ``${VAR}`` substitution can switch between
    the two forms — only a second compose file can. And it is NOT a
    :class:`~lobes.profiles.shapes.Shape` field: which syntax the runtime
    accepts is a fact about the BOARD, true of every shape rendered over it
    (a shape-scoped fix would leave a bare ``lobes init`` on the same board
    broken — the same reasoning as ``host_env``).
    """

    name: str
    summary: str = ""
    roles: Mapping[str, RoleProfile] = field(default_factory=dict)
    host_env: Mapping[str, str] = field(default_factory=dict)
    gpu_access: str = GPU_ACCESS_DEVICES

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))
        object.__setattr__(self, "host_env", MappingProxyType(dict(self.host_env)))

    def role(self, name: str) -> RoleProfile:
        """The declaration for one role; an absent role is fully permissive.

        A profile that says nothing about a role (e.g. a minimal operator
        override touching only ``cortex``) means "no opinion" for the rest,
        not "infeasible" — callers that need to know whether a role was
        EXPLICITLY declared should consult ``name in profile.roles`` instead.
        """
        return self.roles.get(name, RoleProfile())

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, ``{"name", "summary", "roles", "host_env", "gpu_access"}``."""
        return {
            "name": self.name,
            "summary": self.summary,
            "roles": {role: rp.to_dict() for role, rp in self.roles.items()},
            "host_env": dict(self.host_env),
            "gpu_access": self.gpu_access,
        }

    @staticmethod
    def from_dict(name: str, data: Mapping[str, Any]) -> "Profile":
        """Build a :class:`Profile` from a parsed TOML/JSON mapping.

        Validates the top-level shape and every role name before building
        each :class:`RoleProfile` (which validates its own knob names) — an
        unrecognised role (anything outside :data:`ROLES`) is a LOAD ERROR,
        exactly like an unrecognised knob. The optional ``host_env`` table and
        ``gpu_access`` value are validated the same way
        (:func:`_host_env_from_dict` / :func:`_gpu_access_from_value`).
        """
        known_top = {"name", "summary", "roles", "host_env", "gpu_access"}
        unknown_top = set(data.keys()) - known_top
        if unknown_top:
            raise _profile_error(
                message=f"unknown top-level key(s) {sorted(unknown_top)!r} in profile {name!r}",
                remediation=f"known keys: {sorted(known_top)}",
            )
        summary = data.get("summary", "")
        raw_roles = data.get("roles", {})
        if not isinstance(raw_roles, Mapping):
            raise _profile_error(
                message=f"profile {name!r}: 'roles' must be a table/mapping",
                remediation="declare roles as [roles.<name>] tables",
            )
        unknown_roles = set(raw_roles.keys()) - set(ROLES)
        if unknown_roles:
            raise _profile_error(
                message=f"unknown role(s) {sorted(unknown_roles)!r} in profile {name!r}",
                remediation=f"known roles: {', '.join(ROLES)}",
            )
        roles = {
            role: RoleProfile.from_dict(role, role_data) for role, role_data in raw_roles.items()
        }
        host_env = _host_env_from_dict(name, data.get("host_env", {}))
        gpu_access = _gpu_access_from_value(name, data.get("gpu_access", GPU_ACCESS_DEVICES))
        # declared-name wins over an embedded "name" field (loader passes the
        # filename stem, which is the source of truth for a profile's identity).
        declared_name = data.get("name", name)
        if declared_name != name:
            raise _profile_error(
                message=(
                    f"profile file for {name!r} declares name={declared_name!r} "
                    "— the two must match"
                ),
                remediation="rename the file or fix the 'name' field so they agree",
            )
        return Profile(
            name=name,
            summary=summary,
            roles=roles,
            host_env=host_env,
            gpu_access=gpu_access,
        )
