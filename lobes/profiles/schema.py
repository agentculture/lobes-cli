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
    # speculative_config (dspark-adopt-shape-default): the raw
    # ``--speculative-config=...`` argv token a shape/card wants on the lane,
    # threaded to the compose command's ${PREFIX_SPECULATIVE_CONFIG-default}
    # slot. A shape needs this to declare a draft that is NOT the checkpoint's
    # own baked-in MTP head -- e.g. spark-lobe's DSpark block drafter. Like
    # hf_overrides it is a plain string, so the str-knob render path handles
    # it with no special case. See the QUOTING note on the dataclass field
    # below: the value carries its own shell quotes, and that is deliberate.
    "speculative_config",
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
# Roles whose compose lane actually CONSUMES a <PREFIX>_SPECULATIVE_CONFIG
# slot, and therefore the only roles a profile/shape may declare
# ``speculative_config`` for.
#
# This is not a taste judgement about which lanes "should" speculate -- it is a
# fact about lobes/templates/fleet/docker-compose.yml. Three lanes expand the
# variable (vllm-primary, vllm-multimodal, vllm-worker). The rest do not:
# vllm-muse HARDCODES its `--speculative-config` token as a YAML LIST element
# (a list cannot host the dash-only ${VAR-default} idiom, whose whole point is
# to vanish when the value is empty -- an empty list element is an empty argv
# token, not an omitted flag), and the pooling gears (vllm-embed, vllm-rerank)
# plus vllm-hand carry no speculative flag at all.
#
# Rendering the key for one of those roles would put a variable in `.env` that
# NOTHING reads -- a silent no-op, and this repo's rule is that a knob which
# cannot take effect must fail loudly at load rather than pretend it applied
# (the same rule that makes an unknown knob name a load error, above).
# Un-gating a role means giving its lane a real slot FIRST, then adding it
# here; for `muse` that also means converting its command from a list to the
# shell-lexed string form vllm-primary/vllm-worker use.
SPECULATIVE_CONFIG_ROLES: frozenset[str] = frozenset({"cortex", "senses", "worker"})


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


# The keys one ``[[exclusive_roles]]`` entry may carry. ``roles`` is the group
# itself; ``shapes`` names the deployment shapes that resolve it (so the
# refusal can point at a concrete alternative instead of "pass --shape"); and
# ``reason`` is the measured arithmetic that makes the group true, quoted back
# to the operator verbatim.
_EXCLUSIVE_KEYS = ("roles", "shapes", "reason")


@dataclass(frozen=True)
class ExclusiveRoles:
    """One declared **mutually-exclusive** role group on a card.

    Feasibility (:attr:`RoleProfile.feasible`) and CO-RESIDENCY are different
    axes, and only the first is a per-role fact. A board can be perfectly
    capable of serving two roles — each one boots, each one is honestly
    ``feasible = true`` — and still not have the memory to run BOTH at once.
    Nothing in :class:`RoleProfile` can say that, because it is a statement
    about a *pair*, not about either member.

    This is where a card says it. Declaring::

        [[exclusive_roles]]
        roles = ["cortex", "senses"]
        shapes = ["orin-cortex", "orin-lobe"]
        reason = "…the measured arithmetic…"

    means "this board can serve either, never both together". It is read by
    ``lobes init``'s co-residency guard
    (:func:`lobes.profiles.shape_render.overcommitted_groups`), which refuses
    to scaffold a deployment whose resolved shape would host more than one
    member — the ``machine-as-brain`` default being exactly the shape that
    would.

    ``shapes`` is not decoration: the guard's whole value is naming the way
    OUT, and deriving that from the built-in shape set would surface every
    shape on every card (``spark-lobe`` "resolves" a cortex/senses conflict
    too — on the wrong board). It is declared here, and
    ``tests/test_init_coresidency.py`` proves each named shape exists and
    actually resolves the group, so the declaration cannot drift into a lie.
    ``reason`` is likewise carried into the error text — a refusal that states
    the numbers is one an operator can check.

    Frozen and role-name-validated at load, like everything else in this
    module; a group of fewer than two roles, a duplicate member, or a role
    outside :data:`ROLES` is a LOAD ERROR, never a silently dropped
    declaration.
    """

    roles: tuple[str, ...]
    shapes: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view — the exact shape :func:`_exclusive_group` accepts back."""
        return {
            "roles": list(self.roles),
            "shapes": list(self.shapes),
            "reason": self.reason,
        }


def _str_tuple(name: str, key: str, raw: Any, *, minimum: int) -> tuple[str, ...]:
    """A validated tuple of plain strings for one ``exclusive_roles`` key."""
    if not isinstance(raw, (list, tuple)) or not all(isinstance(item, str) for item in raw):
        raise _profile_error(
            message=f"profile {name!r}: exclusive_roles {key!r} must be a list of strings, "
            f"got {raw!r}",
            remediation=f'declare it as {key} = ["…", "…"]',
        )
    values = tuple(raw)
    if len(values) < minimum:
        raise _profile_error(
            message=(
                f"profile {name!r}: exclusive_roles {key!r} needs at least {minimum} "
                f"entr{'y' if minimum == 1 else 'ies'}, got {list(values)!r}"
            ),
            remediation=(
                "a mutual-exclusion group is a statement about a PAIR — one role "
                "cannot conflict with itself"
                if key == "roles"
                else f"name the shape(s) that resolve the group in {key}"
            ),
        )
    if len(set(values)) != len(values):
        raise _profile_error(
            message=f"profile {name!r}: exclusive_roles {key!r} repeats an entry: {list(values)!r}",
            remediation=f"list each name once in {key}",
        )
    return values


def _exclusive_group(name: str, raw: Any) -> ExclusiveRoles:
    """Validate and build ONE ``[[exclusive_roles]]`` entry."""
    if not isinstance(raw, Mapping):
        raise _profile_error(
            message=f"profile {name!r}: each exclusive_roles entry must be a table/mapping",
            remediation="declare each group as a [[exclusive_roles]] table with "
            f"keys: {', '.join(_EXCLUSIVE_KEYS)}",
        )
    unknown = set(raw.keys()) - set(_EXCLUSIVE_KEYS)
    if unknown:
        raise _profile_error(
            message=(f"profile {name!r}: unknown exclusive_roles key(s) {sorted(unknown)!r}"),
            remediation=f"known keys: {', '.join(_EXCLUSIVE_KEYS)}",
        )
    roles = _str_tuple(name, "roles", raw.get("roles", []), minimum=2)
    unknown_roles = set(roles) - set(ROLES)
    if unknown_roles:
        raise _profile_error(
            message=(
                f"profile {name!r}: unknown role(s) {sorted(unknown_roles)!r} "
                "in an exclusive_roles group"
            ),
            remediation=f"known roles: {', '.join(ROLES)}",
        )
    shapes = _str_tuple(name, "shapes", raw.get("shapes", []), minimum=1)
    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        raise _profile_error(
            message=f"profile {name!r}: exclusive_roles 'reason' must be str, "
            f"got {type(reason).__name__} ({reason!r})",
            remediation='quote it — reason = "…the measured arithmetic…"',
        )
    return ExclusiveRoles(roles=roles, shapes=shapes, reason=reason)


def _exclusive_roles_from_value(name: str, raw: Any) -> tuple[ExclusiveRoles, ...]:
    """Validate and build a profile's ``exclusive_roles`` declaration."""
    if not isinstance(raw, (list, tuple)):
        raise _profile_error(
            message=f"profile {name!r}: 'exclusive_roles' must be a list of tables",
            remediation="declare each group as a [[exclusive_roles]] table",
        )
    return tuple(_exclusive_group(name, entry) for entry in raw)


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
    "speculative_config": (_is_optional_str, _STR_OR_NONE),
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
    # The lane's ``--speculative-config=...`` argv token, or None for "no
    # opinion" (the compose template's own default applies -- for cortex that
    # is the checkpoint's self-hosted MTP head at n=2).
    #
    # QUOTING (load-bearing, and the reason this is a raw token rather than a
    # parsed JSON dataclass): the compose slot is UNQUOTED --
    # ``${PRIMARY_SPECULATIVE_CONFIG-'--speculative-config={...}'}`` -- because
    # the DEFAULT supplies its own single quotes. A value substituted there
    # crosses compose's dotenv parser first and its shell-lexer second, so it
    # must carry BOTH layers itself: the .env line is double-quoted (so dotenv
    # keeps the inner quotes) and the value inside is single-quoted (so the
    # shell-lexer yields ONE argv token instead of splitting on the JSON's
    # spaces and eating its double quotes). Authors write those exact bytes
    # here; ``lobes.profiles.render`` renders them verbatim. The empty string
    # is the documented OFF switch (the flag is omitted from the argv, not
    # blanked) -- that is why the dash-only ``${VAR-default}`` operator is used
    # in the template, and why None (silent) and "" (off) mean different
    # things here. See lobes/templates/fleet/env.example's
    # PRIMARY_SPECULATIVE_CONFIG / MULTIMODAL_SPECULATIVE_CONFIG blocks.
    #
    # LIFECYCLE CAVEAT (true of EVERY knob, not just this one; raised by
    # review on PR #202, Qodo finding 3). `None` restores the template default
    # on a FRESH render only. `.env` is merge-only: `lobes init --apply`
    # force-writes the keys the resolved profile renders and leaves every
    # other line alone, so REMOVING a declaration does not remove a key an
    # earlier render already wrote -- the stale line keeps winning. Clearing a
    # previously-rendered knob on a live box means deleting that line from
    # `.env` by hand. Tracked as a lifecycle gap in issue #204, not a
    # property of this field.
    speculative_config: str | None = None

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
        if data.get("speculative_config") is not None and role not in SPECULATIVE_CONFIG_ROLES:
            servable = ", ".join(sorted(SPECULATIVE_CONFIG_ROLES))
            raise _profile_error(
                message=(
                    f"role {role!r}: knob 'speculative_config' has no effect — the "
                    f"{role!r} lane's compose command does not expand "
                    f"<PREFIX>_SPECULATIVE_CONFIG"
                ),
                remediation=(
                    f"declare 'speculative_config' only for: {servable}. "
                    "Serving a different draft on another lane needs that lane's "
                    "compose command to grow a ${PREFIX_SPECULATIVE_CONFIG-default} "
                    "slot first (see SPECULATIVE_CONFIG_ROLES)"
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

    ``exclusive_roles`` is the card's **co-residency** declaration — the one
    thing a per-role table structurally cannot say, because it is a statement
    about a PAIR of roles rather than about either one (see
    :class:`ExclusiveRoles`). It renders NO ``.env`` key and no compose file:
    what it reaches is ``lobes init``'s co-residency guard, which refuses to
    scaffold a deployment whose resolved shape would host two roles the card
    declares mutually exclusive. A card that declares none (every profile but
    ``orin`` today) is completely unaffected — the guard finds no group and
    the render is byte-identical.
    """

    name: str
    summary: str = ""
    roles: Mapping[str, RoleProfile] = field(default_factory=dict)
    host_env: Mapping[str, str] = field(default_factory=dict)
    gpu_access: str = GPU_ACCESS_DEVICES
    exclusive_roles: tuple[ExclusiveRoles, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))
        object.__setattr__(self, "host_env", MappingProxyType(dict(self.host_env)))
        object.__setattr__(self, "exclusive_roles", tuple(self.exclusive_roles))

    def role(self, name: str) -> RoleProfile:
        """The declaration for one role; an absent role is fully permissive.

        A profile that says nothing about a role (e.g. a minimal operator
        override touching only ``cortex``) means "no opinion" for the rest,
        not "infeasible" — callers that need to know whether a role was
        EXPLICITLY declared should consult ``name in profile.roles`` instead.
        """
        return self.roles.get(name, RoleProfile())

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view of every declared field (round-trips through ``from_dict``)."""
        return {
            "name": self.name,
            "summary": self.summary,
            "roles": {role: rp.to_dict() for role, rp in self.roles.items()},
            "host_env": dict(self.host_env),
            "gpu_access": self.gpu_access,
            "exclusive_roles": [group.to_dict() for group in self.exclusive_roles],
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
        known_top = {"name", "summary", "roles", "host_env", "gpu_access", "exclusive_roles"}
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
        exclusive_roles = _exclusive_roles_from_value(name, data.get("exclusive_roles", []))
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
            exclusive_roles=exclusive_roles,
        )
