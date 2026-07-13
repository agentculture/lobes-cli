"""The per-machine :class:`Profile` schema — the third profiling axis.

Where the legacy :class:`~lobes.profiles.MachineProfile` (still exported from
:mod:`lobes.profiles`, see the package ``__init__``) is one flat row of
single-model knobs, a :class:`Profile` is the FLEET-shaped declaration this
package resolves: per :data:`ROLES` entry (``cortex`` / ``senses`` /
``embedder`` / ``reranker``), whether that role is even feasible on the target
box, which model serves it, and every machine knob the compose templates
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
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping

from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError

# The per-machine-profile roles — the four gateway-fronted, generate/pooling
# lanes a compose template actually parameterises per machine. Deliberately a
# SUBSET of lobes.roles.ROLES: stt/tts are fixed audio sidecars (Parakeet /
# Chatterbox) with no machine-dependent vLLM knobs of their own — they are out
# of scope for this schema, matching lobes/roles.py's own
# ROLE_MAX_MODEL_LEN_ENV, which likewise carries no stt/tts entry.
ROLES: tuple[str, ...] = ("cortex", "senses", "embedder", "reranker")

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
)


def _profile_error(message: str, remediation: str) -> ModelGearError:
    return ModelGearError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


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
    """

    name: str
    summary: str = ""
    roles: Mapping[str, RoleProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))

    def role(self, name: str) -> RoleProfile:
        """The declaration for one role; an absent role is fully permissive.

        A profile that says nothing about a role (e.g. a minimal operator
        override touching only ``cortex``) means "no opinion" for the rest,
        not "infeasible" — callers that need to know whether a role was
        EXPLICITLY declared should consult ``name in profile.roles`` instead.
        """
        return self.roles.get(name, RoleProfile())

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, ``{"name": ..., "summary": ..., "roles": {...}}``."""
        return {
            "name": self.name,
            "summary": self.summary,
            "roles": {role: rp.to_dict() for role, rp in self.roles.items()},
        }

    @staticmethod
    def from_dict(name: str, data: Mapping[str, Any]) -> "Profile":
        """Build a :class:`Profile` from a parsed TOML/JSON mapping.

        Validates the top-level shape and every role name before building
        each :class:`RoleProfile` (which validates its own knob names) — an
        unrecognised role (anything outside :data:`ROLES`) is a LOAD ERROR,
        exactly like an unrecognised knob.
        """
        known_top = {"name", "summary", "roles"}
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
        return Profile(name=name, summary=summary, roles=roles)
