"""The :class:`Shape` schema — the DEPLOYMENT-SHAPE axis (brain-shapes t1).

Where a :class:`~lobes.profiles.schema.Profile` says how each role is TUNED on
a given card (the per-machine axis, #108), a :class:`Shape` says which roles a
BOX actually HOSTS at all — the orthogonal deployment-shape axis: does this
box run the whole brain (``machine-as-brain``), or only the lobes it is best
at, leaving the rest to a peer box in the mesh (a ``mesh-brain`` shape, e.g.
``spark-lobe`` / ``thor-lobe``)?

A Shape is composed OVER the #108 Profile schema rather than re-implementing
it: its role vocabulary (:data:`SHAPE_ROLES`) is
:data:`~lobes.profiles.schema.ROLES` (the four Profile-machinery core roles —
``cortex``/``senses``/``embedder``/``reranker``) plus :data:`AUDIO_ROLES`
(``stt``/``tts``, the opt-in audio-overlay sidecars —
``lobes/templates/fleet/docker-compose.audio.yml``) plus :data:`OPT_IN_ROLES`
(``minor``, the opt-in ``vllm-minor`` compose service — added in the
mesh-brain end-state's t2, issue #112, for the ``orin-small`` reference
shape); and its per-role budget ``overrides`` reuse
:class:`~lobes.profiles.schema.RoleProfile` verbatim (the SAME knob
vocabulary, :data:`~lobes.profiles.schema.KNOB_NAMES`, and the SAME
validation), so a shape's override table is never a parallel, re-typed
schema. ``stt``/``tts``/``minor`` carry no Profile knobs of their own (see
``schema.py``'s ``ROLES`` docstring for stt/tts; ``minor`` is the existing
opt-in ``vllm-minor`` service, whose knobs are the compose template's own
fixed ``${MINOR_MODEL:-...}``/``${VLLM_MINOR_GPU_MEM_UTIL:-...}``/
``${VLLM_MINOR_MAX_MODEL_LEN:-...}`` defaults) and so cannot appear in
``overrides`` — only in ``hosts``.

Every built-in shape (:mod:`lobes.profiles.builtin_shapes`) is PURE DATA: the
four shipped TOML files (``machine-as-brain``, ``spark-lobe``, ``thor-lobe``,
``orin-small``) differ from each other only in their ``hosts`` role subset
and their ``overrides`` budget re-derivation — there is no per-shape Python
branch anywhere in this module.

``overrides`` here are DECLARED DATA, never a runtime mutation: this task (t1)
leaves every shape's overrides empty (machine-as-brain must ALWAYS stay
empty — a single box hosting everything re-derives nothing); a later task
(t2) fills in ``spark-lobe``/``thor-lobe``'s re-derived budget knobs directly
in these same TOML files. Rendering a (shape, profile) pair into a concrete
compose/.env (overlaying ``overrides`` onto the resolved
:class:`~lobes.profiles.schema.Profile`) is a later task (t3) — this module
only defines and loads the shape itself.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Mapping

from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.profiles.schema import ROLES as PROFILE_ROLES
from lobes.profiles.schema import RoleProfile

# The two audio-overlay roles -- fixed sidecars (Parakeet STT / Chatterbox
# TTS) with no machine-dependent vLLM knobs of their own, exactly like
# schema.py's ROLES docstring already establishes for the Profile axis. Kept
# here (not imported from lobes.roles) so this module's only substrate
# dependency stays lobes.profiles.schema, per the brain-shapes t1 scope.
AUDIO_ROLES: tuple[str, ...] = ("stt", "tts")

# The six first-class, Colleague-facing roles (issue #81): the four
# Profile-machinery core roles plus the two audio-overlay sidecars. This is
# the "whole brain" set machine-as-brain hosts exactly -- the identity-shape
# invariant (see shape_render.py's module docstring and
# tests/goldens/regen.py's `_shape_needs_goldens`) is defined against THIS
# set, not the broader :data:`SHAPE_ROLES` below, because the opt-in `minor`
# gear (added after this constant, for t2) is deliberately excluded from
# "every role this card can serve" -- machine-as-brain never hosts it.
COLLEAGUE_ROLES: tuple[str, ...] = PROFILE_ROLES + AUDIO_ROLES

# The opt-in `minor` compose service (`vllm-minor`, gated in the fleet
# template by the "minor" Docker Compose profile -- see env.example's
# `COMPOSE_PROFILES=minor`) -- a light 4B bf16 generate gear that is
# DELIBERATELY NOT one of the six first-class Colleague roles (issue #81;
# see CLAUDE.md's "Colleague roles" section: "the 4B minor ... are opt-in
# gears and not first-class Colleague roles"). Added to the Shape schema's
# hostable vocabulary for the mesh-brain end-state's t2 (issue #112): a box
# with no heavy cortex/senses lobe at all (the `orin-small` reference shape)
# still needs SOME generate lane to host, and re-using the `cortex` role slot
# for it would mean the box advertises the 27B Colleague role while actually
# serving a 4B model behind it -- exactly the half-honest posture #92 exists
# to forbid. `minor` carries no Profile/RoleProfile knobs of its own (its
# budget is the compose template's own fixed defaults -- see the module
# docstring), so -- like AUDIO_ROLES -- it can only appear in `hosts`, never
# `overrides`.
OPT_IN_ROLES: tuple[str, ...] = ("minor",)

# Roles that carry no Profile/RoleProfile knobs of their own -- hostable, but
# never valid inside a shape's `overrides` table (there is nothing on them to
# re-derive a budget for).
_NO_OVERRIDE_ROLES: tuple[str, ...] = AUDIO_ROLES + OPT_IN_ROLES

# Every role a Shape may declare hosted: :data:`COLLEAGUE_ROLES` (the six
# first-class, Colleague-facing roles) plus the opt-in `minor` gear
# (:data:`OPT_IN_ROLES`) -- the one addition beyond that six-role vocabulary.
SHAPE_ROLES: tuple[str, ...] = COLLEAGUE_ROLES + OPT_IN_ROLES

BUILTIN_SHAPES_PACKAGE = "lobes.profiles.builtin_shapes"
SHAPE_SUFFIX = ".toml"


def _shape_error(message: str, remediation: str) -> ModelGearError:
    return ModelGearError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _parse(text: str, *, source: str) -> dict:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"malformed shape TOML in {source}: {exc}",
            remediation="fix the TOML syntax (tomllib is the parser -- stdlib, no YAML)",
        ) from exc


@dataclass(frozen=True)
class Shape:
    """A named deployment shape -- the role subset one box hosts.

    ``hosts`` is the role subset (a subset of :data:`SHAPE_ROLES`) this shape
    hosts; a role absent from ``hosts`` is simply not served by a box
    rendering this shape (t3's job; t5 wires the resulting honesty into
    ``lobes capabilities`` / the gateway). ``overrides`` is a per-CORE-role
    (never audio-role) budget re-derivation, reusing
    :class:`~lobes.profiles.schema.RoleProfile` verbatim -- a knob a shape
    stays silent on takes no position (composed with the resolved
    :class:`~lobes.profiles.schema.Profile` at render time, t3).

    Frozen, like :class:`~lobes.profiles.schema.Profile` -- a loaded Shape is
    never mutated in place.
    """

    name: str
    summary: str = ""
    hosts: tuple[str, ...] = ()
    overrides: Mapping[str, RoleProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hosts", tuple(self.hosts))
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    def hosts_role(self, role: str) -> bool:
        """Whether this shape hosts ``role`` at all."""
        return role in self.hosts

    def override(self, role: str) -> RoleProfile:
        """The declared budget override for ``role``; an undeclared role is fully permissive.

        Mirrors :meth:`~lobes.profiles.schema.Profile.role` -- a shape that
        says nothing about a role's budget means "no opinion", not "reset to
        zero", so a caller composing this with a resolved
        :class:`~lobes.profiles.schema.Profile` at render time can always
        call this safely, including for a role this shape doesn't even host.
        """
        return self.overrides.get(role, RoleProfile())

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, ``{"name", "summary", "hosts", "overrides"}``."""
        return {
            "name": self.name,
            "summary": self.summary,
            "hosts": list(self.hosts),
            "overrides": {role: rp.to_dict() for role, rp in self.overrides.items()},
        }

    @staticmethod
    def from_dict(name: str, data: Mapping[str, Any]) -> "Shape":
        """Build a :class:`Shape` from a parsed TOML/JSON mapping.

        An unrecognised top-level key, an unrecognised role in ``hosts``, an
        unrecognised role in ``overrides`` (including the roles in
        :data:`_NO_OVERRIDE_ROLES` -- the audio roles and ``minor`` -- which
        carry no Profile knobs to override), or a malformed knob value inside
        an override is always a LOAD ERROR -- never a silently dropped key or
        a silently ignored value, matching
        :meth:`~lobes.profiles.schema.Profile.from_dict`'s contract exactly.
        """
        known_top = {"name", "summary", "hosts", "overrides"}
        unknown_top = set(data.keys()) - known_top
        if unknown_top:
            raise _shape_error(
                message=f"unknown top-level key(s) {sorted(unknown_top)!r} in shape {name!r}",
                remediation=f"known keys: {sorted(known_top)}",
            )

        summary = data.get("summary", "")

        raw_hosts = data.get("hosts", [])
        if not isinstance(raw_hosts, (list, tuple)):
            raise _shape_error(
                message=f"shape {name!r}: 'hosts' must be a list of role names",
                remediation='declare hosts as hosts = ["cortex", "embedder", ...]',
            )
        unknown_hosts = set(raw_hosts) - set(SHAPE_ROLES)
        if unknown_hosts:
            raise _shape_error(
                message=f"unknown role(s) {sorted(unknown_hosts)!r} in shape {name!r} 'hosts'",
                remediation=f"known roles: {', '.join(SHAPE_ROLES)}",
            )

        raw_overrides = data.get("overrides", {})
        if not isinstance(raw_overrides, Mapping):
            raise _shape_error(
                message=f"shape {name!r}: 'overrides' must be a table/mapping",
                remediation="declare overrides as [overrides.<role>] tables",
            )
        unknown_override_roles = set(raw_overrides.keys()) - set(SHAPE_ROLES)
        if unknown_override_roles:
            raise _shape_error(
                message=(
                    f"unknown role(s) {sorted(unknown_override_roles)!r} "
                    f"in shape {name!r} 'overrides'"
                ),
                remediation=f"known roles: {', '.join(SHAPE_ROLES)}",
            )
        no_override_roles_used = set(raw_overrides.keys()) & set(_NO_OVERRIDE_ROLES)
        if no_override_roles_used:
            raise _shape_error(
                message=(
                    f"role(s) {sorted(no_override_roles_used)!r} in shape {name!r} 'overrides' "
                    "carry no Profile knobs to override (the audio-overlay sidecars stt/tts, "
                    "or the opt-in 'minor' gear)"
                ),
                remediation="only the four Profile-machinery core roles may appear in overrides: "
                f"{', '.join(PROFILE_ROLES)}",
            )
        overrides = {
            role: RoleProfile.from_dict(role, role_data)
            for role, role_data in raw_overrides.items()
        }

        # declared-name wins over an embedded "name" field (the loader passes
        # the filename stem, which is the source of truth for a shape's
        # identity) -- matches Profile.from_dict's mismatch check exactly.
        declared_name = data.get("name", name)
        if declared_name != name:
            raise _shape_error(
                message=(
                    f"shape file for {name!r} declares name={declared_name!r} "
                    "-- the two must match"
                ),
                remediation="rename the file or fix the 'name' field so they agree",
            )

        return Shape(name=name, summary=summary, hosts=tuple(raw_hosts), overrides=overrides)


def builtin_shape_names() -> tuple[str, ...]:
    """The names of every packaged built-in shape, sorted."""
    root = files(BUILTIN_SHAPES_PACKAGE)
    names = [
        entry.name[: -len(SHAPE_SUFFIX)]
        for entry in root.iterdir()
        if entry.name.endswith(SHAPE_SUFFIX)
    ]
    return tuple(sorted(names))


def load_builtin_shape(name: str) -> Shape | None:
    """Load one packaged built-in shape by name, or ``None`` if it doesn't exist."""
    root = files(BUILTIN_SHAPES_PACKAGE)
    node = root / f"{name}{SHAPE_SUFFIX}"
    if not node.is_file():
        return None
    data = _parse(node.read_text(encoding="utf-8"), source=f"builtin-shape:{name}")
    return Shape.from_dict(name, data)


def resolve_shape(name: str) -> Shape:
    """Resolve ``name`` to a built-in :class:`Shape`.

    Raises :class:`ModelGearError` (``EXIT_USER_ERROR``) for an unknown name;
    never falls back to a default shape silently. Mirrors
    :func:`~lobes.profiles.loader.resolve_profile`'s normalisation (trimmed,
    lower-cased) so ``lobes init --shape`` (a later task) gets the same
    forgiving matching as ``--profile`` already does.
    """
    key = (name or "").strip().lower()
    shape = load_builtin_shape(key)
    if shape is not None:
        return shape
    known = ", ".join(builtin_shape_names())
    raise _shape_error(
        message=f"unknown shape {name!r}",
        remediation=f"choose one of: {known}",
    )
