"""Resolve a :class:`~lobes.profiles.schema.Profile` by name.

Two sources, merged with operator precedence:

* **built-ins** — packaged data under :mod:`lobes.profiles.builtin`, read via
  :func:`importlib.resources.files` (so they work from an installed wheel, not
  just a source checkout);
* **operator-defined** — TOML files an operator drops in
  ``<deployment-dir>/profiles/<name>.toml`` (the same filename convention as
  the built-ins: the file's stem is the profile name). A same-named operator
  file OVERRIDES the built-in entirely — the two are never merged field by
  field, so an operator profile is always a complete, self-contained
  declaration of what it *does* say (any role/knob it stays silent on falls
  back to "no opinion", per :meth:`Profile.role`, not to the shadowed
  built-in's value).

Nothing here mutates a :class:`Profile` — every load produces a fresh, frozen
object; the same name resolved twice returns two independent (``==``-equal)
instances, never the same shared one being edited out from under a caller.

Card *detection* (matching a live box to a profile name) is a later task
(``lobes/runtime/_detect.py``); this module only resolves a profile once a
name is already known — "an explicit profile name wins over any detection" is
satisfied trivially here because detection doesn't happen in this module at
all.
"""

from __future__ import annotations

import tomllib
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

from lobes import machines
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.profiles.schema import Profile

BUILTIN_PACKAGE = "lobes.profiles.builtin"
PROFILE_SUFFIX = ".toml"
OPERATOR_SUBDIR = "profiles"

# The one built-in whose knobs are partly DERIVED (not re-typed) from the
# lobes.machines registry — see _apply_machine_registry below.
_MACHINE_DERIVED_BUILTINS = ("thor",)


def _parse(text: str, *, source: str) -> dict:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"malformed profile TOML in {source}: {exc}",
            remediation="fix the TOML syntax (tomllib is the parser — stdlib, no YAML)",
        ) from exc


def _apply_machine_registry(profile: Profile) -> Profile:
    """Overlay live :mod:`lobes.machines` knobs onto a built-in profile.

    Keeps a machine-validated divergence (e.g. Thor's sm_110 pooling quirks)
    SINGLE-SOURCED in the chip-strategy registry rather than re-typed as a
    literal in a ``builtin/*.toml`` file: ``builtin/thor.toml`` on disk only
    carries the template-shared baseline (same models/util/context/quant as
    Spark); this function fills in the knobs the matching
    :class:`~lobes.machines.CardStrategy` actually measured, straight from
    its :meth:`~lobes.machines.CardStrategy.role_knobs`. A change to the
    SM_110 trait or a chip's ``role_overrides`` is therefore reflected here
    with no edit to this module or to the TOML file.
    """
    strategy = machines.get(profile.name)
    if strategy is None:
        return profile
    updated_roles = dict(profile.roles)
    for role, knobs in strategy.role_knobs().items():
        if role not in updated_roles:
            continue
        overrides = {knob_name: knob.value for knob_name, knob in knobs.items()}
        updated_roles[role] = replace(updated_roles[role], **overrides)
    # Construct the Profile directly (rather than dataclasses.replace(profile,
    # ...)) so the declared return type matches what a static checker infers —
    # replace()'s generic signature resolves to the base DataclassInstance
    # protocol for some checkers, not the concrete Profile subtype.
    return Profile(
        name=profile.name, summary=profile.summary, roles=MappingProxyType(updated_roles)
    )


def builtin_names() -> tuple[str, ...]:
    """The names of every packaged built-in profile, sorted."""
    root = files(BUILTIN_PACKAGE)
    names = [
        entry.name[: -len(PROFILE_SUFFIX)]
        for entry in root.iterdir()
        if entry.name.endswith(PROFILE_SUFFIX)
    ]
    return tuple(sorted(names))


def load_builtin(name: str) -> Profile | None:
    """Load one packaged built-in profile by name, or ``None`` if it doesn't exist."""
    root = files(BUILTIN_PACKAGE)
    node = root / f"{name}{PROFILE_SUFFIX}"
    if not node.is_file():
        return None
    data = _parse(node.read_text(encoding="utf-8"), source=f"builtin:{name}")
    profile = Profile.from_dict(name, data)
    if name in _MACHINE_DERIVED_BUILTINS:
        profile = _apply_machine_registry(profile)
    return profile


def _operator_dir(deploy_dir: Path | str) -> Path:
    return Path(deploy_dir).expanduser() / OPERATOR_SUBDIR


def discover_operator_profiles(deploy_dir: Path | str) -> dict[str, Profile]:
    """Operator-authored profiles found in ``<deploy_dir>/profiles/*.toml``.

    Absent/empty directory -> ``{}`` (never raises just for "no operator
    profiles here" — that is the common case, not an error).

    Keyed by the filename stem NORMALISED (``.strip().lower()``), matching how
    :func:`resolve_profile` normalises the requested name — an operator file
    named e.g. ``Thor.toml`` must still be found when a caller asks for
    ``thor`` (or ``THOR``, or `` thor ``). Two files that collide after
    normalisation (``Thor.toml`` and ``thor.toml`` both present) is an
    unresolvable ambiguity — which one wins is a coin flip an operator would
    never want silently made for them — so it is a LOAD ERROR rather than a
    silently-picked winner.
    """
    operator_dir = _operator_dir(deploy_dir)
    if not operator_dir.is_dir():
        return {}
    found: dict[str, Profile] = {}
    raw_names_by_key: dict[str, str] = {}
    for path in sorted(operator_dir.glob(f"*{PROFILE_SUFFIX}")):
        raw_name = path.stem
        key = raw_name.strip().lower()
        if key in found:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=(
                    f"operator profile name collision in {operator_dir}: "
                    f"{raw_names_by_key[key]!r} and {raw_name!r} both normalise "
                    f"to {key!r}"
                ),
                remediation=(
                    "rename one of the files so their names differ after "
                    "case-folding (profile names are matched case-insensitively)"
                ),
            )
        data = _parse(path.read_text(encoding="utf-8"), source=str(path))
        # Pass the RAW stem (not the normalised key) as the profile's declared
        # identity — the file's own casing is what an embedded `name = "..."`
        # field (if present) must match, per Profile.from_dict's mismatch
        # check; only the DICT KEY used for lookup is normalised, matching how
        # resolve_profile() normalises the requested name.
        found[key] = Profile.from_dict(raw_name, data)
        raw_names_by_key[key] = raw_name
    return found


def available_profiles(deploy_dir: Path | str | None = None) -> dict[str, Profile]:
    """Every resolvable profile: built-ins, then operator profiles override by name."""
    merged: dict[str, Profile] = {name: load_builtin(name) for name in builtin_names()}
    if deploy_dir is not None:
        merged.update(discover_operator_profiles(deploy_dir))
    return merged


def resolve_profile(name: str, deploy_dir: Path | str | None = None) -> Profile:
    """Resolve ``name`` to a :class:`Profile` — operator profiles win over built-ins.

    Raises :class:`ModelGearError` (``EXIT_USER_ERROR``) for an unknown name;
    never falls back to a default profile silently.
    """
    key = (name or "").strip().lower()
    if deploy_dir is not None:
        operator = discover_operator_profiles(deploy_dir)
        if key in operator:
            return operator[key]
    builtin = load_builtin(key)
    if builtin is not None:
        return builtin
    known = ", ".join(builtin_names())
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"unknown profile {name!r}",
        remediation=f"choose one of: {known}, or add <deploy-dir>/profiles/{name}.toml",
    )
