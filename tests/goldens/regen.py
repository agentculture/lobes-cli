"""Regeneration helper for ``tests/goldens/*.env`` (t13).

Run from the repo root with::

    uv run python tests/goldens/regen.py

Rewrites every golden this directory owns:

* ``tests/goldens/<profile>.env`` — one per packaged built-in profile
  (:func:`lobes.profiles.loader.builtin_names`, currently ``spark``/``thor``),
  the sorted ``KEY=VALUE`` projection of
  ``profile_env(resolve_profile(<name>))``.
* ``tests/goldens/template-defaults.env`` — the ``${VAR:-default}`` surface of
  ``lobes/templates/fleet/docker-compose.yml``.
* ``tests/goldens/shapes/<shape>__<card>.env`` — one per (deployment-shape,
  card) pair that is NOT the whole-brain identity shape (brain-shapes t3), the
  sorted ``KEY=VALUE`` projection of
  ``render_shape(resolve_shape(<shape>), resolve_profile(<card>)).env``. The
  identity shape ``machine-as-brain`` (hosts every role, no overrides) renders
  byte-identically to the bare ``<card>.env`` above, so it is validated against
  that existing golden by ``tests/test_shape_goldens.py`` rather than copied
  into a drifting duplicate here.

These are the byte-for-byte comparison targets in
``tests/test_profile_goldens.py``. Regenerating is a deliberate act, not a
reflex: if you only meant to change ONE machine (or nothing at all), diff the
regenerated files before committing — a change that also moves a golden you
didn't mean to touch is exactly the signal this suite exists to catch (see
``tests/goldens/README.md``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_GOLDENS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _GOLDENS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # allow `python tests/goldens/regen.py` standalone

from lobes.profiles.loader import builtin_names, resolve_profile  # noqa: E402
from lobes.profiles.render import profile_env  # noqa: E402
from lobes.profiles.shape_render import render_shape  # noqa: E402
from lobes.profiles.shapes import (  # noqa: E402
    COLLEAGUE_ROLES,
    builtin_shape_names,
    resolve_shape,
)

FLEET_COMPOSE = _REPO_ROOT / "lobes" / "templates" / "fleet" / "docker-compose.yml"
_SHAPES_DIR = _GOLDENS_DIR / "shapes"

_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def profile_env_text(name: str) -> str:
    """The sorted ``KEY=VALUE\\n`` projection of a resolved profile's rendering.

    Pure passthrough of ``profile_env(resolve_profile(name))`` — this module
    never reimplements the profile -> env mapping, it only formats what
    ``lobes.profiles.render`` already produced (see the task instructions:
    goldens MUST go through ``profile_env``, never a reimplementation).
    """
    env = profile_env(resolve_profile(name))
    lines = sorted(f"{key}={value}" for key, value in env.items())
    return "\n".join(lines) + "\n"


def extract_template_defaults(text: str) -> set[str]:
    """Every ``${VAR:-default}`` substitution in *text*, as ``"VAR=default"`` strings.

    A brace-depth walk rather than a single regex, because the template nests
    one substitution inside another's default —
    ``${HF_CACHE:-${HOME:-/root}/.cache/huggingface}`` — and a naive
    ``\\$\\{(\\w+):-([^}]*)\\}`` regex stops at the FIRST ``}`` it sees, which
    would mis-parse the outer ``HF_CACHE`` default as ``${HOME:-/root`` (missing
    its closing brace and the ``/.cache/huggingface`` tail). Walking brace depth
    finds the true matching close for the outer substitution, and — because the
    scan resumes at ``start + 2`` rather than past the whole matched span — the
    inner ``${HOME:-/root}`` is then found too, as its own separate entry.

    Bare ``${VAR}`` (no ``:-default``) is skipped on purpose: it isn't part of
    "the defaults surface" (e.g. ``${VLLM_PORT}`` in this template, which has
    no default and errors out with nothing else to substitute).

    Returns a **set** of ``"VAR=default"`` strings, not a ``dict`` keyed by
    ``VAR``. The template legitimately gives the same var name two different
    defaults in two places — e.g. ``MINOR_SERVED_NAME`` defaults to the real
    model id inside the (opt-in, off-by-default) ``vllm-minor`` service's own
    ``--served-model-name`` flag, but to ``""`` in the gateway's env block,
    since the gateway must not silently route to a gear nobody turned on. A
    dict would pick one of the two by write order and hide drift in the other;
    a set of raw ``"VAR=default"`` pairs keeps both distinct entries, while
    identical duplicates (``HF_CACHE`` appears once per service block, always
    with the same default) still collapse to a single golden line.
    """
    results: set[str] = set()
    i = 0
    n = len(text)
    while True:
        start = text.find("${", i)
        if start == -1:
            break
        depth = 1
        j = start + 2
        while j < n and depth > 0:
            if text[j : j + 2] == "${":
                depth += 1
                j += 2
                continue
            if text[j] == "}":
                depth -= 1
                j += 1
                continue
            j += 1
        inner = text[start + 2 : j - 1]
        i = start + 2  # advance past "${" only, so a nested "${" is still found
        name, sep, default = inner.partition(":-")
        if sep and _VAR_NAME_RE.fullmatch(name):
            results.add(f"{name}={default}")
    return results


def template_defaults_text() -> str:
    """The sorted ``VAR=default\\n`` projection of the fleet compose template."""
    text = FLEET_COMPOSE.read_text(encoding="utf-8")
    lines = sorted(extract_template_defaults(text))
    return "\n".join(lines) + "\n"


def _shape_needs_goldens(shape) -> bool:
    """Whether a shape gets its own ``shapes/`` goldens.

    The whole-brain identity shape (hosts every :data:`COLLEAGUE_ROLES` role
    -- the six first-class Colleague roles, NOT the broader
    :data:`~lobes.profiles.shapes.SHAPE_ROLES`, which also admits the opt-in
    `minor` gear that machine-as-brain deliberately never hosts -- with no
    overrides) renders identically to the bare card profile, so it is
    validated against the existing ``tests/goldens/<card>.env`` (see
    ``tests/test_shape_goldens.py``) rather than copied into a drifting
    duplicate. Every shape that DROPS a role, hosts `minor`, or carries an
    override diverges from the bare profile and gets per-card goldens of its
    own. General by construction: a future identity shape is auto-excluded, a
    future mesh-lobe (or small-model reference shape) auto-included.
    """
    return set(shape.hosts) != set(COLLEAGUE_ROLES) or bool(shape.overrides)


def shape_golden_pairs() -> list[tuple[str, str]]:
    """Every ``(shape, card)`` pair that gets a ``shapes/`` golden — sorted, deterministic.

    The cross product of the non-identity built-in shapes with every built-in
    card profile. Enumerated (not hardcoded) so a new shape or card is picked up
    automatically the next time this command runs.
    """
    pairs: list[tuple[str, str]] = []
    for shape_name in builtin_shape_names():
        if not _shape_needs_goldens(resolve_shape(shape_name)):
            continue
        for card_name in builtin_names():
            pairs.append((shape_name, card_name))
    return pairs


def shape_golden_path(shape_name: str, card_name: str) -> Path:
    """The on-disk golden path for a ``(shape, card)`` pair."""
    return _SHAPES_DIR / f"{shape_name}__{card_name}.env"


def shape_env_text(shape_name: str, card_name: str) -> str:
    """The sorted ``KEY=VALUE\\n`` projection of a (shape, card) rendering.

    Pure passthrough of ``render_shape(resolve_shape(shape), resolve_profile(card))``
    -- this module never reimplements the shape x card mapping, it only formats
    what ``lobes.profiles.shape_render`` already produced.
    """
    return render_shape(resolve_shape(shape_name), resolve_profile(card_name)).env_text()


def write_shape_goldens() -> list[Path]:
    """Rewrite every ``shapes/<shape>__<card>.env`` golden; returns the paths written."""
    _SHAPES_DIR.mkdir(exist_ok=True)
    written: list[Path] = []
    for shape_name, card_name in shape_golden_pairs():
        path = shape_golden_path(shape_name, card_name)
        path.write_text(shape_env_text(shape_name, card_name), encoding="utf-8")
        written.append(path)
    return written


def write_goldens() -> list[Path]:
    written: list[Path] = []
    for name in builtin_names():
        path = _GOLDENS_DIR / f"{name}.env"
        path.write_text(profile_env_text(name), encoding="utf-8")
        written.append(path)
    template_path = _GOLDENS_DIR / "template-defaults.env"
    template_path.write_text(template_defaults_text(), encoding="utf-8")
    written.append(template_path)
    written.extend(write_shape_goldens())
    return written


if __name__ == "__main__":
    for path in write_goldens():
        print(f"wrote {path.relative_to(_REPO_ROOT)}")
