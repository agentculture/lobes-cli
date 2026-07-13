"""Golden rendered artifacts per shipped profile (t13).

Tests hold the rendered ``.env``-projection for EVERY shipped profile
(``spark``, ``thor``) and diff it byte-for-byte against ``tests/goldens/``.
Rendering (:func:`lobes.profiles.render.profile_env` over
:func:`lobes.profiles.loader.resolve_profile`) is a pure function of
``(profile, template)`` with no host state, so these goldens run identically
on any dev box — a GPU-less CI runner included.

A change scoped to one machine (e.g. editing ``lobes/profiles/builtin/thor.toml``
or the shared ``SM_110`` trait in ``lobes/machines/_traits.py``) that somehow
alters ANOTHER machine's rendering fails this suite unless that other golden
is deliberately updated in the SAME commit — this is the enforcement
mechanism behind "the Thor box and the Spark box can both change lobes
without regressing each other." See ``tests/goldens/README.md`` for the full
rationale.

Regenerate every golden with::

    uv run python tests/goldens/regen.py

then diff the result before committing — a golden moving that you didn't
intend to touch is the signal this suite exists to catch, not a lint error to
silence.

This module never reimplements the profile -> env mapping: every assertion
goes through :func:`lobes.profiles.render.profile_env` /
:func:`lobes.profiles.loader.resolve_profile`, exactly like ``tests/goldens/regen.py``
does when writing the golden files.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import profile_env
from tests.goldens.regen import FLEET_COMPOSE, profile_env_text, template_defaults_text

_GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"
_REGEN_CMD = "uv run python tests/goldens/regen.py"

# The four measured sm_110 divergences (lobes/machines/thor.py +
# lobes/machines/_traits.py:SM_110) that must show up in thor.env and MUST NOT
# show up in spark.env.
_THOR_DIVERGENCE_LINES = (
    "PRIMARY_KV_CACHE_DTYPE=auto",
    "EMBED_ATTENTION_BACKEND=TRITON_ATTN",
    "RERANK_ATTENTION_BACKEND=TRITON_ATTN",
    "RERANK_ENFORCE_EAGER=--enforce-eager",
)


def _read_golden(name: str) -> str:
    path = _GOLDENS_DIR / name
    if not path.is_file():
        pytest.fail(f"missing golden {path} — generate it with: {_REGEN_CMD}")
    return path.read_text(encoding="utf-8")


# --- byte-for-byte goldens ---------------------------------------------------


@pytest.mark.parametrize("name", builtin_names())
def test_profile_golden_byte_for_byte(name: str) -> None:
    """profile_env(resolve_profile(name)) renders exactly what's committed.

    Enumerates over ``builtin_names()`` (not a hardcoded list) so a future
    built-in profile is caught automatically the first time this test runs
    without a matching golden on disk (see the ``pytest.fail`` in
    ``_read_golden``), rather than being silently skipped.
    """
    actual = profile_env_text(name)
    expected = _read_golden(f"{name}.env")
    assert actual == expected, (
        f"tests/goldens/{name}.env drifted from "
        f"profile_env(resolve_profile({name!r})).\n"
        f"If this is a deliberate change, regenerate with: {_REGEN_CMD}\n"
        "then diff the result — a change to ONE machine's golden should not "
        "move ANOTHER machine's golden or template-defaults.env in the same diff."
    )


def test_template_defaults_golden_byte_for_byte() -> None:
    """The ${VAR:-default} surface of the fleet compose template is pinned.

    Catches a template edit that changes what an UNRESOLVED knob renders to
    for every machine at once (the GB10/spark deployment runs mostly on these
    defaults) — a profile-only golden can't see this class of change since a
    profile only carries the knobs it takes an explicit position on.
    """
    actual = template_defaults_text()
    expected = _read_golden("template-defaults.env")
    assert actual == expected, (
        f"tests/goldens/template-defaults.env drifted from the ${{VAR:-default}} "
        f"surface of {FLEET_COMPOSE}.\n"
        f"If this is a deliberate change, regenerate with: {_REGEN_CMD}"
    )


def test_golden_file_set_matches_builtin_profiles() -> None:
    """Every packaged built-in profile has a golden, and vice versa — no gaps."""
    on_disk = {p.stem for p in _GOLDENS_DIR.glob("*.env")} - {"template-defaults"}
    assert on_disk == set(builtin_names())


# --- cross-machine isolation (acceptance criterion 1) ------------------------


def test_spark_golden_carries_none_of_thors_sm110_divergences() -> None:
    """Editing the thor bundle/SM_110 trait must never leak into spark.env.

    Demonstrates isolation by test design: each profile renders and is
    compared against its OWN golden file, and this test additionally asserts
    the concrete divergent values (TRITON_ATTN / --enforce-eager / kv auto)
    that a thor-scoped edit would introduce are simply absent from spark.env.
    """
    spark_text = _read_golden("spark.env")
    for divergent_line in _THOR_DIVERGENCE_LINES:
        assert divergent_line not in spark_text, (
            f"{divergent_line!r} leaked into spark.env — a thor-only change "
            "must not alter another machine's rendering"
        )


def test_thor_golden_carries_its_own_sm110_divergences() -> None:
    """The inverse check: thor.env DOES carry its four validated divergences."""
    thor_text = _read_golden("thor.env")
    for divergent_line in _THOR_DIVERGENCE_LINES:
        assert divergent_line in thor_text, f"{divergent_line!r} missing from thor.env"


def test_spark_and_thor_share_every_other_knob() -> None:
    """Outside the four sm_110 divergences, spark and thor render identically.

    Both profiles declare the same models/util/context/quant on purpose (both
    are 128 GB unified-memory Blackwell-class boards) — this pins that
    intentional near-duplication so a future accidental divergence (or
    accidental de-duplication) shows up as a targeted diff instead of getting
    lost in the two full-file byte comparisons above.
    """
    spark_lines = set(_read_golden("spark.env").splitlines())
    thor_lines = set(_read_golden("thor.env").splitlines())
    # cortex.kv_cache_dtype is a same-KEY, different-VALUE divergence (fp8 vs
    # auto); the other three are thor-only additions (spark has no opinion, so
    # the compose template's own ${VAR:-default} applies for those knobs).
    only_in_spark = spark_lines - thor_lines
    only_in_thor = thor_lines - spark_lines
    assert only_in_spark == {"PRIMARY_KV_CACHE_DTYPE=fp8"}
    assert only_in_thor == {
        "PRIMARY_KV_CACHE_DTYPE=auto",
        "EMBED_ATTENTION_BACKEND=TRITON_ATTN",
        "RERANK_ATTENTION_BACKEND=TRITON_ATTN",
        "RERANK_ENFORCE_EAGER=--enforce-eager",
    }


# --- purity: no host state, no GPU, no subprocess ----------------------------


def test_rendering_is_pure_repeated_calls_are_identical() -> None:
    """render(profile) twice yields byte-identical output — no hidden state."""
    for name in builtin_names():
        first = profile_env(resolve_profile(name))
        second = profile_env(resolve_profile(name))
        assert first == second


def test_rendering_consults_no_host_state(tmp_path, monkeypatch) -> None:
    """resolve_profile + profile_env complete with an empty HOME, a trimmed
    environment, and subprocess spawning turned into a hard failure.

    Proves the "no GPU or host state" acceptance criterion directly, rather
    than by inference: card *detection* (lobes/runtime/_detect.py) is a
    separate module this test never touches — resolve_profile takes an
    explicit profile name, exactly as lobes init does once detection has
    already run.
    """

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "profile resolution/rendering must not spawn a subprocess "
            f"(called with args={args!r}, kwargs={kwargs!r})"
        )

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(os, "system", _boom)

    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    for key in list(os.environ):
        if key not in ("HOME", "PATH"):
            monkeypatch.delenv(key, raising=False)

    for name in builtin_names():
        profile = resolve_profile(name)  # no deploy_dir -> built-ins only, no FS scan
        env = profile_env(profile)
        assert env, f"profile_env({name!r}) rendered nothing under a trimmed environment"
