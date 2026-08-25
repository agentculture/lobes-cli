"""Verify, by test, the boundaries the associate frame promised NOT to move.

lightning-on-orin plan, t11. Two confirmed claims from the frame are load-
bearing here, and both are boundary claims — this module exists because a
boundary claim is worth nothing until something FAILS if it is ever crossed:

* **c9** — nothing about the mesh's existing ``worker`` seat changes. The
  Spark GB10 hosts ``worker`` today and the Thor reaches it by proxy; a
  local Orin ``associate`` is an ADDITION to that topology, not a
  replacement or a rewire.
* **c10 / h36** (h36 AMENDED 2026-08-25, after #199 merged) — the gateway,
  role and proxy MACHINERY is untouched. Concretely: no gateway BEHAVIOUR
  changes — routing, selection, replica-pool and proxy LOGIC are untouched
  — and the only real edit is the ADDITIVE registration #199 (0.63.0) made
  generic across role prefixes, now extended to a tenth. associate gains
  its ``<PREFIX>_*`` vocabulary in ``_config.py``/``_replicas.py``/
  ``_routing.py``/``_selection.py`` exactly as the existing nine roles have
  it, and no existing role's behaviour changes.

``tests/test_associate_role.py`` (t6) already pins associate's OWN contract
(responsibilities, tier rung, public addresses, pressure policy, per-role env
channels). This module does not repeat that. It instead pins the NEGATIVE
space t6 measured but did not turn into a standing regression test: that
``lobes/gateway/_replicas.py``, ``_selection.py`` and ``_routing.py`` are
fully generic over role prefixes and needed ZERO edits for a tenth role, that
every ``_config.py``/``server.py`` per-role dict is additive-only over the
nine pre-existing roles, and that the Spark's own rendered contract (which
hosts ``worker``, not ``associate``) is untouched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lobes import roles as roles_mod
from lobes.gateway import _replicas as replicas_mod
from lobes.gateway import _routing as routing_mod
from lobes.gateway import _selection as selection_mod
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
)
from lobes.profiles.loader import resolve_profile
from lobes.profiles.render import ROLE_ENV_PREFIX
from lobes.profiles.shape_render import render_shape
from lobes.profiles.shapes import resolve_shape
from tests.goldens.regen import shape_golden_path

_GATEWAY_DIR = Path(routing_mod.__file__).resolve().parent


# ---------------------------------------------------------------------------
# h36 — genericity guard: _replicas.py / _selection.py / _routing.py needed
# ZERO edits to register a tenth role. If either module ever grows a literal
# reference to "associate" (or any other role name used as a branch
# condition), h36 is broken — the machinery stopped being generic and started
# special-casing roles again.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", [replicas_mod, selection_mod, routing_mod], ids=lambda m: m.__name__
)
def test_generic_gateway_modules_carry_no_associate_literal(module) -> None:
    """associate's registration touched _config.py/server.py only.

    The commit that added the role (9fa9440, "associate — the tenth
    Colleague role") did not modify _replicas.py, _selection.py or
    _routing.py at all — this is the standing proof that stays true even
    after future refactors touch those files for other reasons.
    """
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "associate" not in source.lower()


def test_replicas_selection_routing_source_is_unchanged_by_associate_commit() -> None:
    """Cross-check against the commit that actually added the role.

    ``git show`` of the commit that introduced associate
    (``feat(roles): associate — the tenth Colleague role``) must not list
    any of the three generic modules in its changed-files set. This is a
    point-in-time historical fact, not a live behavioural guarantee (a LATER
    commit could still touch these files for an unrelated reason without
    breaking h36) — the two source-content tests above are what must keep
    passing forever; this one documents that the original registration
    really was zero-edit, which is the empirical basis for h36's claim.
    """
    result = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--grep=associate — the tenth Colleague role"],
        cwd=_GATEWAY_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    shas = [line for line in result.stdout.splitlines() if line.strip()]
    if not shas:
        pytest.skip("associate-introducing commit not found in this checkout's history")
    sha = shas[0]
    diff = subprocess.run(
        ["git", "show", "--stat", "--format=", sha],
        cwd=_GATEWAY_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    changed = diff.stdout
    for path in (
        "gateway/_replicas.py",
        "gateway/_selection.py",
        "gateway/_routing.py",
    ):
        assert path not in changed, f"{path} was touched by the associate-introducing commit"


# ---------------------------------------------------------------------------
# c10 / h36 — additive-only: every _config.py / server.py per-role dict, once
# filtered down to the nine pre-existing roles, has EXACTLY the values it had
# before associate. A new key may be ADDED for associate; no existing value
# may be edited, reordered away, or removed.
# ---------------------------------------------------------------------------

_EXPECTED_FEASIBLE_ENV = {
    "primary": "PRIMARY_FEASIBLE",
    "multimodal": "MULTIMODAL_FEASIBLE",
    "muse": "MUSE_FEASIBLE",
    "worker": "WORKER_FEASIBLE",
    "hand": "HAND_FEASIBLE",
    "embed": "EMBED_FEASIBLE",
    "rerank": "RERANK_FEASIBLE",
    "stt": "STT_FEASIBLE",
    "tts": "TTS_FEASIBLE",
}

_EXPECTED_PEER_ORIGIN_ENV = {
    "primary": "PRIMARY_PEER_ORIGIN",
    "multimodal": "MULTIMODAL_PEER_ORIGIN",
    "muse": "MUSE_PEER_ORIGIN",
    "worker": "WORKER_PEER_ORIGIN",
    "hand": "HAND_PEER_ORIGIN",
    "embed": "EMBED_PEER_ORIGIN",
    "rerank": "RERANK_PEER_ORIGIN",
    "stt": "STT_PEER_ORIGIN",
    "tts": "TTS_PEER_ORIGIN",
}

_EXPECTED_PEER_PROXY_ENV = {
    "primary": "PRIMARY_PEER_PROXY",
    "multimodal": "MULTIMODAL_PEER_PROXY",
    "muse": "MUSE_PEER_PROXY",
    "worker": "WORKER_PEER_PROXY",
    "hand": "HAND_PEER_PROXY",
    "embed": "EMBED_PEER_PROXY",
    "rerank": "RERANK_PEER_PROXY",
    "stt": "STT_PEER_PROXY",
    "tts": "TTS_PEER_PROXY",
}

_EXPECTED_PEER_API_KEY_ENV = {
    "primary": "PRIMARY_PEER_API_KEY",
    "multimodal": "MULTIMODAL_PEER_API_KEY",
    "muse": "MUSE_PEER_API_KEY",
    "worker": "WORKER_PEER_API_KEY",
    "hand": "HAND_PEER_API_KEY",
    "embed": "EMBED_PEER_API_KEY",
    "rerank": "RERANK_PEER_API_KEY",
    "stt": "STT_PEER_API_KEY",
    "tts": "TTS_PEER_API_KEY",
}

_EXPECTED_PEER_ORIGINS_ENV = {
    "primary": "PRIMARY_PEER_ORIGINS",
    "multimodal": "MULTIMODAL_PEER_ORIGINS",
    "muse": "MUSE_PEER_ORIGINS",
    "worker": "WORKER_PEER_ORIGINS",
    "hand": "HAND_PEER_ORIGINS",
    "embed": "EMBED_PEER_ORIGINS",
    "rerank": "RERANK_PEER_ORIGINS",
    "stt": "STT_PEER_ORIGINS",
    "tts": "TTS_PEER_ORIGINS",
}

_EXPECTED_PEER_API_KEYS_ENV = {
    "primary": "PRIMARY_PEER_API_KEYS",
    "multimodal": "MULTIMODAL_PEER_API_KEYS",
    "muse": "MUSE_PEER_API_KEYS",
    "worker": "WORKER_PEER_API_KEYS",
    "hand": "HAND_PEER_API_KEYS",
    "embed": "EMBED_PEER_API_KEYS",
    "rerank": "RERANK_PEER_API_KEYS",
    "stt": "STT_PEER_API_KEYS",
    "tts": "TTS_PEER_API_KEYS",
}

_EXPECTED_PEER_SERVED_NAME_ENV = {
    "primary": "PRIMARY_SERVED_NAME",
    "multimodal": "MULTIMODAL_SERVED_NAME",
    "muse": "MUSE_SERVED_NAME",
    "worker": "WORKER_SERVED_NAME",
    "hand": "HAND_SERVED_NAME",
    "embed": "EMBED_SERVED_NAME",
    "rerank": "RERANK_SERVED_NAME",
}

_EXPECTED_PEER_ROLE_HINT = {
    "primary": "primary",
    "multimodal": "multimodal",
    "muse": "muse",
    "worker": "worker",
    "hand": "hand",
    "embed": "embedding",
    "rerank": "reranker",
}


@pytest.mark.parametrize(
    "actual,expected,name",
    [
        (FEASIBLE_ENV, _EXPECTED_FEASIBLE_ENV, "FEASIBLE_ENV"),
        (PEER_ORIGIN_ENV, _EXPECTED_PEER_ORIGIN_ENV, "PEER_ORIGIN_ENV"),
        (PEER_PROXY_ENV, _EXPECTED_PEER_PROXY_ENV, "PEER_PROXY_ENV"),
        (PEER_API_KEY_ENV, _EXPECTED_PEER_API_KEY_ENV, "PEER_API_KEY_ENV"),
        (PEER_ORIGINS_ENV, _EXPECTED_PEER_ORIGINS_ENV, "PEER_ORIGINS_ENV"),
        (PEER_API_KEYS_ENV, _EXPECTED_PEER_API_KEYS_ENV, "PEER_API_KEYS_ENV"),
        (S._PEER_SERVED_NAME_ENV, _EXPECTED_PEER_SERVED_NAME_ENV, "server._PEER_SERVED_NAME_ENV"),
        (S._PEER_ROLE_HINT, _EXPECTED_PEER_ROLE_HINT, "server._PEER_ROLE_HINT"),
    ],
)
def test_per_backend_channel_is_additive_only_over_the_nine_pre_existing_backends(
    actual: dict, expected: dict, name: str
) -> None:
    """Filtering associate back out reproduces exactly the pre-associate dict.

    A regression here means an existing role's env KEY was renamed, an
    existing value was edited, or an existing role was dropped from the
    channel — any of which is a real behavioural change to the nine
    pre-existing roles, not an addition.
    """
    filtered = {k: v for k, v in actual.items() if k != "associate"}
    assert filtered == expected, f"{name} changed for a pre-existing backend"


def test_opt_in_backends_only_gained_associate() -> None:
    assert OPT_IN_BACKENDS - {"associate"} == {"muse", "worker"}


def test_never_proxied_backends_is_still_empty() -> None:
    # d1 (2026-08-20) emptied this set; associate must not re-populate it.
    assert NEVER_PROXIED_BACKENDS == frozenset()


# ---------------------------------------------------------------------------
# c9 — the worker seat itself: its role-registry entry and env prefix are
# untouched.
# ---------------------------------------------------------------------------


def test_workers_own_role_contract_is_untouched() -> None:
    """The full responsibilities/forbidden tuples are already pinned in
    tests/test_roles.py — this only re-asserts the identity/wiring facts
    plus the RELATION to associate that c9 depends on: associate derives
    from worker without worker itself moving."""
    assert roles_mod.ROLE_BACKEND["worker"] == "worker"
    assert roles_mod.ROLE_PATH["worker"] == "/v1/chat/completions"
    assert roles_mod.ROLE_MAX_MODEL_LEN_ENV["worker"] == "WORKER_MAX_MODEL_LEN"
    assert "repo_action" in roles_mod.ROLE_RESPONSIBILITIES["worker"]
    assert "repo_action" not in roles_mod.ROLE_FORBIDDEN["worker"]
    assert ROLE_ENV_PREFIX["worker"] == "WORKER"
    # associate is a strict subset of worker's responsibilities plus one
    # extra forbidden token — never the other way around.
    assert set(roles_mod.ROLE_RESPONSIBILITIES["associate"]) < set(
        roles_mod.ROLE_RESPONSIBILITIES["worker"]
    )


# ---------------------------------------------------------------------------
# c9 — the Spark's own rendered contract hosts `worker`, not `associate`, and
# is byte-for-byte untouched.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_name", ["thor-worker", "spark-lobe"])
def test_spark_shape_render_carries_no_associate_trace(shape_name: str) -> None:
    """Neither Spark-relevant shape mentions associate anywhere in its render.

    associate is declared on the Orin card only (lightning-on-orin, t8's
    MEASURED Orin budget); a box that renders a Spark-hosted shape should
    show no trace of it at all — not even an infeasible marker, since
    associate never enters ``FEASIBLE_ENV``'s per-card TOML declaration on
    a card that doesn't know about it.
    """
    shape = resolve_shape(shape_name)
    profile = resolve_profile("spark")
    rendered = render_shape(shape, profile).env_text()
    assert "ASSOCIATE" not in rendered.upper()


def test_thor_worker_on_spark_golden_still_serves_worker_unchanged() -> None:
    """The live-deployed shape (thor-worker rendered on the Spark card, per
    CLAUDE.md's 2026-08-20 worker-relocation record) keeps its WORKER_* block
    exactly as committed, byte for byte."""
    path = shape_golden_path("thor-worker", "spark")
    golden = path.read_text(encoding="utf-8")
    shape = resolve_shape("thor-worker")
    profile = resolve_profile("spark")
    rendered = render_shape(shape, profile).env_text()
    assert rendered == golden
    assert "WORKER_BASE_URL=http://vllm-worker:8000" in golden
    assert "ASSOCIATE" not in golden.upper()


def test_no_shape_golden_on_the_spark_card_mentions_associate() -> None:
    """Sweep every committed (shape, spark) golden for an associate leak."""
    goldens_dir = Path(__file__).resolve().parent / "goldens" / "shapes"
    spark_goldens = sorted(goldens_dir.glob("*__spark.env"))
    assert spark_goldens, "expected at least one *__spark.env golden to exist"
    for path in spark_goldens:
        text = path.read_text(encoding="utf-8")
        assert "ASSOCIATE" not in text.upper(), f"{path} unexpectedly mentions associate"
