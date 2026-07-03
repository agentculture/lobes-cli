"""TDD guard for docs/gateway-fleet.md (devague plan
`lobes-unifies-its-generate-lane-on-one-vllm-nightl`, task t9 — shipped-state
docs).

t9 is doc-only: it must describe what actually shipped (the fleet's tier
routing, the nightly-unified engine, and the retuned always-on duo budget)
without contradicting the real templates. Like test_nightly_migration_doc.py,
this test does NOT trust the doc's prose on faith — it parses the REAL fleet
env.example / docker-compose.yml and cross-checks the doc's numbers against
them, so a future drift (someone retunes the duo budget without updating the
doc) fails loudly here instead of silently rotting the doc.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "gateway-fleet.md"
FLEET_ENV_EXAMPLE = REPO_ROOT / "lobes" / "templates" / "fleet" / "env.example"
FLEET_COMPOSE = REPO_ROOT / "lobes" / "templates" / "fleet" / "docker-compose.yml"

NIGHTLY_DIGEST_IMAGE = (
    "vllm/vllm-openai@sha256:" "7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9"
)
NGC_IMAGE = "nvcr.io/nvidia/vllm:26.04-py3"


def _env_default(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(\S+)", text, re.MULTILINE)
    assert match is not None, f"{key} not found in {FLEET_ENV_EXAMPLE}"
    return match.group(1)


def test_doc_exists() -> None:
    assert DOC.exists(), f"Expected {DOC} to exist"


def test_doc_names_the_coolthor_default_and_multimodal_coder_alias() -> None:
    """t7 promoted coolthor/gemma-4-12B-it-NVFP4A16 to the default multimodal
    gear (native MTP on) and kept the coder checkpoint reachable only via the
    opt-in `multimodal-coder` alias. The fleet doc's tier table must say so."""
    doc_text = DOC.read_text(encoding="utf-8")

    assert "coolthor/gemma-4-12B-it-NVFP4A16" in doc_text, (
        "doc must name the default multimodal gear's checkpoint "
        "(coolthor/gemma-4-12B-it-NVFP4A16)"
    )
    assert "multimodal-coder" in doc_text, "doc must name the opt-in multimodal-coder alias"
    assert "model=main|minor|multimodal" in doc_text, (
        "doc must still define the tier-alias vocabulary string "
        "docs/vllm-nightly-migration.md cross-checks against"
    )


def test_doc_cites_the_nightly_engine_unification() -> None:
    """Every default-on gear (primary, multimodal, embed, rerank) now pins the
    same vLLM nightly digest (t4) — the doc must say so and cite the pinned
    digest, and the real fleet compose must back that claim up."""
    doc_text = DOC.read_text(encoding="utf-8")
    compose_text = FLEET_COMPOSE.read_text(encoding="utf-8")

    assert "nightly" in doc_text.lower(), "doc must describe the fleet's nightly engine"
    assert NIGHTLY_DIGEST_IMAGE in doc_text, "doc must cite the pinned nightly digest"
    assert (
        "vllm-nightly-migration.md" in doc_text
    ), "doc must cite docs/vllm-nightly-migration.md as the evidence trail"

    # Cross-check against the real template: the default-on services must
    # actually pin that digest today.
    for service in ("vllm-primary", "vllm-embed", "vllm-rerank"):
        assert re.search(
            rf"  {re.escape(service)}:\n(?:.*\n)*?.*image:.*{re.escape(NIGHTLY_DIGEST_IMAGE)}",
            compose_text,
        ), (
            f"{service} no longer pins the nightly digest in {FLEET_COMPOSE} — "
            "doc claim is now stale"
        )


def test_doc_records_the_parked_t8_minor_middle_migration() -> None:
    """t8 (migrate opt-in minor/middle to nightly) is explicitly deferred —
    the doc must say so honestly, and the real template must still show them
    on the pre-migration NGC image (else t8 shipped and this doc is stale)."""
    doc_text = DOC.read_text(encoding="utf-8")
    compose_text = FLEET_COMPOSE.read_text(encoding="utf-8")

    assert "t8" in doc_text, "doc must name t8 as the pending minor/middle migration task"
    assert "parked" in doc_text.lower(), "doc must say t8 is parked/deferred, not silently omitted"
    assert NGC_IMAGE in doc_text, "doc must cite the NGC image minor/middle still pin"

    for service in ("vllm-minor", "vllm-middle"):
        assert re.search(
            rf"  {re.escape(service)}:\n(?:.*\n)*?.*image: {re.escape(NGC_IMAGE)}",
            compose_text,
        ), (
            f"{service} no longer pins {NGC_IMAGE} in {FLEET_COMPOSE} — "
            "t8 may have shipped; the doc's 'parked' framing is now stale"
        )


def test_doc_duo_budget_matches_the_real_templates() -> None:
    """The doc's always-on duo budget (128K/util 0.30 primary, 32K/util 0.14
    multimodal, 0.56 total) must match the real fleet env.example defaults —
    not a stale pre-rebalance number (64K/0.30 primary, 128K/0.22 multimodal,
    0.64), nor the older pre-duo number (128K/0.45 primary, 8K/0.12
    multimodal, 0.69)."""
    doc_text = DOC.read_text(encoding="utf-8")
    env_text = FLEET_ENV_EXAMPLE.read_text(encoding="utf-8")

    assert _env_default(env_text, "PRIMARY_MAX_MODEL_LEN") == "131072"
    assert _env_default(env_text, "PRIMARY_GPU_MEM_UTIL") == "0.30"
    assert _env_default(env_text, "MULTIMODAL_MAX_MODEL_LEN") == "32768"
    assert _env_default(env_text, "MULTIMODAL_GPU_MEM_UTIL") == "0.14"

    for needle in ("131072", "0.30", "32768", "0.14", "0.56"):
        assert needle in doc_text, f"doc must cite the live duo-budget value {needle!r}"

    # The stale pre-duo total (0.69) must not be presented as the current
    # default budget anymore.
    assert (
        "**0.69**" not in doc_text
    ), "doc still presents the stale pre-duo 0.69 total as the current default budget"
