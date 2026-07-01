"""TDD guard for docs/vllm-nightly-migration.md (devague plan
`lobes-unifies-its-generate-lane-on-one-vllm-nightl`, task t1 — before-state
verification + baselines).

t1 is doc-only / verification-only: no compose, catalog, or image pin may
change. This test does NOT assume the doc's factual claims — it parses the
REAL fleet compose template and the REAL gemma Dockerfile and cross-checks
them against what the doc says, so a future drift (someone flips an image pin
without updating the doc) fails loudly here instead of silently rotting the
doc.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "vllm-nightly-migration.md"
FLEET_COMPOSE = REPO_ROOT / "lobes" / "templates" / "fleet" / "docker-compose.yml"
GEMMA_DOCKERFILE = REPO_ROOT / "lobes" / "templates" / "fleet" / "Dockerfile.vllm-gemma4"
CULTURE_YAML = REPO_ROOT / "culture.yaml"
GATEWAY_FLEET_DOC = REPO_ROOT / "docs" / "gateway-fleet.md"

PINNED_TODAY_IMAGE = "nvcr.io/nvidia/vllm:26.04-py3"

# t4 (devague plan `lobes-unifies-its-generate-lane-on-one-vllm-nightl`) flipped
# the primary/embed/rerank services from PINNED_TODAY_IMAGE to this pinned
# nightly digest — the SAME digest Dockerfile.vllm-gemma4 already bases off,
# unifying the fleet's generate/embed/rerank engine on one vLLM nightly
# (0.23.1rc1.dev672). See docs/vllm-nightly-migration.md §4/§5 (t2/t3 GO
# verdicts) for the live validation that preceded this flip.
NIGHTLY_DIGEST_IMAGE = (
    "vllm/vllm-openai@sha256:" "7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9"
)

# Services that the plan says pin the NGC image today (t4 will migrate these
# to nightly later — t1 only verifies where they stand right now).
_NGC_PINNED_SERVICES = ("vllm-primary", "vllm-embed", "vllm-rerank")


def _service_block(text: str, service_name: str) -> str:
    """Extract the YAML block for a top-level compose service (same helper
    shape as tests/test_cli_fleet.py's _fleet_compose_text/_service_block)."""
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if re.match(rf"  {re.escape(service_name)}:", ln)),
        None,
    )
    assert start is not None, f"service '{service_name}' not found in fleet compose"
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"  \S", lines[i])),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_doc_exists() -> None:
    assert DOC.exists(), f"Expected {DOC} to exist (devague plan task t1 deliverable)"


def test_doc_does_not_mutate_image_pins() -> None:
    """t1 is verification-only — it must not touch compose/catalog/templates."""
    for path in (FLEET_COMPOSE, GEMMA_DOCKERFILE, REPO_ROOT / "lobes" / "catalog.py"):
        assert path.exists()
    # No assertion beyond existence: this test documents the constraint. The
    # mutation-avoidance is enforced by the git diff at commit time, not here.


def test_doc_records_the_before_state_ngc_pin_as_history() -> None:
    """docs/vllm-nightly-migration.md §1 is a HISTORICAL before-state record
    (t1, written before t4 flipped the pins) — its prose must still cite the
    NGC 26.04-py3 image and the 0.19.0 engine version it describes. This is a
    doc-content check only; it does NOT cross-check the live compose (that's
    test_fleet_primary_embed_rerank_pin_the_nightly_digest below) — t1's
    before-state narrative doesn't get rewritten just because t4 shipped."""
    doc_text = DOC.read_text(encoding="utf-8")

    assert PINNED_TODAY_IMAGE in doc_text, (
        f"doc must still cite the before-state pinned image {PINNED_TODAY_IMAGE!r} for the "
        "primary/embed/rerank services (historical record, not live state)"
    )
    assert (
        "0.19.0" in doc_text
    ), "doc must cite the vLLM engine version (0.19.0) behind that before-state image tag"


def test_fleet_primary_embed_rerank_pin_the_nightly_digest() -> None:
    """t4 flip: the LIVE fleet compose no longer pins PINNED_TODAY_IMAGE for
    primary/embed/rerank — it now pins the same nightly digest
    Dockerfile.vllm-gemma4 bases off. docs/vllm-nightly-migration.md §1 is
    deliberately left as a before-state record (see the test above); this
    test is the one that must fail loudly if the live template drifts from
    that flip, not the historical-doc check."""
    compose_text = FLEET_COMPOSE.read_text(encoding="utf-8")

    for service in _NGC_PINNED_SERVICES:
        block = _service_block(compose_text, service)
        assert PINNED_TODAY_IMAGE not in block, (
            f"template regression: {service} still pins the pre-t4 image "
            f"{PINNED_TODAY_IMAGE!r} in {FLEET_COMPOSE} — t4 flipped this to the nightly digest"
        )
        assert NIGHTLY_DIGEST_IMAGE in block, (
            f"template regression: {service} does not pin the nightly digest "
            f"{NIGHTLY_DIGEST_IMAGE!r} in {FLEET_COMPOSE}"
        )


def test_doc_claims_match_actual_gemma_nightly_pin() -> None:
    """The doc's claim that the gemma vllm-multimodal service already runs on
    the nightly image must match the REAL Dockerfile.vllm-gemma4 FROM line."""
    doc_text = DOC.read_text(encoding="utf-8")
    compose_text = FLEET_COMPOSE.read_text(encoding="utf-8")
    dockerfile_text = GEMMA_DOCKERFILE.read_text(encoding="utf-8")

    assert "vllm/vllm-openai" in doc_text, "doc must cite the nightly base image (vllm/vllm-openai)"
    assert "nightly" in doc_text.lower(), "doc must say the gemma gear runs the nightly image"
    assert (
        "Dockerfile.vllm-gemma4" in doc_text
    ), "doc must cite the Dockerfile that builds the gemma image"

    # vllm-multimodal must NOT pin the NGC image directly — it builds instead.
    multimodal_block = _service_block(compose_text, "vllm-multimodal")
    assert (
        PINNED_TODAY_IMAGE not in multimodal_block
    ), "vllm-multimodal must not pin the NGC 26.04-py3 image — it builds via Dockerfile.vllm-gemma4"
    assert (
        "Dockerfile.vllm-gemma4" in multimodal_block
    ), "vllm-multimodal must build from Dockerfile.vllm-gemma4 (real template regression check)"

    from_lines = [ln for ln in dockerfile_text.splitlines() if ln.strip().startswith("FROM")]
    assert from_lines, f"no FROM instruction found in {GEMMA_DOCKERFILE}"
    assert any("vllm/vllm-openai" in ln for ln in from_lines), (
        f"template regression: {GEMMA_DOCKERFILE} no longer bases off vllm/vllm-openai "
        "(nightly) — docs/vllm-nightly-migration.md's before-state claim is now stale"
    )


def test_doc_records_baselines_to_beat() -> None:
    """The doc must cite the numbers from the source docs, not paraphrase
    them loosely — a future benchmark has to know exactly what 'winning'
    means."""
    doc_text = DOC.read_text(encoding="utf-8")

    # 27B primary: 18.7-19.1 tok/s decode, 72-79% MTP draft acceptance
    # (docs/qwen3.6-27b-text-nvfp4-mtp.md).
    assert re.search(
        r"19(\.\d+)?\s*tok/s", doc_text
    ), "doc must cite the 27B ~19 tok/s decode baseline"
    assert "72" in doc_text and "79" in doc_text, (
        "doc must cite the 72-79% MTP draft acceptance range from "
        "docs/qwen3.6-27b-text-nvfp4-mtp.md"
    )
    assert "qwen3.6-27b-text-nvfp4-mtp.md" in doc_text, "doc must cite its 27B baseline source"

    # Gemma 12B: ~23 tok/s no-spec (docs/gemma-4-12b-nvfp4.md).
    assert re.search(
        r"23(\.\d+)?\s*tok/s", doc_text
    ), "doc must cite the Gemma ~23 tok/s no-spec baseline"
    assert "gemma-4-12b-nvfp4.md" in doc_text, "doc must cite its Gemma baseline source"


def test_doc_cites_real_mesh_traffic_not_a_hypothetical() -> None:
    """The doc must point at concrete evidence (culture.yaml / gateway
    routing) that the generate lane carries real mesh traffic — not assert it
    as a hypothetical."""
    doc_text = DOC.read_text(encoding="utf-8")
    culture_yaml_text = CULTURE_YAML.read_text(encoding="utf-8")
    gateway_doc_text = GATEWAY_FLEET_DOC.read_text(encoding="utf-8")

    assert "culture.yaml" in doc_text, "doc must cite culture.yaml as evidence"
    # The exact model id culture.yaml's lobes agent is served by — verifies
    # the doc's citation is accurate, not a stale/guessed value.
    assert "vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" in culture_yaml_text
    assert "vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" in doc_text

    assert "model=main" in doc_text or "model=multimodal" in doc_text, (
        "doc must cite the tier-alias routing (model=main / model=multimodal), "
        "not just narrate it"
    )
    assert (
        "model=main|minor|multimodal" in gateway_doc_text
    ), f"expected gateway routing doc {GATEWAY_FLEET_DOC} to define the tier aliases the doc cites"
