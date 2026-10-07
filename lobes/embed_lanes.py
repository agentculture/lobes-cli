"""Specialist embed/rerank lane registry (orin-embedding-specialist, t3).

Pure data plus a name validator; no I/O. Each specialist lane is addressed by
its OWN distinct name (operator decision c35), never as a new Colleague role:
``lobes.roles.ROLES`` is deliberately untouched (adding a role is effectively
irreversible). Names are short model slugs (``gemma2-embed``) and must never
collide with a role, a tier alias, a backend/self-named alias, or the
``{role}-{member}`` member-lane spelling the gateway mesh uses.

Dimensions are declared only where verified; ``dim=0`` / ``mrl_dims=()`` mean
"not declared", never a guess. The catalog need not contain these ids yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lobes.catalog import ENGINE_VLLM, TIER_ROLE
from lobes.roles import ROLE_BACKEND, ROLES

ENGINE_SENTENCE_TRANSFORMERS = "sentence-transformers"  # local literal; catalog owns its own

TASK_EMBED = "embed"
TASK_SCORE = "score"

MODALITIES = ("text", "code", "image", "video", "audio")

# Self-named gateway aliases (lobes/gateway/_config.py) that are not tiers/roles.
_SELF_NAMED_ALIASES = ("multimodal-coder", "embed-deep")

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class EmbedLane:
    name: str
    catalog_id: str
    engine: str
    base_url_env: str
    task: str  # "embed" | "score"
    modalities: tuple[str, ...]
    dim: int = 0  # native dimension; 0 = not declared
    mrl_dims: tuple[int, ...] = ()
    normalization: str = "l2"  # "l2" | "none"


def base_url_env_for(name: str) -> str:
    return name.upper().replace("-", "_") + "_BASE_URL"


def reserved_names() -> frozenset[str]:
    """Every name a lane may not equal (roles, tiers, backends, self-named)."""
    return frozenset(
        (*ROLES, *TIER_ROLE, *TIER_ROLE.values(), *ROLE_BACKEND.values(), *_SELF_NAMED_ALIASES)
    )


def validate_lane_name(name: str) -> None:
    """Raise ``ValueError`` when *name* is unusable as a specialist lane name."""
    if not _NAME_RE.match(name):
        raise ValueError(f"lane name {name!r} must be lowercase alphanumerics joined by '-'")
    reserved = reserved_names()
    if name in reserved:
        raise ValueError(f"lane name {name!r} collides with a role, tier or backend alias")
    for base in reserved:
        if name.startswith(base + "-"):
            raise ValueError(
                f"lane name {name!r} matches the {{role}}-{{member}} member-lane pattern ({base}-*)"
            )


def _lane(
    name: str,
    catalog_id: str,
    *,
    engine: str = ENGINE_VLLM,
    task: str = TASK_EMBED,
    modalities: tuple[str, ...],
    dim: int = 0,
    mrl_dims: tuple[int, ...] = (),
    normalization: str = "l2",
) -> EmbedLane:
    validate_lane_name(name)
    return EmbedLane(
        name=name,
        catalog_id=catalog_id,
        engine=engine,
        base_url_env=base_url_env_for(name),
        task=task,
        modalities=modalities,
        dim=dim,
        mrl_dims=mrl_dims,
        normalization=normalization,
    )


EMBED_LANES: tuple[EmbedLane, ...] = (
    _lane(
        "gemma2-embed",
        "google/embeddinggemma-2",
        engine=ENGINE_SENTENCE_TRANSFORMERS,
        modalities=("text", "code", "image", "video", "audio"),
        dim=768,
        mrl_dims=(128, 256, 512, 768),
    ),
    _lane(
        "qwen3vl-embed",
        "Qwen/Qwen3-VL-Embedding-8B",
        modalities=("text", "image", "video"),
    ),
    _lane(
        "qwen3vl-rerank",
        "Qwen/Qwen3-VL-Reranker-8B",
        task=TASK_SCORE,
        modalities=("text", "image", "video"),
        normalization="none",
    ),
    _lane(
        "nemotron-embed",
        "nvidia/Nemotron-3-Embed-8B-BF16",
        modalities=("text",),
        dim=4096,
    ),
    _lane(
        "nomic-code-embed",
        "nomic-ai/nomic-embed-code",
        modalities=("text", "code"),
    ),
)
