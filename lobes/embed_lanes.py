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
from dataclasses import dataclass, replace
from typing import Mapping

import lobes.gateway  # noqa: F401  (pre-existing roles<->gateway cycle: load gateway first)
from lobes.catalog import ENGINE_SENTENCE_TRANSFORMERS, ENGINE_VLLM, TIER_ROLE
from lobes.roles import ROLE_BACKEND, ROLES

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
    checkpoint_path: str = ""  # operator fine-tunes only: local checkpoint dir


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
    normalization: str = "",
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
        normalization="l2",
    ),
    _lane(
        "qwen3vl-embed",
        "Qwen/Qwen3-VL-Embedding-8B",
        modalities=("text", "image", "video"),
        dim=4096,
    ),
    _lane(
        "qwen3vl-rerank",
        "Qwen/Qwen3-VL-Reranker-8B",
        task=TASK_SCORE,
        modalities=("text", "image", "video"),
    ),
    _lane(
        "nemotron-embed",
        "nvidia/Nemotron-3-Embed-8B-BF16",
        modalities=("text",),
        dim=4096,
        normalization="l2",
    ),
    _lane(
        "nomic-code-embed",
        "nomic-ai/nomic-embed-code",
        modalities=("text", "code"),
        dim=3584,
    ),
)


# --------------------------------------------------------------------------
# Operator-declared fine-tune lanes (orin-embedding-specialist, t11)
# --------------------------------------------------------------------------

FINETUNE_ENV = "EMBED_FINETUNE_LANES"
FINETUNE_BASE_LANE = "gemma2-embed"
# A fine-tune's served identity is ``local:<name>``: an HF repo id always
# contains "/", so this can never equal a catalog id, and it says plainly that
# the weights are an operator-local checkpoint rather than a published model.
FINETUNE_ID_PREFIX = "local:"


def finetune_identity(name: str) -> str:
    return FINETUNE_ID_PREFIX + name


def _parse_finetune_entry(entry: str) -> tuple[str, str]:
    name, sep, path = entry.partition("=")
    name, path = name.strip(), path.strip()
    if not sep or not name or not path:
        raise ValueError(f"{FINETUNE_ENV} entry {entry!r} must be name=/abs/path")
    if not path.startswith("/"):
        raise ValueError(f"{FINETUNE_ENV} path for {name!r} must be absolute, got {path!r}")
    return name, path


def parse_finetune_lanes(env: Mapping[str, str]) -> tuple[EmbedLane, ...]:
    """Derive fine-tune lanes from ``EMBED_FINETUNE_LANES`` (never mutates EMBED_LANES)."""
    raw = env.get(FINETUNE_ENV) or ""
    base = next(lane for lane in EMBED_LANES if lane.name == FINETUNE_BASE_LANE)
    taken = {lane.name for lane in EMBED_LANES}
    lanes: list[EmbedLane] = []
    for entry in (e for e in raw.split(",") if e.strip()):
        name, path = _parse_finetune_entry(entry)
        validate_lane_name(name)
        if name in taken:
            raise ValueError(f"fine-tune lane {name!r} collides with an existing embed lane")
        taken.add(name)
        lanes.append(
            replace(
                base,
                name=name,
                catalog_id=finetune_identity(name),
                base_url_env=base_url_env_for(name),
                checkpoint_path=path,
            )
        )
    return tuple(lanes)


def all_lanes(env: Mapping[str, str]) -> tuple[EmbedLane, ...]:
    """The static registry plus any declared fine-tunes; base entries untouched."""
    return EMBED_LANES + parse_finetune_lanes(env)
