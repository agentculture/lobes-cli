"""Speech-to-text LANE registry (stt-lanes): ``stt-he`` and ``stt-en``.

Pure data plus two lookups; no I/O. Like the specialist embed lanes
(:mod:`lobes.embed_lanes`) these are lanes, never Colleague roles:
``lobes.roles.ROLES`` is untouched (adding a role is effectively irreversible).
The ``stt`` role keeps meaning "whatever this box's ``AUDIO_URL`` stt container
serves"; a lane names a LANGUAGE-specific engine a caller selects explicitly on
``POST /v1/audio/transcriptions``.

A sibling registry rather than entries in ``EMBED_LANES``: every embed-lane
consumer (``/v1/embeddings`` model routing, ``lobes up``/``assess``, the shape
renderer, the deployment lock) reads that tuple and assumes a vector space,
and ``stt-en`` has no fixed checkpoint at all. Keeping a separate tuple of the
same duck-typed shape (``name``/``catalog_id``/``task``/``base_url_env``) lets
the gateway reuse the shared mesh machinery (announce from ``/capabilities``,
verify, ``lane_placement``, the one-hop forwarder) without any of those
embed-only paths ever seeing an stt lane.

``catalog_id == ""`` means "not pinned": ``stt-en`` serves whatever the box's
own ``stt`` container declares (``STT_MODEL``/``STT_RUNTIME``), so its identity
is resolved per box by :func:`served_identity`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import lobes.gateway  # noqa: F401  (pre-existing roles<->gateway cycle: load gateway first)
from lobes.embed_lanes import validate_lane_name
from lobes.gateway._routing import TRANSCRIBE_TASK
from lobes.lane_keys import lane_env_key

TASK_TRANSCRIBE = TRANSCRIBE_TASK

#: The one path an stt lane serves, on the gateway and on the sidecar alike.
TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"

#: The sidecar readiness path the gateway's lane probe dials. The same route
#: the Parakeet and Whisper ``stt`` servers already expose, so ``stt-en`` can
#: point straight at the box's existing ``stt`` container.
READY_PATH = "/v1/health/ready"

#: The role whose ``{role}-<suffix>`` spelling these lanes use (the narrow
#: exception in :func:`lobes.embed_lanes.validate_lane_name`).
AUDIO_ROLE = "stt"


@dataclass(frozen=True)
class SttLane:
    name: str
    language: str  # injected as the multipart `language` field when the caller sends none
    base_url_env: str
    catalog_id: str = ""  # pinned checkpoint; "" = follows the box's STT_MODEL
    runtime: str = ""  # pinned serving stack; "" = follows the box's STT_RUNTIME
    revision: str = ""  # pinned HF revision, when catalog_id is pinned
    licence: str = ""  # declared licence of the pinned weights
    task: str = TASK_TRANSCRIBE


def _lane(name: str, language: str, **pinned: str) -> SttLane:
    validate_lane_name(name, audio_role=AUDIO_ROLE)
    return SttLane(
        name=name, language=language, base_url_env=lane_env_key(name, "BASE_URL"), **pinned
    )


STT_LANES: tuple[SttLane, ...] = (
    # ivrit.ai's streaming Zipformer2 (66M) on sherpa-onnx, int8, CPU-only. The
    # weights are CC-BY-4.0: attribution is required (docs/stt-lanes.md).
    _lane(
        "stt-he",
        "he",
        catalog_id="ivrit-ai/zipformer2-66m-streaming-he",
        runtime="sherpa-onnx",
        revision="ee769a2a13b887f91572eec349458bfb0fe79e7a",
        licence="CC-BY-4.0",
    ),
    # The box's existing `stt` container, asked for English (operator decision
    # 2026-10-23: "we can use existing stt for english for now").
    _lane("stt-en", "en"),
)


def lane_by_name(name: str | None) -> SttLane | None:
    """The registry lane named exactly *name*, else ``None``."""
    return next((lane for lane in STT_LANES if lane.name == name), None) if name else None


def served_identity(lane: SttLane, env: Mapping[str, str]) -> tuple[str, str]:
    """``(model, runtime)`` this box serves *lane* as.

    A pinned lane is its own declaration; an unpinned one reads the box's
    ``STT_MODEL``/``STT_RUNTIME`` exactly as the ``stt`` role's advert does
    (falling back to the same Parakeet default), so the lane and the role it
    rides on can never name different engines.
    """
    if lane.catalog_id:
        return lane.catalog_id, lane.runtime
    from lobes.roles import _STT_MODEL, _STT_RUNTIME, _declared_audio_engine

    return _declared_audio_engine(AUDIO_ROLE, env, _STT_MODEL, _STT_RUNTIME)
