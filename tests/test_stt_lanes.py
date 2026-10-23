"""Speech-to-text LANES (stt-lanes): ``stt-he`` and ``stt-en``.

Registry-level contract: two lanes, NOT roles (``lobes.roles.ROLES`` is
untouched), each declaring a language; ``stt-he`` pins its checkpoint and
revision, ``stt-en`` follows whatever the box's local ``stt`` container
declares (``STT_MODEL``/``STT_RUNTIME``). The ``stt-<suffix>`` spelling is a
narrow, documented exception to the ``{role}-{member}`` name rule, granted only
because ``stt`` has no member lanes.
"""

from __future__ import annotations

import dataclasses

import pytest

from lobes import stt_lanes
from lobes.embed_lanes import EMBED_LANES, validate_lane_name
from lobes.roles import MEMBER_LANE_EXCLUDED_ROLES, ROLES
from lobes.stt_lanes import STT_LANES, SttLane, lane_by_name, served_identity


def test_two_lanes_with_declared_languages():
    assert {lane.name: lane.language for lane in STT_LANES} == {"stt-he": "he", "stt-en": "en"}


def test_lanes_are_not_roles():
    for lane in STT_LANES:
        assert lane.name not in ROLES


def test_stt_he_declares_its_pinned_checkpoint():
    lane = lane_by_name("stt-he")
    assert lane is not None
    assert lane.catalog_id == "ivrit-ai/zipformer2-66m-streaming-he"
    assert lane.revision == "ee769a2a13b887f91572eec349458bfb0fe79e7a"
    assert lane.runtime == "sherpa-onnx"
    assert lane.licence == "CC-BY-4.0"


def test_stt_en_follows_the_local_stt_declaration():
    lane = lane_by_name("stt-en")
    assert lane is not None
    assert lane.catalog_id == ""  # not pinned: the box's own stt container decides
    env = {"STT_MODEL": "ivrit-ai/whisper-large-v3-turbo", "STT_RUNTIME": "transformers"}
    assert served_identity(lane, env) == ("ivrit-ai/whisper-large-v3-turbo", "transformers")


def test_stt_en_unset_falls_back_to_the_stt_role_default():
    from lobes.roles import _STT_MODEL, _STT_RUNTIME

    assert served_identity(lane_by_name("stt-en"), {}) == (_STT_MODEL, _STT_RUNTIME)


def test_stt_he_identity_ignores_the_box_stt_declaration():
    env = {"STT_MODEL": "something/else", "STT_RUNTIME": "nemo"}
    assert served_identity(lane_by_name("stt-he"), env) == (
        "ivrit-ai/zipformer2-66m-streaming-he",
        "sherpa-onnx",
    )


def test_env_keys_derive_from_the_lane_name():
    assert {lane.name: lane.base_url_env for lane in STT_LANES} == {
        "stt-he": "STT_HE_BASE_URL",
        "stt-en": "STT_EN_BASE_URL",
    }


def test_lanes_are_frozen():
    lane = dataclasses.replace(STT_LANES[0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        lane.name = "x"  # type: ignore[misc]
    assert all(isinstance(lane, SttLane) for lane in STT_LANES)


def test_task_is_transcribe_and_path_routed():
    from lobes.gateway._routing import _PATH_ROUTED_TASKS, TRANSCRIBE_TASK

    assert stt_lanes.TASK_TRANSCRIBE == TRANSCRIBE_TASK == "transcribe"
    assert TRANSCRIBE_TASK in _PATH_ROUTED_TASKS
    assert all(lane.task == TRANSCRIBE_TASK for lane in STT_LANES)


def test_no_collision_with_embed_lanes():
    assert not {lane.name for lane in STT_LANES} & {lane.name for lane in EMBED_LANES}


def test_unknown_name_is_none():
    assert lane_by_name("stt") is None
    assert lane_by_name("stt-fr") is None
    assert lane_by_name(None) is None


# --- the narrow name-rule exception ------------------------------------------


def test_stt_prefixed_names_are_still_refused_without_the_exception():
    with pytest.raises(ValueError, match="member-lane"):
        validate_lane_name("stt-he")


def test_the_exception_admits_stt_lanes_only():
    validate_lane_name("stt-he", audio_role="stt")
    validate_lane_name("stt-en", audio_role="stt")


@pytest.mark.parametrize("role", ["tts", "cortex", "embedder", "innereye", "decider"])
def test_the_exception_is_not_available_to_other_roles(role):
    with pytest.raises(ValueError):
        validate_lane_name(f"{role}-he", audio_role=role)


def test_the_exception_does_not_weaken_other_prefixes():
    # An stt lane may still not be named like ANOTHER role's member lane.
    with pytest.raises(ValueError, match="must be named stt-"):
        validate_lane_name("cortex-he", audio_role="stt")
    with pytest.raises(ValueError, match="collides"):
        validate_lane_name("stt", audio_role="stt")


def test_the_exception_rests_on_stt_having_no_member_lanes():
    assert "stt" in MEMBER_LANE_EXCLUDED_ROLES
