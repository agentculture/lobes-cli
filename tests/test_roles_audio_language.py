"""The audio roles' ``language`` advert (hebrew-realtime, approved deviation d3).

``STT_LANGUAGE`` / ``TTS_LANGUAGE`` reach ``GET /capabilities`` and ``lobes
capabilities`` as a ``language`` key on the ``stt`` / ``tts`` roles. The key is
ABSENT — not null — everywhere nothing is declared, so every existing
deployment's advert stays byte-identical.
"""

from __future__ import annotations

import dataclasses

from lobes.gateway._config import build_config
from lobes.roles import ROLES, build_role_registry, role_payload

_ENV = {
    "PRIMARY_URL": "http://vllm-primary:8000",
    "PRIMARY_SERVED_NAME": "x/y",
    "AUDIO_URL": "http://realtime:8080",
}


def _registry(env):
    table, server = build_config(env)
    return build_role_registry(table, server, env=env, gateway_url="http://localhost:8000")


def test_nothing_declared_no_role_carries_the_key_and_the_payload_is_asdict_minus_it():
    registry = _registry(dict(_ENV))
    for role in ROLES:
        payload = role_payload(registry[role])
        assert "language" not in payload, role
        expected = dataclasses.asdict(registry[role])
        expected.pop("language")
        assert payload == expected


def test_declared_languages_are_advertised_per_lane():
    registry = _registry({**_ENV, "STT_LANGUAGE": "he", "TTS_LANGUAGE": " HE "})
    assert role_payload(registry["stt"])["language"] == "he"
    assert role_payload(registry["tts"])["language"] == "he"


def test_the_two_lanes_are_independent():
    registry = _registry({**_ENV, "STT_LANGUAGE": "he"})
    assert role_payload(registry["stt"])["language"] == "he"
    assert "language" not in role_payload(registry["tts"])


def test_a_blank_declaration_is_nothing_declared():
    registry = _registry({**_ENV, "STT_LANGUAGE": "  "})
    assert "language" not in role_payload(registry["stt"])


def test_no_generate_or_pooling_role_ever_gets_a_language():
    registry = _registry({**_ENV, "STT_LANGUAGE": "he", "TTS_LANGUAGE": "he"})
    for role in ROLES:
        if role not in ("stt", "tts"):
            assert "language" not in role_payload(registry[role]), role


def test_both_serializers_use_the_shared_payload():
    import inspect

    from lobes.cli._commands import capabilities
    from lobes.gateway import server

    assert "role_payload(" in inspect.getsource(capabilities._role_payload)
    assert "role_payload(registry[role])" in inspect.getsource(server)
