"""Deviation d2: the hand-defined split/tunnel keys t21 added are gone.

The site reads split brains and the hub from the running containers, so the
gateway must neither receive nor document ``<PREFIX>_SPANS``,
``<PREFIX>_WORLD_SIZE`` or ``LOBES_TUNNEL_PUBLIC_URL``.
"""

from __future__ import annotations

from importlib import resources

import yaml

_FLEET = resources.files("lobes.templates").joinpath("fleet")
_PREFIXES = ("PRIMARY", "MULTIMODAL", "MUSE", "WORKER", "ASSOCIATE", "HAND")
RETIRED = ["LOBES_TUNNEL_PUBLIC_URL"] + [
    f"{p}_{s}" for p in _PREFIXES for s in ("SPANS", "WORLD_SIZE")
]


def _gateway_env() -> dict[str, str]:
    compose = yaml.safe_load(_FLEET.joinpath("docker-compose.yml").read_text(encoding="utf-8"))
    return dict(e.partition("=")[::2] for e in compose["services"]["gateway"]["environment"])


def test_no_retired_key_reaches_the_gateway():
    assert len(RETIRED) == 13
    env = _gateway_env()
    for key in RETIRED:
        assert key not in env, key


def test_no_retired_key_is_documented_in_env_example():
    text = _FLEET.joinpath("env.example").read_text(encoding="utf-8")
    for key in RETIRED:
        assert key not in text, key
