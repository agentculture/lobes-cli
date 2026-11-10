"""t21: the split/tunnel evidence keys reach the gateway container.

The gateway takes an explicit ``environment:`` list, never env_file, so a key
set in .env but absent from that list is silently inert.
"""

from __future__ import annotations

import re
from importlib import resources

import yaml

from lobes.gateway._evidence import SPLIT_ROLE_PREFIX, TUNNEL_ENV

_FLEET = resources.files("lobes.templates").joinpath("fleet")
KEYS = [TUNNEL_ENV] + [
    f"{p}_{s}" for p in SPLIT_ROLE_PREFIX.values() for s in ("SPANS", "WORLD_SIZE")
]


def _gateway_env() -> dict[str, str]:
    compose = yaml.safe_load(_FLEET.joinpath("docker-compose.yml").read_text(encoding="utf-8"))
    return dict(e.partition("=")[::2] for e in compose["services"]["gateway"]["environment"])


def test_every_evidence_key_is_an_empty_default_passthrough():
    env = _gateway_env()
    for key in KEYS:
        assert env.get(key) == f"${{{key}:-}}", key


def test_every_evidence_key_is_documented_in_env_example():
    text = _FLEET.joinpath("env.example").read_text(encoding="utf-8")
    for key in KEYS:
        assert re.search(rf"^# {key}=$", text, re.M), key
