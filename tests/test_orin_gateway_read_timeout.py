"""``GATEWAY_READ_TIMEOUT`` on the Orin card only (plan orin-associate-at-1m, t4).

The Orin's cold, 1.04M-token associate request measured a 2,390.17 s TTFT
(``docs/evidence/2026-09-13-measure-associate-budget-orin-1m.txt``) — nearly
4x the gateway's shipped 600 s socket-read timeout
(``lobes/gateway/_config.py``'s ``GATEWAY_READ_TIMEOUT`` default, applied to
every upstream POST in ``lobes/gateway/server.py``). ``host_env`` is the
schema's designed home for a box-wide gateway fact that is not a per-role vLLM
knob (``lobes/profiles/schema.py``'s module docstring;
``lobes/profiles/render.py``'s ``profile_env`` renders ``host_env`` first).
Decision c30 (operator): only the Orin card changes — Spark, Thor and the
conservative ``base`` fallback keep the 600 s default, since only the Orin
serves a lane whose cold TTFT can run that long.
"""

from __future__ import annotations

from lobes.profiles.loader import resolve_profile
from lobes.profiles.render import profile_env
from lobes.runtime._lock import lock_keys

_KEY = "GATEWAY_READ_TIMEOUT"


def test_orin_host_env_declares_gateway_read_timeout() -> None:
    env = profile_env(resolve_profile("orin"))
    assert env[_KEY] == "7200"


def test_spark_thor_base_do_not_declare_gateway_read_timeout() -> None:
    for name in ("spark", "thor", "base"):
        env = profile_env(resolve_profile(name))
        assert _KEY not in env, f"{name} unexpectedly declares {_KEY}"


def test_gateway_read_timeout_is_in_the_deployment_lock_allowlist() -> None:
    # The lock allowlist derives host_env keys from every packaged built-in
    # profile (lobes/runtime/_lock.py's lock_keys) — no separate declaration
    # needed here; this pins that derivation actually picks the new key up.
    assert _KEY in lock_keys()
