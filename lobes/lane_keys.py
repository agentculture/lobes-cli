"""The ONE spelling of a specialist lane's env keys (``<LANE>_<SUFFIX>``).

Dependency-free on purpose: ``lobes.embed_lanes`` (registry), the gateway
(``lobes.gateway._config``/``server``) and the shape renderer
(``lobes.profiles.shape_render``) all import it, and the roles<->gateway import
cycle means none of them can import each other for this. Changing the rule here
changes it everywhere, so the BASE_URL a shape renders and the key the gateway
reads can never drift apart.
"""

from __future__ import annotations


def lane_env_prefix(lane_name: str) -> str:
    """``gemma2-embed`` -> ``GEMMA2_EMBED``."""
    return lane_name.upper().replace("-", "_")


def lane_env_key(lane_name: str, suffix: str) -> str:
    """``lane_env_key("gemma2-embed", "BASE_URL")`` -> ``GEMMA2_EMBED_BASE_URL``."""
    return f"{lane_env_prefix(lane_name)}_{suffix}"
