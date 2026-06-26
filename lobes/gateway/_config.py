"""Build the gateway's :class:`RoutingTable` + :class:`ServerConfig` from env vars.

Reads a mapping (``os.environ`` by default) and constructs frozen config objects.
No sockets — pass a plain ``dict`` to unit-test it offline. The env keys mirror
the ``gateway`` service's ``environment:`` block in the fleet compose template.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from lobes.gateway._routing import Backend, RoutingTable

_DEFAULT_PRIMARY = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_DEFAULT_FALLBACK = "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4"
_DEFAULT_EMBED = "Qwen/Qwen3-Embedding-0.6B"
_DEFAULT_RERANK = "Qwen/Qwen3-Reranker-0.6B"
_DEFAULT_MINOR = "Qwen/Qwen3.5-4B"


@dataclass(frozen=True)
class ServerConfig:
    """Where the gateway listens and how patient it is with backends."""

    host: str
    port: int
    connect_timeout: float  # short: a refused/down backend fails over fast
    read_timeout: float  # long: a reasoning model's first token is slow
    # The audio/realtime backend that serves /v1/audio/* (+ /v1/realtime in PR2).
    # None on a text-only fleet → those paths 404. Set by the --audio overlay.
    audio_url: str | None = None


def _parse_aliases(raw: str | None) -> dict[str, str]:
    """Parse ``alias=served,other=served`` into a dict; skip blank/malformed pairs."""
    out: dict[str, str] = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        alias, _, target = pair.partition("=")
        alias, target = alias.strip(), target.strip()
        if alias and target:
            out[alias] = target
    return out


def _as_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key) or default)
    except (TypeError, ValueError):
        return float(default)


def _as_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key) or default)
    except (TypeError, ValueError):
        return int(default)


def _optional_backend(
    env: Mapping[str, str],
    *,
    name: str,
    url_key: str,
    name_key: str,
    default_url: str,
    default_name: str,
    task: str = "generate",
) -> Backend | None:
    """A fleet backend wired only when its env (``url_key`` or ``name_key``) is set.

    Returns ``None`` when neither is present — so the default gateway serves the
    primary alone, and each extra backend (fallback / embed / rerank) opts in
    independently via its own env pair.
    """
    if not (env.get(url_key) or env.get(name_key)):
        return None
    return Backend(
        name=name,
        base_url=(env.get(url_key) or default_url).rstrip("/"),
        served_name=env.get(name_key) or default_name,
        task=task,
    )


def build_config(env: Mapping[str, str] | None = None) -> tuple[RoutingTable, ServerConfig]:
    """Construct the routing table and server config from environment variables."""
    env = os.environ if env is None else env

    primary = Backend(
        name="primary",
        base_url=(env.get("PRIMARY_URL") or "http://vllm-primary:8000").rstrip("/"),
        served_name=env.get("PRIMARY_SERVED_NAME") or _DEFAULT_PRIMARY,
    )
    # The primary is always present; fallback / embed / rerank are each wired only
    # when their own env pair is set (so the default gateway serves the primary
    # alone, and a pooling/fallback backend opts in independently).
    optional = (
        _optional_backend(
            env,
            name="fallback",
            url_key="FALLBACK_URL",
            name_key="FALLBACK_SERVED_NAME",
            default_url="http://vllm-fallback:8000",
            default_name=_DEFAULT_FALLBACK,
        ),
        # The minor co-resident generate backend (Qwen/Qwen3.5-4B, bf16).
        # Wired only when MINOR_BASE_URL or MINOR_SERVED_NAME is present in
        # the environment — i.e. when the operator has activated the compose
        # "minor" profile and set these vars (they are absent by default so
        # the routing table is unchanged on a standard fleet startup).
        _optional_backend(
            env,
            name="minor",
            url_key="MINOR_BASE_URL",
            name_key="MINOR_SERVED_NAME",
            default_url="http://vllm-minor:8000",
            default_name=_DEFAULT_MINOR,
        ),
        _optional_backend(
            env,
            name="embed",
            url_key="EMBED_URL",
            name_key="EMBED_SERVED_NAME",
            default_url="http://vllm-embed:8000",
            default_name=_DEFAULT_EMBED,
            task="embed",
        ),
        _optional_backend(
            env,
            name="rerank",
            url_key="RERANK_URL",
            name_key="RERANK_SERVED_NAME",
            default_url="http://vllm-rerank:8000",
            default_name=_DEFAULT_RERANK,
            task="score",
        ),
    )
    backends = [primary, *(b for b in optional if b is not None)]
    table = RoutingTable(
        backends=tuple(backends),
        default_model=env.get("GATEWAY_DEFAULT_MODEL") or primary.served_name,
        aliases=_parse_aliases(env.get("GATEWAY_ALIASES")),
    )
    server = ServerConfig(
        host=env.get("GATEWAY_HOST") or "0.0.0.0",  # nosec B104 — bind all inside the container
        port=_as_int(env, "GATEWAY_PORT", 8000),
        connect_timeout=_as_float(env, "GATEWAY_CONNECT_TIMEOUT", 5.0),
        read_timeout=_as_float(env, "GATEWAY_READ_TIMEOUT", 600.0),
        audio_url=(env.get("AUDIO_URL") or "").rstrip("/") or None,
    )
    return table, server
