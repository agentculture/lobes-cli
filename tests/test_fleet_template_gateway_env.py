"""The base fleet compose must pass GATEWAY_PUBLIC_URL and AUDIO_URL to the
gateway container WITHOUT the ``-f docker-compose.audio.yml`` overlay (issue
#96).

Two bugs this guards against:

  (a) GATEWAY_PUBLIC_URL defaulted to empty (``${GATEWAY_PUBLIC_URL:-}``), so
      the advertised /capabilities origin fell back to inferring from the
      request Host header — and when no Host header is present, fabricated an
      absolute URL from the gateway's INTERNAL listen port
      (GATEWAY_PORT=8000), not the published host port (VLLM_PORT, default
      8001 on the reference rig). The default must instead be built from the
      published ``VLLM_PORT`` mapping (``"${VLLM_PORT:-8000}:8000"``), so the
      advertised origin is configured truth.

  (b) AUDIO_URL only reached the gateway via docker-compose.audio.yml's
      override block. On a base-only deployment (no --audio overlay),
      ServerConfig.audio_url is empty and POST /v1/audio/speech 404s, while
      `lobes capabilities` (reading the merged .env) reports stt/tts as
      ready=true. AUDIO_URL must be present in the base template's gateway
      environment, defaulted to EMPTY (``${AUDIO_URL:-}``) — the audio overlay
      is what supplies the real ``http://realtime:8080`` value; a base-only
      deployment must resolve audio_url to unset, not a URL of a container
      that was never started.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"


def _load_fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


def _gateway_env_map() -> dict[str, str]:
    """The gateway's ``environment:`` list, as a ``{KEY: raw-value}`` dict.

    Values keep their raw ``${...}`` interpolation text (PyYAML expands
    nothing) — exactly what a caller composing WITHOUT the audio overlay
    would see baked into the base template.
    """
    compose = _load_fleet()
    env: list[str] = compose["services"]["gateway"]["environment"]
    out: dict[str, str] = {}
    for entry in env:
        assert "=" in entry, f"non key=value gateway environment entry: {entry!r}"
        key, _, value = entry.partition("=")
        out[key] = value
    return out


class TestGatewayEnvKeysPresent:
    """Both vars must reach the gateway from the BASE template alone."""

    def test_gateway_public_url_key_present(self) -> None:
        env = _gateway_env_map()
        assert "GATEWAY_PUBLIC_URL" in env

    def test_audio_url_key_present(self) -> None:
        env = _gateway_env_map()
        assert "AUDIO_URL" in env


class TestGatewayPublicUrlDefaultsFromVllmPort:
    """The GATEWAY_PUBLIC_URL default must be built from the published
    VLLM_PORT mapping (``"${VLLM_PORT:-8000}:8000"``), not left empty."""

    def test_default_is_not_empty(self) -> None:
        env = _gateway_env_map()
        value = env["GATEWAY_PUBLIC_URL"]
        # The old, buggy form: an empty default that falls through to
        # Host-header inference (and, absent a Host header, the internal port).
        assert value != "${GATEWAY_PUBLIC_URL:-}", (
            "GATEWAY_PUBLIC_URL must not default to empty — an empty default "
            "lets the advertised origin fall back to the gateway's internal "
            "listen port when no Host header is present (issue #96)"
        )

    def test_default_references_vllm_port(self) -> None:
        env = _gateway_env_map()
        value = env["GATEWAY_PUBLIC_URL"]
        assert "${VLLM_PORT" in value, (
            "GATEWAY_PUBLIC_URL's default must be derived from ${VLLM_PORT:-...} "
            f"(the published host port) — got {value!r}"
        )

    def test_still_overridable_by_operator(self) -> None:
        # The var name itself must still gate on GATEWAY_PUBLIC_URL, so an
        # operator-set GATEWAY_PUBLIC_URL in .env continues to win (a tunnel /
        # Host-rewriting reverse proxy).
        env = _gateway_env_map()
        value = env["GATEWAY_PUBLIC_URL"]
        assert value.startswith(
            "${GATEWAY_PUBLIC_URL:-"
        ), f"GATEWAY_PUBLIC_URL must remain operator-overridable — got {value!r}"

    def test_ports_mapping_is_the_vllm_port_default_this_test_assumes(self) -> None:
        # Guard the assumption this whole class is built on: the published
        # mapping is "${VLLM_PORT:-8000}:8000". If that ever changes, this
        # test (and the GATEWAY_PUBLIC_URL default it drives) must be revisited.
        compose = _load_fleet()
        ports = compose["services"]["gateway"]["ports"]
        assert "${VLLM_PORT:-8000}:8000" in ports

    def test_rendered_default_matches_published_port_shape(self) -> None:
        # Exact string check on the template as documented in the task: the
        # nested-default form composing GATEWAY_PUBLIC_URL from VLLM_PORT.
        env = _gateway_env_map()
        assert (
            env["GATEWAY_PUBLIC_URL"]
            == "${GATEWAY_PUBLIC_URL:-http://localhost:${VLLM_PORT:-8000}}"
        )


class TestAudioUrlDefaultsEmpty:
    """AUDIO_URL must default to EMPTY on the base template — the audio
    overlay (docker-compose.audio.yml) supplies the real
    http://realtime:8080 value; a base-only deployment has no realtime
    container to point at."""

    def test_default_is_empty_not_a_url(self) -> None:
        env = _gateway_env_map()
        assert env["AUDIO_URL"] == "${AUDIO_URL:-}", (
            "AUDIO_URL must default to empty on the base template — a non-empty "
            "default (e.g. http://realtime:8080) would advertise a realtime "
            "container that a base-only (no --audio) deployment never starts"
        )
