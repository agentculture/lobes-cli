"""The base fleet compose must pass GATEWAY_PUBLIC_URL and AUDIO_URL to the
gateway container WITHOUT the ``-f docker-compose.audio.yml`` overlay
(issues #92 / #96).

Two bugs this guards against:

  (a) AUDIO_URL only reached the gateway via docker-compose.audio.yml's
      override block. On a base-only deployment (no --audio overlay),
      ServerConfig.audio_url is empty and POST /v1/audio/speech 404s, while
      `lobes capabilities` (reading the merged .env) reports stt/tts as
      ready=true. AUDIO_URL must be present in the base template's gateway
      environment, defaulted to EMPTY (``${AUDIO_URL:-}``) — the audio overlay
      is what supplies the real ``http://realtime:8080`` value; a base-only
      deployment must resolve audio_url to unset, not a URL of a container
      that was never started (issue #96).

  (b) GATEWAY_PUBLIC_URL must default to EMPTY. It is an operator override
      ONLY — a tunnel URL or a Host-rewriting reverse proxy. It must NOT be
      defaulted to a localhost/published-port URL, because
      ``lobes.gateway.server.reachable_origin`` prefers a set ``public_url``
      OVER the request Host header::

          if public_url:   return public_url.rstrip("/")   # a default here wins
          if host_header:  return f"{scheme}://{host_header}"
          return None

      So ANY default (e.g. ``http://localhost:${VLLM_PORT}``) would advertise
      loopback to every remote client in GET /capabilities: a LAN/tunnel
      caller dialing ``spark.local:8001`` would be told to dial
      ``http://localhost:8001`` — a foreign service on THEIR machine. That is
      the #92 defect (advertised endpoint points at a foreign daemon). The
      correct precedence is: explicit operator override > request Host header
      > empty (never an internal-port-derived URL). The compose side owns
      keeping the default empty; the resolver side is enforced elsewhere
      (plan task t6).
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


class TestGatewayPublicUrlDefaultsEmpty:
    """GATEWAY_PUBLIC_URL must default to EMPTY — it is an operator override
    only (a tunnel / Host-rewriting reverse proxy). A defaulted value would
    outrank the request Host header and advertise loopback to remote clients
    (the #92 defect)."""

    def test_default_is_exactly_empty(self) -> None:
        env = _gateway_env_map()
        assert env["GATEWAY_PUBLIC_URL"] == "${GATEWAY_PUBLIC_URL:-}", (
            "GATEWAY_PUBLIC_URL must default to empty (${GATEWAY_PUBLIC_URL:-}) "
            "— any non-empty default outranks the request Host header in "
            "reachable_origin and advertises the wrong origin to remote "
            "clients (issue #92)"
        )

    def test_default_has_no_localhost_or_vllm_port(self) -> None:
        # THE regression guard: the amended requirement (c29) forbids a
        # localhost/published-port default. If someone "helpfully" restores a
        # ${VLLM_PORT}-derived default, this fails. Neither token may appear.
        env = _gateway_env_map()
        value = env["GATEWAY_PUBLIC_URL"]
        assert "localhost" not in value, (
            "GATEWAY_PUBLIC_URL default must not contain 'localhost' — a "
            "loopback default is advertised to remote clients (issue #92)"
        )
        assert "VLLM_PORT" not in value, (
            "GATEWAY_PUBLIC_URL default must not be derived from VLLM_PORT — "
            "it is an operator override only, defaulted empty (issue #92)"
        )

    def test_still_operator_overridable(self) -> None:
        # The var name itself must still gate on GATEWAY_PUBLIC_URL, so an
        # operator-set GATEWAY_PUBLIC_URL in .env still wins (a tunnel /
        # Host-rewriting reverse proxy) — the override path is preserved.
        env = _gateway_env_map()
        value = env["GATEWAY_PUBLIC_URL"]
        assert value.startswith(
            "${GATEWAY_PUBLIC_URL:-"
        ), f"GATEWAY_PUBLIC_URL must remain operator-overridable — got {value!r}"


class TestAudioUrlDefaultsEmpty:
    """AUDIO_URL must default to EMPTY on the base template — the audio
    overlay (docker-compose.audio.yml) supplies the real
    http://realtime:8080 value; a base-only deployment has no realtime
    container to point at (issue #96)."""

    def test_default_is_empty_not_a_url(self) -> None:
        env = _gateway_env_map()
        assert env["AUDIO_URL"] == "${AUDIO_URL:-}", (
            "AUDIO_URL must default to empty on the base template — a non-empty "
            "default (e.g. http://realtime:8080) would advertise a realtime "
            "container that a base-only (no --audio) deployment never starts"
        )

    def test_still_operator_overridable(self) -> None:
        # The audio overlay overrides AUDIO_URL by re-declaring the key; the
        # base default must still gate on ${AUDIO_URL:-...} so an operator or
        # the overlay can supply the real value.
        env = _gateway_env_map()
        assert env["AUDIO_URL"].startswith("${AUDIO_URL:-")


class TestPortsMappingDistinguishesPublishedFromInternal:
    """The published host port (VLLM_PORT) differs from the gateway's internal
    container port (GATEWAY_PORT=8000). This distinction is what makes a
    localhost:GATEWAY_PORT default wrong on a rig that publishes elsewhere
    (e.g. :8001) — kept as documentation of the root cause."""

    def test_ports_mapping_uses_vllm_port_published_to_internal_8000(self) -> None:
        compose = _load_fleet()
        ports = compose["services"]["gateway"]["ports"]
        assert "${VLLM_PORT:-8000}:8000" in ports, (
            "gateway must publish ${VLLM_PORT:-8000} -> internal 8000; the "
            "published port can differ from the internal one, which is exactly "
            "why GATEWAY_PUBLIC_URL must not be built from the internal port"
        )
