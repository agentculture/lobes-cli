"""The associate lane's AUTHENTICATED front (lightning-on-orin plan, t10).

This module exists because of an incident, not a hypothesis. On 2026-08-25 the
spike ran NVIDIA's published Jetson recipe verbatim (frame boundary c18 funded
exactly that, for the SPIKE only) and the recipe binds an OpenAI-compatible
generate endpoint on the box's tailnet with no credential. Within seconds of
the API server starting, TWO DISTINCT tailnet peers queried it, neither
initiated by the operator --
``docs/evidence/2026-08-25-spike-lightning-vllm-orin.txt`` records::

    INFO:     100.127.105.72:50652 - "GET /v1/models HTTP/1.1" 200 OK
    INFO:     100.105.216.63:55196 - "GET /v1/models HTTP/1.1" 200 OK

Frame claim c30 predicted the exposure; c46 records that it was REALISED. The
shipped lane therefore departs from the vendor recipe in one NAMED way (frame
boundary c29 requires the departure be named as a correction, not smuggled):
**the recipe's host-network binding is not inherited**. The container still
passes ``--host=0.0.0.0``, because inside its own bridge network namespace
that means "this container's interfaces", not "the box's tailnet address" --
and the service publishes NO host port, so the only way in is the gateway,
which carries the opt-in ``GATEWAY_API_KEY`` bearer gate.

Three things are pinned here, one per acceptance criterion:

1. **Refused without a credential** -- with ``GATEWAY_API_KEY`` set, an
   unauthenticated ``model=associate`` request 401s at the gateway's inbound
   edge with ZERO upstream dials, so the lane is never touched.
2. **No open port** -- the RENDERED compose (``docker compose config`` with
   the ``associate`` profile active, plus the static YAML) publishes no host
   port for any generate lane and sets ``network_mode: host`` nowhere; the
   gateway is the single published surface.
3. **The incident is on the record** -- the evidence transcript quotes both
   observed tailnet client IPs.

Plus the deployment-level half: ``lobes doctor`` FAILS an associate-hosting
deployment that has no inbound key at all, so "binds behind ``GATEWAY_API_KEY``"
is checkable rather than a thing the operator is trusted to have remembered.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lobes.cli import main
from lobes.cli._commands import doctor as doctor_module
from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.runtime import _compose, _detect, _env

_REPO = Path(__file__).resolve().parents[1]
_FLEET_COMPOSE = _REPO / "lobes" / "templates" / "fleet" / "docker-compose.yml"
_EVIDENCE = _REPO / "docs" / "evidence" / "2026-08-26-associate-gateway-auth-front.txt"

#: The two tailnet clients that queried the unauthenticated spike endpoint.
#: Quoted verbatim from the spike transcript -- neither was initiated by the
#: operator's session (honesty condition h32).
_OBSERVED_TAILNET_CLIENTS = ("100.127.105.72", "100.105.216.63")
_SPIKE_TRANSCRIPT = _REPO / "docs" / "evidence" / "2026-08-25-spike-lightning-vllm-orin.txt"

_KEY = "sk-lobes-associate-inbound-0001"
_ASSOCIATE_URL = "http://vllm-associate:8000"

#: Every lane in the fleet template that serves a generate/pooling model. None
#: of them may publish a host port -- the gateway is the only front door.
_MODEL_SERVICES = (
    "vllm-primary",
    "vllm-multimodal",
    "vllm-muse",
    "vllm-worker",
    "vllm-associate",
    "vllm-hand",
    "vllm-embed",
    "vllm-rerank",
)


def _fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# criterion 2 -- the lane binds through the gateway, never an open port
# ---------------------------------------------------------------------------


class TestNoUnauthenticatedGeneratePort:
    def test_gateway_is_the_only_service_publishing_a_host_port(self) -> None:
        services = _fleet()["services"]
        publishing = sorted(name for name, svc in services.items() if svc.get("ports"))
        assert publishing == ["gateway"], (
            "only the gateway may publish a host port -- every model lane is reachable "
            "solely over the compose network, behind the GATEWAY_API_KEY gate"
        )

    @pytest.mark.parametrize("service", _MODEL_SERVICES)
    def test_model_lane_publishes_nothing_and_only_exposes(self, service: str) -> None:
        svc = _fleet()["services"][service]
        assert "ports" not in svc, f"{service} must not publish a host port"
        assert "8000" in [str(p) for p in svc.get("expose", [])]

    def test_no_service_uses_host_networking(self) -> None:
        """The NAMED departure from the vendor recipe (frame boundary c29).

        NVIDIA's published Jetson recipe runs the container with
        ``--network host``, which is what put an uncredentialed 30B endpoint
        on this box's tailnet during the spike. The shipped lane does not
        inherit it: with a bridge network namespace, the recipe's own
        ``--host=0.0.0.0`` binds only the container's interfaces.
        """
        for name, svc in _fleet()["services"].items():
            assert svc.get("network_mode") != "host", f"{name} must not use host networking"

    def test_associate_command_keeps_host_0_0_0_0_inside_the_namespace(self) -> None:
        # Kept deliberately: the flag is correct *given* no host networking,
        # and changing it would break the gateway's own compose-network dial.
        # This test is what makes the pairing (bridge net + 0.0.0.0) explicit
        # rather than an accident of copying the recipe.
        command = _fleet()["services"]["vllm-associate"]["command"]
        assert "--host=0.0.0.0" in command
        assert "network_mode" not in _fleet()["services"]["vllm-associate"]

    def test_rendered_compose_publishes_no_generate_port(self) -> None:
        """The RENDERED (substituted) compose, associate profile active."""
        if shutil.which("docker") is None:  # pragma: no cover - env dependent
            pytest.skip("docker not available")
        proc = subprocess.run(
            ["docker", "compose", "-f", str(_FLEET_COMPOSE), "--profile", "associate", "config"],
            capture_output=True,
            text=True,
            cwd=str(_FLEET_COMPOSE.parent),
            env={"PATH": __import__("os").environ.get("PATH", "")},
        )
        if proc.returncode != 0:  # pragma: no cover - env dependent
            pytest.skip(f"docker compose config unavailable: {proc.stderr[-300:]}")
        rendered = yaml.safe_load(proc.stdout)["services"]
        assert "vllm-associate" in rendered, "the associate profile must render the lane"
        publishing = sorted(name for name, svc in rendered.items() if svc.get("ports"))
        assert publishing == ["gateway"]
        assert not rendered["vllm-associate"].get("ports")


def _fake_orin_card() -> "_detect.DetectedCard":
    return _detect.DetectedCard(
        resolved="orin",
        device_name="NVIDIA Test",
        compute_capability="sm_87",
        total_memory_gb=61.3,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )


class TestTheRenderedDeploymentAddsNoPort:
    """`lobes init --shape orin-associate --apply` must not open one either.

    The shape overlay is the only other file that can change the deployment's
    service definitions, so an exposure guarantee proved against the packaged
    template alone would be incomplete.
    """

    def test_shape_overlay_publishes_nothing(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(_detect, "detect_card", _fake_orin_card)
        assert (
            main(
                ["init", str(tmp_path), "--profile", "orin", "--shape", "orin-associate", "--apply"]
            )
            == 0
        )
        # Read as TEXT, not YAML: the overlay carries compose's own `!reset`
        # tag, which no stdlib/PyYAML safe loader knows. The assertion is a
        # plain absence, so text is the honest instrument here.
        overlay = (tmp_path / _compose.SHAPE_OVERLAY).read_text(encoding="utf-8")
        assert "ports:" not in overlay, "the shape overlay must never publish a host port"
        assert "network_mode" not in overlay

    def test_scaffolded_fleet_compose_still_publishes_only_the_gateway(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(_detect, "detect_card", _fake_orin_card)
        assert (
            main(
                ["init", str(tmp_path), "--profile", "orin", "--shape", "orin-associate", "--apply"]
            )
            == 0
        )
        deployed = yaml.safe_load((tmp_path / _compose.COMPOSE_FILE).read_text(encoding="utf-8"))[
            "services"
        ]
        publishing = sorted(name for name, svc in deployed.items() if svc.get("ports"))
        assert publishing == ["gateway"]


# ---------------------------------------------------------------------------
# criterion 1 -- an unauthenticated peer is refused, and never reaches the lane
# ---------------------------------------------------------------------------


class _FakeUpstream:
    def __init__(self, name: str) -> None:
        self.status = 200
        self.headers = [("Content-Type", "application/json")]
        self._body = json.dumps({"served": name}).encode()

    def read_all(self):
        return self._body

    def read(self, _n):
        data, self._body = self._body, b""
        return data

    def close(self):
        pass


def _spawn(monkeypatch, **env_over):
    env = {
        "PRIMARY_SERVED_NAME": "P",
        "ASSOCIATE_BASE_URL": _ASSOCIATE_URL,
        "ASSOCIATE_SERVED_NAME": "A",
        "ASSOCIATE_FEASIBLE": "true",
        "GATEWAY_DEFAULT_MODEL": "P",
    }
    env.update(env_over)
    table, cfg = build_config(env)
    opened: list[str] = []

    def fake_open(backend, path, body, headers, *, connect_timeout, read_timeout):
        opened.append(backend.name)
        return _FakeUpstream(backend.name)

    monkeypatch.setattr(S, "open_upstream", fake_open)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    return SimpleNamespace(base=f"http://{host}:{port}", opened=opened, httpd=httpd)


@pytest.fixture
def gated(monkeypatch):
    """An associate-hosting gateway with the inbound bearer gate ARMED."""
    gw = _spawn(monkeypatch, GATEWAY_API_KEY=_KEY)
    try:
        yield gw
    finally:
        gw.httpd.shutdown()
        gw.httpd.server_close()


def _post_associate(base, *, headers=None):
    body = json.dumps({"model": "associate", "messages": []}).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=body,
        method="POST",
        headers=headers or {},
    )
    return urllib.request.urlopen(req, timeout=5)


class TestUnauthenticatedPeerIsRefused:
    @pytest.mark.parametrize(
        "headers",
        [
            None,  # a peer that holds no key at all -- the incident's shape
            {"Authorization": "Bearer sk-some-other-boxes-key"},
            {"Authorization": _KEY},  # bare token, no scheme
        ],
        ids=["no-credential", "wrong-key", "malformed"],
    )
    def test_generate_request_401s_and_never_dials_the_lane(self, gated, headers) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post_associate(gated.base, headers=headers)
        assert exc.value.code == 401
        assert exc.value.headers.get("WWW-Authenticate", "").startswith("Bearer")
        assert gated.opened == [], "a rejected request must never reach the associate lane"

    def test_model_listing_is_also_gated(self, gated) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(gated.base + "/v1/models", timeout=5)
        assert exc.value.code == 401
        # The incident WAS a GET /v1/models from two peers; that exact probe
        # is what the gate must refuse.
        assert gated.opened == []

    def test_401_body_never_echoes_key_material(self, gated) -> None:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post_associate(gated.base)
        body = exc.value.read().decode()
        assert _KEY not in body
        assert "invalid_api_key" in body

    def test_the_credentialed_caller_reaches_the_lane(self, gated) -> None:
        resp = _post_associate(gated.base, headers={"Authorization": f"Bearer {_KEY}"})
        assert resp.status == 200
        assert gated.opened == ["associate"]


# ---------------------------------------------------------------------------
# the deployment half -- "behind GATEWAY_API_KEY" is CHECKED, not assumed
# ---------------------------------------------------------------------------


def _scaffold_fleet(path):
    _compose.write_scaffold(path, force=True, templates=dict(_compose.FLEET_TEMPLATES))
    _compose.write_plugin_file(path, force=True)
    _env.set_env(path / ".env", "LOBES_PROFILE", "orin")
    return path


def _doctor_json(capsys) -> dict:
    main(["doctor", "--json"])
    return json.loads(capsys.readouterr().out)


class TestDoctorRefusesAnUnauthenticatedAssociateDeployment:
    def test_wired_associate_without_any_inbound_key_fails(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "ASSOCIATE_BASE_URL", _ASSOCIATE_URL)

        payload = _doctor_json(capsys)
        check = {c["id"]: c for c in payload["checks"]}["associate_auth_gate"]
        assert check["passed"] is False
        assert check["severity"] == "error"
        assert payload["healthy"] is False
        assert "GATEWAY_API_KEY" in check["remediation"]

    @pytest.mark.parametrize("key_var", ["GATEWAY_API_KEY", "CULTURE_VLLM_API_KEY"])
    def test_either_inbound_key_channel_satisfies_the_gate(
        self, tmp_path, monkeypatch, capsys, key_var
    ):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "ASSOCIATE_BASE_URL", _ASSOCIATE_URL)
        _env.set_env(tmp_path / ".env", key_var, _KEY)

        payload = _doctor_json(capsys)
        check = {c["id"]: c for c in payload["checks"]}["associate_auth_gate"]
        assert check["passed"] is True
        assert _KEY not in json.dumps(payload), "doctor must never echo key material"

    def test_a_blank_key_does_not_count_as_set(self, tmp_path, monkeypatch, capsys):
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "ASSOCIATE_BASE_URL", _ASSOCIATE_URL)
        _env.set_env(tmp_path / ".env", "GATEWAY_API_KEY", "   ")

        payload = _doctor_json(capsys)
        assert {c["id"]: c for c in payload["checks"]}["associate_auth_gate"]["passed"] is False

    def test_deployment_not_hosting_associate_is_unaffected(self, tmp_path, monkeypatch, capsys):
        """No ASSOCIATE_BASE_URL -> the check does not fire at all.

        Every pre-associate deployment (and every box that refers/proxies
        associate to a peer rather than hosting it) must be byte-identically
        unaffected -- this is an ADDITIVE finding, not a new fleet-wide
        requirement that an existing worker/muse box suddenly fails.
        """
        _scaffold_fleet(tmp_path)
        monkeypatch.setenv("LOBES_DIR", str(tmp_path))
        monkeypatch.setattr(_compose, "docker_available", lambda: True)
        _env.set_env(tmp_path / ".env", "WORKER_BASE_URL", "http://vllm-worker:8000")

        payload = _doctor_json(capsys)
        assert "associate_auth_gate" not in {c["id"] for c in payload["checks"]}

    def test_a_referred_associate_peer_origin_does_not_trip_the_gate(self, tmp_path):
        """Declaring a PEER that hosts associate is not hosting it locally."""
        (tmp_path / ".env").write_text(
            "ASSOCIATE_PEER_ORIGIN=http://peer.example:8000\n", encoding="utf-8"
        )
        assert doctor_module._associate_auth_gate_check(tmp_path) is None


# ---------------------------------------------------------------------------
# criterion 3 -- the incident is quoted, not paraphrased
# ---------------------------------------------------------------------------


class TestEvidenceQuotesTheIncident:
    def test_evidence_file_exists(self) -> None:
        assert _EVIDENCE.is_file(), f"missing evidence file: {_EVIDENCE}"

    @pytest.mark.parametrize("client_ip", _OBSERVED_TAILNET_CLIENTS)
    def test_both_observed_tailnet_client_ips_are_quoted(self, client_ip: str) -> None:
        assert client_ip in _EVIDENCE.read_text(encoding="utf-8")

    @pytest.mark.parametrize("client_ip", _OBSERVED_TAILNET_CLIENTS)
    def test_the_quoted_ips_really_are_in_the_spike_transcript(self, client_ip: str) -> None:
        """The quote must be traceable to the primary source, not invented."""
        assert f"{client_ip}:" in _SPIKE_TRANSCRIPT.read_text(encoding="utf-8")

    def test_evidence_names_the_vendor_recipe_departure(self) -> None:
        text = _EVIDENCE.read_text(encoding="utf-8")
        assert "--network host" in text
        assert "GATEWAY_API_KEY" in text
