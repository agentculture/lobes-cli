"""Tests for ``lobes capabilities`` / ``lobes endpoint`` (issue #81, tasks t5/t7).

These verbs are the CLI-side view of the six first-class Colleague-facing
roles (``cortex``/``senses``/``embedder``/``reranker``/``stt``/``tts``). Since
issue #96 (plan "advertised implies reachable", task t7) they are CLIENTS of
the gateway's own ``GET /capabilities`` rather than a second, independent
derivation from the deployment's ``.env`` — see the module docstring in
``lobes.cli._commands.capabilities`` for why re-deriving the same contract
twice from two different config sources is exactly what let issue #92 and
issue #96 drift in opposite directions. The tests below that don't care about
the live-vs-offline distinction hit the *offline* fallback (the autouse
``offline_runtime`` fixture in ``tests/conftest.py`` neutralises the gateway
probe, matching how it already neutralises ``/health``); the fake-gateway
tests near the bottom of this file explicitly restore the real probe against
a real loopback server. Both verbs are strictly read-only — no compose/docker
call, no ``--apply``.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lobes.cli import main
from lobes.cli._commands import capabilities as capabilities_module
from lobes.gateway._config import build_config
from lobes.gateway.server import capabilities_payload
from lobes.roles import ROLES, STT_REALTIME_RESPONSIBILITY
from lobes.runtime import _compose, _env

_ROLE_INFO_FIELDS = {
    "role",
    "model",
    "runtime",
    "endpoint",
    "path",
    "context",
    "quant",
    "mtp",
    "feasible",
    "responsibilities",
    "forbidden_responsibilities",
    "ready",
    "loaded",
}

# Captured at import time — BEFORE tests/conftest.py's autouse ``offline_runtime``
# fixture (correctly) neutralises ``capabilities_module._fetch_gateway_capabilities``
# for every other test in this file. The fake-gateway tests near the bottom
# restore this real implementation via ``monkeypatch`` (which shares one instance
# across a test's whole fixture graph, including the autouse fixture — the same
# pattern ``tests/test_cli_tunnel.py`` uses to re-enable ``_health.is_healthy``
# for its own tests) so they exercise the actual HTTP round trip, not a stub.
_REAL_FETCH_GATEWAY_CAPABILITIES = capabilities_module._fetch_gateway_capabilities


def _scaffold_fleet(path):
    """Write the packaged fleet templates verbatim — the SAME .env `lobes init
    --fleet` would scaffold, so the served-context overlay assertions below
    exercise the real shipped defaults (PRIMARY_MAX_MODEL_LEN=131072,
    MULTIMODAL_MAX_MODEL_LEN=32768, ...), not a hand-rolled fixture."""
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


# ---------------------------------------------------------------------------
# lobes capabilities
# ---------------------------------------------------------------------------


def test_capabilities_json_returns_all_six_roles_with_full_metadata(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    # The JSON payload is the bare six-role dict — no "source"/mode key is
    # ever mixed in, in ANY mode (Qodo action-required finding on PR #102:
    # the gateway's own GET /capabilities returns exactly these six keys, so
    # a strict `set(payload) == ROLES` check must hold here too). No gateway
    # is listening on the resolved port in this offline fixture, so the CLI
    # degrades to the .env-derived fallback and says so on stderr instead.
    assert set(payload) == set(ROLES)
    assert "offline" in err
    for role in ROLES:
        info = payload[role]
        assert _ROLE_INFO_FIELDS <= set(info)
        assert info["role"] == role
        assert info["model"]  # never blank


def test_capabilities_json_reports_served_context_not_catalog_native(tmp_path, capsys) -> None:
    """The #81 contract: context is the SERVED --max-model-len from the
    deployment env, not the catalog native (t5's core behaviour change)."""
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cortex"]["context"] == 131072
    assert payload["senses"]["context"] == 32768
    assert payload["cortex"]["loaded"] is True
    assert payload["senses"]["loaded"] is True
    assert payload["embedder"]["loaded"] is True
    assert payload["reranker"]["loaded"] is True
    # Audio overlay not scaffolded here (no --audio) → present, unloaded.
    assert payload["stt"]["loaded"] is False
    assert payload["tts"]["loaded"] is False
    assert payload["stt"]["context"] == 0
    assert payload["tts"]["context"] == 0


def test_capabilities_json_endpoint_is_gateway_base_url(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # VLLM_PORT=8000 in the packaged fleet env.example.
    assert payload["cortex"]["endpoint"] == "http://localhost:8000"
    assert payload["embedder"]["endpoint"] == "http://localhost:8000"


def test_capabilities_non_json_renders_readable_table_with_all_six_roles(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    for role in ROLES:
        assert role in out
    assert "responsibilities:" in out
    assert "131072" in out  # served cortex context visible in the table


def test_capabilities_json_marks_hardware_infeasible_role_unserved(tmp_path, capsys) -> None:
    """End to end (profile → env → CLI), task t6: a role this machine's
    per-machine profile declared infeasible (``PRIMARY_FEASIBLE=false`` in
    the deployment's ``.env`` — the same channel ``lobes.gateway._config.
    FEASIBLE_ENV`` reads) is never advertised ready by the offline registry
    either — present (not omitted, per the #92 convention), but marked
    unserved."""
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_FEASIBLE", "false")
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(ROLES)  # still present, not omitted
    assert payload["cortex"]["feasible"] is False
    assert payload["cortex"]["ready"] is False
    # Every sibling role is unaffected.
    for role in ("senses", "embedder", "reranker"):
        assert payload[role]["feasible"] is True


def test_capabilities_non_json_table_flags_hardware_infeasible_role(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_FEASIBLE", "false")
    rc = main(["capabilities", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "infeasible on this machine" in out


def test_capabilities_table_distinguishes_proxied_from_referral_only(tmp_path, capsys) -> None:
    # Proxy-lobes (#115/#127): a dropped role with origin + proxy knob renders
    # the PROXIED wording (this gateway forwards), never the referral-only
    # "dial it directly" wording — and vice versa when the knob is absent.
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "MULTIMODAL_FEASIBLE", "false")
    _env.set_env(tmp_path / _compose.ENV_FILE, "MULTIMODAL_PEER_ORIGIN", "http://peer.example:8000")
    rc = main(["capabilities", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hosted by peer: http://peer.example:8000 (dial it directly)" in out
    assert "proxied via this gateway" not in out

    _env.set_env(tmp_path / _compose.ENV_FILE, "MULTIMODAL_PEER_PROXY", "true")
    rc = main(["capabilities", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "proxied via this gateway from peer: http://peer.example:8000" in out
    assert "dial it directly" not in out


def test_capabilities_unscaffolded_still_answers_all_six_roles(capsys) -> None:
    """Read-only: with nothing scaffolded, capabilities degrades gracefully to
    catalog defaults (all unloaded except the always-present cortex) instead of
    erroring — mirrors 'lobes overview --live' on an empty deployment."""
    rc = main(["capabilities", "--json"])
    assert rc == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert set(payload) == set(ROLES)
    assert "offline" in err
    assert payload["cortex"]["loaded"] is True  # primary is always wired
    assert payload["senses"]["loaded"] is False
    assert payload["embedder"]["loaded"] is False
    assert payload["reranker"]["loaded"] is False
    # No overlay env available either → catalog native.
    from lobes.catalog import SUPPORTED_MODELS

    primary_native = next(
        m.native_max_model_len for m in SUPPORTED_MODELS if m.id == payload["cortex"]["model"]
    )
    assert payload["cortex"]["context"] == primary_native


def test_capabilities_offline_fallback_never_reports_ready_true(tmp_path, capsys) -> None:
    """Job 3 / issue #96: a config file is not evidence of health.

    ``AUDIO_URL`` present in the deployment's ``.env`` makes ``stt``/``tts``
    ``loaded`` (a config fact — the overlay is configured), but the gateway is
    unreachable in this test (the autouse ``offline_runtime`` fixture stubs
    the probe, matching a real down-gateway), so nothing was ever probed. The
    offline fallback must therefore report ``ready=false`` for EVERY role —
    not just stt/tts — no matter what ``loaded``/config truth it can compute.
    This is the literal issue #96 scenario: ``AUDIO_URL`` was in ``.env`` but
    never reached the gateway container's own environment, and the CLI's old
    ``.env``-derived registry advertised ``ready=true`` on a path that
    actually 404s/503s.
    """
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "AUDIO_URL", "http://realtime:8080")
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert set(payload) == set(ROLES)
    assert "offline" in err
    # AUDIO_URL is configured, so stt/tts ARE "loaded" (a config fact) ...
    assert payload["stt"]["loaded"] is True
    assert payload["tts"]["loaded"] is True
    # cortex/senses/embedder/reranker are also loaded in the scaffolded fleet.
    assert payload["cortex"]["loaded"] is True
    # ... but NOTHING was probed, so every single role's `ready` is False.
    for role in ROLES:
        assert payload[role]["ready"] is False, role


def test_capabilities_non_json_table_marks_offline_source(capsys) -> None:
    """The human-readable table must also say this is a configured-defaults
    view, not a live one — not just the JSON ``source`` key (Job 1)."""
    rc = main(["capabilities"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "offline" in out
    assert "gateway unreachable" in out


def test_capabilities_never_touches_docker(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)

    def boom(*a, **k):
        raise AssertionError("capabilities must never invoke docker/compose")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    monkeypatch.setattr(_compose, "compose_down", boom)
    monkeypatch.setattr(_compose, "_run", boom)
    monkeypatch.setattr(_compose, "_probe", boom)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0


def test_capabilities_has_no_apply_flag(capsys) -> None:
    """Read-only verb: no --apply, unlike switch/serve/stop/init/fleet/tunnel."""
    with pytest.raises(SystemExit) as exc:
        main(["capabilities", "--apply"])
    assert exc.value.code == 1  # EXIT_USER_ERROR via the structured argparse error


# ---------------------------------------------------------------------------
# stt realtime/VAD session capability (issue #149, task t4)
# ---------------------------------------------------------------------------
#
# Acceptance criterion 1: on an audio-enabled registry, BOTH `lobes
# capabilities` (the CLI's offline .env-derived fallback) and `GET
# /capabilities` (lobes.gateway.server.capabilities_payload — the SAME
# function the gateway's own HTTP route calls, see server.py's
# _get_capabilities) show the realtime capability under `stt`. Acceptance
# criterion 2: a text-only fleet (no audio overlay) shows no realtime claim
# on either surface — the negative control, without which the positive
# assertions below would be vacuous.


def test_capabilities_json_stt_advertises_realtime_when_audio_enabled(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "AUDIO_URL", "http://realtime:8080")
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stt"]["loaded"] is True
    assert STT_REALTIME_RESPONSIBILITY in payload["stt"]["responsibilities"]
    # The capability is stt-only — tts never claims it.
    assert STT_REALTIME_RESPONSIBILITY not in payload["tts"]["responsibilities"]


def test_capabilities_json_stt_no_realtime_claim_on_text_only_fleet(tmp_path, capsys) -> None:
    """Negative control: the packaged fleet scaffold has no `--audio` overlay
    (AUDIO_URL unset) — `stt` must not claim the realtime capability."""
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stt"]["loaded"] is False
    assert payload["stt"]["responsibilities"] == ["transcribe", "audio_input_to_text"]
    assert STT_REALTIME_RESPONSIBILITY not in payload["stt"]["responsibilities"]


def test_capabilities_json_stt_no_realtime_claim_when_lane_declared_infeasible(
    tmp_path, capsys
) -> None:
    """Honesty: STT_FEASIBLE=false must withhold the claim even with the
    overlay wired — a capability this machine declared off is never
    advertised, exactly like the existing feasible/loaded discipline."""
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "AUDIO_URL", "http://realtime:8080")
    _env.set_env(tmp_path / _compose.ENV_FILE, "STT_FEASIBLE", "false")
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stt"]["feasible"] is False
    assert STT_REALTIME_RESPONSIBILITY not in payload["stt"]["responsibilities"]


def test_gateway_get_capabilities_stt_advertises_realtime_when_audio_enabled() -> None:
    """The gateway side of criterion 1: `capabilities_payload` — the exact
    function `GET /capabilities` dispatches to — claims the capability when
    the audio overlay is wired."""
    env = {"PRIMARY_URL": "http://vllm-primary:8000", "AUDIO_URL": "http://realtime:8080"}
    table, cfg = build_config(env)
    payload = capabilities_payload(table, cfg, env=env, gateway_url="http://localhost:8000")
    assert payload["stt"]["loaded"] is True
    assert STT_REALTIME_RESPONSIBILITY in payload["stt"]["responsibilities"]


def test_gateway_get_capabilities_stt_no_realtime_claim_on_text_only_fleet() -> None:
    """The gateway side of criterion 2: a text-only fleet's `GET
    /capabilities` must not claim the realtime capability either."""
    env = {"PRIMARY_URL": "http://vllm-primary:8000"}
    table, cfg = build_config(env)
    payload = capabilities_payload(table, cfg, env=env, gateway_url="http://localhost:8000")
    assert payload["stt"]["loaded"] is False
    assert STT_REALTIME_RESPONSIBILITY not in payload["stt"]["responsibilities"]


# ---------------------------------------------------------------------------
# lobes endpoint <role>
# ---------------------------------------------------------------------------


def test_endpoint_prints_gateway_base_url_for_cortex(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["endpoint", "cortex", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "http://localhost:8000"


def test_endpoint_json_shape(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["endpoint", "embedder", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"role": "embedder", "endpoint": "http://localhost:8000"}


def test_endpoint_works_for_every_role(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    # The four gateway-fronted roles resolve to the reachable gateway URL; the
    # audio roles (stt/tts) are unwired here (no --audio overlay) → blank, but
    # 'lobes endpoint' still exits 0 for every known role, wired or not.
    expected = {
        "cortex": "http://localhost:8000",
        "senses": "http://localhost:8000",
        "muse": "http://localhost:8000",
        "embedder": "http://localhost:8000",
        "reranker": "http://localhost:8000",
        "stt": "",
        "tts": "",
    }
    assert set(expected) == set(ROLES)
    for role in ROLES:
        rc = main(["endpoint", role, "--compose-dir", str(tmp_path)])
        assert rc == 0
        assert capsys.readouterr().out.strip() == expected[role]


def test_endpoint_unknown_role_exits_user_error_with_hint(capsys) -> None:
    rc = main(["endpoint", "bogus"])
    assert rc == 1  # EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err
    for role in ROLES:
        assert role in err


def test_endpoint_unknown_role_json_error_shape(capsys) -> None:
    rc = main(["endpoint", "bogus", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == 1
    assert "bogus" in payload["message"]
    for role in ROLES:
        assert role in payload["remediation"]


def test_endpoint_never_touches_docker(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)

    def boom(*a, **k):
        raise AssertionError("endpoint must never invoke docker/compose")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    monkeypatch.setattr(_compose, "compose_down", boom)
    monkeypatch.setattr(_compose, "_run", boom)
    monkeypatch.setattr(_compose, "_probe", boom)
    rc = main(["endpoint", "cortex", "--compose-dir", str(tmp_path)])
    assert rc == 0


# ---------------------------------------------------------------------------
# Gateway-client mode (issue #96, task t7): a real loopback fake gateway
# ---------------------------------------------------------------------------
#
# Every test above exercises the OFFLINE fallback (the autouse
# ``offline_runtime`` fixture neutralises the live probe). The tests below
# restore the real ``_fetch_gateway_capabilities`` and point it at an actual
# ``ThreadingHTTPServer`` on an ephemeral port, proving the CLI performs a
# genuine HTTP round trip and renders exactly what the gateway said — not a
# re-derivation, not a re-shaped version of it.


class _FakeGatewayHandler(BaseHTTPRequestHandler):
    """Serves a fixed JSON body on GET /capabilities; 404s everything else."""

    payload: dict = {}

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/capabilities":
            body = json.dumps(self.payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_a) -> None:  # silence test noise
        pass


def _known_capabilities_payload() -> dict:
    """A hand-built payload that deliberately does NOT match anything the
    offline `.env`-derived fallback would ever compute (fake models, fake
    ports, fake context sizes) — so an exact match against it proves the CLI
    rendered the gateway's own answer, not its own guess."""
    payload: dict[str, dict] = {}
    for i, role in enumerate(ROLES):
        payload[role] = {
            "role": role,
            "model": f"fake/{role}-model",
            "runtime": "vllm",
            "endpoint": "http://localhost:9999",
            "path": "/v1/fake",
            "context": 1000 + i,
            "quant": "fake-quant",
            "mtp": bool(i % 2),
            "feasible": True,
            "responsibilities": [f"{role}-thing"],
            "forbidden_responsibilities": [],
            "ready": bool(i % 2 == 0),
            "loaded": True,
        }
    return payload


@pytest.fixture
def fake_gateway(monkeypatch):
    """A real gateway stand-in on an ephemeral port, with the real gateway
    probe restored (the autouse fixture stubs it to `None` by default)."""
    payload = _known_capabilities_payload()
    handler = type("_BoundFakeGatewayHandler", (_FakeGatewayHandler,), {"payload": payload})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    # Restores the REAL implementation (captured at module-import time, before
    # tests/conftest.py's autouse fixture stubbed it) for this test only —
    # mirrors how tests/test_cli_tunnel.py re-enables `_health.is_healthy`.
    monkeypatch.setattr(
        capabilities_module, "_fetch_gateway_capabilities", _REAL_FETCH_GATEWAY_CAPABILITIES
    )
    try:
        yield httpd.server_address[1], payload
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_capabilities_json_reproduces_live_gateway_payload_exactly(fake_gateway, capsys) -> None:
    """Job 1's core assertion: against a fake gateway serving a known
    /capabilities payload, `lobes capabilities --json` reproduces it exactly
    byte-for-byte (key-for-key) — no added/removed keys, no reshaping. This
    is also the fix for the Qodo action-required finding on PR #102: a prior
    revision added a top-level ``source`` sibling here, which broke exact
    reproduction of the gateway's own contract."""
    port, known_payload = fake_gateway
    rc = main(["capabilities", "--port", str(port), "--json"])
    assert rc == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload == known_payload
    # Gateway mode is a live, authoritative answer — nothing to caveat, so no
    # offline notice (or anything else) is written to stderr.
    assert err == ""


def test_capabilities_json_gateway_mode_keys_exactly_match_roles(fake_gateway, capsys) -> None:
    """Regression guard for the Qodo action-required finding on PR #102: a
    strict consumer doing ``set(payload.keys()) == ROLES`` — precisely the
    kind of contract check this CLI-as-gateway-client rewrite (issue #96,
    t7) promotes — must pass in gateway mode. The gateway's own
    ``GET /capabilities`` returns exactly ``{cortex, senses, embedder,
    reranker, stt, tts}`` and nothing else; the CLI's ``--json`` rendering of
    a live gateway response must match byte-for-byte, with no extra
    top-level key (e.g. no ``source``) ever mixed in."""
    port, _known_payload = fake_gateway
    rc = main(["capabilities", "--port", str(port), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) == set(ROLES)


def test_capabilities_non_json_table_marks_gateway_source(fake_gateway, capsys) -> None:
    port, known_payload = fake_gateway
    rc = main(["capabilities", "--port", str(port)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gateway" in out
    assert "live GET /capabilities" in out
    # A fake model id from the known payload shows up verbatim in the table.
    assert known_payload["cortex"]["model"] in out


# --- version skew: a NEWER CLI against an OLDER gateway ---------------------


def test_older_gateway_payload_missing_additive_fields_is_still_authoritative(
    monkeypatch, capsys
) -> None:
    """A newer CLI must not demote an older gateway to "unreachable".

    The CLI and the gateway are separately-versioned processes, and on a mesh of
    mixed-version boxes a newer CLI routinely probes an older gateway whose
    payload predates a field. Requiring every CURRENT RoleInfo field would read
    that as "a foreign daemon", silently swap the gateway's authoritative answer
    for offline .env guesses, and print "gateway unreachable" — false, since it
    answered. That is #92's dishonesty inverted, so it is pinned here.
    """
    payload = _known_capabilities_payload()
    for role in payload:  # simulate a pre-`tools`, pre-`feasible` gateway
        payload[role].pop("tools", None)
        payload[role].pop("feasible", None)
    handler = type("_OldGatewayHandler", (_FakeGatewayHandler,), {"payload": payload})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        capabilities_module, "_fetch_gateway_capabilities", _REAL_FETCH_GATEWAY_CAPABILITIES
    )
    try:
        rc = main(["capabilities", "--port", str(httpd.server_address[1])])
        assert rc == 0
        out = capsys.readouterr().out
        assert "live GET /capabilities" in out  # trusted, not demoted to offline
        assert "offline" not in out
        # The old gateway's own answer is what got rendered — not a local guess.
        assert payload["cortex"]["model"] in out
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_payload_missing_a_CORE_field_is_still_rejected(monkeypatch, capsys) -> None:
    """The flip side: tolerating additive fields must not blunt the check's real
    job — telling a real gateway from a stray daemon answering on a guessed port
    (a live hazard on this rig, see lobes.roles._gateway_base_url). A body
    missing a CORE field is still malformed => fall back to offline."""
    payload = _known_capabilities_payload()
    for role in payload:
        payload[role].pop("endpoint")  # a core field no real gateway omits
    handler = type("_ForeignDaemonHandler", (_FakeGatewayHandler,), {"payload": payload})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        capabilities_module, "_fetch_gateway_capabilities", _REAL_FETCH_GATEWAY_CAPABILITIES
    )
    try:
        rc = main(["capabilities", "--port", str(httpd.server_address[1])])
        assert rc == 0
        assert "offline" in capsys.readouterr().out  # not trusted
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_endpoint_gateway_mode_uses_live_payload_not_offline_guess(fake_gateway, capsys) -> None:
    """`lobes endpoint` also asks the gateway first (Job 1: both verbs)."""
    port, known_payload = fake_gateway
    rc = main(["endpoint", "cortex", "--port", str(port), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"role": "cortex", "endpoint": known_payload["cortex"]["endpoint"]}


def test_capabilities_gateway_mode_never_touches_docker(fake_gateway, monkeypatch, capsys) -> None:
    port, _known_payload = fake_gateway

    def boom(*a, **k):
        raise AssertionError("capabilities must never invoke docker/compose")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    monkeypatch.setattr(_compose, "compose_down", boom)
    monkeypatch.setattr(_compose, "_run", boom)
    monkeypatch.setattr(_compose, "_probe", boom)
    rc = main(["capabilities", "--port", str(port), "--json"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Registration — both verbs show up in --help / overview, don't break either
# ---------------------------------------------------------------------------


def test_capabilities_and_endpoint_appear_in_top_level_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "capabilities" in out
    assert "endpoint" in out


def test_overview_still_works_and_lists_the_new_verbs(capsys) -> None:
    rc = main(["overview", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    verbs_section = next(s for s in payload["sections"] if s["title"] == "Verbs")
    joined = " ".join(verbs_section["items"])
    assert "capabilities" in joined
    assert "endpoint" in joined
