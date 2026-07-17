"""First-class stt/tts: peer channels + per-endpoint audio routing (issue #129).

The live trigger: the Spark GB10 wants Chatterbox (tts) served from the Thor
while Parakeet (stt) stays local — not expressible via the one namespace-wide
``AUDIO_URL``, and pointing that at a peer breaks four proxy-lobes guarantees
(all-or-nothing, credential leak, no loop guard/attribution, dishonest
capabilities). These tests pin the fix: stt/tts ride the SAME feasibility /
peer-origin / proxy / key channels as the five core backends, route
per-endpoint, and forward through the same data-plane machinery.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._readiness import probe_audio_peer_ready
from lobes.gateway._routing import audio_role_for_path
from lobes.roles import _STT_MODEL, _TTS_MODEL, annotate_peer_referrals, build_role_registry

_THOR_ORIGIN = "http://thor.tail:8000"
_PEER_KEY = "mg-thor-inbound-key"
_SPEECH = "/v1/audio/speech"
_TRANSCRIBE = "/v1/audio/transcriptions"


def _audio_env(**over) -> dict:
    """A Spark-shaped fleet env: audio overlay wired, tts declared off +
    proxied to the Thor (the live ask), stt served locally."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
        "AUDIO_URL": "http://realtime:8080",
        "TTS_FEASIBLE": "false",
        "TTS_PEER_ORIGIN": _THOR_ORIGIN,
        "TTS_PEER_PROXY": "true",
        "TTS_PEER_API_KEY": _PEER_KEY,
    }
    env.update(over)
    return env


def _build(env):
    table, cfg = build_config(env)
    return table, cfg, S.peer_specs_from_table(table, env)


class _FakeUpstream:
    def __init__(self, status, body=b'{"ok":1}', headers=None):
        self.status = status
        self.headers = headers if headers is not None else [("Content-Type", "application/json")]
        self._body = body
        self.closed = False

    def read_all(self):
        return self._body

    def read(self, _n):
        data, self._body = self._body, b""
        return data

    def close(self):
        self.closed = True


def _opener(outcome=200, body=b'{"ok":1}'):
    calls = []

    def opener(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
        calls.append(
            SimpleNamespace(backend=backend, path=path, body=fwd_body, headers=list(headers))
        )
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeUpstream(outcome, body=body)

    return opener, calls


def _audio_request(table, cfg, specs, path, *, headers=(), body=b'{"model":"tts-1"}'):
    opener, calls = _opener()
    resp = S.handle_audio_request(
        table, cfg, specs, path, list(headers), body, opener, audio_ready_probe=lambda: True
    )
    return resp, calls


# ============================================================================
# routing: per-endpoint role resolution
# ============================================================================


def test_audio_role_for_path_maps_the_two_lanes() -> None:
    assert audio_role_for_path(_SPEECH) == "tts"
    assert audio_role_for_path(_TRANSCRIBE) == "stt"
    assert audio_role_for_path(_SPEECH + "?format=wav") == "tts"
    assert audio_role_for_path("/v1/audio/translations") is None  # legacy namespace route
    assert audio_role_for_path("/v1/chat/completions") is None


# ============================================================================
# config: the seven-name channels arm identically for audio roles
# ============================================================================


def test_tts_declared_off_and_proxied_arms_exactly_like_a_core_role() -> None:
    table, _cfg, specs = _build(_audio_env())
    assert "tts" in table.infeasible
    assert "stt" not in table.infeasible
    assert table.peer_origins["tts"] == _THOR_ORIGIN
    assert table.peer_proxied == frozenset({"tts"})
    assert table.peer_api_keys["tts"] == _PEER_KEY
    assert specs["tts"].served_name == _TTS_MODEL  # fixed sidecar id, never blank


def test_origin_without_knob_stays_referral_only_for_audio() -> None:
    env = _audio_env()
    del env["TTS_PEER_PROXY"]
    table, _cfg, specs = _build(env)
    assert "tts" in table.infeasible
    assert table.peer_proxied == frozenset()
    assert specs == {}


def test_knob_on_a_feasible_audio_lane_is_inert() -> None:
    # tts NOT declared off → the local overlay serves it; the knob is ignored.
    env = _audio_env(TTS_FEASIBLE="")
    table, _cfg, _specs = _build(env)
    assert "tts" not in table.infeasible
    assert table.peer_proxied == frozenset()


def test_no_audio_env_is_byte_identical_default() -> None:
    # Every pre-#129 deployment: no STT_/TTS_ knob anywhere → both lanes stay
    # feasible (the sleeping-lobe contract), no peer channel, no spec.
    table, _cfg, specs = _build({"PRIMARY_SERVED_NAME": "m"})
    assert "stt" not in table.infeasible and "tts" not in table.infeasible
    assert specs == {}


# ============================================================================
# data plane: the four AUDIO_URL violations are impossible on the new lane
# ============================================================================


def test_speech_forwards_to_peer_while_transcriptions_stay_local() -> None:
    # The all-or-nothing violation, fixed: tts-remote + stt-local in ONE deployment.
    table, cfg, specs = _build(_audio_env())
    resp, calls = _audio_request(table, cfg, specs, _SPEECH)
    assert calls[0].backend.base_url == _THOR_ORIGIN
    assert dict(resp.headers).get(S.PROXIED_BY_HEADER) == _THOR_ORIGIN

    resp, calls = _audio_request(table, cfg, specs, _TRANSCRIBE, body=b"--multipart--")
    assert calls[0].backend.base_url == cfg.audio_url  # the LOCAL bridge
    assert S.PROXIED_BY_HEADER not in dict(resp.headers)


def test_forwarded_body_is_verbatim_no_model_rewrite() -> None:
    table, cfg, specs = _build(_audio_env())
    body = b'{"model":"tts-1","input":"hello","voice":"alloy"}'
    _resp, calls = _audio_request(table, cfg, specs, _SPEECH, body=body)
    assert calls[0].body == body  # path-routed lane: never rewritten


def test_callers_credential_never_reaches_the_audio_peer() -> None:
    # The credential-leak violation, fixed: strip inbound, inject pairwise.
    table, cfg, specs = _build(_audio_env())
    _resp, calls = _audio_request(
        table, cfg, specs, _SPEECH, headers=[("Authorization", "Bearer caller-secret")]
    )
    sent = calls[0].headers
    auth = [v for k, v in sent if k.lower() == "authorization"]
    assert auth == [f"Bearer {_PEER_KEY}"]
    assert all("caller-secret" not in v for _k, v in sent)


def test_marked_audio_arrival_refused_508_zero_outbound() -> None:
    # The no-loop-guard violation, fixed: single hop, refused with no dial.
    table, cfg, specs = _build(_audio_env())
    resp, calls = _audio_request(table, cfg, specs, _SPEECH, headers=[(S.PROXIED_HEADER, "tts")])
    assert resp.status == 508
    assert json.loads(resp.body)["error"]["code"] == "proxy_loop"
    assert calls == []


def test_outbound_audio_forward_carries_single_hop_marker() -> None:
    table, cfg, specs = _build(_audio_env())
    _resp, calls = _audio_request(table, cfg, specs, _SPEECH)
    assert (S.PROXIED_HEADER, "tts") in calls[0].headers


def test_declared_off_unproxied_lane_404s_role_infeasible_with_referral() -> None:
    env = _audio_env()
    del env["TTS_PEER_PROXY"]
    table, cfg, specs = _build(env)
    resp, calls = _audio_request(table, cfg, specs, _SPEECH)
    assert resp.status == 404
    error = json.loads(resp.body)["error"]
    assert error["code"] == "role_infeasible"
    assert error["hosted_by"] == _THOR_ORIGIN
    assert calls == []  # referral-only: this gateway never dials the peer


def test_peer_down_yields_retryable_503_with_attribution() -> None:
    table, cfg, specs = _build(_audio_env())
    opener, _calls = _opener(outcome=S.UpstreamError("connection refused"))
    resp = S.handle_audio_request(
        table, cfg, specs, _SPEECH, [], b"{}", opener, audio_ready_probe=lambda: True
    )
    assert resp.status == 503
    assert dict(resp.headers).get(S.PROXIED_BY_HEADER) == _THOR_ORIGIN


def test_other_audio_paths_keep_the_legacy_namespace_route() -> None:
    # An /v1/audio/* path outside the two role lanes stays on AUDIO_URL even
    # when tts is proxied — byte-identical to pre-#129 behaviour.
    table, cfg, specs = _build(_audio_env())
    resp, calls = _audio_request(table, cfg, specs, "/v1/audio/translations")
    assert calls[0].backend.base_url == cfg.audio_url


# ============================================================================
# capabilities honesty: hosted_by + proxied + peer-probed ready
# ============================================================================


def test_capabilities_annotate_proxied_tts_and_local_stt() -> None:
    table, cfg, _specs = _build(_audio_env())
    registry = build_role_registry(table, cfg, peer_ready={"tts": True})
    payload = {
        role: {"feasible": info.feasible, "ready": info.ready} for role, info in registry.items()
    }
    annotate_peer_referrals(payload, table)

    assert payload["tts"]["feasible"] is False
    assert payload["tts"]["hosted_by"] == _THOR_ORIGIN
    assert payload["tts"]["proxied"] is True
    assert payload["tts"]["ready"] is True  # the live PEER probe verdict

    assert payload["stt"]["feasible"] is True
    assert "hosted_by" not in payload["stt"]
    assert registry["stt"].model == _STT_MODEL


def test_proxied_tts_without_peer_signal_is_honestly_not_ready() -> None:
    table, cfg, _specs = _build(_audio_env())
    registry = build_role_registry(table, cfg)  # no peer_ready supplied
    assert registry["tts"].ready is False
    assert registry["tts"].loaded is False
    registry = build_role_registry(table, cfg, peer_ready={"tts": False})
    assert registry["tts"].ready is False


def test_default_registry_is_byte_identical_without_audio_knobs() -> None:
    # Every pre-#129 deployment: the audio entries keep their sleeping-lobe
    # shape (feasible:true; loaded/ready track the overlay) exactly as before.
    table, cfg = build_config({"PRIMARY_SERVED_NAME": "m", "AUDIO_URL": "http://rt:8080"})
    registry = build_role_registry(table, cfg, gateway_url="http://gw:8000")
    for role in ("stt", "tts"):
        assert registry[role].feasible is True
        assert registry[role].loaded is True
    table, cfg = build_config({"PRIMARY_SERVED_NAME": "m"})
    registry = build_role_registry(table, cfg)
    for role in ("stt", "tts"):
        assert registry[role].feasible is True
        assert registry[role].loaded is False
        assert registry[role].ready is False


# ============================================================================
# the audio peer probe (capabilities-based, not /v1/models)
# ============================================================================


def _caps_opener(status=200, payload=None):
    def opener(url, timeout, api_key):
        assert url.endswith("/capabilities")
        body = json.dumps(payload if payload is not None else {}).encode()
        return status, body

    return opener


def test_audio_peer_probe_true_iff_peer_reports_role_ready() -> None:
    ready = {"roles": {"tts": {"ready": True, "feasible": True}}}
    assert probe_audio_peer_ready(_THOR_ORIGIN, "tts", opener=_caps_opener(200, ready)) is True
    not_ready = {"roles": {"tts": {"ready": False}}}
    assert probe_audio_peer_ready(_THOR_ORIGIN, "tts", opener=_caps_opener(200, not_ready)) is False
    assert probe_audio_peer_ready(_THOR_ORIGIN, "tts", opener=_caps_opener(503, {})) is False
    assert probe_audio_peer_ready(_THOR_ORIGIN, "tts", opener=_caps_opener(200, {})) is False


def test_audio_peer_probe_accepts_flat_role_payloads_and_never_raises() -> None:
    flat = {"tts": {"ready": True}}
    assert probe_audio_peer_ready(_THOR_ORIGIN, "tts", opener=_caps_opener(200, flat)) is True

    def raising(url, timeout, api_key):
        raise OSError("refused")

    assert probe_audio_peer_ready(_THOR_ORIGIN, "tts", opener=raising) is False


# ============================================================================
# advertisement honesty
# ============================================================================


def test_endpoints_reflect_per_lane_serving() -> None:
    # tts proxied + stt local → both lanes advertised.
    table, cfg, _specs = _build(_audio_env())
    eps = S._endpoints_for(table, bool(cfg.audio_url))
    assert "POST /v1/audio/speech" in eps
    assert "POST /v1/audio/transcriptions" in eps
    # tts declared off, NOT proxied → speech not advertised (it 404s).
    env = _audio_env()
    del env["TTS_PEER_PROXY"]
    table, cfg, _specs = _build(env)
    eps = S._endpoints_for(table, bool(cfg.audio_url))
    assert "POST /v1/audio/speech" not in eps
    assert "POST /v1/audio/transcriptions" in eps


def test_v1_models_never_lists_audio_sidecar_ids() -> None:
    # The audio lanes are path-routed: their fixed sidecar ids must not appear
    # as requestable `model` values even while proxied-and-ready.
    from lobes.gateway._routing import list_models_payload

    table, _cfg, specs = _build(_audio_env())
    peer_served = {
        name: spec.served_name for name, spec in specs.items() if name not in ("stt", "tts")
    }
    payload = list_models_payload(table, {"tts": True}, peer_served)
    ids = {m["id"] for m in payload["data"]}
    assert _TTS_MODEL not in ids and _STT_MODEL not in ids
