"""Gateway audio routing: handle_audio_post (no sockets) + a loopback relay.

/v1/audio/* is path-routed to a single audio backend with NO model parse/rewrite
and NO failover (one backend) — the inverse of handle_post. The response (a whole
audio file or small JSON) is relayed buffered.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from model_gear.gateway import server as S
from model_gear.gateway._config import build_config
from model_gear.gateway._routing import is_audio_path


def _cfg(**over):
    env = {"PRIMARY_SERVED_NAME": "P", "FALLBACK_SERVED_NAME": "F"}
    env.update(over)
    return build_config(env)


class _FakeUpstream:
    def __init__(self, status=200, body=b"AUDIO", headers=None):
        self.status = status
        self.headers = headers or [("Content-Type", "audio/wav")]
        self._body = body
        self.closed = False

    def read_all(self):
        return self._body

    def read(self, _n):
        data, self._body = self._body, b""
        return data

    def close(self):
        self.closed = True


# --- is_audio_path (pure) -------------------------------------------------


def test_is_audio_path() -> None:
    assert is_audio_path("/v1/audio/speech")
    assert is_audio_path("/v1/audio/transcriptions?language=en")
    assert not is_audio_path("/v1/chat/completions")
    assert not is_audio_path("/v1/models")
    assert not is_audio_path("/health")


# --- handle_audio_post (no sockets) ---------------------------------------


def test_404_when_no_audio_backend_configured() -> None:
    _, cfg = _cfg()  # AUDIO_URL unset → text-only fleet
    resp = S.handle_audio_post(cfg, "/v1/audio/speech", [], b"{}", None)
    assert resp.status == 404 and resp.upstream is None
    assert "not configured" in json.loads(resp.body)["error"]["message"]


def test_forwards_body_verbatim_without_model_rewrite() -> None:
    _, cfg = _cfg(AUDIO_URL="http://realtime:8080")
    calls = []

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append((backend.name, backend.base_url, path, body))
        return _FakeUpstream(200)

    multipart = b'--b\r\nContent-Disposition: form-data; name="model"\r\n\r\nignore\r\n--b--'
    resp = S.handle_audio_post(
        cfg,
        "/v1/audio/transcriptions",
        [("Content-Type", "multipart/form-data; boundary=b")],
        multipart,
        opener,
    )
    name, url, path, fwd_body = calls[0]
    assert name == "audio" and url == "http://realtime:8080"
    assert path == "/v1/audio/transcriptions"
    assert fwd_body == multipart  # verbatim — never JSON-parsed or model-rewritten
    # Streamed (chunked), not buffered: a large audio body must not be read whole
    # into the gateway's memory.
    assert resp.status == 200 and resp.streaming is True and resp.upstream is not None


def test_no_failover_relays_single_backend_status() -> None:
    # A 400 (or any status) from the one audio backend is relayed as-is.
    _, cfg = _cfg(AUDIO_URL="http://realtime:8080")

    def opener(*a, **k):
        return _FakeUpstream(400, b'{"error":1}', [("Content-Type", "application/json")])

    resp = S.handle_audio_post(cfg, "/v1/audio/speech", [], b"{}", opener)
    assert resp.status == 400 and resp.upstream is not None


def test_502_when_audio_backend_unreachable() -> None:
    _, cfg = _cfg(AUDIO_URL="http://realtime:8080")

    def opener(*a, **k):
        raise S.UpstreamError("refused")

    resp = S.handle_audio_post(cfg, "/v1/audio/speech", [], b"{}", opener)
    assert resp.status == 502 and resp.upstream is None
    assert json.loads(resp.body)["error"]["attempts"] == ["refused"]


# --- loopback: the real handler routes /v1/audio/* and relays binary -------


@pytest.fixture
def audio_gateway(monkeypatch):
    table, cfg = _cfg(AUDIO_URL="http://audio-backend:8080")
    seen = {}

    def fake_open(backend, path, body, headers, *, connect_timeout, read_timeout):
        seen["backend"] = backend.name
        seen["path"] = path
        seen["body"] = body
        return _FakeUpstream(200, body=b"RIFFwav-bytes", headers=[("Content-Type", "audio/wav")])

    monkeypatch.setattr(S, "open_upstream", fake_open)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}", seen
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_integration_audio_speech_routes_to_audio_backend(audio_gateway) -> None:
    base, seen = audio_gateway
    req = urllib.request.Request(
        base + "/v1/audio/speech",
        data=b'{"input":"hi","response_format":"wav"}',
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
        assert r.headers.get("Content-Type") == "audio/wav"
        # Streamed relay → chunked transfer, no Content-Length (urllib still
        # transparently de-chunks the body for us).
        assert r.headers.get("Transfer-Encoding") == "chunked"
        assert r.headers.get("Content-Length") is None
        assert r.read() == b"RIFFwav-bytes"
    assert seen["backend"] == "audio"  # path-routed to the audio backend, not a vLLM
    assert seen["path"] == "/v1/audio/speech"
