"""Inbound gateway auth tests (proxy-lobes t2, issues #115/#127).

The gate is OPT-IN and lives at the handler's inbound edge only:

* ``ServerConfig.api_key`` **unset** (no ``GATEWAY_API_KEY`` /
  ``CULTURE_VLLM_API_KEY``, the default) ⇒ every route behaves byte-identically
  to the pre-auth gateway — the ``Authorization`` header is never even read,
  let alone compared (proved below by routing a garbage-header request
  normally AND by spying on :func:`hmac.compare_digest`).
* ``api_key`` **set** ⇒ every DATA-PLANE route (all POSTs + the ``GET /v1/*``
  model listings) requires ``Authorization: Bearer <key>``. Missing /
  malformed / wrong-key each 401 with an OpenAI-shaped ``invalid_api_key``
  body + ``WWW-Authenticate: Bearer``, and the check runs BEFORE any body
  parse, model resolution, readiness probe, or upstream connection — the
  counting fakes below assert ZERO upstream activity on every rejected
  request. ``/health`` (the container-probe endpoint) and ``/capabilities``
  (the control-plane discovery/honesty surface peers read before they hold
  any key) stay KEYLESS by design.
* the comparison is :func:`hmac.compare_digest` over utf-8 bytes
  (timing-safe), and the 401 body/headers never echo any key material —
  neither the expected key nor whatever the caller sent.

Loopback fixtures mirror ``tests/test_gateway_server.py``'s ``gateway``
fixture: a real ``ThreadingHTTPServer`` on an ephemeral port with
``open_upstream`` (and ``probe_audio_ready``) monkeypatched to counting
fakes, so no real backend is needed and every upstream dial is observable.
"""

from __future__ import annotations

import hmac
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config

_KEY = "sk-lobes-inbound-0001"
_WRONG_KEY = "sk-caller-sent-wrong-key"


def _cfg(**over):
    env = {
        "PRIMARY_SERVED_NAME": "P",
        "FALLBACK_URL": "http://vllm-fallback:8000",
        "FALLBACK_SERVED_NAME": "F",
        "GATEWAY_DEFAULT_MODEL": "P",
    }
    env.update(over)
    return build_config(env)


class _FakeUpstream:
    """Duck-typed stand-in for server._Upstream (no socket)."""

    def __init__(self, status, body=b'{"ok":1}'):
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body

    def read_all(self):
        return self._body

    def read(self, _n):
        data, self._body = self._body, b""
        return data

    def close(self):
        pass


def _spawn_gateway(monkeypatch, env_over):
    """A loopback gateway whose upstream dials + audio readiness probes COUNT.

    ``opened`` records every ``open_upstream`` call (backend name) and
    ``probed`` every ``probe_audio_ready`` call — a rejected request must add
    to NEITHER (the zero-upstream-sockets acceptance criterion).
    """
    table, cfg = _cfg(**env_over)
    opened: list[str] = []
    probed: list[str] = []

    def fake_open(backend, path, body, headers, *, connect_timeout, read_timeout):
        opened.append(backend.name)
        return _FakeUpstream(200, body=b'{"echo": "' + backend.name.encode() + b'"}')

    def fake_probe(url, **_kw):
        probed.append(url)
        return True

    monkeypatch.setattr(S, "open_upstream", fake_open)
    monkeypatch.setattr(S, "probe_audio_ready", fake_probe)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    return SimpleNamespace(base=f"http://{host}:{port}", opened=opened, probed=probed, httpd=httpd)


@pytest.fixture
def auth_gateway(monkeypatch):
    """api_key SET (via GATEWAY_API_KEY) + an audio backend wired, so the
    gate's precedence over the audio readiness probe is observable."""
    gw = _spawn_gateway(monkeypatch, {"GATEWAY_API_KEY": _KEY, "AUDIO_URL": "http://realtime:8080"})
    try:
        yield gw
    finally:
        gw.httpd.shutdown()
        gw.httpd.server_close()


@pytest.fixture
def open_gateway(monkeypatch):
    """api_key UNSET (neither env var) — auth disabled, today's exact gateway."""
    gw = _spawn_gateway(monkeypatch, {})
    try:
        yield gw
    finally:
        gw.httpd.shutdown()
        gw.httpd.server_close()


def _request(base, path, *, method="GET", body=None, headers=None):
    req = urllib.request.Request(base + path, data=body, method=method, headers=headers or {})
    return urllib.request.urlopen(req, timeout=5)


def _expect_401(base, path, *, method="GET", body=None, headers=None):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _request(base, path, method=method, body=body, headers=headers)
    assert exc.value.code == 401
    return exc.value


def _bearer(key=_KEY):
    return {"Authorization": f"Bearer {key}"}


# --- the pure parser/comparator (no sockets) ---------------------------------


def test_bearer_token_matches_exact_key() -> None:
    assert S.bearer_token_matches(_KEY, f"Bearer {_KEY}") is True


def test_bearer_scheme_is_case_insensitive() -> None:
    # RFC 7235 §2.1: auth-scheme comparison is case-insensitive.
    assert S.bearer_token_matches(_KEY, f"bearer {_KEY}") is True
    assert S.bearer_token_matches(_KEY, f"BEARER {_KEY}") is True


@pytest.mark.parametrize(
    "authorization",
    [
        None,  # header absent
        "",  # header blank
        _KEY,  # bare token, no scheme
        f"Basic {_KEY}",  # foreign scheme
        "Bearer",  # scheme with no token at all
        "Bearer   ",  # scheme with an empty token
        "Bearer wrong-key",  # well-formed, wrong key
        f"Bearer {_KEY} trailing",  # trailing junk is not the key
    ],
)
def test_bearer_token_matches_fails_closed(authorization) -> None:
    assert S.bearer_token_matches(_KEY, authorization) is False


def test_comparison_is_hmac_compare_digest_over_utf8_bytes(monkeypatch) -> None:
    # Acceptance criterion (c): the comparison is hmac.compare_digest, fed
    # utf-8 BYTES on both sides (compare_digest is only timing-safe for
    # ascii-compatible str; bytes make it unconditionally safe).
    calls: list[tuple[object, object]] = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(S.hmac, "compare_digest", spy)
    assert S.bearer_token_matches(_KEY, f"Bearer {_KEY}") is True
    assert S.bearer_token_matches(_KEY, "Bearer wrong-key") is False
    assert len(calls) == 2
    assert all(isinstance(side, bytes) for pair in calls for side in pair)


def test_malformed_header_never_reaches_compare_digest(monkeypatch) -> None:
    # A missing/foreign-scheme/empty-token credential fails CLOSED before the
    # comparison — compare_digest only ever sees a well-formed Bearer token.
    calls: list[object] = []
    monkeypatch.setattr(S.hmac, "compare_digest", lambda a, b: calls.append(a) or True)
    assert S.bearer_token_matches(_KEY, None) is False
    assert S.bearer_token_matches(_KEY, f"Basic {_KEY}") is False
    assert S.bearer_token_matches(_KEY, "Bearer   ") is False
    assert calls == []


# --- api_key set: the POST data plane is gated -------------------------------


def test_post_no_header_401_openai_shape_zero_upstreams(auth_gateway) -> None:
    # Acceptance criterion (a), "no header": 401 with an OpenAI-shaped
    # invalid_api_key error, a WWW-Authenticate: Bearer challenge, and ZERO
    # upstream connections — the gate runs before any body parse, model
    # resolution, or backend dial.
    err = _expect_401(
        auth_gateway.base, "/v1/chat/completions", method="POST", body=b'{"model":"P"}'
    )
    payload = json.loads(err.read())
    assert payload["error"]["type"] == "invalid_api_key"
    assert payload["error"]["code"] == "invalid_api_key"
    assert isinstance(payload["error"]["message"], str) and payload["error"]["message"]
    assert err.headers.get("WWW-Authenticate") == "Bearer"
    # The gate runs BEFORE the request body is read, so the 401 closes the
    # connection — an unread body must not poison keep-alive framing.
    assert err.headers.get("Connection") == "close"
    assert auth_gateway.opened == []
    assert auth_gateway.probed == []


@pytest.mark.parametrize(
    "authorization",
    [
        f"Basic {_KEY}",  # malformed: foreign scheme
        "Bearer",  # malformed: no token
        "Bearer   ",  # malformed: empty token
        _KEY,  # malformed: bare token, no scheme
        f"Bearer {_WRONG_KEY}",  # well-formed, wrong key
    ],
)
def test_post_malformed_or_wrong_key_401_zero_upstreams(auth_gateway, authorization) -> None:
    # Acceptance criterion (a), "malformed / wrong key": same 401 shape, same
    # zero-upstream guarantee, for every rejected credential form.
    err = _expect_401(
        auth_gateway.base,
        "/v1/chat/completions",
        method="POST",
        body=b'{"model":"P"}',
        headers={"Authorization": authorization},
    )
    assert json.loads(err.read())["error"]["code"] == "invalid_api_key"
    assert auth_gateway.opened == []
    assert auth_gateway.probed == []


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/score",
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
    ],
)
def test_every_post_data_plane_route_is_gated(auth_gateway, path) -> None:
    # Every POST route is data plane — including /v1/audio/* (whose keyless
    # readiness probe would itself be an upstream socket; `probed` stays empty).
    _expect_401(auth_gateway.base, path, method="POST", body=b"{}")
    assert auth_gateway.opened == []
    assert auth_gateway.probed == []


def test_post_right_key_routes_normally(auth_gateway) -> None:
    # The happy path: a correct Bearer key passes through to the NORMAL
    # routing path — the same primary-owner dial an ungated gateway performs.
    with _request(
        auth_gateway.base,
        "/v1/chat/completions",
        method="POST",
        body=b'{"model":"P"}',
        headers=_bearer(),
    ) as r:
        assert r.status == 200
        assert json.load(r)["echo"] == "primary"
    assert auth_gateway.opened == ["primary"]


# --- api_key set: the GET /v1/* model listings are gated ----------------------


def test_get_v1_models_401_without_key_serves_with_key(auth_gateway) -> None:
    # Model listings are part of the OpenAI surface callers script against —
    # gated like the POST data plane, not left keyless like /health.
    err = _expect_401(auth_gateway.base, "/v1/models")
    assert json.loads(err.read())["error"]["code"] == "invalid_api_key"
    with _request(auth_gateway.base, "/v1/models", headers=_bearer()) as r:
        assert r.status == 200
        assert [m["id"] for m in json.load(r)["data"]] == ["P", "F"]


def test_get_supported_models_401_without_key_serves_with_key(auth_gateway) -> None:
    err = _expect_401(auth_gateway.base, "/v1/models/supported")
    assert json.loads(err.read())["error"]["code"] == "invalid_api_key"
    with _request(auth_gateway.base, "/v1/models/supported", headers=_bearer()) as r:
        assert r.status == 200
        assert json.load(r)["object"] == "lobes.supported_models"


def test_unknown_v1_get_route_is_gated_401_before_404(auth_gateway) -> None:
    # The WHOLE /v1/* GET namespace is data plane: an unauthenticated caller
    # gets 401 before learning whether a /v1 route even exists; with the key
    # the pre-auth 404 verdict is unchanged.
    _expect_401(auth_gateway.base, "/v1/does-not-exist")
    headers = _bearer()  # hoisted: exactly one call inside the raises block (S5778)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _request(auth_gateway.base, "/v1/does-not-exist", headers=headers)
    assert exc.value.code == 404


# --- api_key set: /health and /capabilities stay keyless ----------------------


def test_health_answers_keyless_while_api_key_set(auth_gateway) -> None:
    # Acceptance criterion (b): /health is the container-probe endpoint —
    # compose healthchecks and peer boxes reach it before any key exists.
    with urllib.request.urlopen(auth_gateway.base + "/health", timeout=5) as r:
        assert r.status == 200
        assert json.load(r)["status"] == "ok"


def test_capabilities_answers_keyless_while_api_key_set(auth_gateway) -> None:
    # /capabilities is the control-plane discovery/honesty surface (#81, #112):
    # peers and referral-followers read it BEFORE they hold any key.
    with urllib.request.urlopen(auth_gateway.base + "/capabilities", timeout=5) as r:
        assert r.status == 200
        payload = json.load(r)
    assert "cortex" in payload and "senses" in payload


# --- api_key unset: byte-identical to the pre-auth gateway --------------------


def test_api_key_unset_garbage_auth_header_still_routes_normally(open_gateway) -> None:
    # Requirement 1: with no key configured the gateway NEVER inspects
    # Authorization — a garbage credential routes exactly like today.
    with _request(
        open_gateway.base,
        "/v1/chat/completions",
        method="POST",
        body=b'{"model":"P"}',
        headers={"Authorization": "Bearer totally-garbage"},
    ) as r:
        assert r.status == 200
        assert json.load(r)["echo"] == "primary"
    assert open_gateway.opened == ["primary"]


def test_api_key_unset_get_models_ignores_auth_header(open_gateway) -> None:
    with _request(
        open_gateway.base, "/v1/models", headers={"Authorization": f"Basic {_WRONG_KEY}"}
    ) as r:
        assert r.status == 200
        assert [m["id"] for m in json.load(r)["data"]] == ["P", "F"]


def test_api_key_unset_never_consults_compare_digest(open_gateway, monkeypatch) -> None:
    # Provable no-inspection: the disabled gate short-circuits before ANY
    # header read, so the timing-safe comparator is never reached either.
    calls: list[object] = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(S.hmac, "compare_digest", spy)
    with _request(
        open_gateway.base,
        "/v1/chat/completions",
        method="POST",
        body=b'{"model":"P"}',
        headers={"Authorization": "Bearer totally-garbage"},
    ) as r:
        assert r.status == 200
    assert calls == []


# --- the 401 never leaks key material -----------------------------------------


def test_401_never_echoes_key_material(auth_gateway) -> None:
    # Acceptance criterion (c): neither the CONFIGURED key nor what the CALLER
    # sent appears anywhere in the 401 status line, headers, or body — a 401
    # must not become a key-material oracle.
    err = _expect_401(
        auth_gateway.base,
        "/v1/chat/completions",
        method="POST",
        body=b'{"model":"P"}',
        headers={"Authorization": f"Bearer {_WRONG_KEY}"},
    )
    raw_body = err.read().decode("utf-8")
    raw_headers = str(err.headers)
    for secret in (_KEY, _WRONG_KEY):
        assert secret not in raw_body
        assert secret not in raw_headers
