"""Tests for peer probing in the readiness cache (task t4, issues #115/#127).

Proxy-lobes t1 (already merged) added ``RoutingTable.peer_origins`` /
``peer_proxied`` / ``peer_api_keys`` as CONFIG ONLY — nothing dials a peer
yet. This task teaches :class:`~lobes.gateway._readiness.ReadinessCache` to
actually probe a proxied role's declared peer, so a later task can fold the
result into ``GET /v1/models`` and ``GET /capabilities`` (the #92
"advertised implies reachable" rule, extended across a box boundary): a
proxied role may only be advertised ready when the PEER genuinely serves the
model this box would forward to it — not merely when the peer's process
answers HTTP at all.

Two properties get the most scrutiny, because a regression here would be a
silent honesty violation or a silent availability regression:

* :func:`probe_peer_ready` returns ``True`` **only** on ``200`` + the
  expected served id actually present in ``data[]`` — every other observed
  outcome (non-200, malformed body, id missing, connect refused, timeout)
  collapses to ``False``. ``None`` is reserved for "never probed yet", not
  for a probe *outcome* — mirrored by :class:`ReadinessCache` itself.
* a hanging/flapping peer must NEVER delay or fail local-backend probing —
  proved by running local and peer probing on independent background
  threads with independently injectable, blocking probes.

Stdlib only, mirroring the gateway's dependency-free discipline (the
Authorization-header tests below spin up a real ``http.server`` instead of
a mocking library).
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from lobes.gateway import _readiness as R

# --- probe_peer_ready: the peer honesty check -------------------------------


def test_probe_peer_ready_200_with_id_is_true() -> None:
    def opener(_url, _timeout, _api_key):
        return 200, json.dumps({"data": [{"id": "gemma-4-12b"}]}).encode()

    assert R.probe_peer_ready("http://thor:8001", "gemma-4-12b", opener=opener) is True


def test_probe_peer_ready_200_without_id_is_false() -> None:
    def opener(_url, _timeout, _api_key):
        return 200, json.dumps({"data": [{"id": "some-other-model"}]}).encode()

    assert R.probe_peer_ready("http://thor:8001", "gemma-4-12b", opener=opener) is False


def test_probe_peer_ready_200_empty_data_is_false() -> None:
    def opener(_url, _timeout, _api_key):
        return 200, json.dumps({"data": []}).encode()

    assert R.probe_peer_ready("http://thor:8001", "gemma-4-12b", opener=opener) is False


def test_probe_peer_ready_non_200_is_false() -> None:
    def opener(_url, _timeout, _api_key):
        return 503, b"{}"

    assert R.probe_peer_ready("http://thor:8001", "gemma-4-12b", opener=opener) is False


def test_probe_peer_ready_connect_refused_is_false() -> None:
    def boom(_url, _timeout, _api_key):
        raise OSError("connection refused")

    assert R.probe_peer_ready("http://thor:8001", "gemma-4-12b", opener=boom) is False


def test_probe_peer_ready_timeout_is_false() -> None:
    def boom(_url, _timeout, _api_key):
        raise TimeoutError("timed out")

    assert R.probe_peer_ready("http://thor:8001", "gemma-4-12b", opener=boom) is False


def test_probe_peer_ready_malformed_json_is_false() -> None:
    def opener(_url, _timeout, _api_key):
        return 200, b"not json"

    assert R.probe_peer_ready("http://thor:8001", "gemma-4-12b", opener=opener) is False


def test_probe_peer_ready_non_dict_payload_is_false() -> None:
    def opener(_url, _timeout, _api_key):
        return 200, json.dumps(["unexpected", "list"]).encode()

    assert R.probe_peer_ready("http://thor:8001", "gemma-4-12b", opener=opener) is False


def test_probe_peer_ready_malformed_url_no_crash_no_socket() -> None:
    # DEFAULT opener path: a non-numeric port raises ValueError before any
    # socket opens, so this degrades to False offline instead of crashing.
    assert R.probe_peer_ready("http://peer:notaport/", "gemma-4-12b") is False


def test_probe_peer_ready_hits_v1_models_path() -> None:
    seen = {}

    def opener(url, _timeout, _api_key):
        seen["url"] = url
        return 200, json.dumps({"data": []}).encode()

    R.probe_peer_ready("http://thor:8001/", "gemma-4-12b", opener=opener)
    assert seen["url"] == "http://thor:8001/v1/models"


def test_probe_peer_ready_bound_by_its_own_timeout_argument() -> None:
    seen = {}

    def opener(_url, timeout, _api_key):
        seen["timeout"] = timeout
        return 200, json.dumps({"data": [{"id": "gemma"}]}).encode()

    R.probe_peer_ready("http://thor:8001", "gemma", timeout=1.23, opener=opener)
    assert seen["timeout"] == 1.23


# --- probe_peer_ready: Authorization header, over a real HTTP server --------


class _CapturingHandler(BaseHTTPRequestHandler):
    """Records every request's headers; answers 200 with a fixed body."""

    captured_headers: list[dict] = []
    response_body: bytes = b'{"data": []}'

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name
        _CapturingHandler.captured_headers.append(dict(self.headers))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(_CapturingHandler.response_body)

    def log_message(self, format, *args) -> None:  # noqa: A002 - silence test noise
        pass


def _start_capturing_server() -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_authorization_header_present_iff_key_declared() -> None:
    _CapturingHandler.captured_headers = []
    _CapturingHandler.response_body = json.dumps({"data": [{"id": "gemma"}]}).encode()
    server, thread = _start_capturing_server()
    try:
        origin = f"http://127.0.0.1:{server.server_port}"

        assert R.probe_peer_ready(origin, "gemma", api_key="sk-peer-secret") is True
        assert R.probe_peer_ready(origin, "gemma") is True

        assert len(_CapturingHandler.captured_headers) == 2
        with_key, without_key = _CapturingHandler.captured_headers
        assert with_key.get("Authorization") == "Bearer sk-peer-secret"
        assert "Authorization" not in without_key
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


# --- PeerSpec: no key material ever surfaces in repr/str --------------------


def test_peer_spec_repr_never_contains_key_material() -> None:
    spec = R.PeerSpec(
        name="multimodal",
        origin="http://thor:8001",
        served_name="gemma-4-12b",
        api_key="sk-peer-super-secret",  # nosec B105 - test fixture, not a credential
    )
    assert "sk-peer-super-secret" not in repr(spec)
    assert "sk-peer-super-secret" not in str(spec)


def test_peer_spec_defaults_api_key_to_none() -> None:
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma-4-12b")
    assert spec.api_key is None


# --- ReadinessCache: peer specs merged into .current() ----------------------


def test_peer_seed_is_unknown_before_first_probe() -> None:
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache(
        {}, peer_specs=[spec], peer_probe=lambda s: True, interval=1000, start=False
    )
    try:
        assert cache.current() == {"multimodal": None}
    finally:
        cache.stop()


def test_no_peer_specs_behaves_exactly_as_before() -> None:
    # No peer_specs at all -> no peer thread, .current() unaffected -
    # byte-identical to the pre-t4 cache for every existing caller.
    cache = R.ReadinessCache(
        {"primary": "http://primary:8000"}, probe=lambda u: True, interval=0.01, start=True
    )
    try:
        assert cache._peer_thread is None
        deadline = time.time() + 2.0
        while time.time() < deadline and cache.current() != {"primary": True}:
            time.sleep(0.005)
        assert cache.current() == {"primary": True}
    finally:
        cache.stop()


def test_peer_probe_true_lands_in_current_keyed_by_name() -> None:
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache(
        {"primary": "http://primary:8000"},
        probe=lambda u: True,
        peer_specs=[spec],
        peer_probe=lambda s: True,
        interval=0.01,
        start=True,
    )
    try:
        expected = {"primary": True, "multimodal": True}
        deadline = time.time() + 2.0
        while time.time() < deadline and cache.current() != expected:
            time.sleep(0.005)
        assert cache.current() == expected
    finally:
        cache.stop()


def test_peer_probe_false_reported() -> None:
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache(
        {}, peer_specs=[spec], peer_probe=lambda s: False, interval=0.01, start=True
    )
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and cache.current().get("multimodal") is None:
            time.sleep(0.005)
        assert cache.current() == {"multimodal": False}
    finally:
        cache.stop()


def test_peer_probe_receives_the_spec_it_was_registered_with() -> None:
    seen = []
    spec = R.PeerSpec(
        name="multimodal", origin="http://thor:8001", served_name="gemma", api_key="sk-x"
    )

    def peer_probe(received_spec):
        seen.append(received_spec)
        return True

    cache = R.ReadinessCache(
        {}, peer_specs=[spec], peer_probe=peer_probe, interval=0.01, start=True
    )
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and not seen:
            time.sleep(0.005)
        assert seen and seen[0] is spec
    finally:
        cache.stop()


def test_peer_probe_raising_degrades_to_false_not_none() -> None:
    # Once a peer probe has actually RUN, a raising probe must degrade to
    # False (the peer honesty check FAILED and the daemon survives) - None
    # is reserved for "never probed yet", not "probed and errored".
    def boom(_spec):
        raise RuntimeError("boom")

    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache({}, peer_specs=[spec], peer_probe=boom, interval=0.01, start=True)
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and cache.current().get("multimodal") is None:
            time.sleep(0.005)
        assert cache.current() == {"multimodal": False}
        assert cache._peer_thread is not None and cache._peer_thread.is_alive()
    finally:
        cache.stop()


def test_one_peer_raising_does_not_stop_others_being_probed() -> None:
    def peer_probe(spec):
        if spec.name == "bad":
            raise RuntimeError("boom")
        return True

    specs = [
        R.PeerSpec(name="bad", origin="http://bad:8001", served_name="x"),
        R.PeerSpec(name="good", origin="http://good:8001", served_name="y"),
    ]
    cache = R.ReadinessCache({}, peer_specs=specs, peer_probe=peer_probe, interval=0.01, start=True)
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and cache.current().get("good") is not True:
            time.sleep(0.005)
        snap = cache.current()
        assert snap["good"] is True
        assert snap["bad"] is False
    finally:
        cache.stop()


def test_multiple_peer_refresh_cycles_do_not_corrupt_local_snapshot() -> None:
    # Local values live in an independent internal store from peer values,
    # merged only at .current() read time - a peer refresh cycle must never
    # wipe out (or corrupt) the local backend's snapshot, and vice versa.
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache(
        {"primary": "http://primary:8000"},
        probe=lambda u: True,
        peer_specs=[spec],
        peer_probe=lambda s: True,
        interval=0.01,
        start=True,
    )
    try:
        expected = {"primary": True, "multimodal": True}
        deadline = time.time() + 2.0
        while time.time() < deadline and cache.current() != expected:
            time.sleep(0.005)
        # Let several more refresh cycles run on BOTH loops.
        time.sleep(0.15)
        assert cache.current() == expected
    finally:
        cache.stop()


# --- Isolation: a hanging peer never delays local-backend readiness --------


def test_slow_hanging_peer_does_not_block_local_probe_readiness() -> None:
    """Acceptance criterion (b): a peer probe that hangs well past any
    reasonable deadline (an injected ``peer_probe`` that ignores every
    configured timeout, worse than a real hung socket) must never delay the
    LOCAL backend's snapshot from becoming ready. Local and peer probing run
    on independent background threads precisely so a peer stuck inside a
    blocking call can never share a thread — and therefore never a deadline
    — with local probing.
    """
    peer_entered = threading.Event()
    release_peer = threading.Event()

    def hanging_peer_probe(_spec):
        peer_entered.set()
        release_peer.wait(timeout=10.0)
        return True

    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache(
        {"primary": "http://primary:8000"},
        probe=lambda u: True,
        peer_specs=[spec],
        peer_probe=hanging_peer_probe,
        interval=0.01,
        start=True,
    )
    try:
        assert peer_entered.wait(2.0), "peer probe never started"
        deadline = time.time() + 1.0
        while time.time() < deadline and cache.current().get("primary") is not True:
            time.sleep(0.005)
        assert (
            cache.current().get("primary") is True
        ), "a hanging peer probe delayed local readiness"
        # The peer itself is honestly still unknown - it has not returned yet.
        assert cache.current().get("multimodal") is None
    finally:
        release_peer.set()
        cache.stop()


def test_slow_hanging_local_probe_does_not_block_peer_readiness() -> None:
    """The isolation invariant cuts both ways: a hanging LOCAL probe must
    never delay a peer's snapshot from becoming ready either.
    """
    local_entered = threading.Event()
    release_local = threading.Event()

    def hanging_local_probe(_url):
        local_entered.set()
        release_local.wait(timeout=10.0)
        return True

    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache(
        {"primary": "http://primary:8000"},
        probe=hanging_local_probe,
        peer_specs=[spec],
        peer_probe=lambda s: True,
        interval=0.01,
        start=True,
    )
    try:
        assert local_entered.wait(2.0), "local probe never started"
        deadline = time.time() + 1.0
        while time.time() < deadline and cache.current().get("multimodal") is not True:
            time.sleep(0.005)
        assert (
            cache.current().get("multimodal") is True
        ), "a hanging local probe delayed peer readiness"
        assert cache.current().get("primary") is None
    finally:
        release_local.set()
        cache.stop()


# --- own timeout: peer probing must not borrow the local timeout budget ----


def test_default_peer_probe_binds_its_own_peer_timeout_not_locals(monkeypatch) -> None:
    captured = {}

    def fake_probe_peer_ready(origin, served_name, *, timeout, api_key=None, opener=None):
        captured["origin"] = origin
        captured["served_name"] = served_name
        captured["timeout"] = timeout
        captured["api_key"] = api_key
        return True

    monkeypatch.setattr(R, "probe_peer_ready", fake_probe_peer_ready)
    spec = R.PeerSpec(
        name="multimodal", origin="http://thor:8001", served_name="gemma", api_key="sk-x"
    )
    cache = R.ReadinessCache(
        {},
        peer_specs=[spec],
        peer_timeout=0.42,
        timeout=9.0,  # deliberately different: a bug that borrows this fails the assert below
        interval=1000,
        start=False,
    )
    try:
        cache._refresh_once_peers()
        assert captured["timeout"] == 0.42
        assert captured["origin"] == "http://thor:8001"
        assert captured["served_name"] == "gemma"
        assert captured["api_key"] == "sk-x"
        assert cache.current() == {"multimodal": True}
    finally:
        cache.stop()


# --- refresh(): the synchronous startup seed covers peers too --------------


def test_refresh_performs_a_synchronous_peer_probe_pass_too() -> None:
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache(
        {}, peer_specs=[spec], peer_probe=lambda s: True, interval=1000, start=False
    )
    try:
        assert cache.current() == {"multimodal": None}
        cache.refresh()
        assert cache.current() == {"multimodal": True}
    finally:
        cache.stop()


# --- clean shutdown: stop()/close() take the peer thread down too ----------


def test_stop_terminates_the_peer_thread_too() -> None:
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache(
        {}, peer_specs=[spec], peer_probe=lambda s: True, interval=0.01, start=True
    )
    assert cache._peer_thread is not None and cache._peer_thread.is_alive()
    cache.stop()
    assert cache._peer_thread is None


def test_close_is_an_alias_for_stop_for_peer_thread_too() -> None:
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache(
        {}, peer_specs=[spec], peer_probe=lambda s: True, interval=0.01, start=True
    )
    assert cache._peer_thread is not None and cache._peer_thread.is_alive()
    cache.close()
    assert cache._peer_thread is None


def test_stop_is_idempotent_with_peer_specs_and_safe_before_start() -> None:
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache({}, peer_specs=[spec], peer_probe=lambda s: True, start=False)
    cache.stop()  # no threads yet - must not raise
    cache.stop()  # idempotent
    assert cache._peer_thread is None


def test_current_returns_a_copy_isolated_from_caller_mutation_for_peers_too() -> None:
    spec = R.PeerSpec(name="multimodal", origin="http://thor:8001", served_name="gemma")
    cache = R.ReadinessCache({}, peer_specs=[spec], peer_probe=lambda s: True, start=False)
    try:
        snapshot = cache.current()
        snapshot["multimodal"] = "corrupted"
        snapshot["injected"] = True
        assert cache.current() == {"multimodal": None}
    finally:
        cache.stop()
