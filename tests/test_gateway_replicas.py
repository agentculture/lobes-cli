"""Tests for the per-role replica snapshot cache (issue #199, task t4).

The module under test (:mod:`lobes.gateway._replicas`) is the *load and
fingerprint* substrate the replica pool selects over. Four properties get
the most scrutiny here, because a regression in any of them is a silent
honesty violation rather than a visible failure:

* :meth:`ReplicaCache.current` is a pure dict/tuple read — it must NEVER
  open a socket, so a request-path selection costs no I/O. Proved by an
  injected opener that fails the test if it is called from the reading
  thread.
* A peer only pools when its LIVE fingerprint matches the local one on the
  four disqualifying fields (served id, quantization, max context,
  runtime). ``"unknown"`` on either side never pools silently.
* Nothing ever dials a peer's vLLM port: every peer URL is the declared
  gateway origin, path ``/status`` or ``/capabilities``.
* A served id the catalog does not know still reports the LIVE id and
  ``max_model_len`` and ``"unknown"`` for everything undeclared — the
  module never consults :mod:`lobes.catalog` at all.

Stdlib only, mirroring the gateway's dependency-free discipline and
``tests/test_readiness_peer_probe.py``'s injected-opener style.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time

import pytest

from lobes.gateway import _replicas as R
from lobes.gateway import _selection as S

# --- fixtures / builders ---------------------------------------------------

LOCAL_URL = "http://vllm-primary:8000"
SERVED = "unsloth/Qwen3.8-27B-NVFP4"
PEER = "http://thor:8000"

DECLARED = {
    "runtime": "vllm",
    "quantization": "compressed-tensors",
    "kv_cache_dtype": "fp8",
    "reasoning_parser": "qwen3",
    "tool_parser": "qwen3_coder",
    "speculative_config": "dspark",
}


def _lane(**kw) -> R.LocalLane:
    return R.LocalLane(
        base_url=kw.pop("base_url", LOCAL_URL),
        served_name=kw.pop("served_name", SERVED),
        declared=kw.pop("declared", dict(DECLARED)),
        **kw,
    )


def _models_body(served: str = SERVED, max_model_len: int | None = 262144) -> bytes:
    entry: dict = {"id": served, "object": "model"}
    if max_model_len is not None:
        entry["max_model_len"] = max_model_len
    return json.dumps({"object": "list", "data": [entry]}).encode()


def _metrics_body(running: int = 0, waiting: int = 0) -> bytes:
    return (
        "# TYPE vllm:num_requests_running gauge\n"
        f'vllm:num_requests_running{{model_name="m"}} {running}.0\n'
        "# TYPE vllm:num_requests_waiting gauge\n"
        f'vllm:num_requests_waiting{{model_name="m"}} {waiting}.0\n'
    ).encode()


def _status_body(
    *,
    busy: bool = False,
    health: str = "ok",
    running: int = 0,
    waiting: int = 0,
    name: str = "primary",
    served_name: str = SERVED,
    capacity: object = None,
) -> bytes:
    backend: dict = {
        "name": name,
        "task": "generate",
        "served_name": served_name,
        "health": health,
        "metrics": {"running": running, "waiting": waiting},
    }
    if capacity is not None:
        backend["capacity"] = capacity
    payload = {
        "object": "lobes.fleet_status",
        "default_model": served_name,
        "busy": {"running": running, "waiting": waiting},
        "backends": [backend],
        "endpoints": [],
        "pressure": {
            "mode": "busy" if busy else "warm",
            "shed": busy,
            "reason": "swap" if busy else "",
            "swap_used_percent": 90.0 if busy else 1.0,
            "iowait_percent": 0.0,
        },
    }
    return json.dumps(payload).encode()


def _caps_body(fingerprint: dict | None = None, *, role: str = "cortex", **role_kw) -> bytes:
    entry: dict = {
        "role": role,
        "model": role_kw.pop("model", SERVED),
        "context": role_kw.pop("context", 262144),
        "runtime": "vllm",
        "quant": "modelopt",
        "mtp": True,
        "ready": True,
    }
    entry.update(role_kw)
    if fingerprint is not None:
        entry["fingerprint"] = fingerprint
    return json.dumps({"object": "lobes.capabilities", "roles": {role: entry}}).encode()


def _fingerprint(**kw) -> dict:
    fp = {
        "served_id": SERVED,
        "max_model_len": 262144,
        "quantization": "compressed-tensors",
        "kv_cache_dtype": "fp8",
        "runtime": "vllm",
        "reasoning_parser": "qwen3",
        "tool_parser": "qwen3_coder",
        "speculative_config": "dspark",
    }
    fp.update(kw)
    return fp


def _router(routes: dict):
    """Build an opener that maps URL → ``(status, body)`` from *routes*.

    Records every ``(url, api_key)`` it was asked for on ``.calls``.
    """

    calls: list[tuple[str, str | None]] = []

    def opener(url: str, _timeout: float, api_key: str | None):
        calls.append((url, api_key))
        try:
            outcome = routes[url]
        except KeyError:  # pragma: no cover - a test asked for an unrouted URL
            raise AssertionError(f"unrouted URL requested: {url}")
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome()
        return outcome

    opener.calls = calls  # type: ignore[attr-defined]
    return opener


def _default_routes(**kw) -> dict:
    return {
        LOCAL_URL + "/v1/models": (200, _models_body()),
        LOCAL_URL + "/metrics": (200, _metrics_body(**kw.pop("local_load", {}))),
        PEER + "/status": (200, _status_body(**kw.pop("status", {}))),
        PEER + "/capabilities": (200, _caps_body(kw.pop("fingerprint", _fingerprint()))),
    }


def _cache(routes: dict, *, peers=None, local=None, **kw) -> R.ReplicaCache:
    opener = kw.pop("urlopen", None) or _router(routes)
    return R.ReplicaCache(
        "cortex",
        _lane() if local is None else local,
        peers if peers is not None else (R.PeerReplica(PEER, ""),),
        urlopen=opener,
        start=False,
        **kw,
    )


def _by_origin(states) -> dict:
    return {s.origin: s for s in states}


def _refreshed(cache: R.ReplicaCache) -> R.ReplicaCache:
    cache.refresh()
    return cache


# --- local lane fingerprint ------------------------------------------------


def test_local_fingerprint_merges_live_models_with_declared() -> None:
    cache = _cache(_default_routes())
    cache.refresh()
    local = _by_origin(cache.current())[LOCAL_URL]
    assert local.local is True
    assert local.ready is True
    assert local.health == "ok"
    fp = local.fingerprint
    assert fp is not None
    assert fp.served_id == SERVED
    assert fp.max_model_len == 262144
    assert fp.runtime == "vllm"
    assert fp.quantization == "compressed-tensors"
    assert fp.kv_cache_dtype == "fp8"
    assert fp.reasoning_parser == "qwen3"
    assert fp.tool_parser == "qwen3_coder"
    assert fp.speculative_config == "dspark"


def test_local_load_is_read_from_its_own_metrics() -> None:
    cache = _cache(_default_routes(local_load={"running": 3, "waiting": 2}))
    cache.refresh()
    local = _by_origin(cache.current())[LOCAL_URL]
    assert (local.running, local.waiting) == (3, 2)


def test_unknown_served_id_reports_live_values_and_unknown_for_undeclared() -> None:
    """A served id the catalog never heard of: live id + context, unknown rest.

    ``never a catalog value`` is structural here — this module imports no
    catalog at all (asserted below), so there is nothing to fall back to.
    """
    routes = _default_routes()
    routes[LOCAL_URL + "/v1/models"] = (200, _models_body("cortex", 262144))
    cache = _cache(routes, local=R.LocalLane(LOCAL_URL, "cortex", {}))
    cache.refresh()
    fp = _by_origin(cache.current())[LOCAL_URL].fingerprint
    assert fp is not None
    assert fp.served_id == "cortex"
    assert fp.max_model_len == 262144
    assert fp.runtime == R.UNKNOWN
    assert fp.quantization == R.UNKNOWN
    assert fp.kv_cache_dtype == R.UNKNOWN
    assert fp.speculative_config == R.UNKNOWN


def test_module_needs_no_catalog_import() -> None:
    source = R.__file__
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "lobes.catalog" not in text
    assert "from ..catalog" not in text
    assert "from .catalog" not in text


def test_local_lane_unreachable_is_not_ready() -> None:
    routes = _default_routes()
    routes[LOCAL_URL + "/v1/models"] = OSError("connection refused")
    cache = _cache(routes)
    cache.refresh()
    local = _by_origin(cache.current())[LOCAL_URL]
    assert local.ready is False
    assert local.health == "unreachable"
    assert local.fingerprint is None


def test_no_local_lane_yields_peers_only() -> None:
    routes = _default_routes()
    cache = R.ReplicaCache(
        "cortex", None, (R.PeerReplica(PEER, ""),), urlopen=_router(routes), start=False
    )
    cache.refresh()
    states = cache.current()
    assert [s.origin for s in states] == [PEER]
    assert states[0].local is False


# --- peer probing ----------------------------------------------------------


def test_matching_peer_is_ready_and_compatible() -> None:
    cache = _cache(_default_routes(status={"running": 1, "waiting": 0}))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.local is False
    assert peer.ready is True
    assert peer.busy is False
    assert peer.health == "ok"
    assert (peer.running, peer.waiting) == (1, 0)
    assert peer.compatible is True
    assert peer.reason == ""
    assert peer.last_seen > 0.0


@pytest.mark.parametrize(
    "field,value,needle",
    [
        ("served_id", "unsloth/Qwen3.6-27B-NVFP4", "served_id"),
        ("quantization", "q4_k_m", "quantization"),
        ("max_model_len", 131072, "max_model_len"),
        ("runtime", "llamacpp", "runtime"),
    ],
)
def test_disqualifying_field_difference_marks_incompatible(field, value, needle) -> None:
    cache = _cache(_default_routes(fingerprint=_fingerprint(**{field: value})))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.compatible is False
    assert needle in peer.reason
    assert str(value) in peer.reason


def test_informational_field_difference_stays_compatible_and_is_recorded() -> None:
    cache = _cache(
        _default_routes(fingerprint=_fingerprint(kv_cache_dtype="auto", speculative_config="mtp"))
    )
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.compatible is True
    assert peer.reason == ""
    assert peer.fingerprint is not None
    assert peer.fingerprint.kv_cache_dtype == "auto"
    assert peer.fingerprint.speculative_config == "mtp"


def test_parser_difference_stays_compatible() -> None:
    cache = _cache(_default_routes(fingerprint=_fingerprint(reasoning_parser="x", tool_parser="y")))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.compatible is True
    assert peer.fingerprint is not None
    assert peer.fingerprint.tool_parser == "y"


def test_unknown_quantization_on_either_side_never_pools() -> None:
    cache = _cache(_default_routes(fingerprint=_fingerprint(quantization="unknown")))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.compatible is False
    assert "quantization: unknown" in peer.reason

    routes = _default_routes()
    local_unknown = R.LocalLane(LOCAL_URL, SERVED, {"runtime": "vllm"})
    cache2 = _cache(routes, local=local_unknown)
    cache2.refresh()
    peer2 = _by_origin(cache2.current())[PEER]
    assert peer2.compatible is False
    assert "quantization: unknown" in peer2.reason


def test_unknown_max_model_len_never_pools() -> None:
    cache = _cache(_default_routes(fingerprint=_fingerprint(max_model_len=None)))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.compatible is False
    assert "max_model_len: unknown" in peer.reason


def test_capabilities_without_fingerprint_uses_live_id_and_context_only() -> None:
    routes = _default_routes()
    routes[PEER + "/capabilities"] = (200, _caps_body(None))
    cache = _cache(routes)
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    fp = peer.fingerprint
    assert fp is not None
    assert fp.served_id == SERVED
    assert fp.max_model_len == 262144
    # The role entry's own quant/runtime are CATALOG-derived (they mislabel an
    # unknown served id) so they are never read: unknown, and therefore never
    # silently pooled.
    assert fp.quantization == R.UNKNOWN
    assert fp.runtime == R.UNKNOWN
    assert peer.compatible is False
    assert "quantization: unknown" in peer.reason


def test_peer_gateway_down_is_not_selectable() -> None:
    routes = _default_routes()
    routes[PEER + "/status"] = OSError("connection refused")
    routes[PEER + "/capabilities"] = OSError("connection refused")
    cache = _cache(routes)
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.ready is False
    assert peer.health == "unreachable"
    assert peer.last_seen == 0.0


def test_peer_timeout_is_reported_as_timeout() -> None:
    routes = _default_routes()
    routes[PEER + "/status"] = TimeoutError("timed out")
    routes[PEER + "/capabilities"] = TimeoutError("timed out")
    cache = _cache(routes)
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.ready is False
    assert peer.health == "timeout"


def test_peer_non_200_status_is_error() -> None:
    routes = _default_routes()
    routes[PEER + "/status"] = (503, b"{}")
    cache = _cache(routes)
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.ready is False
    assert peer.health == "error"


def test_peer_malformed_status_body_is_error() -> None:
    routes = _default_routes()
    routes[PEER + "/status"] = (200, b"not json")
    cache = _cache(routes)
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.ready is False
    assert peer.health == "error"


def test_peer_reporting_busy_is_never_selectable() -> None:
    cache = _cache(_default_routes(status={"busy": True}))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.busy is True


def test_plain_boolean_busy_field_is_honoured() -> None:
    routes = _default_routes()
    payload = json.loads(_status_body())
    payload["busy"] = True
    routes[PEER + "/status"] = (200, json.dumps(payload).encode())
    cache = _cache(routes)
    cache.refresh()
    assert _by_origin(cache.current())[PEER].busy is True


def test_peer_backend_health_not_ok_is_not_ready() -> None:
    cache = _cache(_default_routes(status={"health": "unreachable"}))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.ready is False
    assert peer.health == "unreachable"


def test_peer_not_serving_the_role_is_not_ready() -> None:
    routes = _default_routes()
    routes[PEER + "/status"] = (
        200,
        _status_body(name="multimodal", served_name="google/gemma-4-12b"),
    )
    cache = _cache(routes)
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.ready is False
    assert peer.health == "unknown"


def test_backend_matched_by_role_backend_name_when_served_id_differs() -> None:
    routes = _default_routes()
    routes[PEER + "/status"] = (200, _status_body(name="cortex", served_name="something-else"))
    cache = _cache(routes)
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.ready is True


def test_explicit_backend_name_is_matched() -> None:
    routes = _default_routes()
    routes[PEER + "/status"] = (200, _status_body(name="primary", served_name="something-else"))
    cache = _cache(routes, backend_name="primary")
    cache.refresh()
    assert _by_origin(cache.current())[PEER].ready is True


def test_declared_weight_rides_through_untouched() -> None:
    routes = _default_routes()
    cache = _cache(routes, peers=(R.PeerReplica(PEER, "", weight=2.5),))
    cache.refresh()
    assert _by_origin(cache.current())[PEER].weight == 2.5
    assert _by_origin(cache.current())[LOCAL_URL].weight == 1.0


def test_last_seen_is_carried_forward_across_a_failing_cycle() -> None:
    clock = iter([10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0])
    routes = _default_routes()
    opener = _router(routes)
    cache = R.ReplicaCache(
        "cortex",
        _lane(),
        (R.PeerReplica(PEER, ""),),
        urlopen=opener,
        monotonic=lambda: next(clock),
        start=False,
    )
    cache.refresh()
    first = _by_origin(cache.current())[PEER].last_seen
    assert first == 10.0
    routes[PEER + "/status"] = OSError("down")
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.ready is False
    assert peer.last_seen == first  # carried forward, never reset by a failure


# --- honesty: no socket on the read path, no vLLM port on a peer -----------


def test_current_opens_no_socket() -> None:
    cache = _cache(_default_routes())
    cache.refresh()
    opened: list[str] = []

    def exploding(url, _timeout, _api_key):  # pragma: no cover - must never run
        opened.append(url)
        raise AssertionError("current() opened a socket")

    cache._urlopen = exploding  # noqa: SLF001 - deliberate: prove current() never probes
    for _ in range(200):
        states = cache.current()
        assert len(states) == 2
    assert opened == []


def test_current_returns_a_tuple_of_frozen_states() -> None:
    cache = _cache(_default_routes())
    cache.refresh()
    states = cache.current()
    assert isinstance(states, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        states[0].ready = False  # type: ignore[misc]


def test_no_probe_dials_a_vllm_port_on_a_peer() -> None:
    opener = _router(_default_routes())
    cache = _cache({}, urlopen=opener)
    cache.refresh()
    peer_urls = [url for url, _ in opener.calls if url.startswith(PEER)]
    assert peer_urls, "the peer was never probed"
    for url in peer_urls:
        path = url[len(PEER) :]
        assert path in ("/status", "/capabilities"), path
    # and nothing outside the local lane / the declared origin was dialed
    for url, _ in opener.calls:
        assert url.startswith(PEER) or url.startswith(LOCAL_URL)


def test_authorization_is_sent_only_when_a_key_is_declared() -> None:
    opener = _router(_default_routes())
    cache = _cache({}, urlopen=opener, peers=(R.PeerReplica(PEER, "s3cret"),))
    cache.refresh()
    peer_keys = {key for url, key in opener.calls if url.startswith(PEER)}
    assert peer_keys == {"s3cret"}
    local_keys = {key for url, key in opener.calls if url.startswith(LOCAL_URL)}
    assert local_keys == {None}


def test_empty_key_slot_sends_no_authorization() -> None:
    opener = _router(_default_routes())
    cache = _cache({}, urlopen=opener, peers=(R.PeerReplica(PEER, ""),))
    cache.refresh()
    assert {key for url, key in opener.calls if url.startswith(PEER)} == {None}


def test_peer_replica_repr_never_leaks_key_material() -> None:
    assert "s3cret" not in repr(R.PeerReplica(PEER, "s3cret"))


# --- daemon-thread lifecycle (mirrors tests/test_readiness_peer_probe.py) ---


def test_seeded_state_before_any_probe_is_unknown_and_not_ready() -> None:
    cache = _cache(_default_routes())
    states = _by_origin(cache.current())
    assert states[PEER].health == "unknown"
    assert states[PEER].ready is False
    assert states[PEER].fingerprint is None
    assert states[PEER].last_seen == 0.0
    assert states[LOCAL_URL].ready is False


def test_start_and_stop_run_and_join_both_threads() -> None:
    cache = _cache(_default_routes(), refresh_interval=0.01)
    cache.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _by_origin(cache.current())[PEER].ready:
                break
            time.sleep(0.01)
        assert cache.is_alive() is True
        assert _by_origin(cache.current())[PEER].ready is True
    finally:
        cache.stop()
    assert cache.is_alive() is False


def test_stop_is_idempotent_and_safe_before_start() -> None:
    cache = _cache(_default_routes())
    cache.stop()
    cache.stop()
    assert cache.is_alive() is False


def test_close_is_an_alias_for_stop() -> None:
    cache = _cache(_default_routes(), refresh_interval=0.01)
    cache.start()
    cache.close()
    assert cache.is_alive() is False


def test_refresh_interval_paces_repeated_probes() -> None:
    counter = {"n": 0}
    routes = _default_routes()

    def counting(url, timeout, api_key):
        if url.endswith("/status"):
            counter["n"] += 1
        return routes[url]

    cache = R.ReplicaCache(
        "cortex",
        _lane(),
        (R.PeerReplica(PEER, ""),),
        urlopen=counting,
        refresh_interval=0.01,
        start=False,
    )
    cache.start()
    try:
        deadline = time.monotonic() + 5.0
        while counter["n"] < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        cache.stop()
    assert counter["n"] >= 3


def test_a_hung_peer_never_delays_the_local_probe() -> None:
    release = threading.Event()
    routes = _default_routes()

    def opener(url, _timeout, _api_key):
        if url.startswith(PEER):
            release.wait(5.0)
        return routes[url]

    cache = R.ReplicaCache(
        "cortex",
        _lane(),
        (R.PeerReplica(PEER, ""),),
        urlopen=opener,
        refresh_interval=0.01,
        start=False,
    )
    cache.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _by_origin(cache.current())[LOCAL_URL].ready:
                break
            time.sleep(0.01)
        # The peer probe is still blocked, yet the local lane is live.
        assert _by_origin(cache.current())[LOCAL_URL].ready is True
        assert _by_origin(cache.current())[PEER].ready is False
    finally:
        release.set()
        cache.stop()


def test_a_raising_opener_degrades_and_never_crashes_the_daemon() -> None:
    routes = _default_routes()
    routes[PEER + "/status"] = RuntimeError("boom")
    cache = _cache(routes, refresh_interval=0.01)
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.ready is False
    assert peer.health == "error"


def test_one_failing_peer_does_not_stop_the_others() -> None:
    other = "http://orin:8000"
    routes = _default_routes()
    routes[other + "/status"] = OSError("down")
    routes[other + "/capabilities"] = OSError("down")
    cache = _cache(
        routes,
        peers=(R.PeerReplica(other, ""), R.PeerReplica(PEER, "")),
    )
    cache.refresh()
    states = _by_origin(cache.current())
    assert states[other].ready is False
    assert states[PEER].ready is True


def test_no_peers_spawns_no_peer_thread() -> None:
    cache = _cache(_default_routes(), peers=())
    cache.start()
    try:
        assert cache._peer_thread is None  # noqa: SLF001 - structural assertion
    finally:
        cache.stop()


# --- pure comparison helper -------------------------------------------------


def test_compare_fingerprints_is_pure_and_names_every_differing_field() -> None:
    local = R.Fingerprint(SERVED, 262144, "vllm", "nvfp4", "fp8", "qwen3", "qwen3_coder", "none")
    peer = R.Fingerprint("other", 131072, "llamacpp", "q4_k_m", "auto", "x", "y", "z")
    ok, reason = R.compare_fingerprints(local, peer)
    assert ok is False
    for needle in ("served_id", "max_model_len", "runtime", "quantization"):
        assert needle in reason
    assert "kv_cache_dtype" not in reason


def test_compare_fingerprints_missing_local_is_incompatible() -> None:
    peer = R.Fingerprint(SERVED, 262144, "vllm", "nvfp4", "fp8", "", "", "")
    ok, reason = R.compare_fingerprints(None, peer)
    assert ok is False
    assert reason
    ok2, reason2 = R.compare_fingerprints(peer, None)
    assert ok2 is False
    assert reason2


def test_local_runtime_falls_back_to_owned_by_when_undeclared():
    """No lane declares ``runtime`` today; the engine's own ``owned_by`` is live truth."""
    from lobes.gateway._replicas import _runtime_from

    assert _runtime_from({"id": "m", "owned_by": "vllm"}, {}) == "vllm"
    assert _runtime_from({"id": "m", "owned_by": "llama.cpp"}, {}) == "llamacpp"
    assert _runtime_from({"id": "m", "owned_by": "someone"}, {}) == "unknown"
    assert _runtime_from({"id": "m"}, {}) == "unknown"
    assert _runtime_from({"id": "m", "owned_by": "vllm"}, {"runtime": "llamacpp"}) == "llamacpp"


# --- capacity ingest + clamp (capacity-relative pool routing, t4 stage 1) ---
#
# `weight` stopped being a hardcoded 1.0 placeholder: it now carries the
# replica's measured max-active-requests CAPACITY, ingested from the peer's
# own `/status` (q1: peer self-published) or, for the local seed, from this
# box's declared `<PREFIX>_MAX_ACTIVE`. Because that number is peer-CONTROLLED
# and `_selection.estimated_wait` DIVIDES by it, an inflated value would rank
# as near-zero wait at every load level and vacuum the whole pool — so ingest
# clamps, and says so in `reason`.


def test_a_peer_published_capacity_populates_weight() -> None:
    cache = _cache(_default_routes(status={"capacity": 8}))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.weight == 8.0
    assert peer.capacity == 8.0
    assert peer.compatible is True


def test_the_local_seed_populates_from_this_boxs_own_declared_capacity() -> None:
    cache = _cache(_default_routes(), local=_lane(weight=6.0))
    local = _by_origin(cache.current())[LOCAL_URL]
    assert local.weight == 6.0  # seeded, before any probe
    cache.refresh()
    local = _by_origin(cache.current())[LOCAL_URL]
    assert local.weight == 6.0
    assert local.capacity == 6.0


def test_a_peer_publishing_no_capacity_falls_back_and_stays_routable() -> None:
    """h3: an unpublished capacity must never make a replica unselectable."""
    cache = _cache(_default_routes())
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.weight == R.UNCALIBRATED_WEIGHT
    assert peer.capacity is None
    assert peer.ready is True and peer.compatible is True
    assert S.is_calibrated(peer) is False
    assert S.is_full(peer) is False


def test_an_inflated_peer_capacity_is_clamped_and_the_clamp_is_in_the_reason() -> None:
    cache = _cache(_default_routes(status={"capacity": 10_000}))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.weight == R.CAPACITY_CLAMP_MAX
    assert "clamped" in peer.reason
    assert "10000" in peer.reason
    # Still poolable — a clamp bounds a peer's share, it does not evict it.
    assert peer.compatible is True and peer.ready is True


def test_a_clamped_capacity_is_observable_rather_than_silent() -> None:
    quiet = _by_origin(_refreshed(_cache(_default_routes(status={"capacity": 4}))).current())[PEER]
    assert quiet.reason == ""


def test_the_local_declared_capacity_is_clamped_too() -> None:
    cache = _cache(_default_routes(), local=_lane(weight=10_000.0))
    cache.refresh()
    local = _by_origin(cache.current())[LOCAL_URL]
    assert local.weight == R.CAPACITY_CLAMP_MAX
    assert "clamped" in local.reason


def test_a_non_positive_or_unparseable_capacity_is_ignored_with_a_reason() -> None:
    for published in (0, -4, "lots", float("nan")):
        cache = _cache(_default_routes(status={"capacity": published}))
        cache.refresh()
        peer = _by_origin(cache.current())[PEER]
        assert peer.weight == R.UNCALIBRATED_WEIGHT, published
        assert "capacity ignored" in peer.reason, published


def test_the_kill_switch_pins_every_capacity_local_and_peer_alike() -> None:
    cache = _cache(
        _default_routes(status={"capacity": 8}),
        local=_lane(weight=6.0),
        capacity_kill_switch=True,
    )
    cache.refresh()
    states = _by_origin(cache.current())
    assert states[PEER].weight == R.UNCALIBRATED_WEIGHT
    assert states[LOCAL_URL].weight == R.UNCALIBRATED_WEIGHT
    assert "clamped" not in states[PEER].reason


def test_the_clamp_maximum_is_configurable_per_cache() -> None:
    cache = _cache(_default_routes(status={"capacity": 8}), capacity_max=4.0)
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.weight == 4.0
    assert "clamped" in peer.reason


def test_resolve_capacity_is_pure_over_its_inputs() -> None:
    assert R.resolve_capacity(8.0) == (8.0, "")
    assert R.resolve_capacity(None) == (R.UNCALIBRATED_WEIGHT, "")
    weight, note = R.resolve_capacity(99.0, capacity_max=8.0)
    assert (weight, "clamped" in note) == (8.0, True)
    assert R.resolve_capacity(8.0, kill_switch=True) == (R.UNCALIBRATED_WEIGHT, "")


# --- capacity is keyed to a fingerprint (t4 stage 2) -----------------------
#
# A calibrated capacity is only valid for the (box, checkpoint, window,
# speculative config) it was MEASURED on. `lobes switch` is a down+up with a
# model swap and a shape re-render force-writes keys, so a stored capacity
# outlives its conditions unless it is keyed to the live fingerprint the
# module already probes. It is: the capacity is pinned to the fingerprint it
# arrived under and DISCARDED when that fingerprint stops matching, falling
# back to the safe default until a new number is published (spec c23/h16).


def _switched_routes(**kw) -> dict:
    """The same fleet after a `lobes switch`: a different served id, everywhere."""
    other = "unsloth/Qwen3.6-27B-NVFP4"
    routes = _default_routes(**kw)
    status = dict(kw.get("status") or {})
    status["served_name"] = other
    routes[LOCAL_URL + "/v1/models"] = (200, _models_body(served=other))
    routes[PEER + "/status"] = (200, _status_body(**status))
    routes[PEER + "/capabilities"] = (200, _caps_body(_fingerprint(served_id=other)))
    return routes


def test_a_capacity_is_pinned_to_the_fingerprint_it_arrived_under() -> None:
    cache = _cache(_default_routes(status={"capacity": 8}), local=_lane(weight=6.0))
    cache.refresh()
    states = _by_origin(cache.current())
    assert states[PEER].capacity_fingerprint == states[PEER].fingerprint
    assert states[LOCAL_URL].capacity_fingerprint == states[LOCAL_URL].fingerprint


def test_a_local_capacity_is_discarded_when_the_local_checkpoint_changes() -> None:
    routes = _default_routes(status={"capacity": 8})
    cache = _cache(routes, local=_lane(weight=6.0))
    cache.refresh()
    assert _by_origin(cache.current())[LOCAL_URL].weight == 6.0
    # `lobes switch`: the lane now serves a different checkpoint.
    routes[LOCAL_URL + "/v1/models"] = (200, _models_body(served="unsloth/Qwen3.6-27B-NVFP4"))
    cache.refresh()
    local = _by_origin(cache.current())[LOCAL_URL]
    assert local.weight == R.UNCALIBRATED_WEIGHT
    assert "capacity discarded" in local.reason
    assert "served_id" in local.reason


def test_a_peer_capacity_is_discarded_when_the_peers_fingerprint_changes() -> None:
    routes = _default_routes(status={"capacity": 8})
    cache = _cache(routes)
    cache.refresh()
    assert _by_origin(cache.current())[PEER].weight == 8.0
    routes[PEER + "/capabilities"] = (200, _caps_body(_fingerprint(max_model_len=131072)))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.weight == R.UNCALIBRATED_WEIGHT
    assert "capacity discarded" in peer.reason
    assert S.is_calibrated(peer) is False


def test_a_republished_stale_capacity_stays_discarded() -> None:
    """The same number under a new fingerprint must not resurrect next pass."""
    routes = _default_routes(status={"capacity": 8})
    cache = _cache(routes)
    cache.refresh()
    routes[PEER + "/capabilities"] = (200, _caps_body(_fingerprint(max_model_len=131072)))
    for _ in range(3):
        cache.refresh()
        assert _by_origin(cache.current())[PEER].weight == R.UNCALIBRATED_WEIGHT


def test_a_recalibrated_capacity_repins_to_the_new_fingerprint() -> None:
    routes = _default_routes(status={"capacity": 8})
    cache = _cache(routes)
    cache.refresh()
    routes[PEER + "/capabilities"] = (200, _caps_body(_fingerprint(max_model_len=131072)))
    cache.refresh()
    assert _by_origin(cache.current())[PEER].weight == R.UNCALIBRATED_WEIGHT
    # The operator recalibrates on the new checkpoint and the peer publishes it.
    routes[PEER + "/status"] = (200, _status_body(capacity=12))
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.weight == 12.0
    assert "capacity discarded" not in peer.reason
    assert peer.capacity_fingerprint == peer.fingerprint


def test_an_unprobed_fingerprint_never_forces_a_discard() -> None:
    """Silence is not evidence of a change: an unreachable peer keeps its capacity."""
    routes = _default_routes(status={"capacity": 8})
    cache = _cache(routes)
    cache.refresh()
    routes[PEER + "/status"] = OSError("connection refused")
    cache.refresh()
    peer = _by_origin(cache.current())[PEER]
    assert peer.weight == 8.0
    assert peer.ready is False  # unreachable, so it is not a pool candidate anyway


def test_an_informational_fingerprint_change_never_discards_a_capacity() -> None:
    """Only the four disqualifying fields key a capacity — the drafter/parsers
    differ across the Spark+Thor pair by operator policy (spec c32)."""
    routes = _default_routes(status={"capacity": 8})
    cache = _cache(routes)
    cache.refresh()
    routes[PEER + "/capabilities"] = (200, _caps_body(_fingerprint(speculative_config="mtp")))
    cache.refresh()
    assert _by_origin(cache.current())[PEER].weight == 8.0


def test_both_sides_switching_together_discards_both_capacities() -> None:
    routes = _default_routes(status={"capacity": 8})
    cache = _cache(routes, local=_lane(weight=6.0))
    cache.refresh()
    switched = _switched_routes(status={"capacity": 8})
    routes.update(switched)
    cache.refresh()
    states = _by_origin(cache.current())
    assert states[LOCAL_URL].weight == R.UNCALIBRATED_WEIGHT
    assert states[PEER].weight == R.UNCALIBRATED_WEIGHT


# --- local in-flight accounting (t4 stage 3) -------------------------------
#
# Load is otherwise probe-sourced ONLY, on a 5s refresh, with nothing counted
# at dispatch — so a burst of concurrent arrivals all read one stale snapshot
# and stampede the same replica. Accurate capacity makes that WORSE (a
# genuinely least-full peer attracts the whole burst), so the snapshot has to
# self-correct between refreshes: this box counts its own outstanding
# dispatches and folds them into the `waiting` the selection policy divides by.


class _Clock:
    """A monotonic clock a test can advance by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def test_no_dispatches_means_no_in_flight_anywhere() -> None:
    cache = _cache(_default_routes())
    cache.refresh()
    for state in cache.current():
        assert state.in_flight == 0
    assert cache.in_flight(PEER) == 0
    assert cache.in_flight("http://nowhere:8000") == 0


def test_the_dispatch_context_manager_counts_while_open_and_releases_on_exit() -> None:
    cache = _cache(_default_routes())
    cache.refresh()
    with cache.dispatch(PEER):
        assert cache.in_flight(PEER) == 1
        assert _by_origin(cache.current())[PEER].in_flight == 1
    assert cache.in_flight(PEER) == 0
    assert _by_origin(cache.current())[PEER].in_flight == 0


def test_the_dispatch_context_manager_releases_on_an_exception() -> None:
    cache = _cache(_default_routes())
    cache.refresh()
    with pytest.raises(RuntimeError):
        with cache.dispatch(PEER):
            raise RuntimeError("upstream blew up mid-stream")
    assert cache.in_flight(PEER) == 0


def test_end_dispatch_is_idempotent_and_tolerates_none() -> None:
    cache = _cache(_default_routes())
    token = cache.begin_dispatch(PEER)
    cache.end_dispatch(token)
    cache.end_dispatch(token)  # a double release must never go negative
    cache.end_dispatch(None)
    cache.end_dispatch(-1)
    assert cache.in_flight(PEER) == 0


def test_in_flight_is_folded_into_waiting_so_estimated_wait_self_corrects() -> None:
    cache = _cache(_default_routes(local_load={"running": 1, "waiting": 0}))
    cache.refresh()
    before = _by_origin(cache.current())[LOCAL_URL]
    with cache.dispatch(LOCAL_URL):
        during = _by_origin(cache.current())[LOCAL_URL]
        assert during.waiting == before.waiting + 1
        assert during.in_flight == 1
        assert S.estimated_wait(during) > S.estimated_wait(before)


def test_a_dispatch_older_than_the_last_probe_is_not_counted_twice() -> None:
    """Once a probe has SEEN the request, the probed number is authoritative."""
    clock = _Clock()
    cache = _cache(_default_routes(), monotonic=clock)
    cache.refresh()
    token = cache.begin_dispatch(LOCAL_URL)
    assert _by_origin(cache.current())[LOCAL_URL].in_flight == 1
    clock.t += 1.0
    cache.refresh()  # the probe now reports the request itself
    assert _by_origin(cache.current())[LOCAL_URL].in_flight == 0
    cache.end_dispatch(token)


def test_a_leaked_dispatch_expires_and_never_makes_a_box_look_full_forever() -> None:
    """The leak guard: a counter that is never released must still decay."""
    clock = _Clock()
    cache = _cache(_default_routes(), monotonic=clock)
    cache.refresh()
    cache.begin_dispatch(LOCAL_URL)  # deliberately never released
    assert cache.in_flight(LOCAL_URL) == 1
    clock.t += R.INFLIGHT_MAX_AGE + 1.0
    assert cache.in_flight(LOCAL_URL) == 0
    assert _by_origin(cache.current())[LOCAL_URL].in_flight == 0


def test_concurrent_arrivals_spread_across_two_idle_replicas() -> None:
    """N arrivals against two idle replicas do not all land on one, WITHOUT
    waiting for a probe refresh (the spec's own honesty condition)."""
    cache = _cache(_default_routes())
    cache.refresh()
    picked: list[str] = []
    held = []
    for _ in range(4):
        choice = S.select_replica(cache.current())
        assert choice.origin is not None
        picked.append(choice.origin)
        held.append(cache.begin_dispatch(choice.origin))
    assert set(picked) == {LOCAL_URL, PEER}
    assert picked.count(LOCAL_URL) == 2 and picked.count(PEER) == 2
    for token in held:
        cache.end_dispatch(token)
    assert all(state.in_flight == 0 for state in cache.current())


def test_in_flight_counting_opens_no_socket() -> None:
    def _forbidden(url, _timeout, _key):  # pragma: no cover - must never run
        raise AssertionError(f"current() opened a socket: {url}")

    cache = _cache(_default_routes())
    cache.refresh()
    with cache.dispatch(PEER):
        cache._urlopen = _forbidden  # noqa: SLF001 - structural assertion
        assert _by_origin(cache.current())[PEER].in_flight == 1


def test_dispatch_counts_are_thread_safe() -> None:
    cache = _cache(_default_routes())
    cache.refresh()
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        for _ in range(50):
            with cache.dispatch(PEER):
                pass

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache.in_flight(PEER) == 0
