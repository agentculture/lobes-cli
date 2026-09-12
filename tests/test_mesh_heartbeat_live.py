"""The heartbeat actually announces: end-to-end against a fake seed gateway.

Regression for the 2026-09-12 silent mesh on the live fleet — every announce
raised inside the best-effort catch (str join key decoded as bytes) and the
reannounce endpoint had no stored announcement. Neither the unit tests nor the
handler tests exercised the real thread, so this one does.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from lobes.gateway._config import build_config
from lobes.gateway.server import build_mesh_wiring


def _seed(hits: list) -> tuple[HTTPServer, int]:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_a):  # noqa: D401
            pass

        def _send(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/mesh/roster"):
                self._send(b'{"members": [], "ledger": {}}')
            else:
                self._send(b'{"mesh": true, "name": "seed", "schema_version": 1}')

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            hits.append((self.path, self.headers.get("Authorization"), self.rfile.read(n)))
            self._send(b'{"status": "ok"}')

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_heartbeat_posts_a_real_announcement_to_the_seed_with_the_join_key() -> None:
    hits: list = []
    srv, port = _seed(hits)
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "unsloth/Qwen3.8-27B-NVFP4",
        "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
        "LOBES_MESH_KEY": "sk-test",
        "LOBES_MESH_NAME": "me",
        "LOBES_MESH_SEEDS": f"http://127.0.0.1:{port}",
        "LOBES_MESH_HEARTBEAT_S": "1",
    }
    table, cfg = build_config(env)
    routes, _holder = build_mesh_wiring(table, cfg, None, {}, env=env)
    try:
        assert routes._announcement_bytes is not None  # stored at start, for reannounce
        deadline = time.monotonic() + 8.0
        while not hits and time.monotonic() < deadline:
            time.sleep(0.05)
        assert hits, "no announce reached the seed within 8 s"
        path, auth, body = hits[0]
        assert path == "/mesh/announce"
        assert auth == "Bearer sk-test"
        ann = json.loads(body)
        assert ann["name"] == "me"
        assert ann["origin"] == "http://me.local:8000"
        assert "cortex" in ann["roles"]  # role names, never backend names
    finally:
        routes._stop.set()
        srv.shutdown()


def test_a_slow_peer_probe_never_blocks_the_roster_or_inbound_announces() -> None:
    """Regression: the verification pass ran under routes._lock (live Spark, dev518)."""
    import io

    from lobes.gateway._mesh_wire import Announcement, Fingerprint, RoleInfo, encode

    hits: list = []
    srv, port = _seed(hits)
    # A member whose /capabilities probe is SLOW: 2.5 s per dial.
    slow_hits: list = []

    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            slow_hits.append(self.path)
            time.sleep(2.5)
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    slow = HTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=slow.serve_forever, daemon=True).start()
    slow_origin = f"http://127.0.0.1:{slow.server_address[1]}"
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "unsloth/Qwen3.8-27B-NVFP4",
        "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
        "LOBES_MESH_KEY": "sk-test",
        "LOBES_MESH_NAME": "me",
        "LOBES_MESH_SEEDS": f"http://127.0.0.1:{port}",
        "LOBES_MESH_HEARTBEAT_S": "1",
    }
    table, cfg = build_config(env)
    routes, _holder = build_mesh_wiring(table, cfg, None, {}, env=env)

    class Req:
        def __init__(self, body: bytes):
            self.rfile = io.BytesIO(body)
            self.headers = {"Authorization": "Bearer sk-test", "Content-Length": str(len(body))}
            self.client_address = ("127.0.0.1", 1)

    ann = Announcement(
        name="slowbox",
        origin=slow_origin,
        schema_version="1",
        roles={
            "associate": RoleInfo(
                model="m",
                runtime="vllm",
                context=1,
                quant="q",
                responsibilities=(),
                forbidden_responsibilities=(),
                fingerprint=Fingerprint(
                    served_id="m", quantization="q", max_model_len=1, runtime="vllm"
                ),
            )
        },
    )
    try:
        assert routes.announce(Req(encode(ann)))[0] == 200
        # Let the heartbeat start a verification pass against the slow member.
        deadline = time.monotonic() + 6.0
        while not slow_hits and time.monotonic() < deadline:
            time.sleep(0.05)
        assert slow_hits, "the verification pass never probed the member"
        # While that probe is in flight, the roster must still answer fast.
        t0 = time.monotonic()
        status, _h, body = routes.roster_list(Req(b""))
        assert status == 200
        assert (time.monotonic() - t0) < 1.0
        assert any(m["name"] == "slowbox" for m in json.loads(body)["members"])
        # ...and a second inbound announce must not be blocked either.
        t0 = time.monotonic()
        assert routes.announce(Req(encode(ann)))[0] == 200
        assert (time.monotonic() - t0) < 1.0
    finally:
        routes._stop.set()
        srv.shutdown()
        slow.shutdown()


def test_a_non_empty_seed_roster_is_merged_without_deadlocking_the_roster() -> None:
    """Regression: the seed merge wrapped Roster.announce in the roster's own
    non-reentrant lock — the heartbeat deadlocked on the first real seed roster
    (live Orin, 2026-09-12) and /mesh/roster hung forever."""
    import io

    from lobes.gateway._mesh_routes import _fetch_seed_roster

    class Seed(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            body = json.dumps(
                {
                    "members": [
                        {"name": "peerbox", "origin": "http://peer.local:8000", "capacity": 1.0}
                    ],
                    "ledger": {},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), Seed)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    seed = f"http://127.0.0.1:{srv.server_address[1]}"
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "unsloth/Qwen3.8-27B-NVFP4",
        "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
        "LOBES_MESH_KEY": "sk-test",
        "LOBES_MESH_NAME": "me",
    }
    table, cfg = build_config(env)
    routes, _ = build_mesh_wiring(table, cfg, None, {}, start=False, env=env)
    done: list = []

    def run():
        _fetch_seed_roster([seed], "sk-test", routes.roster, timeout=3.0, routes=routes)
        done.append(True)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(5.0)
    try:
        assert done, "seed roster merge deadlocked"
        assert "peerbox" in routes.roster.members()

        class Req:
            rfile = io.BytesIO(b"")
            headers = {"Authorization": "Bearer sk-test", "Content-Length": "0"}
            client_address = ("127.0.0.1", 1)

        t0 = time.monotonic()
        status, _h, body = routes.roster_list(Req())
        assert status == 200
        assert (time.monotonic() - t0) < 1.0
        assert any(m["name"] == "peerbox" for m in json.loads(body)["members"])
    finally:
        srv.shutdown()


def test_announced_and_advertised_fingerprints_carry_a_known_runtime() -> None:
    """Regression: without a declared pool the lane is never live-probed, so every
    fingerprint read runtime=unknown and the unknown rule made verification
    impossible on the live fleet (2026-09-12)."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "unsloth/Qwen3.8-27B-NVFP4",
        "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
        "LOBES_MESH_KEY": "sk-test",
        "LOBES_MESH_NAME": "me",
    }
    table, cfg = build_config(env)
    routes, _ = build_mesh_wiring(table, cfg, None, {}, start=False, env=env)
    fresh = routes._announcement_builder()
    assert fresh.roles["cortex"].fingerprint.runtime == "vllm"


def test_a_box_never_lists_itself_via_announce_or_seed_merge() -> None:
    """Regression: the live Spark's roster listed 'spark' after merging a seed
    roster that (correctly) listed it as a member."""
    import io

    from lobes.gateway._mesh_routes import _fetch_seed_roster
    from lobes.gateway._mesh_wire import Announcement, encode

    class Seed(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            body = json.dumps(
                {
                    "members": [
                        {"name": "me", "origin": "http://me.local:8000", "capacity": 1.0},
                        {"name": "other", "origin": "http://other.local:8000", "capacity": 1.0},
                    ],
                    "ledger": {},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), Seed)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "unsloth/Qwen3.8-27B-NVFP4",
        "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
        "LOBES_MESH_KEY": "sk-test",
        "LOBES_MESH_NAME": "me",
    }
    table, cfg = build_config(env)
    routes, _ = build_mesh_wiring(table, cfg, None, {}, start=False, env=env)
    try:
        _fetch_seed_roster(
            [f"http://127.0.0.1:{srv.server_address[1]}"],
            "sk-test",
            routes.roster,
            timeout=3.0,
            routes=routes,
        )
        assert "other" in routes.roster.members()
        assert "me" not in routes.roster.members()

        class Req:
            def __init__(self, body: bytes):
                self.rfile = io.BytesIO(body)
                self.headers = {"Authorization": "Bearer sk-test", "Content-Length": str(len(body))}
                self.client_address = ("127.0.0.1", 1)

        ann = Announcement(name="me", origin="http://me.local:8000", schema_version="1", roles={})
        status, _h, body = routes.announce(Req(encode(ann)))
        assert status == 409
        assert json.loads(body)["error"]["type"] == "mesh_name_conflict"
        assert "me" not in routes.roster.members()
    finally:
        srv.shutdown()


def test_seed_merge_never_refreshes_a_known_member_s_liveness() -> None:
    """Regression: a stopped member stayed alive mesh-wide because every seed
    roster still listed it and the merge re-announced it each tick."""
    from lobes.gateway._mesh_roster import Roster
    from lobes.gateway._mesh_routes import _fetch_seed_roster

    class Seed(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            body = json.dumps(
                {
                    "members": [
                        {"name": "dead", "origin": "http://dead.local:8000", "capacity": 1.0}
                    ],
                    "ledger": {},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), Seed)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    clock = {"t": 0.0}
    roster = Roster(clock=lambda: clock["t"], missed_max=2)
    roster.announce("dead", "http://dead.local:8000", None, now=0.0)
    try:
        for _ in range(4):  # four ticks, each preceded by a seed merge that still lists 'dead'
            clock["t"] += 60.0
            _fetch_seed_roster(
                [f"http://127.0.0.1:{srv.server_address[1]}"],
                "sk-test",
                roster,
                timeout=3.0,
                routes=None,
            )
            roster.tick(now=clock["t"])
        assert "dead" not in roster.members(), "a seed listing must not keep a silent member alive"
    finally:
        srv.shutdown()


def test_a_dropped_member_is_not_revived_by_a_peer_roster_that_still_lists_it() -> None:
    """Regression (live, dev526, 2026-09-12): the heartbeat pass ticks FIRST and
    fetches seed rosters SECOND, so the very pass that dropped the stopped
    Thor re-learned it from the Orin's roster — which still listed it — and
    the two survivors revived the dead member for each other forever
    (``thor[v=False]`` never left the Spark roster in 260 s)."""
    from lobes.gateway._mesh_roster import Roster
    from lobes.gateway._mesh_routes import _fetch_seed_roster

    class Seed(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            body = json.dumps(
                {
                    "members": [
                        {"name": "dead", "origin": "http://dead.local:8000", "capacity": 1.0}
                    ],
                    "ledger": {},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), Seed)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    clock = {"t": 0.0}
    roster = Roster(clock=lambda: clock["t"], missed_max=2)
    roster.announce("dead", "http://dead.local:8000", None, now=0.0)
    seen_absent = False
    try:
        for _ in range(6):  # the LIVE pass order: tick, then merge the seed rosters
            clock["t"] += 60.0
            roster.tick(now=clock["t"])
            if "dead" not in roster.members():
                seen_absent = True
            _fetch_seed_roster(
                [f"http://127.0.0.1:{srv.server_address[1]}"],
                "sk-test",
                roster,
                timeout=3.0,
                routes=None,
            )
        assert seen_absent, "missed_max=2 must drop a silent member within two ticks"
        assert "dead" not in roster.members(), "a peer roster must not revive a dropped member"
        # A heartbeat FROM the member itself always re-admits it, hold-down or not.
        roster.announce("dead", "http://dead.local:8000", None, now=clock["t"])
        assert "dead" in roster.members()
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# t2 — the boot window closes on its own: a pass on the loop's FIRST
# iteration, an event-woken (never handler-run) verify pass on announce and
# on seed discovery, and single-flight verification.
# ---------------------------------------------------------------------------


def _peer(probe_log: list, *, delay: float = 0.0, label: str = "peer") -> tuple[HTTPServer, str]:
    """A fake peer gateway that records every ``/capabilities`` probe.

    Each peer gets its OWN single-threaded server, so a slow peer stalls only
    its own probe — exactly the live shape the verification pool assumes.
    """

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_a):  # noqa: D401
            pass

        def _send(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/capabilities"):
                probe_log.append((label, time.monotonic()))
                if delay:
                    time.sleep(delay)
                self._send(b"{}")
            else:
                self._send(b'{"members": [], "ledger": {}}')

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(n)
            self._send(b'{"status": "ok"}')

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _env(**over) -> dict:
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "unsloth/Qwen3.8-27B-NVFP4",
        "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
        "LOBES_MESH_KEY": "sk-test",
        "LOBES_MESH_NAME": "me",
    }
    env.update({k: str(v) for k, v in over.items()})
    return env


def _peer_announcement(name: str, origin: str, *, served: str = "m"):
    from lobes.gateway._mesh_wire import Announcement, Fingerprint, RoleInfo

    return Announcement(
        name=name,
        origin=origin,
        schema_version="1",
        roles={
            "associate": RoleInfo(
                model=served,
                runtime="vllm",
                context=1,
                quant="q",
                responsibilities=(),
                forbidden_responsibilities=(),
                fingerprint=Fingerprint(
                    served_id=served, quantization="q", max_model_len=1, runtime="vllm"
                ),
            )
        },
    )


class _Req:
    def __init__(self, body: bytes):
        import io

        self.rfile = io.BytesIO(body)
        self.headers = {"Authorization": "Bearer sk-test", "Content-Length": str(len(body))}
        self.client_address = ("127.0.0.1", 1)


def _wiring(env: dict):
    table, cfg = build_config(env)
    return build_mesh_wiring(table, cfg, None, {}, start=False, env=env)


def test_the_first_capabilities_probe_lands_within_one_second_of_start() -> None:
    """Criterion 1: the loop's FIRST iteration runs a pass (it used to
    `continue`, so the very first verification waited a whole heartbeat
    interval and the boot window was at least one tick long)."""
    from lobes.gateway._mesh_routes import start_mesh

    probes: list = []
    peer_srv, peer_origin = _peer(probes)
    env = _env(LOBES_MESH_HEARTBEAT_S=1)
    routes, _holder = _wiring(env)
    routes.roster.announce("peerbox", peer_origin, 1.0)
    routes._announcements[peer_origin] = _peer_announcement("peerbox", peer_origin)
    try:
        t0 = time.monotonic()
        start_mesh(routes, routes._announcement_builder())
        deadline = t0 + 5.0
        while not probes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert probes, "no /capabilities probe within 5 s of start"
        assert probes[0][1] - t0 < 1.0, f"first probe took {probes[0][1] - t0:.3f}s"
    finally:
        routes._stop.set()
        peer_srv.shutdown()


def test_an_announce_during_a_slow_probe_returns_fast_and_is_probed_before_the_next_tick() -> None:
    """Criterion 2: POST /mesh/announce answers in < 50 ms while a 3 s probe
    is in flight, and the announcing member is probed long before the next
    periodic tick (the heartbeat here is 30 s, so a tick-driven probe is
    impossible within the window this asserts)."""
    from lobes.gateway._mesh_routes import start_mesh
    from lobes.gateway._mesh_wire import encode

    slow_probes: list = []
    fast_probes: list = []
    slow_srv, slow_origin = _peer(slow_probes, delay=3.0, label="slow")
    fast_srv, fast_origin = _peer(fast_probes, label="fast")
    env = _env(LOBES_MESH_HEARTBEAT_S=30)
    routes, _holder = _wiring(env)
    routes.roster.announce("slowbox", slow_origin, 1.0)
    routes._announcements[slow_origin] = _peer_announcement("slowbox", slow_origin)
    try:
        start_mesh(routes, routes._announcement_builder())
        deadline = time.monotonic() + 5.0
        while not slow_probes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert slow_probes, "the first pass never probed the slow member"

        body = encode(_peer_announcement("fastbox", fast_origin))
        t0 = time.monotonic()
        status, _h, _b = routes.announce(_Req(body))
        elapsed = time.monotonic() - t0
        assert status == 200
        assert elapsed < 0.05, f"announce blocked for {elapsed:.3f}s"

        deadline = time.monotonic() + 15.0
        while not fast_probes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fast_probes, "the announcing member was never probed"
        assert fast_probes[0][1] - t0 < 15.0  # << the 30 s tick: event-driven, not tick-driven
    finally:
        routes._stop.set()
        slow_srv.shutdown()
        fast_srv.shutdown()


def test_a_member_discovered_from_a_seed_roster_is_probed_without_waiting_for_a_tick() -> None:
    """Criterion 2 (seed half): discovery sets the verify-now event, so the
    newly learned member is probed in the pass that immediately follows the
    seed merge rather than one heartbeat later (30 s here).

    The member's announcement is pre-seeded because `_collect_members_to_verify`
    only probes members that HAVE one — on a live mesh that announcement
    arrives from the member's own heartbeat; here it is placed directly so the
    test isolates the discovery wake.
    """
    from lobes.gateway._mesh_routes import start_mesh

    probes: list = []
    peer_srv, peer_origin = _peer(probes)

    class Seed(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            body = json.dumps(
                {
                    "members": [{"name": "peerbox", "origin": peer_origin, "capacity": 1.0}],
                    "ledger": {},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    seed_srv = HTTPServer(("127.0.0.1", 0), Seed)
    threading.Thread(target=seed_srv.serve_forever, daemon=True).start()
    env = _env(
        LOBES_MESH_HEARTBEAT_S=30,
        LOBES_MESH_SEEDS=f"http://127.0.0.1:{seed_srv.server_address[1]}",
    )
    routes, _holder = _wiring(env)
    routes._announcements[peer_origin] = _peer_announcement("peerbox", peer_origin)
    try:
        t0 = time.monotonic()
        start_mesh(routes, routes._announcement_builder())
        deadline = t0 + 10.0
        while not probes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert probes, "a seed-discovered member was never probed"
        assert probes[0][1] - t0 < 10.0  # << the 30 s tick
        assert "peerbox" in routes.roster.members()
    finally:
        routes._stop.set()
        seed_srv.shutdown()
        peer_srv.shutdown()


def test_repeated_announces_do_not_reprobe_beyond_the_first_announce_plus_one_per_tick() -> None:
    """Criterion 2 (storm half): three members announcing every second for 6 s
    are each probed at most once per member's first announce (the events
    coalesce, so this is an upper bound of three passes) plus once per tick —
    never once per announce.  Ungated, the 18 announces in this window would
    drive up to 18 passes.
    """
    from lobes.gateway._mesh_routes import start_mesh
    from lobes.gateway._mesh_wire import encode

    probes: list = []
    servers = []
    peers = []
    for i in range(3):
        srv, origin = _peer(probes, label=f"p{i}")
        servers.append(srv)
        peers.append((f"box{i}", origin))
    env = _env(LOBES_MESH_HEARTBEAT_S=3)
    routes, _holder = _wiring(env)
    try:
        start_mesh(routes, routes._announcement_builder())
        end = time.monotonic() + 6.0
        while time.monotonic() < end:
            for name, origin in peers:
                assert routes.announce(_Req(encode(_peer_announcement(name, origin))))[0] == 200
            time.sleep(1.0)
        counts = {label: 0 for label, _ in peers}
        for label, _ts in probes:
            idx = int(label[1:])
            counts[peers[idx][0]] += 1
        assert all(c >= 1 for c in counts.values()), counts
        # 3 first-announce wakes (worst case, uncoalesced) + 2 ticks + 1 slack.
        assert all(c <= 6 for c in counts.values()), counts
    finally:
        routes._stop.set()
        for srv in servers:
            srv.shutdown()


def test_verification_passes_never_overlap_and_run_only_on_the_heartbeat_thread() -> None:
    """Criterion 3: single flight.  No request handler ever runs a pass, and
    two passes never overlap even under an announce storm."""
    from unittest.mock import patch

    from lobes.gateway import _mesh_routes as mod
    from lobes.gateway._mesh_routes import start_mesh
    from lobes.gateway._mesh_wire import encode

    probes: list = []
    peer_srv, peer_origin = _peer(probes)
    state = {"active": 0, "max": 0}
    thread_names: set = set()
    guard = threading.Lock()

    def wrapper(members, key, timeout, log):
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            thread_names.add(threading.current_thread().name)
        try:
            time.sleep(0.3)
            return {}, {}, {}
        finally:
            with guard:
                state["active"] -= 1

    env = _env(LOBES_MESH_HEARTBEAT_S=1)
    routes, _holder = _wiring(env)
    routes.roster.announce("peerbox", peer_origin, 1.0)
    routes._announcements[peer_origin] = _peer_announcement("peerbox", peer_origin)
    try:
        with patch.object(mod, "_run_verification_probes", wrapper):
            start_mesh(routes, routes._announcement_builder())
            end = time.monotonic() + 3.0
            i = 0
            while time.monotonic() < end:
                i += 1
                ann = _peer_announcement("peerbox", peer_origin, served=f"m{i}")
                assert routes.announce(_Req(encode(ann)))[0] == 200
                time.sleep(0.05)
            routes._stop.set()
            time.sleep(0.5)
        assert state["max"] == 1, f"verification passes overlapped ({state['max']} at once)"
        assert thread_names == {"lobes-mesh-heartbeat"}, thread_names
    finally:
        routes._stop.set()
        peer_srv.shutdown()


def _replying_peer(probes: list, name: str) -> tuple[HTTPServer, str]:
    """A fake peer whose POST /mesh/announce reply carries ITS OWN announcement
    (d1) and whose /capabilities agrees with it, so the announcer can verify
    it on the very pass that discovered it."""
    from lobes.gateway._mesh_wire import encode

    holder: dict = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_a):  # noqa: D401
            pass

        def _send(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/capabilities"):
                probes.append((name, time.monotonic()))
                self._send(
                    json.dumps(
                        {
                            "associate": {
                                "ready": True,
                                "fingerprint": {
                                    "served_id": "m",
                                    "quantization": "q",
                                    "max_model_len": 1,
                                    "runtime": "vllm",
                                },
                            }
                        }
                    ).encode()
                )
            else:
                self._send(b'{"members": [], "ledger": {}}')

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(n)
            ann = json.loads(encode(_peer_announcement(name, holder["origin"])))
            self._send(json.dumps({"status": "announced", "announcement": ann}).encode())

    srv = HTTPServer(("127.0.0.1", 0), H)
    holder["origin"] = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, holder["origin"]


def test_a_cold_box_learns_a_seed_peers_announcement_from_the_reply_and_verifies_it() -> None:
    """d1 (measured gap, 2026-09-12): a recreated gateway holds NO peer
    announcements, so its first pass had nothing to verify and requests 404'd
    until each peer's next heartbeat (~60 s live). Now the seed's announce
    reply carries the seed's own announcement, and it is verified within the
    first seconds — with the heartbeat at 30 s, a tick-driven path cannot
    explain the timing this asserts."""
    from lobes.gateway._mesh_routes import start_mesh

    probes: list = []
    peer_srv, peer_origin = _replying_peer(probes, "seedbox")
    env = _env(LOBES_MESH_HEARTBEAT_S=30, LOBES_MESH_SEEDS=peer_origin)
    routes, holder = _wiring(env)
    try:
        t0 = time.monotonic()
        start_mesh(routes, routes._announcement_builder())
        deadline = t0 + 5.0
        while time.monotonic() < deadline:
            view = holder.current()
            snap = getattr(view, "snapshot", None)
            if snap is not None and any(m.name == "seedbox" and m.probed for m in snap.members):
                break
            time.sleep(0.02)
        view = holder.current()
        members = {m.name: m for m in view.snapshot.members}
        assert "seedbox" in members, "seed never entered the roster from its reply"
        assert members["seedbox"].probed, "seed was not probed within 5 s of start"
        assert "associate" in members["seedbox"].verified_roles
        assert probes and probes[0][1] - t0 < 5.0
    finally:
        routes._stop.set()
        peer_srv.shutdown()
