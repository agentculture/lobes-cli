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
        assert ann["name"] == "me" and ann["origin"] == "http://me.local:8000"
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
        assert status == 200 and (time.monotonic() - t0) < 1.0
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
        assert status == 200 and (time.monotonic() - t0) < 1.0
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
        assert "other" in routes.roster.members() and "me" not in routes.roster.members()

        class Req:
            def __init__(self, body: bytes):
                self.rfile = io.BytesIO(body)
                self.headers = {"Authorization": "Bearer sk-test", "Content-Length": str(len(body))}
                self.client_address = ("127.0.0.1", 1)

        ann = Announcement(name="me", origin="http://me.local:8000", schema_version="1", roles={})
        status, _h, body = routes.announce(Req(encode(ann)))
        assert status == 409 and json.loads(body)["error"]["type"] == "mesh_name_conflict"
        assert "me" not in routes.roster.members()
    finally:
        srv.shutdown()
