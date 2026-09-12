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
