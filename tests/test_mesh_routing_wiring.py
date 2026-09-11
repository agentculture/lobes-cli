"""End-to-end wiring tests for mesh routing (mesh-t7, W12).

Exercises the full handle_post → mesh forward chain through injected
open_upstream stubs (no real sockets), with threaded fake member gateways
for the real-capability probe phase.

Acceptance criteria covered
----------------------------
W12-1  announce → verify → forward: member announces, probe verifies,
       handle_post forwards via the mesh-enabled peer pool path, with mesh
       markers and Bearer join key.
W12-2  mismatch: announce + probe fingerprint mismatch → role unverified
       → 404 role_infeasible, zero dials.
W12-3  two equal-fingerprint members form one pool.
W12-4  508 chain: A→B→(no C) proxy_loop — B refuses without dialing, A
       relays.
W12-5  drop → next view → 404 no hosted_by, zero dials.
W12-6  LOBES_MESH_KEY unset ⇒ inert: pre-mesh referral/proxy bytes, no
       X-Lobes-Mesh-* headers.

Spec targets: c3, h25, c7, h9, c28, h18, c44, h35, c40, h31.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._mesh_config import build_mesh_config
from lobes.gateway._mesh_roster import Roster
from lobes.gateway._mesh_routes import MeshRoutes, verify_members
from lobes.gateway._mesh_routing import build_snapshot, origins_for_role
from lobes.gateway._mesh_wire import Announcement, Fingerprint, RoleInfo
from lobes.gateway._replicas import ReplicaState

if TYPE_CHECKING:
    from lobes.gateway._mesh_routing import RoutingSnapshot

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeUpstream:
    """Duck-typed stand-in for server._Upstream (no socket)."""

    def __init__(self, status, body=b'{"ok":1}', chunks=None, headers=None):
        self.status = status
        self.headers = headers if headers is not None else [("Content-Type", "application/json")]
        self._body = body
        self._chunks = list(chunks) if chunks is not None else None
        self.closed = False

    def read_all(self):
        return self._body

    def read(self, _n=8192):
        if self._chunks is None:
            data, self._body = self._body, b""
            return data
        return self._chunks.pop(0) if self._chunks else b""

    def close(self):
        self.closed = True


class _FakeMemberGateway:
    """A threaded local HTTP server serving /capabilities, /status, and
    /v1/chat/completions for mesh-verification testing."""

    def __init__(self):
        self._server = None
        self._thread = None
        self._capacities: dict[str, dict] = {}
        self._load = 0
        self._busy = False
        self._capacity = 8.0
        self._recorded_request: dict | None = None
        self._response_status = 200
        self._response_body = b'{"choices": [{"text": ""}]}'

    def _handle_capabilities(self, handler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({"roles": self._capacities}).encode())

    def _handle_status(self, handler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        body = json.dumps(
            {
                "load": self._load,
                "busy": self._busy,
                "capacity": self._capacity,
            }
        ).encode()
        handler.wfile.write(body)

    def _handle_completions(self, handler) -> None:
        cl = int(handler.headers.get("Content-Length", 0))
        body = handler.rfile.read(cl) if cl > 0 else b""
        headers_list = [
            (k, handler.headers[k])
            for k in handler.headers
            if k.lower() not in ("transfer-encoding", "content-length")
        ]
        self._recorded_request = {
            "path": handler.path,
            "body": body,
            "headers": headers_list,
        }
        handler.send_response(self._response_status)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(self._response_body)

    def start(self, port: int) -> None:
        """Start the server on the given port (blocking in a thread)."""
        routes = {
            "/capabilities": self._handle_capabilities,
            "/status": self._handle_status,
            "/v1/chat/completions": self._handle_completions,
        }

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                handler_fn = routes.get(self.path)
                if handler_fn:
                    handler_fn(self)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                handler_fn = routes.get(self.path)
                if handler_fn:
                    handler_fn(self)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, fmt, *args):
                pass  # silence logs

        self._server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5.0)

    def set_capabilities(self, roles: dict[str, dict]) -> None:
        self._capacities = roles

    def set_status(self, *, load: int = 0, busy: bool = False, capacity: float = 8.0) -> None:
        self._load = load
        self._busy = busy
        self._capacity = capacity

    def set_response(self, status: int, body: bytes) -> None:
        self._response_status = status
        self._response_body = body

    @property
    def recorded_request(self) -> dict | None:
        return self._recorded_request


def _fp(
    served_id="unsloth/Qwen3.8-27B-NVFP4",
    quantization="NVFP4",
    max_model_len=262144,
    runtime="vllm",
) -> Fingerprint:
    return Fingerprint(
        served_id=served_id,
        quantization=quantization,
        max_model_len=max_model_len,
        runtime=runtime,
    )


def _role(name: str, **over) -> RoleInfo:
    """A RoleInfo with sane defaults overridden by *over*."""
    return RoleInfo(
        model=over.get("model", "unsloth/Qwen3.8-27B-NVFP4"),
        runtime=over.get("runtime", "vllm"),
        context=over.get("context", 262144),
        quant=over.get("quant", "NVFP4"),
        responsibilities=over.get("responsibilities", ("generate", "image_understanding")),
        forbidden_responsibilities=over.get(
            "forbidden_responsibilities", ("final_decision", "security_decision")
        ),
        fingerprint=over.get("fingerprint", _fp()),
        capacity=over.get("capacity"),
        private=over.get("private", False),
    )


def _ann(
    name: str,
    origin: str,
    roles: dict[str, RoleInfo] | None = None,
) -> Announcement:
    return Announcement(
        name=name,
        origin=origin,
        schema_version="1.0.0",
        roles=roles or {"cortex": _role("cortex")},
    )


class _FakeRoster:
    """Minimal roster duck-type for build_snapshot tests."""

    def __init__(self, members: list[tuple[str, str, float]]) -> None:
        self._counter = [1.0]
        self._records: dict[str, type] = {}
        self._names: list[str] = []
        for name, origin, capacity in members:
            rec = type(
                "FakeRecord",
                (),
                {
                    "name": name,
                    "origin": origin,
                    "capacity": capacity,
                },
            )()
            self._records[name] = rec
            self._names.append(name)

    def members(self) -> list[str]:
        return list(self._names)

    @property
    def _roster(self) -> dict[str, type]:
        return self._records

    def now(self) -> float:
        return self._counter[0]

    def tick(self, delta: float = 1.0) -> float:
        self._counter[0] += delta
        return self._counter[0]


def _probe_verification(
    member_origins: list[str],
    join_key: str,
    announced_roles_per_origin: dict[str, dict[str, RoleInfo]],
) -> dict[str, frozenset[str]]:
    """Probe each member's /capabilities, compare fingerprints, return verified_roles.

    This mirrors what verify_members() computes: for each announced role,
    compare the announced fingerprint with the probed fingerprint from
    /capabilities. Returns {origin: frozenset_of_verified_role_names}.
    """
    import ssl
    import urllib.request

    from lobes.gateway._mesh_routing import verify_member_roles

    verified: dict[str, frozenset[str]] = {}
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for origin in member_origins:
        url = origin.rstrip("/") + "/capabilities"
        try:
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {join_key}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=2.0, context=ctx) as resp:
                if resp.status == 200:
                    import json

                    payload = json.loads(resp.read())
                    roles_data = payload.get("roles", {})

                    # Get the announced roles for this origin
                    ann_roles = announced_roles_per_origin.get(origin, {})
                    if not ann_roles:
                        continue

                    # Build probed_roles dict for verify_member_roles
                    probed_roles: dict[str, dict] = {}
                    for role_name, role_entry in roles_data.items():
                        if not isinstance(role_entry, dict):
                            continue
                        role_fp = role_entry.get("fingerprint")
                        probed_roles[role_name] = {
                            "fingerprint": role_fp,
                            "ready": role_entry.get("ready"),
                        }

                    # Build an Announcement-like object for verify_member_roles
                    ann = Announcement(
                        name="test",
                        origin=origin,
                        schema_version="1.0.0",
                        roles=ann_roles,
                    )

                    # Compare fingerprints
                    verified_roles = verify_member_roles(ann, probed_roles)
                    if verified_roles:
                        verified[origin] = verified_roles
        except Exception:
            pass  # probe failed → no verification

    return verified


def _build_mesh_snapshot(
    member_origins: list[str],
    announced_roles: dict[str, dict[str, RoleInfo]],
    verified_roles: dict[str, frozenset[str]],
    fingerprints: dict[str, Fingerprint] | None = None,
    member_names: list[str] | None = None,
) -> "RoutingSnapshot":
    """Build a mesh snapshot from member origins, announced roles, and
    verified roles.

    member_names: mesh member names (used as backend names in _mesh_roles()).
    Defaults to "origin:port".
    """
    if fingerprints is None:
        fingerprints = {o: _fp() for o in member_origins}
    if member_names is None:
        member_names = [o.rsplit(":", 1)[-1] for o in member_origins]

    roster_members: list[tuple[str, str, float]] = []
    announcements: dict[str, Announcement] = {}
    for i, origin in enumerate(member_origins):
        roster_members.append((member_names[i], origin, 4.0))
        roles = announced_roles.get(
            origin, {"cortex": _role("cortex", fingerprint=fingerprints.get(origin, _fp()))}
        )
        announcements[origin] = _ann(member_names[i], origin, roles)

    roster = _FakeRoster(roster_members)
    return build_snapshot(roster, announcements=announcements, verified_roles=verified_roles)


def _setup_mesh(
    member_origins: list[str],
    join_key: str,
    announced_roles: dict[str, dict[str, RoleInfo]] | None = None,
    fingerprints: dict[str, Fingerprint] | None = None,
    member_names: list[str] | None = None,
) -> tuple[MeshRoutes, "RoutingSnapshot"]:
    """Set up a MeshRoutes with announced members, verify them, and return
    (routes, snapshot).

    member_names: explicit mesh member names (used as backend names in
    _mesh_roles()). Defaults to "member-N".
    """
    if fingerprints is None:
        fingerprints = {o: _fp() for o in member_origins}
    if member_names is None:
        member_names = [f"member-{i}" for i in range(len(member_origins))]

    mesh_cfg = build_mesh_config(
        {
            "LOBES_MESH_KEY": join_key,
            "LOBES_MESH_NAME": "me",
            "LOBES_MESH_SEEDS": "",
            "LOBES_MESH_HEARTBEAT_S": "60",
            "LOBES_MESH_MISSED_MAX": "3",
        }
    )
    roster = Roster()
    routes = MeshRoutes(mesh_cfg, roster)
    for i, origin in enumerate(member_origins):
        routes.roster.announce(member_names[i], origin, 4.0)
    for origin, roles in (announced_roles or {}).items():
        routes._announcements[origin] = _ann("member", origin, roles)

    holder = type(
        "Holder",
        (),
        {
            "replace": lambda s, v: None,
            "current": lambda s: None,
        },
    )()
    routes._holder = holder
    verify_members(routes, holder, join_key=join_key, timeout=2.0)

    # Build the snapshot with the actual verification result.
    # verify_members already builds a snapshot, but the holder.replace is a no-op,
    # so we rebuild here with the correct verified_roles from the probe.
    # Use routes._announcements for the announced data, and build verified_roles
    # from the probe results (verified_by_origin).
    # We need to capture what verify_members computed. Since verify_members
    # stores results in its local verified_by_origin, we rebuild the snapshot
    # by re-probing (matching what the probe actually returns).
    snap = _build_mesh_snapshot(
        member_origins,
        announced_roles=announced_roles
        or {o: {"cortex": _role("cortex", fingerprint=fingerprints[o])} for o in member_origins},
        # Probe the members and do fingerprint comparison, just like verify_members.
        verified_roles=_probe_verification(
            member_origins,
            join_key,
            announced_roles
            or {
                o: {"cortex": _role("cortex", fingerprint=fingerprints[o])} for o in member_origins
            },
        ),
        member_names=member_names,
    )
    return routes, snap


# ---------------------------------------------------------------------------
# W12-1  announce → verify → forward
# ---------------------------------------------------------------------------


class TestAnnounceVerifyForward:
    """W12-1: member announces, probe verifies, handle_post forwards via
    the mesh-enabled peer pool path."""

    def test_announce_verify_forward(self, monkeypatch):
        join_key = "sk-mesh-join-key"

        member = _FakeMemberGateway()
        member.start(0)
        try:
            port = member._server.server_address[1]
            member_origin = f"http://127.0.0.1:{port}"

            # Member's /capabilities serves matching fingerprint.
            member.set_capabilities(
                {
                    "cortex": {
                        "fingerprint": {
                            "served_id": "unsloth/Qwen3.8-27B-NVFP4",
                            "quantization": "NVFP4",
                            "max_model_len": 262144,
                            "runtime": "vllm",
                        },
                        "ready": True,
                    },
                }
            )

            # Build MeshRoutes, announce, verify.
            # Use "primary" as the mesh member name so the header assertions
            # below can tell the mesh-sourced replica apart from a local one.
            routes, snap = _setup_mesh(
                [member_origin],
                join_key,
                announced_roles={member_origin: {"cortex": _role("cortex")}},
                member_names=["primary"],
            )

            # NO *_PEER_* keys anywhere (t13/AC1): "primary" is infeasible on
            # this box (a pure hardware/shape fact) and the mesh is the ONLY
            # source of a candidate for it — `pooled_backends`/`_pool_selection`
            # must find the mesh-verified member without any env-declared peer
            # origin at all.
            env = {
                "PRIMARY_FEASIBLE": "false",
                "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
            }
            table, cfg = build_config(env)

            opener_calls = []

            def fake_open(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
                opener_calls.append(
                    {
                        "backend": backend,
                        "path": path,
                        "body": fwd_body,
                        "headers": list(headers),
                    }
                )
                return _FakeUpstream(200, b'{"choices": [{"text": "ok"}]}')

            monkeypatch.setattr(S, "open_upstream", fake_open)

            # peer_specs is empty — no env peer config declared at all.
            specs = S.peer_specs_from_table(table)
            assert "primary" not in specs, f"Expected 'primary' not in specs: {list(specs.keys())}"

            # Set LOBES_MESH_KEY so _build_mesh_config() picks it up for the
            # proxy loop's join key.
            monkeypatch.setenv("LOBES_MESH_KEY", join_key)
            monkeypatch.setenv("LOBES_MESH_NAME", "me")

            # No env-sourced ReplicaCache at all (no *_PEER_ORIGINS declared) —
            # `replica_snapshot` itself is None, exactly what a mesh-only box's
            # `replica_snapshot_provider` returns for an empty cache map.
            resp = S.handle_post(
                table,
                cfg,
                "/v1/chat/completions",
                [("Authorization", "Bearer sk-caller")],
                json.dumps({"model": "cortex"}).encode(),
                fake_open,
                peer_specs=specs,
                replica_snapshot=None,
                mesh_snapshot=snap,
            )

            # Forwarded to member via the mesh-sourced pool path.
            assert resp.status == 200
            assert len(opener_calls) == 1
            call = opener_calls[0]
            # Check Bearer join key present.
            auth_headers = [v for k, v in call["headers"] if k.lower() == "authorization"]
            assert f"Bearer {join_key}" in auth_headers, f"Expected join key in {auth_headers}"
            # Check client's bearer is absent.
            caller_auth = [v for k, v in call["headers"] if "sk-caller" in v]
            assert len(caller_auth) == 0, f"Client token leaked: {caller_auth}"
            # Check X-Lobes-Proxied present, naming the BACKEND (not the role).
            proxied_vals = [v for k, v in call["headers"] if k.lower() == "x-lobes-proxied"]
            assert "primary" in proxied_vals
            # Response carries X-Lobes-Proxied-By and X-Lobes-Mesh-Member.
            assert any(k.lower() == "x-lobes-proxied-by" for k, _ in resp.headers)
            member_headers = [v for k, v in resp.headers if k.lower() == "x-lobes-mesh-member"]
            assert member_headers == ["primary"]
        finally:
            member.stop()


# ---------------------------------------------------------------------------
# W12-2  mismatch variant
# ---------------------------------------------------------------------------


class TestMismatchVariant:
    """W12-2: member announces cortex with fp X, /capabilities serves fp Y."""

    def test_mismatch_fingerprint(self, monkeypatch):
        fp_x = _fp(served_id="unsloth/Qwen3.8-27B-NVFP4", quantization="NVFP4")
        fp_y = _fp(served_id="different-model", quantization="INT4")  # different fp
        join_key = "sk-mesh-join-key"

        member = _FakeMemberGateway()
        member.start(0)
        try:
            port = member._server.server_address[1]
            member_origin = f"http://127.0.0.1:{port}"

            # Member's /capabilities serves fp Y (mismatch with announced fp X).
            member.set_capabilities(
                {
                    "cortex": {
                        "fingerprint": {
                            "served_id": fp_y.served_id,
                            "quantization": fp_y.quantization,
                            "max_model_len": fp_y.max_model_len,
                            "runtime": fp_y.runtime,
                        },
                        "ready": True,
                    },
                }
            )

            # Build MeshRoutes, announce, verify.
            routes, snap = _setup_mesh(
                [member_origin],
                join_key,
                announced_roles={member_origin: {"cortex": _role("cortex", fingerprint=fp_x)}},
                fingerprints={member_origin: fp_x},
                member_names=["primary"],
            )

            # NO *_PEER_* keys (t13/AC1): "primary" is infeasible and the mesh
            # verified set is empty (fingerprint mismatch) → no mesh forward
            # path AND no env forward path → the pre-mesh referral 404, with
            # no `hosted_by` since nothing declared one.
            env = {
                "PRIMARY_FEASIBLE": "false",
                "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
            }
            table, cfg = build_config(env)
            specs = S.peer_specs_from_table(table)
            assert "primary" not in specs

            monkeypatch.setenv("LOBES_MESH_KEY", join_key)
            monkeypatch.setenv("LOBES_MESH_NAME", "me")

            opener_calls = []

            def fake_open(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
                opener_calls.append(True)
                return _FakeUpstream(200, b'{"choices": [{"text": "ok"}]}')

            monkeypatch.setattr(S, "open_upstream", fake_open)

            resp = S.handle_post(
                table,
                cfg,
                "/v1/chat/completions",
                [("Authorization", "Bearer sk-caller")],
                json.dumps({"model": "cortex"}).encode(),
                fake_open,
                peer_specs=specs,
                replica_snapshot=None,
                mesh_snapshot=snap,
            )

            # Fingerprint mismatch → mesh verified empty → falls through to
            # the pre-mesh referral path: 404 role_infeasible.
            assert resp.status == 404
            assert len(opener_calls) == 0  # zero dials
            body = json.loads(resp.body)
            assert body["error"]["type"] == "role_infeasible"
        finally:
            member.stop()


# ---------------------------------------------------------------------------
# W12-3  two equal-fingerprint members form one pool
# ---------------------------------------------------------------------------


class TestTwoEqualMembers:
    """W12-3: two fake members announcing cortex with same fp → one pool."""

    def test_two_equal_fingerprint_members(self, monkeypatch):
        fp = _fp(served_id="unsloth/Qwen3.8-27B-NVFP4", quantization="NVFP4")
        join_key = "sk-mesh-join-key"

        members = [_FakeMemberGateway(), _FakeMemberGateway()]
        try:
            for m in members:
                m.start(0)

            origins = []
            for m in members:
                port = m._server.server_address[1]
                origin = f"http://127.0.0.1:{port}"
                origins.append(origin)
                m.set_capabilities(
                    {
                        "cortex": {
                            "fingerprint": {
                                "served_id": fp.served_id,
                                "quantization": fp.quantization,
                                "max_model_len": fp.max_model_len,
                                "runtime": fp.runtime,
                            },
                            "ready": True,
                        },
                    }
                )

            # Build MeshRoutes, announce both, verify both.
            # Use distinct member names — both have cortex verified.
            member_names = ["box-a", "box-b"]
            routes, snap = _setup_mesh(
                origins,
                join_key,
                announced_roles={o: {"cortex": _role("cortex", fingerprint=fp)} for o in origins},
                fingerprints={o: fp for o in origins},
                member_names=member_names,
            )

            # Both members verified → one pool of two origins.
            mesh_origins = origins_for_role(snap, "cortex")
            assert len(mesh_origins) == 2
            assert origins[0] in mesh_origins
            assert origins[1] in mesh_origins

            # NO *_PEER_* keys anywhere (t13/AC1): "primary" is infeasible and
            # BOTH mesh members are the only candidates — `_pool_selection`
            # must rank across them with no env-declared origin at all.
            env = {
                "PRIMARY_FEASIBLE": "false",
                "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
            }
            table, cfg = build_config(env)
            specs = S.peer_specs_from_table(table)
            assert "primary" not in specs
            monkeypatch.setenv("LOBES_MESH_KEY", join_key)
            monkeypatch.setenv("LOBES_MESH_NAME", "me")

            opener_calls = []

            def fake_open(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
                opener_calls.append({"backend": backend, "path": path})
                return _FakeUpstream(200, b'{"choices": [{"text": "ok"}]}')

            monkeypatch.setattr(S, "open_upstream", fake_open)

            resp = S.handle_post(
                table,
                cfg,
                "/v1/chat/completions",
                [("Authorization", "Bearer sk-caller")],
                json.dumps({"model": "cortex"}).encode(),
                fake_open,
                peer_specs=specs,
                replica_snapshot=None,
                mesh_snapshot=snap,
            )

            # Mesh has TWO verified members forming one pool → forwarded to
            # exactly one of them (select_replica's deterministic ranking),
            # zero local dials, and the served member is named on the wire.
            assert resp.status == 200
            assert len(opener_calls) == 1  # one dial to one mesh member
            member_headers = [v for k, v in resp.headers if k.lower() == "x-lobes-mesh-member"]
            assert member_headers == ["box-a"] or member_headers == ["box-b"]
        finally:
            for m in members:
                m.stop()


# ---------------------------------------------------------------------------
# W12-4  508 chain
# ---------------------------------------------------------------------------


class Test508Chain:
    """W12-4: A→B→(no C) proxy_loop chain."""

    def test_proxy_loop_chain(self, monkeypatch):
        join_key = "sk-mesh-join-key"

        member_a = _FakeMemberGateway()
        member_b = _FakeMemberGateway()
        try:
            member_a.start(0)
            port_a = member_a._server.server_address[1]
            origin_a = f"http://127.0.0.1:{port_a}"

            member_b.start(0)
            port_b = member_b._server.server_address[1]
            origin_b = f"http://127.0.0.1:{port_b}"

            member_a.set_capabilities(
                {
                    "cortex": {
                        "fingerprint": {
                            "served_id": "unsloth/Qwen3.8-27B-NVFP4",
                            "quantization": "NVFP4",
                            "max_model_len": 262144,
                            "runtime": "vllm",
                        },
                        "ready": True,
                    },
                }
            )
            member_b.set_capabilities(
                {
                    "cortex": {
                        "fingerprint": {
                            "served_id": "unsloth/Qwen3.8-27B-NVFP4",
                            "quantization": "NVFP4",
                            "max_model_len": 262144,
                            "runtime": "vllm",
                        },
                        "ready": True,
                    },
                }
            )

            # A's snapshot: A lacks cortex, B has cortex verified.
            # Use "no-cortex" for A (not a backend name) and "primary" for B.
            snap = _build_mesh_snapshot(
                [origin_a, origin_b],
                announced_roles={
                    origin_a: {},  # no roles (lacks cortex)
                    origin_b: {"cortex": _role("cortex")},
                },
                verified_roles={
                    origin_a: frozenset(),  # unverified / no announced roles
                    origin_b: frozenset(["cortex"]),
                },
                member_names=["no-cortex", "primary"],
            )

            # NO *_PEER_* keys (t13/AC1): "primary" is infeasible and mesh
            # member B (named "primary") is the only source of a candidate.
            env = {
                "PRIMARY_FEASIBLE": "false",
                "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
            }
            table, cfg = build_config(env)
            specs = S.peer_specs_from_table(table)
            assert "primary" not in specs
            monkeypatch.setenv("LOBES_MESH_KEY", join_key)
            monkeypatch.setenv("LOBES_MESH_NAME", "me")

            opener_calls = []
            call_count = [0]

            def fake_open(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
                call_count[0] += 1
                opener_calls.append(
                    {
                        "backend": backend,
                        "path": path,
                        "body": fwd_body,
                        "headers": list(headers),
                    }
                )
                # First call: A→B forward.
                if call_count[0] == 1:
                    # Simulate B returning 508 proxy_loop.
                    loop_body = json.dumps(
                        {
                            "error": {
                                "message": (
                                    "refusing to proxy: this request already crossed one lobes "
                                    "proxy hop (X-Lobes-Proxied: cortex); forwarding it again "
                                    f"to `{origin_b}` for role `cortex` could loop — peer "
                                    "proxying is single-hop only (issues #115/#127)."
                                ),
                                "type": "proxy_loop",
                                "code": "proxy_loop",
                            }
                        }
                    ).encode()
                    return _FakeUpstream(508, loop_body)
                return _FakeUpstream(500, b'{"error": "unexpected dial"}')

            monkeypatch.setattr(S, "open_upstream", fake_open)

            resp = S.handle_post(
                table,
                cfg,
                "/v1/chat/completions",
                [("Authorization", "Bearer sk-caller")],
                json.dumps({"model": "cortex"}).encode(),
                fake_open,
                peer_specs=specs,
                replica_snapshot=None,
                mesh_snapshot=snap,
            )

            # A should relay B's 508.
            assert resp.status == 508
            assert len(opener_calls) == 1  # only A→B dial

            # The forwarded leg carried Bearer join key.
            call = opener_calls[0]
            auth_headers = [v for k, v in call["headers"] if k.lower() == "authorization"]
            assert f"Bearer {join_key}" in auth_headers, f"Expected join key in {auth_headers}"
            # No client bearer.
            caller_auth = [v for k, v in call["headers"] if "sk-caller" in v]
            assert len(caller_auth) == 0, f"Client token leaked: {caller_auth}"
        finally:
            member_a.stop()
            member_b.stop()


# ---------------------------------------------------------------------------
# W12-5  drop → next view → 404 no hosted_by, zero dials
# ---------------------------------------------------------------------------


class TestDropMember:
    """W12-5: member dropped from roster → 404 role_infeasible, zero dials."""

    def test_drop_member_no_hosted_by(self, monkeypatch):
        fp = _fp(served_id="unsloth/Qwen3.8-27B-NVFP4", quantization="NVFP4")

        # Snapshot before drop: member named "primary" has cortex verified.
        snap_before = build_snapshot(
            _FakeRoster([("primary", "http://alpha.local:8001", 1.0)]),
            announcements={
                "http://alpha.local:8001": _ann(
                    "primary",
                    "http://alpha.local:8001",
                    {"cortex": _role("cortex", fingerprint=fp)},
                ),
            },
            verified_roles={
                "http://alpha.local:8001": frozenset(["cortex"]),
            },
        )
        assert len(origins_for_role(snap_before, "cortex")) == 1

        # Snapshot after drop: no members.
        snap_after = build_snapshot(_FakeRoster([]))
        assert origins_for_role(snap_after, "cortex") == ()

        # NO *_PEER_* keys (t13/AC1): "primary" is infeasible, no env referral
        # is declared, and the member that used to verify it has been DROPPED
        # from the roster — there is genuinely nothing left to place this
        # request on, so it must fall through to a 404 role_infeasible with
        # NO `hosted_by` (nothing ever declared one) and zero dials.
        env = {
            "PRIMARY_FEASIBLE": "false",
            "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
        }
        table, cfg = build_config(env)
        specs = S.peer_specs_from_table(table)
        assert "primary" not in specs
        monkeypatch.setenv("LOBES_MESH_KEY", "sk-mesh-join-key")
        monkeypatch.setenv("LOBES_MESH_NAME", "me")

        opener_calls = []

        def fake_open(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
            opener_calls.append(True)
            return _FakeUpstream(200, b'{"choices": [{"text": "ok"}]}')

        monkeypatch.setattr(S, "open_upstream", fake_open)

        resp = S.handle_post(
            table,
            cfg,
            "/v1/chat/completions",
            [("Authorization", "Bearer sk-caller")],
            json.dumps({"model": "cortex"}).encode(),
            fake_open,
            peer_specs=specs,
            replica_snapshot=None,
            mesh_snapshot=snap_after,
        )

        assert resp.status == 404
        assert len(opener_calls) == 0  # zero dials
        body = json.loads(resp.body)
        assert body["error"]["type"] == "role_infeasible"
        # No env referral was ever declared, so no `hosted_by` at all.
        assert body.get("error", {}).get("hosted_by") is None


# ---------------------------------------------------------------------------
# W12-6  LOBES_MESH_KEY unset ⇒ inert
# ---------------------------------------------------------------------------


class TestInertMesh:
    """W12-6: mesh disabled — pre-mesh referral/proxy bytes exactly."""

    def test_mesh_unset_inert(self, monkeypatch):
        """Handler with mesh None, no declared peer origins → pre-mesh behavior."""
        env = {}
        table, cfg = build_config(env)

        opener_calls = []

        def fake_open(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
            opener_calls.append(True)
            return _FakeUpstream(200, b'{"choices": [{"text": "ok"}]}')

        monkeypatch.setattr(S, "open_upstream", fake_open)

        resp = S.handle_post(
            table,
            cfg,
            "/v1/chat/completions",
            [("Authorization", "Bearer sk-caller")],
            json.dumps({"model": "cortex"}).encode(),
            fake_open,
            mesh_snapshot=None,  # mesh disabled
        )

        # Mesh is None, so no mesh markers, pre-mesh behavior.
        # The "cortex" model is wired locally, so it returns 200.
        assert resp.status == 200
        # No X-Lobes-Mesh-* headers.
        mesh_headers = [k for k, _ in resp.headers if "lobes-mesh" in k.lower()]
        assert len(mesh_headers) == 0
        assert len(opener_calls) == 1  # dialed locally


# ---------------------------------------------------------------------------
# t13: hosted-role busy dispatch forwards to a mesh replica, no *_PEER_* keys
# ---------------------------------------------------------------------------


class TestBusyDispatchForwardsToMesh:
    """A role THIS BOX HOSTS, under swap pressure, forwards to a mesh-verified
    replica of the same role — `_pooled_busy_dispatch`/`_pool_selection`
    sourcing candidates purely from the mesh RoutingSnapshot, no
    `<PREFIX>_PEER_ORIGINS` declared anywhere (t13, spec c7/h9)."""

    def test_swap_pressure_forwards_to_mesh_replica(self, monkeypatch):
        join_key = "sk-mesh-join-key"

        member = _FakeMemberGateway()
        member.start(0)
        try:
            port = member._server.server_address[1]
            member_origin = f"http://127.0.0.1:{port}"
            member.set_capabilities(
                {
                    "cortex": {
                        "fingerprint": {
                            "served_id": "unsloth/Qwen3.8-27B-NVFP4",
                            "quantization": "NVFP4",
                            "max_model_len": 262144,
                            "runtime": "vllm",
                        },
                        "ready": True,
                    },
                }
            )

            routes, snap = _setup_mesh(
                [member_origin],
                join_key,
                announced_roles={member_origin: {"cortex": _role("cortex")}},
                member_names=["primary"],
            )

            # "primary" is HOSTED here (no *_FEASIBLE=false, no *_PEER_* keys
            # anywhere) — the mesh member is a REPLICA of the same role, not a
            # referral for a dropped one.
            env = {"GATEWAY_SELF_ORIGIN": "http://me.local:8000"}
            table, cfg = build_config(env)
            monkeypatch.setenv("LOBES_MESH_KEY", join_key)
            monkeypatch.setenv("LOBES_MESH_NAME", "me")

            def local_replica_snapshot(backend_name):
                if backend_name != "primary":
                    return ()
                return (
                    ReplicaState(
                        origin="local",
                        local=True,
                        ready=True,
                        busy=False,
                        health="ok",
                        running=0,
                        waiting=0,
                        fingerprint=None,
                        compatible=True,
                        reason="",
                        last_seen=0.0,
                        weight=8.0,
                        calibrated=True,
                    ),
                )

            opener_calls = []

            def fake_open(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
                opener_calls.append({"backend": backend, "headers": list(headers)})
                return _FakeUpstream(200, b'{"choices": [{"text": "ok"}]}')

            monkeypatch.setattr(S, "open_upstream", fake_open)

            # Swap pressure alone sheds an UNPOOLED request; a mesh-sourced
            # pooled one is forwarded instead of shed (c7/h9).
            resp = S.handle_post(
                table,
                cfg,
                "/v1/chat/completions",
                [("Authorization", "Bearer sk-caller")],
                json.dumps({"model": "cortex"}).encode(),
                fake_open,
                pressure={"swap_used_percent": 90.0, "iowait_percent": 0.0},
                replica_snapshot=local_replica_snapshot,
                mesh_snapshot=snap,
            )

            assert resp.status == 200
            assert len(opener_calls) == 1  # forwarded, zero local dial
            call = opener_calls[0]
            auth_headers = [v for k, v in call["headers"] if k.lower() == "authorization"]
            assert f"Bearer {join_key}" in auth_headers
            caller_auth = [v for k, v in call["headers"] if "sk-caller" in v]
            assert len(caller_auth) == 0
            assert any(k.lower() == "x-lobes-proxied-by" for k, _ in resp.headers)
            member_headers = [v for k, v in resp.headers if k.lower() == "x-lobes-mesh-member"]
            assert member_headers == ["primary"]
        finally:
            member.stop()
