"""``lobes site`` HTTP server: the page and its read-only JSON API on one origin.

Pattern: heat-governor's ``mesh_server`` -- a stdlib ``ThreadingHTTPServer``
serving one ``index.html``, allow-listed ``/vendor/<file>`` assets, and JSON.

Routes (GET and HEAD; every other method is 405):

- ``/`` -- ``lobes/site/web/index.html``
- ``/vendor/<file>`` -- one plain filename from ``web/vendor/`` (no ``/``, no
  ``..``, allow-listed extension); anything else is 404
- ``/api/mesh`` -- members, versions, memory, roles, lobes (model ids), split
  brains, pairs and the outside-access hub
- ``/api/rollout`` -- the rollout journal's status projection, its step order, the
  resume point and the rollback command for a failed member
- ``/api/residents`` -- the residents file (detected / needs_you per field)
- ``/api/lobe-export?role=&member=`` -- the export bundle listing (no writes)

Every figure comes from an injected :class:`Sources` (live roster, ``/health``,
``/capabilities``, residents file, journal); nothing is hard-coded. Three
derived facts follow strict evidence rules, never a guess:

- **split brain** -- one model whose serving spans several members. Evidence is
  a member's own ``/capabilities`` entry: ``spans`` (member names), or a
  ``world_size`` N > 1 advertised for the same model by exactly N hosting
  members. Without evidence the model is reported single (or a pair).
- **pair** -- one model hosted (not mesh-proxied) by several members, a pool.
- **hub** -- the member running the outside tunnel, only when evidenced: a
  member's ``/health`` ``tunnel`` declaration (``LOBES_TUNNEL_PUBLIC_URL``,
  reported with ``declared: true``) or, for the local box only, a tunnel probe;
  otherwise ``null``.

Memory is reported only where a source measured it: a member's own ``/health``
``memory`` (its gateway reads ``/proc/meminfo``), falling back to this box's
``/proc/meminfo`` for the local member; anything else reads ``null``.

Every ``/api`` body is scrubbed with :func:`lobes.rollout._redact.redact`
against the deployment's secret values. The default bind is loopback and the
server is unauthenticated, so binding elsewhere exposes it to that network.
The site writes nothing.
"""

from __future__ import annotations

import json
import re
import subprocess  # nosec B404
import sys
import threading
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlsplit

from lobes.rollout import _export, _residents
from lobes.rollout._journal import FAILED, STEPS, Journal
from lobes.rollout._observe import MemberObservation, observe_member
from lobes.rollout._redact import redact
from lobes.rollout._rollback import rollback_command
from lobes.rollout._transport import Target

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9490
ROUTES = ("/", "/vendor/<file>", "/api/mesh", "/api/rollout", "/api/residents", "/api/lobe-export")
JSON_TYPE = "application/json"
MEMINFO = Path("/proc/meminfo")
MAX_EXPORT_CONCURRENCY = 2
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_VENDOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VENDOR_TYPES = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
}
_SECRET_KEY = re.compile(r"(?i)(key|token|secret|passw|credential)")
_CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'"
)
_NOT_FOUND = {"error": "not found"}


class ExportUnavailable(RuntimeError):
    """A lobe export could not be listed (member unreachable, no ssh target...)."""


@dataclass
class Sources:
    """Everything the site reads. Tests inject fixtures; :func:`default_sources` is live.

    ``observe(origin)`` -> MemberObservation; ``residents()`` -> Residents;
    ``journal()`` -> Journal-like (``status()``, ``resume_point()``);
    ``export_listing(role, member)`` -> dict; ``memory(name, is_local, health)``
    -> ``{"total_bytes", "available_bytes"}`` or None; ``tunnel(name, is_local,
    health)`` -> evidence string, ``{"evidence": str, "declared": bool}`` or
    None. ``health`` is the member's already-fetched ``/health`` body.
    """

    local_origin: str
    local_name: Optional[str]
    observe: Callable[[str], MemberObservation]
    residents: Callable[[], _residents.Residents]
    journal: Callable[[], Any]
    export_listing: Callable[[str, str], dict]
    memory: Callable[[str, bool, dict], Optional[dict]] = lambda name, local, health: None
    tunnel: Callable[[str, bool, dict], Any] = lambda name, local, health: None
    secrets: tuple = field(default_factory=tuple)


# ------------------------------------------------------------------ scrubbing


def scrub(value: Any, secrets: tuple) -> Any:
    """Redact every string in a JSON-able structure."""
    if isinstance(value, str):
        return redact(value, secrets)
    if isinstance(value, dict):
        return {k: scrub(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v, secrets) for v in value]
    return value


# ------------------------------------------------------------------ /api/mesh


def _hosted(entry: dict) -> bool:
    """A lobe this member serves itself: feasible, not mesh-proxied, has a model."""
    model = entry.get("model")
    return (
        isinstance(model, str)
        and bool(model)
        and entry.get("feasible", True) is not False
        and not entry.get("proxied")
        and not entry.get("hosted_by")
    )


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _lobes(obs: MemberObservation) -> list:
    out = []
    for role, entry in obs.capabilities.items():
        if not _hosted(entry):
            continue
        lobe = {
            "role": role,
            "model": entry["model"],
            "runtime": entry.get("runtime") or "",
            "ready": bool(entry.get("ready")),
            "shape": "single",
        }
        if _int(entry.get("world_size")) is not None:
            lobe["world_size"] = entry["world_size"]
        if isinstance(entry.get("spans"), list):
            lobe["spans"] = [s for s in entry["spans"] if isinstance(s, str)]
        out.append(lobe)
    return out


def _memory(raw: Optional[dict]) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    total, avail = _int(raw.get("total_bytes")), _int(raw.get("available_bytes"))
    if total is None or avail is None or total <= 0 or not 0 <= avail <= total:
        return None
    return {"total_bytes": total, "available_bytes": avail, "used_bytes": total - avail}


def _split_by_spans(hosts: list) -> Optional[tuple]:
    for member, lobe in hosts:
        spans = lobe.get("spans") or []
        if len(set(spans)) >= 2:
            return list(dict.fromkeys(spans)), f"spans advertised by {member}"
    return None


def _split_by_world_size(hosts: list) -> Optional[tuple]:
    sizes: dict = {}
    for member, lobe in hosts:
        n = lobe.get("world_size")
        if n is not None and n > 1:
            sizes.setdefault(n, []).append(member)
    for n, members in sizes.items():
        if len(members) == n:  # exactly the advertised world; more or fewer is ambiguous
            return members, f"world_size {n} advertised by {', '.join(members)}"
    return None


def _shapes(members: list) -> tuple:
    """Group hosted lobes by model; mark split brains and pairs on each lobe."""
    by_model: dict = {}
    for m in members:
        for lobe in m["lobes"]:
            by_model.setdefault(lobe["model"], []).append((m["name"], lobe))
    splits, pairs = [], []
    for model, hosts in by_model.items():
        found = _split_by_spans(hosts) or _split_by_world_size(hosts)
        split_members = set(found[0]) if found else set()
        if found:
            splits.append(
                {
                    "model": model,
                    "roles": sorted({lobe["role"] for _, lobe in hosts}),
                    "members": found[0],
                    "evidence": found[1],
                }
            )
        pooled = [(n, lobe) for n, lobe in hosts if n not in split_members]
        for n, lobe in hosts:
            if n in split_members:
                lobe["shape"] = "split"
        if len({n for n, _ in pooled}) >= 2:
            for _, lobe in pooled:
                lobe["shape"] = "pair"
            pairs.append(
                {
                    "model": model,
                    "roles": sorted({lobe["role"] for _, lobe in pooled}),
                    "members": list(dict.fromkeys(n for n, _ in pooled)),
                }
            )
    return splits, pairs


def _residency(name: str, res: _residents.Residents) -> str:
    return "resident" if name in res.residents else "guest"


def _member(sources: Sources, entry: dict, obs: MemberObservation, local: bool, res) -> dict:
    name = entry["name"]
    roles = entry.get("roles")
    if not isinstance(roles, list):
        roles = sorted(r for r, e in obs.capabilities.items() if _hosted(e))
    return {
        "name": name,
        "origin": entry["origin"],
        "local": local,
        "residency": _residency(name, res),
        "verified": bool(entry.get("verified", local)),
        "reachable": obs.reachable,
        "version": obs.version,
        "memory": _memory(sources.memory(name, local, obs.health)),
        "roles": [r for r in roles if isinstance(r, str)],
        "lobes": _lobes(obs) if obs.reachable else [],
        "_health": obs.health,
    }


def _peers(sources: Sources, local_obs: MemberObservation) -> list:
    local = sources.local_name
    out, seen = [], {local}
    for m in local_obs.roster:
        name, origin = m.get("name"), m.get("origin")
        if not isinstance(name, str) or not isinstance(origin, str) or name in seen:
            continue
        if origin.rstrip("/") == sources.local_origin.rstrip("/"):
            continue
        seen.add(name)
        out.append(m)
    return out


def _hub_entry(name: str, evidence: Any) -> Optional[dict]:
    if isinstance(evidence, dict):
        text = evidence.get("evidence")
        if not isinstance(text, str) or not text:
            return None
        entry = {"member": name, "evidence": text}
        if evidence.get("declared") is True:
            entry["declared"] = True
        return entry
    return {"member": name, "evidence": str(evidence)} if evidence else None


def _hub(sources: Sources, members: list) -> Optional[dict]:
    found = []
    for m in members:
        entry = _hub_entry(m["name"], sources.tunnel(m["name"], m["local"], m["_health"]))
        if entry:
            found.append(entry)
    return found[0] if len(found) == 1 else None


def build_mesh(sources: Sources) -> dict:
    """The ``/api/mesh`` document, from live observations only."""
    local_obs = sources.observe(sources.local_origin)
    res = sources.residents()
    peers = _peers(sources, local_obs)
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(peers)))) as pool:
        peer_obs = list(pool.map(lambda m: sources.observe(m["origin"]), peers))
    local_name = sources.local_name or "local"
    local_entry = {"name": local_name, "origin": sources.local_origin, "verified": True}
    local_entry["roles"] = None
    members = [_member(sources, local_entry, local_obs, True, res)]
    members += [_member(sources, m, o, False, res) for m, o in zip(peers, peer_obs)]
    splits, pairs = _shapes(members)
    hub = _hub(sources, members)
    for m in members:
        del m["_health"]
    versions: dict = {}
    for m in members:
        versions.setdefault(m["version"] or "unknown", []).append(m["name"])
    return {
        "self": local_name,
        "members": members,
        "versions": versions,
        "split_brains": splits,
        "pairs": pairs,
        "hub": hub,
    }


# ------------------------------------------------------------ rollout/residents


def build_rollout(sources: Sources) -> dict:
    journal = sources.journal()
    members = [
        {
            "member": st.member,
            "target": st.target,
            "current": st.current,
            "last_step": st.last_step,
            "last_result": st.last_result,
            "done": st.done,
            "state": st.state,
            "rollback_command": (
                rollback_command(st.member) if st.last_result == FAILED and not st.done else None
            ),
        }
        for st in journal.status().values()
    ]
    rp = journal.resume_point()
    return {
        "steps": list(getattr(journal, "steps", STEPS)),
        "members": members,
        "resume": {"member": rp.member, "step": rp.step} if rp else None,
    }


def _resident_rows(group: dict) -> list:
    rows = []
    for name, entry in group.items():
        fields = {f: {"value": entry[f].value, "state": entry[f].state} for f in _residents.FIELDS}
        needs = [f for f in _residents.FIELDS if entry[f].state == _residents.NEEDS_YOU]
        rows.append({"name": name, "fields": fields, "needs_you": needs})
    return rows


def build_residents(sources: Sources) -> dict:
    res = sources.residents()
    return {
        "file": _residents.DEFAULT_NAME,
        "residents": _resident_rows(res.residents),
        "guests": _resident_rows(res.guests),
    }


# ------------------------------------------------------------------ HTTP


def _web_root() -> Any:
    return resources.files("lobes.site").joinpath("web")


class _Handler(BaseHTTPRequestHandler):
    server_version = "lobes-site"

    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, doc: object, extra: Optional[dict] = None) -> None:
        site = self.server.site  # type: ignore[attr-defined]
        body = json.dumps(scrub(doc, site.sources.secrets)).encode()
        self._send(code, body, JSON_TYPE, {"Cache-Control": "no-store", **(extra or {})})

    def _file(self, *parts: str) -> Optional[bytes]:
        entry = self.server.site.web_root  # type: ignore[attr-defined]
        for p in parts:
            entry = entry.joinpath(p)
        return entry.read_bytes() if entry.is_file() else None

    def _index(self) -> None:
        body = self._file("index.html")
        if body is None:
            return self._json(404, _NOT_FOUND)
        self._send(200, body, "text/html; charset=utf-8", {"Content-Security-Policy": _CSP})

    def _vendor(self, name: str) -> None:
        ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
        body = None
        if _VENDOR_NAME.match(name) and ".." not in name and ext in _VENDOR_TYPES:
            body = self._file("vendor", name)
        if body is None:
            return self._json(404, _NOT_FOUND)
        self._send(200, body, _VENDOR_TYPES[ext])

    def _export(self, query: dict) -> None:
        site = self.server.site  # type: ignore[attr-defined]
        role = (query.get("role") or [""])[0]
        member = (query.get("member") or [""])[0]
        if not _NAME.match(role) or not _NAME.match(member):
            return self._json(400, {"error": "role and member are required plain names"})
        res = site.sources.residents()
        known = set(res.residents) | set(res.guests) | {site.sources.local_name}
        if member not in known:
            return self._json(404, {"error": f"unknown member: {member}"})
        if not site.export_slots.acquire(blocking=False):
            return self._json(503, {"error": "busy, retry shortly"}, {"Retry-After": "2"})
        try:
            listing = dict(site.sources.export_listing(role, member))
        except ExportUnavailable as exc:
            return self._json(502, {"error": str(exc), "role": role, "member": member})
        finally:
            site.export_slots.release()
        listing.setdefault("role", role)
        listing.setdefault("member", member)
        listing["cli"] = f"lobes lobe export {role} --member {member}"
        self._json(200, listing)

    def _route(self) -> None:
        parts = urlsplit(self.path)
        path = parts.path
        site = self.server.site  # type: ignore[attr-defined]
        if path == "/":
            return self._index()
        if path.startswith("/vendor/"):
            return self._vendor(path[len("/vendor/") :])
        builders = {
            "/api/mesh": build_mesh,
            "/api/rollout": build_rollout,
            "/api/residents": build_residents,
        }
        if path in builders:
            return self._json(200, builders[path](site.sources))
        if path == "/api/lobe-export":
            return self._export(parse_qs(parts.query))
        return self._json(404, _NOT_FOUND)

    def _guarded(self) -> None:
        try:
            self._route()
        except Exception:  # noqa: BLE001 - never leak a traceback to the client
            traceback.print_exc(file=sys.stderr)
            self._json(500, {"error": "internal error"})

    def _not_allowed(self) -> None:
        self._send(405, b'{"error":"method not allowed"}', JSON_TYPE, {"Allow": "GET, HEAD"})

    do_GET = do_HEAD = _guarded  # noqa: N815 - http.server API
    do_POST = do_PUT = do_DELETE = do_PATCH = _not_allowed  # noqa: N815

    def log_message(self, *_args: object) -> None:
        return


class SiteServer:
    """The site on a stdlib ``ThreadingHTTPServer``; port 0 picks an ephemeral port."""

    def __init__(
        self,
        sources: Sources,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        web_root: Any = None,
    ):
        self.sources = sources
        self.web_root = Path(web_root) if web_root is not None else _web_root()
        self.export_slots = threading.BoundedSemaphore(MAX_EXPORT_CONCURRENCY)
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.site = self  # type: ignore[attr-defined]
        self.address = self.httpd.server_address
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self.address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self.httpd.shutdown()
        self.httpd.server_close()


def run_until_interrupted(srv: SiteServer) -> None:
    """Serve on the calling thread until Ctrl-C."""
    try:
        srv.httpd.serve_forever()
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------ live sources


def _read_meminfo(path: Path = MEMINFO) -> Optional[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    kb = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        bits = rest.split()
        if bits and bits[0].isdigit():
            kb[key.strip()] = int(bits[0]) * 1024
    if "MemTotal" not in kb or "MemAvailable" not in kb:
        return None
    return {"total_bytes": kb["MemTotal"], "available_bytes": kb["MemAvailable"]}


def _local_tunnel(deploy_dir: Path) -> Optional[str]:
    from lobes.runtime import _tunnel

    pid = _tunnel.tunnel_pid(deploy_dir)
    if pid:
        return f"lobes tunnel: cloudflared pid {pid}"
    try:
        # fixed argv, read-only `docker ps`
        p = subprocess.run(  # nosec B603 B607
            ["docker", "ps", "--filter", "name=cloudflared", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = [n for n in p.stdout.split() if n] if p.returncode == 0 else []
    return f"container {names[0]} running" if names else None


def _health_memory(health: dict) -> Optional[dict]:
    mem = health.get("memory") if isinstance(health, dict) else None
    return mem if _memory(mem) is not None else None  # inconsistent -> next source


def _declared_tunnel(health: dict) -> Optional[dict]:
    """``/health`` ``tunnel`` (operator-declared) as hub evidence, or None."""
    t = health.get("tunnel") if isinstance(health, dict) else None
    if not isinstance(t, dict) or t.get("declared") is not True:
        return None
    url = t.get("public_url")
    if not isinstance(url, str) or not url.strip():
        return None
    host = urlsplit(url.strip()).netloc or url.strip()
    return {"evidence": f"declared: {host}", "declared": True}


def _secret_values(env: dict) -> tuple:
    return tuple(sorted({v for k, v in env.items() if _SECRET_KEY.search(k) and v and len(v) >= 6}))


def _export_target(member: str, local_name: Optional[str], deploy_dir: Path) -> Target:
    res = _residents.load(deploy_dir)
    entry = res.residents.get(member) or res.guests.get(member) or {}

    def val(name: str) -> str:
        f = entry.get(name)
        return str(f.value) if f is not None and f.value not in (None, "") else ""

    if val("ssh") == "local" or (member == local_name and not val("ssh")):
        return Target(local=True, lobes_path=val("lobes_path") or "lobes")
    if not val("ssh") or not val("lobes_path"):
        raise ExportUnavailable(f"{member}: ssh or lobes_path is needs_you in residents.toml")
    return Target(ssh_target=val("ssh"), user=val("user"), lobes_path=val("lobes_path"))


def default_sources(
    deploy_dir: Path | str,
    *,
    gateway: Optional[str] = None,
    opener: Callable[..., Any] = urllib.request.urlopen,  # nosec B310
    timeout: float = 5.0,
) -> Sources:
    """Live sources: this box's gateway, its .env keys, residents file and journal."""
    from lobes.runtime import _env

    base = Path(deploy_dir).expanduser()
    env = _env.read_env_file(base / ".env")
    env.update(_env.read_env_file(base / ".secrets.env"))
    secrets = _secret_values(env)
    port = env.get("VLLM_PORT") or "8000"
    local_origin = (gateway or f"http://127.0.0.1:{port}").rstrip("/")
    local_name = env.get("LOBES_MESH_NAME") or None
    mesh_key = env.get("LOBES_MESH_KEY") or None
    gateway_key = env.get("GATEWAY_API_KEY") or None

    def observe(origin: str) -> MemberObservation:
        local = origin.rstrip("/") == local_origin
        return observe_member(
            origin,
            gateway_key=gateway_key if local else None,  # our key never goes to a peer
            mesh_key=mesh_key if local else None,
            opener=opener,
            timeout=timeout,
        )

    def export_listing(role: str, member: str) -> dict:
        target = _export_target(member, local_name, base)
        try:
            bundle = _export.export_member([role], target, member=member, secrets=secrets)
        except _export.ExportError as exc:
            raise ExportUnavailable(str(exc)) from exc
        e = bundle.exports[0]
        return {
            "role": role,
            "member": member,
            "status": e.status,
            "service": e.service,
            "image": e.image,
            "reason": e.reason,
            "notes": e.notes,
            "files": bundle.paths(),
        }

    return Sources(
        local_origin=local_origin,
        local_name=local_name,
        observe=observe,
        residents=lambda: _residents.load(base),
        journal=lambda: Journal(base, redact=lambda s: redact(s, secrets)),
        export_listing=export_listing,
        # remote members: only what their own /health advertised (already
        # fetched by observe); local: the same, then /proc + the tunnel probe
        memory=lambda name, local, health: (
            _health_memory(health) or (_read_meminfo() if local else None)
        ),
        tunnel=lambda name, local, health: (
            _declared_tunnel(health) or (_local_tunnel(base) if local else None)
        ),
        secrets=secrets,
    )
