"""Lobe export: role -> compose service -> Dockerfile + build inputs.

Reads a member's compose chain (``lobes fleet files --json``) over the rollout
transport, maps a role to the thing that serves it, and returns a bundle of
text files. Nothing here writes to the member, and no ``.env`` /
``.secrets.env`` content is ever read: ``${VAR}`` references stay as written and
every exported text goes through :func:`redact_text`.

Role -> server mapping is derived, never stored, in this order:

1. what the member's gateway targets (its ``*_BASE_URL`` env): a compose
   service/container name, or a container outside compose (``docker inspect``);
2. naming conventions over the compose chain (service name, ``embed-<lane>``,
   a unique ``<role>-*`` service, the role->service table, ``container_name``).

A role neither finds is reported ``unmapped``; it is never guessed.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

from lobes.rollout._redact import REDACTED, redact
from lobes.rollout._transport import (
    Runner,
    as_target,
    build_argv,
    default_runner,
    run,
    run_lobes,
)

BUILT = "built"
IMAGE_ONLY = "image-only"
UNMAPPED = "unmapped"

_NEVER_READ = {".env", ".secrets.env"}
# role -> gateway env prefix (``<PREFIX>_BASE_URL``)
_GATEWAY_PREFIX = {
    "cortex": "PRIMARY",
    "senses": "MULTIMODAL",
    "worker": "WORKER",
    "muse": "MUSE",
    "associate": "ASSOCIATE",
    "hand": "HAND",
    "embedder": "EMBED",
    "reranker": "RERANK",
    "stt": "STT",
    "tts": "TTS",
}
_ROLE_SERVICE = {
    "cortex": "vllm-primary",
    "senses": "vllm-multimodal",
    "worker": "vllm-worker",
    "muse": "vllm-muse",
    "associate": "vllm-associate",
    "hand": "vllm-hand",
    "embedder": "vllm-embed",
    "reranker": "vllm-rerank",
    "tts": "chatterbox",
}
_LANE_ALIAS = {"stt-en": "stt"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"}  # nosec B104
_SECRET_NAME = re.compile(r"(?i)(key|token|secret|passw|credential)")
_SECRET_ASSIGN = re.compile(
    r"(?<![\w{])[A-Za-z0-9_]*(?:API_KEY|TOKEN|MESH_KEY|GATEWAY_KEY)"
    r"[\"']?\s*[=:]\s*[\"']?(?P<ref>\$\{)?"
)


def local_lobes_path(configured: str = "") -> str:
    """Absolute path of the ``lobes`` to run on THIS box.

    The configured residents.toml value when it is absolute, else the
    ``lobes`` on PATH resolved to an absolute path. The transport refuses a
    relative path, so a bare "lobes" fallback would only crash later.
    """
    if configured and configured.startswith("/"):
        return configured
    found = shutil.which("lobes")
    if found:
        return os.path.abspath(found)
    raise ExportError(
        "no absolute lobes path for this box: set lobes_path in residents.toml "
        "or put lobes on PATH"
    )


class ExportError(RuntimeError):
    """The member could not be read (not a per-role problem)."""


# ---------------------------------------------------------------- redaction


def redact_text(text: str, secrets: Sequence[str] = ()) -> str:
    """redact(), but a ``KEY=${REF}`` compose reference is left as written."""
    out = []
    for line in text.splitlines(keepends=True):
        known = line
        for s in sorted({s for s in secrets if s}, key=len, reverse=True):
            known = known.replace(s, REDACTED)
        matches = list(_SECRET_ASSIGN.finditer(known))
        if matches and all(m.group("ref") for m in matches):
            out.append(known)
        else:
            out.append(redact(known, ()))
    return "".join(out)


# ---------------------------------------------------------------- Dockerfile


@dataclass
class CopySources:
    files: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def dockerfile_copy_sources(text: str) -> CopySources:
    """Local COPY/ADD sources (not ``--from=``, URLs, globs or whole-context)."""
    res = CopySources()
    logical = re.sub(r"\\\s*\n", " ", text)
    for line in logical.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or parts[0].upper() not in ("COPY", "ADD"):
            continue
        rest = parts[1].strip()
        if rest.startswith("["):
            try:
                toks = [str(t) for t in json.loads(rest)]
            except ValueError:
                res.skipped.append(rest)
                continue
        else:
            toks = shlex.split(rest)
        flags = [t for t in toks if t.startswith("--")]
        toks = [t for t in toks if not t.startswith("--")]
        if any(f.startswith("--from") for f in flags):
            continue
        for src in toks[:-1]:
            if "://" in src:
                res.skipped.append(f"{src} (url)")
            elif any(c in src for c in "*?[") or src in (".", "./", "/"):
                res.skipped.append(f"{src} (glob or whole context)")
            elif src not in res.files:
                res.files.append(src)
    return res


# ---------------------------------------------------------------- the chain


@dataclass
class Chain:
    deployment_dir: str
    files: list
    services: dict


def _safe_join(base: str, rel: str) -> str | None:
    if rel.startswith("/"):
        return None
    joined = posixpath.normpath(posixpath.join(base, rel))
    if joined != base and not joined.startswith(base.rstrip("/") + "/"):
        return None
    return joined


def _rel(path: Any, base: str) -> Any:
    """``docker compose config`` absolutises paths; make ones under base relative again."""
    if isinstance(path, str) and path.startswith(base.rstrip("/") + "/"):
        return "./" + posixpath.relpath(path, base)
    if path == base:
        return "."
    return path


def _relativize(spec: dict, base: str) -> dict:
    spec = dict(spec)
    build = spec.get("build")
    if isinstance(build, dict):
        spec["build"] = {**build, "context": _rel(build.get("context", "."), base)}
    vols = []
    for v in spec.get("volumes") or []:
        if isinstance(v, dict) and v.get("type") == "bind":
            v = {**v, "source": _rel(v.get("source"), base)}
        vols.append(v)
    if vols:
        spec["volumes"] = vols
    return spec


def _cat(target, path: str, runner, secrets) -> str | None:
    if posixpath.basename(path) in _NEVER_READ:
        return None
    # Raw read (transport redaction would mangle ${VAR} refs); every exported
    # text is passed through redact_text() before it leaves this module.
    rc, out, _ = runner(build_argv(as_target(target), ["cat", path], 30.0), 30.0)
    return out if rc == 0 else None


def load_chain(target, *, runner: Runner | None = None, secrets: Sequence[str] = ()) -> Chain:
    runner = runner or default_runner
    res = run_lobes(target, ["fleet", "files", "--json"], runner=runner, secrets=secrets)
    if not res.ok:
        raise ExportError(f"`lobes fleet files --json` failed on the member: {res.stderr.strip()}")
    try:
        data = json.loads(res.stdout)
        base = str(data["deployment_dir"]).rstrip("/")
        names = [f for f in data["files"] if f != "-f"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ExportError("unreadable `lobes fleet files --json` output") from exc
    paths = []
    for name in names:
        path = name if name.startswith("/") else _safe_join(base, name)
        if path is None:
            raise ExportError(f"compose file {name} escapes the deployment dir")
        paths.append(path)
    argv = ["docker", "compose", "--project-directory", base]
    for p in paths:
        argv += ["-f", p]
    # Docker does the merge. --no-interpolate keeps ${VAR} references, so no
    # .env value comes back; --no-env-resolution keeps env_file contents out
    # (older docker lacks it, so retry without).
    rc, out, err = 1, "", ""
    for extra in (["--no-interpolate", "--no-env-resolution"], ["--no-interpolate"]):
        cmd = argv + ["config", *extra, "--format", "json"]
        rc, out, err = runner(build_argv(as_target(target), cmd, 60.0), 60.0)
        if rc == 0:
            break
    if rc != 0:
        raise ExportError(
            f"`docker compose config` failed on the member: {redact_text(err, secrets).strip()}"
        )
    try:
        services = json.loads(out).get("services") or {}
    except (ValueError, AttributeError) as exc:
        raise ExportError("unreadable `docker compose config` output") from exc
    services = {n: _relativize(spec or {}, base) for n, spec in services.items()}
    return Chain(base, paths, services)


# ------------------------------------------------------------- role mapping


def _gateway_urls(target, chain: Chain, runner, secrets) -> dict:
    gw = chain.services.get("gateway")
    if gw is None:
        return {}
    name = gw.get("container_name") or "gateway"
    res = run(
        target,
        ["docker", "inspect", "--format", "{{json .Config.Env}}", name],
        runner=runner,
        secrets=secrets,
    )
    if not res.ok:
        return {}
    try:
        parsed = json.loads(res.stdout)
        env = parsed[0]["Config"]["Env"] if parsed and isinstance(parsed[0], dict) else parsed
    except (ValueError, KeyError, TypeError, IndexError):
        return {}
    urls = {}
    for item in env or []:
        k, _, v = str(item).partition("=")
        if k.endswith("_BASE_URL"):  # wiring only; every other value is dropped
            urls[k] = v
    return urls


def _service_for_host(chain: Chain, host: str) -> str | None:
    if host in chain.services:
        return host
    for name, spec in chain.services.items():
        if spec.get("container_name") == host:
            return name
    return None


def _candidates(role: str, chain: Chain, urls: dict) -> list:
    """Ordered ``(kind, name, how)``; kind is service|container."""
    out = []
    keys = [role.upper().replace("-", "_") + "_BASE_URL"]
    base = _LANE_ALIAS.get(role, role)
    if base in _GATEWAY_PREFIX:
        keys.append(_GATEWAY_PREFIX[base] + "_BASE_URL")
    for k in keys:
        url = urls.get(k)
        host = urlparse(url).hostname if url else None
        if not host or host in _LOCAL_HOSTS:
            continue
        svc = _service_for_host(chain, host)
        out.append(("service", svc, f"gateway {k}") if svc else ("container", host, f"gateway {k}"))
    svcs = chain.services
    conv = []
    for name in (role, _LANE_ALIAS.get(role), f"embed-{role}", _ROLE_SERVICE.get(role)):
        if name and name in svcs:
            conv.append(name)
    prefixed = [s for s in svcs if s.startswith(role + "-")]
    if len(prefixed) == 1:
        conv.append(prefixed[0])
    for name, spec in svcs.items():
        if spec.get("container_name") in (role, f"model-gear-{role}"):
            conv.append(name)
    for name in conv:
        out.append(("service", name, "convention"))
    return out


# ------------------------------------------------------------ docker run


def docker_run_command(info: dict) -> str:
    """Rebuild a ``docker run`` from ``docker inspect`` JSON (secret values dropped)."""
    cfg, host = info.get("Config") or {}, info.get("HostConfig") or {}
    parts = ["docker", "run", "-d", "--name", info.get("Name", "").lstrip("/")]
    restart = (host.get("RestartPolicy") or {}).get("Name")
    if restart and restart != "no":
        parts += ["--restart", restart]
    if host.get("IpcMode") not in (None, "", "private", "shareable"):
        parts += ["--ipc", host["IpcMode"]]
    if host.get("NetworkMode") not in (None, "", "default", "bridge"):
        parts += ["--network", host["NetworkMode"]]
    if any(
        "gpu" in c
        for r in host.get("DeviceRequests") or []
        for grp in r.get("Capabilities") or []
        for c in grp
    ):
        parts += ["--gpus", "all"]
    for cport, binds in sorted((host.get("PortBindings") or {}).items()):
        for b in binds or []:
            ip = b.get("HostIp") or ""
            hp = b.get("HostPort") or ""
            parts += ["-p", f"{ip + ':' if ip else ''}{hp}:{cport.split('/')[0]}"]
    for bind in host.get("Binds") or []:
        parts += ["-v", bind]
    for item in cfg.get("Env") or []:
        k, _, v = str(item).partition("=")
        parts += ["-e", f"{k}=${{{k}}}" if _SECRET_NAME.search(k) else item]
    ep = cfg.get("Entrypoint")
    if ep:
        parts += ["--entrypoint", ep[0]]
    parts.append(cfg.get("Image", ""))
    if ep:
        parts += list(ep[1:])
    parts += list(cfg.get("Cmd") or [])
    quoted = []
    for p in parts:
        # keep ${VAR} references expandable by the shell, quote everything else
        quoted.append(
            re.sub(r"\\\$\{(\w+)\}", r"${\1}", shlex.quote(p)) if "${" in p else shlex.quote(p)
        )
    return " ".join(quoted)


# ------------------------------------------------------------------ export


@dataclass
class RoleExport:
    role: str
    member: str
    status: str
    source: str = ""
    service: str = ""
    image: str = ""
    reason: str = ""
    how: str = ""
    files: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def manifest(self) -> dict:
        return {
            "role": self.role,
            "member": self.member,
            "status": self.status,
            "source": self.source,
            "service": self.service,
            "image": self.image,
            "mapped_by": self.how,
            "reason": self.reason,
            "files": sorted(self.files),
            "notes": self.notes,
        }


def _finish(r: RoleExport, secrets) -> RoleExport:
    r.files = {k: redact_text(v, secrets) for k, v in r.files.items()}
    r.files["manifest.json"] = redact_text(json.dumps(r.manifest(), indent=2) + "\n", secrets)
    return r


def _bundle_key(rel: str) -> str:
    """Where a deployment-relative file lands in the bundle."""
    return posixpath.basename(rel) if "/" not in rel.strip("./") else rel.lstrip("./")


def _add_file(r, target, chain, rel, runner, secrets, what) -> Optional[str]:
    """Copy one deployment file into the bundle; its bundle key, or None."""
    path = _safe_join(chain.deployment_dir, rel)
    if path is None:
        r.notes.append(f"{what} {rel}: path would escape the deployment dir; not exported")
        return None
    if posixpath.basename(path) in _NEVER_READ:
        r.notes.append(f"{what} {rel}: secrets file; not exported")
        return None
    text = _cat(target, path, runner, secrets)
    if text is None:
        r.notes.append(f"{what} {rel}: not a readable file (directory or missing); not exported")
        return None
    key = _bundle_key(rel)
    r.files[key] = text
    return key


def _export_service(r, name, spec, target, chain, runner, secrets):
    r.service, r.source = name, "compose"
    r.files["compose-service.json"] = json.dumps({"services": {name: spec}}, indent=2) + "\n"
    build = spec.get("build")
    if build:
        r.status = BUILT
        ctx, dockerfile = ".", "Dockerfile"
        if isinstance(build, str):
            ctx = build
        else:
            ctx = str(build.get("context", "."))
            dockerfile = str(build.get("dockerfile", "Dockerfile"))
        base = _safe_join(chain.deployment_dir, ctx)
        if base is None:
            r.notes.append(
                f"build context {ctx}: path would escape the deployment dir (escape); not exported"
            )
            return
        rel_ctx = posixpath.relpath(base, chain.deployment_dir)
        join = lambda p: posixpath.normpath(posixpath.join(rel_ctx, p))  # noqa: E731
        dpath = join(dockerfile)
        key = _add_file(r, target, chain, dpath, runner, secrets, "Dockerfile")
        if key is not None:
            # look the Dockerfile up by the key it was stored under: a build
            # context in a subdirectory stores it as "<dir>/Dockerfile"
            sources = dockerfile_copy_sources(r.files[key])
            for src in sources.files:
                _add_file(r, target, chain, join(src), runner, secrets, "build input")
            for sk in sources.skipped:
                r.notes.append(f"COPY {sk}: not exported")
    else:
        r.status = IMAGE_ONLY
        r.image = str(spec.get("image", ""))
        r.notes.append("image-only: no build; the image reference is in compose-service.json")
    for vol in spec.get("volumes") or []:
        src = (vol.get("source") if isinstance(vol, dict) else str(vol).split(":")[0]) or ""
        if src.startswith("./") or src.startswith("../"):
            _add_file(r, target, chain, src, runner, secrets, "mounted file")


def _export_container(r, name, target, runner, secrets) -> bool:
    res = run(target, ["docker", "inspect", name], runner=runner, secrets=secrets)
    if not res.ok:
        return False
    try:
        info = json.loads(res.stdout)[0]
    except (ValueError, IndexError, TypeError):
        return False
    r.status, r.source, r.service = IMAGE_ONLY, "container", name
    r.image = str((info.get("Config") or {}).get("Image", ""))
    r.files["docker-run.sh"] = (
        "#!/bin/sh\n# reconstructed from `docker inspect`; image-only (not built by lobes)\n"
        + docker_run_command(info)
        + "\n"
    )
    r.notes.append(
        "image-only: container started outside compose; secret-named env vars are ${VAR} references"
    )
    return True


def export_role(
    role: str,
    target,
    *,
    member: str = "",
    runner: Runner | None = None,
    secrets: Sequence[str] = (),
    chain: Chain | None = None,
) -> RoleExport:
    runner = runner or default_runner
    chain = chain or load_chain(target, runner=runner, secrets=secrets)
    urls = _gateway_urls(target, chain, runner, secrets)
    r = RoleExport(role=role, member=member, status=UNMAPPED)
    for kind, name, how in _candidates(role, chain, urls):
        if kind == "service":
            r.how = how
            _export_service(r, name, chain.services[name], target, chain, runner, secrets)
        elif _export_container(r, name, target, runner, secrets):
            r.how = how
        else:
            continue
        if how == "convention":
            r.notes.append("mapped by convention (the gateway did not name this role's server)")
        return _finish(r, secrets)
    r.reason = "no gateway target, compose service or convention maps this role"
    return _finish(RoleExport(role, member, UNMAPPED, reason=r.reason), secrets)


@dataclass
class Bundle:
    member: str
    exports: list

    def _all(self) -> dict:
        out = {}
        for e in self.exports:
            for name, text in e.files.items():
                out[f"{self.member}/{e.role}/{name}"] = text
        summary = {
            "member": self.member,
            "roles": {
                e.role: {"status": e.status, "service": e.service, "reason": e.reason}
                for e in self.exports
            },
        }
        out[f"{self.member}/manifest.json"] = json.dumps(summary, indent=2) + "\n"
        return out

    def paths(self) -> list:
        return sorted(self._all())

    def write(self, out_dir: Path | str) -> list:
        base = Path(out_dir).resolve()
        written = []
        for rel, text in sorted(self._all().items()):
            dest = (base / rel).resolve()
            if base not in dest.parents:
                raise ExportError(f"refusing to write outside the output dir: {rel}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            written.append(rel)
        return written


def export_member(
    roles: Sequence[str],
    target,
    *,
    member: str,
    runner: Runner | None = None,
    secrets: Sequence[str] = (),
) -> Bundle:
    runner = runner or default_runner
    chain = load_chain(target, runner=runner, secrets=secrets)
    return Bundle(
        member,
        [
            export_role(r, target, member=member, runner=runner, secrets=secrets, chain=chain)
            for r in roles
        ],
    )
