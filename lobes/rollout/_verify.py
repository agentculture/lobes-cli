"""Post-member verification through the member's own gateway (mesh path).

Passes only if /health reports the target version, the capability diff has no
regression, and one real request per feasible+ready role (hosted locally or
forwarded) succeeds. The HTTP opener is injected; the key is never logged.
"""

from __future__ import annotations

import io
import json
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from typing import Any, Callable

from lobes.rollout._observe import MemberObservation, Opener, diff, observe_member

_CHAT_ROLES = {"cortex", "worker", "hand", "muse", "senses", "associate"}


@dataclass
class Failure:
    member: str
    check: str
    role: str | None = None
    detail: str = ""


@dataclass
class VerifyResult:
    member: str
    failures: list[Failure] = field(default_factory=list)
    probed: list[str] = field(default_factory=list)
    not_probed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def family(role: str, entry: dict) -> str | None:
    """Map a role/lane name to a probe family; None means not probed."""
    if role in _CHAT_ROLES:
        return "chat"
    if role in ("embedder",) or (entry.get("lane") and role.endswith("-embed")):
        return "embed"
    if role == "reranker":
        return "rerank"
    if role == "stt" or role.startswith("stt-"):
        return "stt"
    return None


def _silent_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)
    return buf.getvalue()


def _json_req(role: str, path: str, payload: dict) -> tuple[str, bytes, str]:
    return path, json.dumps(payload).encode(), "application/json"


def _build(fam: str, role: str) -> tuple[str, bytes, str]:
    if fam == "chat":
        return _json_req(
            role,
            "/v1/chat/completions",
            {"model": role, "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 8},
        )
    if fam == "embed":
        return _json_req(role, "/v1/embeddings", {"model": role, "input": "rollout probe"})
    if fam == "rerank":
        return _json_req(
            role,
            "/v1/rerank",
            {"model": role, "query": "cat", "documents": ["a cat sat", "a car drove"]},
        )
    boundary = uuid.uuid4().hex
    b = boundary.encode()
    body = (
        b"--"
        + b
        + b'\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
        + role.encode()
        + b"\r\n--"
        + b
        + b'\r\nContent-Disposition: form-data; name="file"; filename="probe.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n" + _silent_wav() + b"\r\n--" + b + b"--\r\n"
    )
    return "/v1/audio/transcriptions", body, f"multipart/form-data; boundary={boundary}"


def _valid(fam: str, data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if fam == "chat":
        return bool(data.get("choices"))
    if fam == "embed":
        d = data.get("data")
        return bool(d) and bool(d[0].get("embedding"))
    if fam == "rerank":
        return bool(data.get("results") or data.get("data"))
    return isinstance(data.get("text"), str)


def _probe(opener: Opener, base: str, key: str | None, fam: str, role: str, timeout: float) -> str:
    """Return '' on success, else a short failure detail (never includes the key)."""
    path, body, ctype = _build(fam, role)
    req = urllib.request.Request(base + path, data=body, method="POST")  # nosec B310
    req.add_header("Content-Type", ctype)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with opener(req, timeout=timeout) as r:
            status = getattr(r, "status", 200)
            if not 200 <= status < 300:
                return f"HTTP {status}"
            data = json.loads(r.read())
    except (OSError, ValueError, TypeError, AttributeError, IndexError) as exc:
        return f"{type(exc).__name__}"
    return "" if _valid(fam, data) else "unexpected response body"


def verify_member(
    origin: str,
    *,
    target_version: str,
    before: MemberObservation | None = None,
    gateway_key: str | None = None,
    mesh_key: str | None = None,
    opener: Opener = urllib.request.urlopen,  # nosec B310
    timeout: float = 30.0,
    probe_filter: Callable[[str, dict], bool] | None = None,
) -> VerifyResult:
    """Verify one member after rollout; every failure names member, check, role."""
    res = VerifyResult(member=origin)
    after = observe_member(
        origin, gateway_key=gateway_key, mesh_key=mesh_key, opener=opener, timeout=timeout
    )
    if not after.reachable or after.version != target_version:
        res.failures.append(
            Failure(origin, "health", None, f"version {after.version!r} != {target_version!r}")
        )
        return res
    if before is not None:
        for role in diff(before, after).regressions:
            res.failures.append(Failure(origin, "capability_regression", role, "no longer up"))
    base = origin.rstrip("/")
    for role in sorted(after.capabilities):
        entry = after.capabilities[role]
        if not (entry.get("feasible") and entry.get("ready")):
            continue
        if probe_filter and not probe_filter(role, entry):
            continue
        fam = family(role, entry)
        if fam is None:
            res.not_probed.append(role)
            continue
        detail = _probe(opener, base, gateway_key, fam, role, timeout)
        if detail:
            res.failures.append(Failure(origin, "probe", role, detail))
        else:
            res.probed.append(role)
    return res
