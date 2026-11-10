"""What each member's running containers serve, read over the residents' SSH (d2).

``lobes site`` must read how the system is set, never hand-typed declarations
(deviation d2, 2026-11-10). Two facts are invisible to the gateways -- an
engine's ``/health`` and ``/v1/models`` do not expose its world/rank, and a
gateway container cannot see other containers -- so they are read from the
container runtime itself:

* which containers run (``docker ps``), their image, and the served model name
  parsed from Entrypoint+Cmd (``--served-model-name``, ``--name``, ``--model``,
  else the positional after ``serve``);
* world/rank config for an engine that spans boxes: env keys matching
  ``TF_<X>_WORLD`` / ``TF_<X>_RANK`` (TensorFold) with a plain integer value,
  and vLLM's ``--tensor-parallel-size`` / ``--pipeline-parallel-size`` (``-tp``
  / ``-pp``) from the command line. TP/PP are recorded only: a TP>1 engine on
  one box is not a cross-box split.

**Nothing else leaves the member.** ONE command per member (``ssh`` with
BatchMode via :mod:`lobes.rollout._transport`, or local ``sh`` for this box)
runs ``docker ps`` + ``docker inspect`` and pipes the inspect JSON -- held only
in a remote shell variable -- through a ``python3 -I`` one-liner that prints
the allowlisted projection above. No other env value, and no raw command line,
is printed. The projection is then re-validated and redacted locally
(:func:`parse_projection`) in case the member's ``python3`` is not ours.

A failed, timed-out or unparseable read means that member's topology is
UNKNOWN (``None``), never guessed. Results are cached per member for ``ttl``
seconds (default 120) so a page load does not SSH every time; a failure is
cached for the same time so a dead box does not stall every load.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from lobes.rollout import _transport
from lobes.rollout._redact import redact
from lobes.rollout._transport import Target

DEFAULT_TIMEOUT = 10.0
DEFAULT_TTL = 120.0
MAX_PARALLEL = 8

# Shared by the remote one-liner (copied verbatim below) and the local re-check.
ENV_ALLOW = re.compile(r"^TF_([A-Z0-9]+(?:_[A-Z0-9]+)*)_(WORLD|RANK)$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9/][A-Za-z0-9._/:@+-]{0,199}$")
INT_RE = re.compile(r"^\d{1,4}$")

# The projection, run ON THE MEMBER by ``python3 -I`` with the inspect JSON on
# stdin. Keep it self-contained (stdlib only) and print nothing but the
# allowlisted keys.
REMOTE_PY = r"""
import json, re, shlex, sys
W = re.compile(r"^TF_[A-Z0-9]+(?:_[A-Z0-9]+)*_(WORLD|RANK)$")
N = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
I = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}$")
M = re.compile(r"^[A-Za-z0-9/][A-Za-z0-9._/:@+-]{0,199}$")
D = re.compile(r"^[0-9]{1,4}$")
def flag(a, names):
    for i, t in enumerate(a):
        for n in names:
            if t == n and i + 1 < len(a):
                return a[i + 1]
            if t.startswith(n + "="):
                return t[len(n) + 1:]
    return None
def args(cfg):
    out = []
    for t in (cfg.get("Entrypoint") or []) + (cfg.get("Cmd") or []):
        if not isinstance(t, str):
            continue
        if out and out[-1] == "-c":
            try:
                out += shlex.split(t)
                continue
            except ValueError:
                pass
        out.append(t)
    return out
def model(a):
    m = flag(a, ["--served-model-name", "--name", "--model"])
    if m is None and "serve" in a:
        i = a.index("serve")
        if i + 1 < len(a) and not a[i + 1].startswith("-"):
            m = a[i + 1]
    return m if isinstance(m, str) and M.match(m) else None
raw = sys.stdin.read().strip()
rows = []
for c in (json.loads(raw) if raw else []):
    cfg = c.get("Config") or {}
    a = args(cfg)
    name = (c.get("Name") or "").lstrip("/")
    image = cfg.get("Image") or ""
    env = {}
    for kv in cfg.get("Env") or []:
        k, _, v = kv.partition("=")
        if W.match(k) and D.match(v):
            env[k] = v
    row = {
        "name": name if N.match(name) else "",
        "image": image if I.match(image) else "",
        "model": model(a),
        "env": env,
    }
    for key, names in (("tp", ["--tensor-parallel-size", "-tp"]),
                       ("pp", ["--pipeline-parallel-size", "-pp"])):
        v = flag(a, names)
        if v is not None and D.match(v):
            row[key] = int(v)
    rows.append(row)
print(json.dumps(rows))
"""

# The inspect JSON lives only in the remote shell variable ``j``; a docker
# failure exits non-zero BEFORE anything is printed (no ``pipefail`` needed).
REMOTE_SH = (
    'ids=$(docker ps -q --no-trunc) || exit 3; j="[]"; '
    'if [ -n "$ids" ]; then j=$(docker inspect $ids) || exit 4; fi; '
    'printf "%s" "$j" | python3 -I -c "$0"'
)


def build_remote_argv() -> list:
    """``sh -c <script> <python>`` -- ``$0`` carries the projection code."""
    return ["sh", "-c", REMOTE_SH, REMOTE_PY]


@dataclass(frozen=True)
class Container:
    """One running container's allowlisted facts."""

    name: str
    image: str
    model: Optional[str]
    ranks: tuple = ()  # ((group, world, rank), ...) e.g. ("TF_GLM", 2, 1)
    tp: Optional[int] = None
    pp: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "image": self.image,
            "model": self.model,
            "world_rank": [{"group": g, "world": w, "rank": r} for g, w, r in self.ranks],
            "tp": self.tp,
            "pp": self.pp,
        }


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value <= 9999:
        return value
    if isinstance(value, str) and INT_RE.match(value):
        return int(value)
    return None


def _ranks(env: Any) -> tuple:
    if not isinstance(env, dict):
        return ()
    world: dict = {}
    rank: dict = {}
    for key, value in env.items():
        m = ENV_ALLOW.match(key) if isinstance(key, str) else None
        n = _int(value) if isinstance(value, str) else None
        if m is None or n is None:
            continue
        (world if m.group(2) == "WORLD" else rank)[f"TF_{m.group(1)}"] = n
    return tuple((g, world[g], rank[g]) for g in sorted(world) if g in rank)


def _clean(value: Any, pattern: re.Pattern, secrets: Sequence[str]) -> Optional[str]:
    if not isinstance(value, str) or not pattern.match(value):
        return None
    return value if redact(value, secrets) == value else None


def parse_projection(stdout: str, secrets: Sequence[str] = ()) -> Optional[list]:
    """Re-validate the member's projection; ``None`` when it is not a list."""
    try:
        rows = json.loads(stdout)
    except (TypeError, ValueError):
        return None
    if not isinstance(rows, list):
        return None
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _clean(row.get("name"), NAME_RE, secrets)
        if name is None:
            continue
        out.append(
            Container(
                name=name,
                image=_clean(row.get("image"), IMAGE_RE, secrets) or "",
                model=_clean(row.get("model"), MODEL_RE, secrets),
                ranks=_ranks(row.get("env")),
                tp=_int(row.get("tp")),
                pp=_int(row.get("pp")),
            )
        )
    return out


Topology = Mapping[str, Optional[list]]


class TopologyReader:
    """Read every member's containers in parallel, cached for ``ttl`` seconds.

    ``targets()`` -> ``{member: Target | None}`` (``None``: no ssh target known,
    so unknown). ``runner`` is :data:`lobes.rollout._transport.Runner`.
    """

    def __init__(
        self,
        targets: Callable[[], Mapping[str, Optional[Target]]],
        *,
        runner: _transport.Runner = _transport.default_runner,
        timeout: float = DEFAULT_TIMEOUT,
        ttl: float = DEFAULT_TTL,
        clock: Callable[[], float] = time.monotonic,
        secrets: Sequence[str] = (),
    ):
        self._targets = targets
        self._runner = runner
        self._timeout = timeout
        self._ttl = ttl
        self._clock = clock
        self._secrets = tuple(secrets)
        self._cache: dict = {}  # member -> (read_at, Optional[list])
        self._lock = threading.Lock()

    def _read_one(self, target: Optional[Target]) -> Optional[list]:
        if target is None:
            return None
        try:
            result = _transport.run(
                target,
                build_remote_argv(),
                timeout=self._timeout,
                runner=self._runner,
                secrets=self._secrets,
            )
        except Exception:  # noqa: BLE001 - any transport failure means unknown
            return None
        if not result.ok:
            return None
        return parse_projection(result.stdout, self._secrets)

    def read(self) -> dict:
        """``{member: [Container, ...] | None}`` -- ``None`` is unknown."""
        try:
            targets = dict(self._targets())
        except Exception:  # noqa: BLE001 - unreadable residents: nothing known
            return {}
        with self._lock:  # one refresh at a time; concurrent loads share it
            now = self._clock()
            stale = [
                n for n in targets if n not in self._cache or now - self._cache[n][0] >= self._ttl
            ]
            if stale:
                with ThreadPoolExecutor(max_workers=max(1, min(MAX_PARALLEL, len(stale)))) as pool:
                    fresh = list(pool.map(lambda n: self._read_one(targets[n]), stale))
                for name, value in zip(stale, fresh):
                    self._cache[name] = (now, value)
            return {n: self._cache[n][1] for n in targets}
