"""Per-role REPLICA snapshot — load + serving fingerprint for every replica of one role.

Why this module exists (issue #199, task t4)
--------------------------------------------
Until now a role had exactly one owner per box: ``order_backends`` returns a
0-or-1 list "never a failover chain" (#91), and a peer was consulted only for a
role this box does **not** host (proxy-lobes, #115/#127). The cortex replica
pool changes that: the same logical ``cortex`` is served by several boxes, and
each gateway must pick one **per request**. Picking needs two facts the fleet
never cached before:

* **load** — how busy each replica is right now (running + waiting), and
* **fingerprint** — what each replica is actually serving, so a box does not
  silently pool a different model, quantization, context window or engine.

This module supplies both, as a background snapshot. It is deliberately the
*substrate only*: it decides nothing. :func:`lobes.gateway._selection.select_replica`
(task t5) reads :class:`ReplicaState` and chooses; ``server.py`` (t7/t8)
dispatches. Nothing here forwards a request, mutates config, or touches the
routing table.

What it deliberately does NOT do
--------------------------------
* **It never dials a peer's engine.** A peer is reached ONLY through its
  gateway origin, on ``GET /status`` and ``GET /capabilities``. The vLLM port
  is unpublished cross-box on the Spark; a probe that assumed otherwise would
  false-negative every peer. (Spec c34.)
* **It never consults the catalog.** ``RoleInfo``'s ``model``/``quant``/
  ``runtime``/``mtp`` are catalog-derived and mislabel an unknown served id
  (the Orin's llama.cpp ``cortex`` reads back as ``quant=modelopt,
  runtime=vllm``). A fingerprint field is either LIVE (served id and
  ``max_model_len`` from the lane's own ``/v1/models``), DECLARED (passed into
  the gateway env by the operator), or :data:`UNKNOWN`. It is never invented.
  This module imports no catalog at all, so there is nothing to fall back to.
  (Spec c33/h25.)
* **It never pools an unknown.** ``"unknown"`` on either side of a
  disqualifying field makes the pair incompatible with a reason naming it —
  silence is not evidence of a match. (Spec c13/h11.)
* **It never probes on the request path.** :meth:`ReplicaCache.current` is a
  tuple read under a lock: no socket, no blocking. Probing happens on two
  daemon threads (local and peer, separately, so a hung peer can never delay
  the local lane), exactly as :class:`lobes.gateway._readiness.ReadinessCache`
  does. (Spec c5/h5.)

Consuming t2's config
---------------------
The routing-table fields this cache is fed from (``replica_origins``,
``replica_api_keys``, ``self_origin``, the per-lane declared fingerprint keys)
land in a sibling task. This module takes them as PLAIN constructor inputs —
a :class:`LocalLane`, a sequence of :class:`PeerReplica` — and imports nothing
from :mod:`lobes.gateway._config` or :mod:`lobes.gateway._routing`, mirroring
how :class:`~lobes.gateway._readiness.PeerSpec` stays decoupled from the table
it is built from.

Stdlib only — this gateway is deliberately dependency-free.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from time import monotonic as _monotonic
from typing import Callable
from urllib.parse import urlsplit

from .. import _metrics

# The sentinel for "this field was never declared and cannot be probed". A
# string (not ``None``) so it survives JSON rendering into ``/capabilities``
# unchanged and reads the same in a CLI table.
UNKNOWN = "unknown"

# How often each daemon thread re-probes. Replica load moves faster than
# readiness, but probing costs a cross-box round trip (~110 ms measured on the
# Thor), so the ReadinessCache interval is reused rather than tightened —
# consistency with the existing cache matters more than freshness we cannot
# justify with a measurement.
_DEFAULT_REFRESH_INTERVAL: float = 5.0

# Per-probe socket budget. The local lane and the peers get the SAME default
# but hold them in separate constants-in-spirit (two constructor arguments), so
# a deployment can bound a slow cross-box link without shortening the local
# one — the ``_READINESS_PROBE_TIMEOUT`` / ``_PEER_PROBE_TIMEOUT`` split.
_LOCAL_PROBE_TIMEOUT: float = 3.0
_PEER_PROBE_TIMEOUT: float = 3.0

_MODELS_PATH = "/v1/models"
_METRICS_PATH = "/metrics"
_STATUS_PATH = "/status"
_CAPABILITIES_PATH = "/capabilities"

# The peer gateway paths this module is allowed to dial. Named so the
# "never dial a peer's vLLM port" rule is one grep away from its enforcement.
PEER_PATHS: tuple[str, ...] = (_STATUS_PATH, _CAPABILITIES_PATH)

# The four fields a mismatch DISQUALIFIES on. Everything else in a
# :class:`Fingerprint` is recorded for the operator and never gates pooling.
DISQUALIFYING_FIELDS: tuple[str, ...] = (
    "served_id",
    "quantization",
    "max_model_len",
    "runtime",
)

# The declared-lane keys a :class:`LocalLane` may carry. Absent → UNKNOWN.
DECLARED_KEYS: tuple[str, ...] = (
    "runtime",
    "quantization",
    "kv_cache_dtype",
    "reasoning_parser",
    "tool_parser",
    "speculative_config",
)

# ``(url, timeout, api_key | None) -> (status, body)``. The ONLY thing in this
# module that opens a socket; injected so every test runs offline. Same shape
# as :data:`lobes.gateway._readiness.PeerOpener`.
Opener = Callable[[str, float, "str | None"], "tuple[int, bytes]"]

# Injected clock, so ``last_seen`` is deterministic under test.
Clock = Callable[[], float]


def _default_opener(
    url: str, timeout: float, api_key: str | None
) -> tuple[int, bytes]:  # pragma: no cover - opens a socket
    """GET *url* scheme-aware, attach ``Authorization`` iff *api_key* is set.

    Mirrors :func:`lobes.gateway._readiness._default_peer_opener` exactly: a
    peer origin is an operator-typed cross-box URL that may legitimately be
    ``https://`` (a tunnel or TLS-terminating proxy), so the scheme is
    honoured rather than assumed. The key is used once to build the header
    and never logged or persisted.
    """
    parts = urlsplit(url)
    if parts.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            parts.hostname, parts.port or 443, timeout=timeout
        )
    else:
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        target = parts.path or "/"
        if parts.query:
            target = f"{target}?{parts.query}"
        conn.request("GET", target, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


# --- the pinned data contract (t5/t6/t7 consume these names) ---------------


@dataclass(frozen=True)
class Fingerprint:
    """What one replica is ACTUALLY serving.

    Two provenances, never mixed with a third:

    * ``served_id`` / ``max_model_len`` are **live** — read from the lane's own
      ``GET /v1/models`` (vLLM reports ``max_model_len``), or from a peer's
      published fingerprint.
    * everything else is **declared** — the operator's own lane config passed
      into the gateway env — or :data:`UNKNOWN`.

    There is no third provenance: the catalog is never consulted (see the
    module docstring). ``max_model_len`` is ``None`` when the engine did not
    report one, which is a disqualifier, not a zero.
    """

    served_id: str
    max_model_len: int | None
    runtime: str  # "vllm" | "llamacpp" | UNKNOWN
    quantization: str
    kv_cache_dtype: str  # informational
    reasoning_parser: str  # informational
    tool_parser: str  # informational
    speculative_config: str  # informational (draft mode)


@dataclass(frozen=True)
class ReplicaState:
    """One replica of one role, as of the last background probe.

    ``ready`` and ``busy`` are deliberately separate: ``ready`` is "reachable
    and serving this role", ``busy`` is the peer's OWN pressure verdict. A
    replica is selectable only when it is ``ready and compatible and not
    busy`` — but the *policy* that says so lives in
    :mod:`lobes.gateway._selection` (t5), not here.

    ``compatible``/``reason`` are the honesty pair: ``reason`` is empty exactly
    when ``compatible`` is true, and otherwise names every differing field, so
    ``/capabilities`` and the CLI can show an operator WHY a declared replica
    is not pooling instead of leaving it silently absent.
    """

    origin: str
    local: bool
    ready: bool
    busy: bool
    health: str  # "ok" | "unreachable" | "timeout" | "error" | "unknown"
    running: int
    waiting: int
    fingerprint: Fingerprint | None
    compatible: bool
    reason: str
    last_seen: float  # monotonic timestamp of the last SUCCESSFUL probe (0.0 = never)
    weight: float = 1.0  # declared decode weight for the selection policy


@dataclass(frozen=True)
class LocalLane:
    """This box's own replica of the role.

    ``base_url`` is the internal backend URL (``http://vllm-primary:8000``),
    ``served_name`` the model id this box serves under the role, and
    ``declared`` the operator-typed lane config keyed by :data:`DECLARED_KEYS`
    (any absent key reads back as :data:`UNKNOWN`).
    """

    base_url: str
    served_name: str
    declared: Mapping[str, str] = field(default_factory=dict)
    weight: float = 1.0


@dataclass(frozen=True)
class PeerReplica:
    """One operator-declared peer replica of the role.

    ``origin`` is the peer's GATEWAY origin (never an engine URL — see the
    module docstring) and ``api_key`` this box's outbound credential for it: a
    copy of that peer's own inbound ``GATEWAY_API_KEY``, or ``""`` when the
    peer has no inbound gate (the legal empty slot, spec h13). ``repr=False``
    on the key — a secret must never reach a log, traceback or ``--json``
    dump.
    """

    origin: str
    api_key: str = field(default="", repr=False)
    weight: float = 1.0


# --- pure helpers ----------------------------------------------------------


def _as_int(value: object) -> int | None:
    """Best-effort positive int, else ``None`` (unknown — never a silent 0)."""
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _declared(declared: Mapping[str, str], key: str) -> str:
    value = str(declared.get(key, "") or "").strip()
    return value or UNKNOWN


def _known(value: object) -> bool:
    return value is not None and value != UNKNOWN and value != ""


def compare_fingerprints(local: Fingerprint | None, peer: Fingerprint | None) -> tuple[bool, str]:
    """Is *peer* poolable with *local*? Pure, and the reason names every diff.

    Only :data:`DISQUALIFYING_FIELDS` gate pooling. ``kv_cache_dtype``,
    the parsers and ``speculative_config`` are recorded and IGNORED here: the
    Spark/Thor pair is an explicit operator compatibility policy (same
    checkpoint, id, runtime and window) and differs on exactly those
    informational fields (spec c32).

    An UNKNOWN (or missing) value on EITHER side is incompatible, with the
    reason ``"<field>: unknown"`` — an unknown fingerprint never pools
    silently (spec h11).
    """
    if local is None or peer is None:
        side = "local" if local is None else "peer"
        return False, f"fingerprint: unknown ({side} replica not probed)"
    reasons: list[str] = []
    for name in DISQUALIFYING_FIELDS:
        mine = getattr(local, name)
        theirs = getattr(peer, name)
        if not _known(mine) or not _known(theirs):
            reasons.append(f"{name}: unknown")
        elif mine != theirs:
            reasons.append(f"{name}: {mine} != {theirs}")
    return (not reasons), "; ".join(reasons)


def _classify(exc: BaseException) -> str:
    """Map a probe exception to a :class:`ReplicaState` ``health`` value."""
    if isinstance(exc, TimeoutError):  # socket.timeout is TimeoutError since 3.10
        return "timeout"
    if isinstance(exc, OSError):
        return "unreachable"
    return "error"


def _busy_from_status(payload: Mapping[str, object]) -> bool:
    """The peer's own pressure verdict, from its ``GET /status`` payload.

    ``fleet_status_payload`` renders ``busy`` as an aggregate ``{running,
    waiting}`` dict and the shed decision under ``pressure`` (``mode`` /
    ``shed``), so the verdict is read from ``pressure`` — while a plain
    boolean ``busy`` is also honoured, so a peer serving a simpler status
    shape is understood rather than silently read as idle.
    """
    busy = payload.get("busy")
    if isinstance(busy, bool):
        return busy
    pressure = payload.get("pressure")
    if isinstance(pressure, Mapping):
        return bool(pressure.get("shed")) or pressure.get("mode") == "busy"
    return False


class ReplicaCache:
    """Background load + fingerprint snapshot for every replica of ONE role.

    Construction opens no socket and never blocks: every replica is seeded
    ``health="unknown"``, ``ready=False``, ``fingerprint=None`` — the honest
    pre-probe state, and the same seed-without-probing divergence
    :class:`~lobes.gateway._readiness.ReadinessCache` documents. Call
    :meth:`refresh` once before binding (so the first request has a snapshot),
    then :meth:`start` to hand subsequent passes to the daemon threads.

    Two threads, never one: the local lane and the peers refresh independently
    so a hung peer can never delay the local probe (they cannot share a
    deadline if they do not share a thread).
    """

    def __init__(
        self,
        role: str,
        local: LocalLane | None,
        peers: Sequence[PeerReplica] = (),
        *,
        refresh_interval: float = _DEFAULT_REFRESH_INTERVAL,
        probe_timeout: float = _LOCAL_PROBE_TIMEOUT,
        peer_probe_timeout: float = _PEER_PROBE_TIMEOUT,
        backend_name: str | None = None,
        urlopen: Opener | None = None,
        monotonic: Clock | None = None,
        start: bool = True,
    ) -> None:
        self._role = role
        self._local_lane = local
        # Copy so a caller mutating their sequence cannot change what we probe.
        self._peers: tuple[PeerReplica, ...] = tuple(peers)
        self._interval = refresh_interval
        self._timeout = probe_timeout
        self._peer_timeout = peer_probe_timeout
        # A peer's /status names backends by BACKEND name ("primary"), not by
        # role — but a peer may equally publish the role name. Both are
        # matched; the served id is the primary key and this the fallback.
        self._backend_name = backend_name or role
        self._urlopen: Opener = urlopen or _default_opener
        self._now: Clock = monotonic or _monotonic
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._peer_stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peer_thread: threading.Thread | None = None
        # Two independent stores, merged only for reading in current() — a
        # local pass replaces its store wholesale and must never be able to
        # wipe peer values, or vice versa (ReadinessCache's discipline).
        self._local_state: ReplicaState | None = (
            self._seed(local.base_url, True, local.weight) if local is not None else None
        )
        self._peer_states: dict[str, ReplicaState] = {
            peer.origin: self._seed(peer.origin, False, peer.weight) for peer in self._peers
        }
        if start:
            self.start()

    # --- seeding / reading -------------------------------------------------

    @staticmethod
    def _seed(origin: str, local: bool, weight: float) -> ReplicaState:
        return ReplicaState(
            origin=origin,
            local=local,
            ready=False,
            busy=False,
            health="unknown",
            running=0,
            waiting=0,
            fingerprint=None,
            compatible=local,  # the local replica is trivially compatible with itself
            reason="",
            last_seen=0.0,
            weight=weight,
        )

    def current(self) -> tuple[ReplicaState, ...]:
        """The latest snapshot: local replica first, then peers in declared order.

        A pure read under the lock — never probes, never blocks. Every
        :class:`ReplicaState` is frozen, so the returned tuple is safe to hand
        straight to the selection policy on the request path.
        """
        with self._lock:
            states: list[ReplicaState] = []
            if self._local_state is not None:
                states.append(self._local_state)
            states.extend(self._peer_states[peer.origin] for peer in self._peers)
            return tuple(states)

    def refresh(self) -> None:
        """Probe the local lane AND every peer once, synchronously.

        The public one-shot ``server.serve`` calls before it binds, so the
        very first request reads a real snapshot instead of the unknown seed.
        """
        self._refresh_local()
        if self._peers:
            self._refresh_peers()

    # --- probing -----------------------------------------------------------

    def _get_json(self, url: str, timeout: float, api_key: str | None) -> tuple[dict | None, str]:
        """GET *url* → ``(payload | None, health)``. Never raises."""
        try:
            status, body = self._urlopen(url, timeout, api_key or None)
        except BaseException as exc:  # noqa: BLE001 - every failure degrades, never crashes
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return None, _classify(exc)
        if status != 200:
            return None, "error"
        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            return None, "error"
        if not isinstance(payload, dict):
            return None, "error"
        return payload, "ok"

    def _local_fingerprint(self, payload: Mapping[str, object], lane: LocalLane):
        entries = [e for e in (payload.get("data") or []) if isinstance(e, Mapping)]
        entry = next((e for e in entries if e.get("id") == lane.served_name), None)
        if entry is None:
            entry = entries[0] if entries else None
        if entry is None:
            return None
        return Fingerprint(
            served_id=str(entry.get("id") or UNKNOWN),
            max_model_len=_as_int(entry.get("max_model_len")),
            runtime=_declared(lane.declared, "runtime"),
            quantization=_declared(lane.declared, "quantization"),
            kv_cache_dtype=_declared(lane.declared, "kv_cache_dtype"),
            reasoning_parser=_declared(lane.declared, "reasoning_parser"),
            tool_parser=_declared(lane.declared, "tool_parser"),
            speculative_config=_declared(lane.declared, "speculative_config"),
        )

    def _local_load(self, lane: LocalLane) -> tuple[int, int]:
        """This box's own in-flight counts, from the lane's ``/metrics``.

        The peer half of the snapshot reads load through the peer's gateway
        ``/status`` (the engine port is not reachable cross-box); locally the
        engine IS reachable, so the same numbers come from the same exposition
        ``/status`` aggregates, one hop shorter. An unreachable or
        non-vLLM-shaped scrape degrades to ``(0, 0)`` — the load signal is
        best-effort and never gates local dispatch on its own (t5 takes the
        authoritative local pressure verdict separately).
        """
        try:
            status, body = self._urlopen(
                lane.base_url.rstrip("/") + _METRICS_PATH, self._timeout, None
            )
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return 0, 0
        if status != 200:
            return 0, 0
        try:
            metrics = _metrics.parse_metrics(body.decode("utf-8", "replace"))
        except Exception:  # nosec B110 - best-effort; a bad scrape is not a crash
            return 0, 0
        return int(metrics.get("running", 0) or 0), int(metrics.get("waiting", 0) or 0)

    def _probe_local(self, lane: LocalLane, previous: ReplicaState) -> ReplicaState:
        payload, health = self._get_json(
            lane.base_url.rstrip("/") + _MODELS_PATH, self._timeout, None
        )
        if payload is None:
            return replace(
                previous, ready=False, health=health, fingerprint=None, running=0, waiting=0
            )
        fingerprint = self._local_fingerprint(payload, lane)
        if fingerprint is None:
            return replace(previous, ready=False, health="error", fingerprint=None)
        running, waiting = self._local_load(lane)
        return replace(
            previous,
            ready=True,
            busy=False,  # local pressure is the caller's own signal, not a probe
            health="ok",
            running=running,
            waiting=waiting,
            fingerprint=fingerprint,
            compatible=True,
            reason="",
            last_seen=self._now(),
        )

    def _peer_backend_entry(
        self, payload: Mapping[str, object], served_id: str | None
    ) -> Mapping[str, object] | None:
        backends = payload.get("backends")
        if not isinstance(backends, list):
            return None
        for entry in backends:
            if not isinstance(entry, Mapping):
                continue
            if served_id and entry.get("served_name") == served_id:
                return entry
            if entry.get("name") in (self._backend_name, self._role):
                return entry
        return None

    def _peer_fingerprint(self, payload: Mapping[str, object]):
        """Read the peer's per-role fingerprint, or fall back HONESTLY.

        When the peer publishes an explicit ``fingerprint`` object (t6), every
        field is taken from it. Otherwise only the two fields whose provenance
        is trustworthy are read from the role entry — the served id and the
        context — and every other field is :data:`UNKNOWN`: the role entry's
        own ``quant``/``runtime``/``mtp`` are CATALOG-derived and mislabel an
        unknown served id, so trusting them would pool a llama.cpp
        ``Q4_K_M`` replica as an NVFP4 vLLM one (spec c13).
        """
        roles = payload.get("roles")
        container: Mapping[str, object] = roles if isinstance(roles, Mapping) else payload
        entry = container.get(self._role)
        if not isinstance(entry, Mapping):
            return None
        raw = entry.get("fingerprint")
        if isinstance(raw, Mapping):
            return Fingerprint(
                served_id=str(raw.get("served_id") or UNKNOWN),
                max_model_len=_as_int(raw.get("max_model_len")),
                runtime=str(raw.get("runtime") or UNKNOWN),
                quantization=str(raw.get("quantization") or UNKNOWN),
                kv_cache_dtype=str(raw.get("kv_cache_dtype") or UNKNOWN),
                reasoning_parser=str(raw.get("reasoning_parser") or UNKNOWN),
                tool_parser=str(raw.get("tool_parser") or UNKNOWN),
                speculative_config=str(raw.get("speculative_config") or UNKNOWN),
            )
        return Fingerprint(
            served_id=str(entry.get("model") or UNKNOWN),
            max_model_len=_as_int(entry.get("context")),
            runtime=UNKNOWN,
            quantization=UNKNOWN,
            kv_cache_dtype=UNKNOWN,
            reasoning_parser=UNKNOWN,
            tool_parser=UNKNOWN,
            speculative_config=UNKNOWN,
        )

    def _probe_peer(
        self, peer: PeerReplica, previous: ReplicaState, local_fp: Fingerprint | None
    ) -> ReplicaState:
        origin = peer.origin.rstrip("/")
        key = peer.api_key or None
        status_payload, health = self._get_json(origin + _STATUS_PATH, self._peer_timeout, key)
        if status_payload is None:
            return replace(
                previous,
                ready=False,
                health=health,
                running=0,
                waiting=0,
                compatible=False,
                reason=f"peer gateway {health}",
            )
        last_seen = self._now()
        busy = _busy_from_status(status_payload)
        served_id = local_fp.served_id if local_fp is not None else None
        entry = self._peer_backend_entry(status_payload, served_id)
        if entry is None:
            # The peer answered, but nothing on it serves this role.
            return replace(
                previous,
                ready=False,
                busy=busy,
                health="unknown",
                running=0,
                waiting=0,
                fingerprint=None,
                compatible=False,
                reason=f"peer does not serve role {self._role}",
                last_seen=last_seen,
            )
        metrics = entry.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        backend_health = str(entry.get("health") or "unknown")
        caps_payload, _caps_health = self._get_json(
            origin + _CAPABILITIES_PATH, self._peer_timeout, key
        )
        fingerprint = self._peer_fingerprint(caps_payload) if caps_payload is not None else None
        compatible, reason = compare_fingerprints(local_fp, fingerprint)
        return replace(
            previous,
            ready=backend_health == "ok",
            busy=busy,
            health=backend_health,
            running=int(metrics.get("running", 0) or 0),
            waiting=int(metrics.get("waiting", 0) or 0),
            fingerprint=fingerprint,
            compatible=compatible,
            reason=reason,
            last_seen=last_seen,
        )

    # --- refresh passes ----------------------------------------------------

    def _refresh_local(self) -> None:
        lane = self._local_lane
        if lane is None:
            return
        with self._lock:
            previous = self._local_state
        if previous is None:  # pragma: no cover - set together with _local_lane
            return
        try:
            state = self._probe_local(lane, previous)
        except Exception:  # nosec B110 - best-effort; never kill the daemon thread
            state = replace(previous, ready=False, health="error", fingerprint=None)
        with self._lock:
            self._local_state = state

    def _refresh_peers(self) -> None:
        with self._lock:
            previous = dict(self._peer_states)
            local_fp = self._local_state.fingerprint if self._local_state else None
        updated: dict[str, ReplicaState] = {}
        for peer in self._peers:
            seed = previous.get(peer.origin) or self._seed(peer.origin, False, peer.weight)
            try:
                updated[peer.origin] = self._probe_peer(peer, seed, local_fp)
            except Exception:  # nosec B110 - one bad peer never aborts the pass
                updated[peer.origin] = replace(
                    seed, ready=False, health="error", compatible=False, reason="probe failed"
                )
        with self._lock:
            self._peer_states = updated

    # --- daemon threads (mirrors ReadinessCache) ---------------------------

    def start(self) -> None:
        """Start the background refresh thread(s) (idempotent).

        The peer thread is spawned only when peers are declared — a box with
        no replica set spawns exactly one thread, and a box with no local lane
        and no peers spawns none.
        """
        if self._local_lane is not None:
            self._thread = self._start_thread(
                self._thread, self._stop, self._loop, "lobes-replica-cache"
            )
        if self._peers:
            self._peer_thread = self._start_thread(
                self._peer_thread, self._peer_stop, self._peer_loop, "lobes-replica-peer-cache"
            )

    @staticmethod
    def _start_thread(
        existing: threading.Thread | None,
        stop: threading.Event,
        target: Callable[[], None],
        name: str,
    ) -> threading.Thread | None:
        """Single-live-thread invariant, shared by both loops.

        A still-ALIVE thread is left alone (never a second, overlapping
        refresh thread on top of one still probing); a stale reference to an
        already-exited thread — left behind by :meth:`stop` when its bounded
        join timed out — is cleared so the cache restarts cleanly.
        """
        if existing is not None and existing.is_alive():
            return existing
        stop.clear()
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread

    def _loop(self) -> None:
        self._refresh_local()
        while not self._stop.wait(self._interval):
            self._refresh_local()

    def _peer_loop(self) -> None:
        self._refresh_peers()
        while not self._peer_stop.wait(self._interval):
            self._refresh_peers()

    def stop(self) -> None:
        """Signal both daemon threads to exit and join them (idempotent).

        Each join is bounded by that pass's own worst case (sequential probes
        × timeout, plus a margin) so a clean shutdown normally completes here.
        A thread still alive afterwards keeps its reference rather than being
        orphaned — clearing it would let a later :meth:`start` spawn a second,
        overlapping refresher; the stop flag stays set, so it exits at its
        next wait boundary and :meth:`_start_thread` clears the dead reference
        then. Both threads are daemons, so a straggler never blocks exit.
        """
        self._stop.set()
        self._peer_stop.set()
        # A local pass makes at most two requests (models + metrics).
        self._thread = self._join(self._thread, 2 * self._timeout + 1.0)
        self._peer_thread = self._join(
            self._peer_thread, max(len(self._peers), 1) * 2 * self._peer_timeout + 1.0
        )

    @staticmethod
    def _join(thread: threading.Thread | None, bound: float) -> threading.Thread | None:
        if thread is None:
            return None
        thread.join(timeout=bound)
        return None if not thread.is_alive() else thread

    # Explicit alias — server shutdown code reads more naturally as close().
    close = stop

    def is_alive(self) -> bool:
        """True while the LOCAL refresh thread is running (mirrors ReadinessCache)."""
        return self._thread is not None and self._thread.is_alive()
