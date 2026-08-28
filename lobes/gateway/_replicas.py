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

Local in-flight accounting
--------------------------
Probed load is up to one refresh interval (5 s) stale, and nothing used to be
counted when a request was actually DISPATCHED — so a burst of concurrent
arrivals all read one snapshot and stampeded the same replica. An accurate
capacity makes that worse, not better (today's uniform weight of 1.0 at least
made ties resolve to local, keeping bursts put). So this module now counts
this box's own outstanding dispatches — :meth:`ReplicaCache.dispatch` /
:meth:`~ReplicaCache.begin_dispatch` / :meth:`~ReplicaCache.end_dispatch`,
driven by ``server.py`` — and RECONCILES them against the probed number
(:meth:`~ReplicaCache._reconcile`), so the snapshot self-corrects between
refreshes in BOTH directions. Any dispatch that outlives
:data:`INFLIGHT_MAX_AGE` is dropped, so a leaked counter can never make a
healthy box look permanently full.

Reconciliation, not addition: a probe and a local token can each see the same
request, and simply adding them double-counts while simply preferring the
probe undercounts. Both errors were observed. The rule is at
:meth:`~ReplicaCache._reconcile`; the two failures it exists to close are:

* **stale-LOW** — a dispatch is registered BEFORE the dial, so a probe that
  completes in the window between the token and the request reaching the
  engine samples neither, and discarding the token because it predates
  ``last_seen`` loses the request from both sides at once. That is the burst
  undercount this accounting exists to prevent, recreated by its own
  de-duplication.
* **stale-HIGH** — measured live in the t10 acceptance AMENDMENT: a probe
  reported the peer at ``run=2`` from a burst that had ALREADY completed. At
  capacity 2 that read as full, the peer became unselectable, and one run in
  five forwarded nothing and fell back to single-box (50.82 tok/s against
  98.6). Our OWN completions are first-hand evidence that probed load is
  stale, so they are reconciled against it too.

Capacity (capacity-relative pool routing)
-----------------------------------------
Beyond load and fingerprint this module now supplies a third fact: each
replica's CAPACITY — its measured max active requests — carried on
``ReplicaState.weight``, which `_selection.py` divides load by to rank
replicas by UTILISATION rather than by raw queue depth. A peer publishes its
own (spec q1) on the ``/status`` body already probed here; the local lane's
comes from this box's declared ``<PREFIX>_MAX_ACTIVE``. Every capacity —
local or peer, published or declared — passes through the one
:func:`resolve_capacity` gate, which CLAMPS it and records the clamp in
``reason``: capacity is peer-controlled input that the ranking arithmetic
divides by, so an unbounded value would rank as near-zero wait at every load
level and vacuum the whole pool. A capacity is also KEYED to the fingerprint
it was measured under and discarded when that fingerprint changes, since a
number measured against one checkpoint at one window says nothing about the
next one.

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
import itertools
import json
import threading
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from time import monotonic as _monotonic
from typing import Callable
from urllib.parse import urlsplit

from .. import _metrics
from ._selection import UNCALIBRATED_WEIGHT

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
# How long an OUTSTANDING dispatch keeps counting before it is presumed
# leaked and dropped from the in-flight tally. A leaked counter is the one
# serious failure mode of dispatch accounting: it can only ever grow, it makes
# a perfectly healthy box look more loaded than it is, and — once the tally
# reaches the box's capacity — permanently FULL, with no path back. The
# context-manager API makes a leak hard to write in the first place; this
# expiry makes even a leaked one self-heal. Generous on purpose: it must sit
# well beyond the longest legitimate generation (a 256K-context reasoning turn
# on the Thor's 12.1 tok/s cortex is minutes, not seconds), because expiring a
# LIVE request would under-count load and re-create the stampede.
INFLIGHT_MAX_AGE: float = 900.0

# How long a COMPLETED dispatch stays on the reconciliation ledger. A
# completion is evidence against a probe only until a probe newer than it
# lands, so this only has to outlast the probe interval by a healthy margin;
# past that the entry can say nothing a fresh probe has not already said.
# Kept small deliberately — unlike the in-flight tally this ledger grows with
# THROUGHPUT, not with concurrency.
COMPLETION_RETENTION: float = 60.0

# Hard cap on ledger entries, so a burst far larger than any measured fleet
# concurrency still cannot grow it without bound between prunes. Oldest
# entries fall off first, which is also the right order to lose them in: the
# oldest are the ones a newer probe has already superseded.
_COMPLETION_LEDGER_MAX: int = 1024

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

# --- capacity (capacity-relative pool routing) ------------------------------
#
# ``weight`` on every dataclass below is the replica's CAPACITY: its measured
# max active requests, the throughput knee `lobes/assess.py`'s
# ``calibration_knee`` produces — deliberately NOT vLLM's ``--max-num-seqs``
# OOM-safety cap. :data:`~lobes.gateway._selection.UNCALIBRATED_WEIGHT` (1.0)
# is the "nothing published" SENTINEL, not a measured one-slot capacity; it is
# imported rather than re-typed here so the producer (this module) and the
# consumer (`_selection.py`) can never drift apart on its value.
#
# A peer publishes its own capacity on the ``/status`` body this module
# already probes (spec q1), under this key on the role's backend entry:
#
#     {"backends": [{"name": "primary", ..., "capacity": 8}]}
#
# ``lobes/gateway/server.py``'s ``fleet_status_payload`` is what writes it
# (task t5); an older lobes, or a non-lobes replica, publishes nothing and
# falls back to the sentinel — an unpublished capacity NEVER makes a replica
# unselectable (spec h3).
PEER_CAPACITY_KEY = "capacity"

# The ceiling any ingested capacity is CLAMPED to, local and peer alike.
#
# Why a clamp at all: capacity arrives as peer-CONTROLLED input (nothing
# validates what a declared peer answers — spec c19/s11, and note the Thor
# currently serves /status with no inbound gate at all), and
# ``_selection.estimated_wait`` DIVIDES by it. A peer publishing 10000 would
# score a near-zero wait at every load level and silently vacuum the entire
# pool — a black hole with no error anywhere. So a received capacity is
# bounded on ingest, and the clamp is RECORDED in the replica row's ``reason``
# rather than applied silently (spec h13).
#
# Why 64: it sits above every concurrency figure this fleet has ever
# MEASURED — the worker lane's 54.33x KV-derived concurrency at 65K is the
# largest number in docs/evidence/, and the cortex pair's measured knees are
# single digits — so the clamp cannot clip an honest calibration, while an
# absurd published value still captures at most a bounded share of traffic.
# It is a constructor argument (``capacity_max``), not a hard constant, so a
# deployment whose hardware genuinely outgrows it can raise it without a code
# change.
CAPACITY_CLAMP_MAX: float = 64.0

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

    ``compatible``/``reason`` are the honesty pair: ``reason`` names every
    differing field, so ``/capabilities`` and the CLI can show an operator WHY
    a declared replica is not pooling instead of leaving it silently absent.
    ``reason`` ALSO carries capacity notes (a clamp, a refused value, a
    capacity discarded on a fingerprint change) — see :func:`_with_note` for
    why that field and not a new one — so it is no longer empty exactly when
    ``compatible`` is true.
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
    # The RESOLVED capacity `_selection.py` ranks by: the ingested value after
    # the clamp and the kill switch, or `UNCALIBRATED_WEIGHT` when nothing was
    # published (or the published value was refused/discarded).
    weight: float = UNCALIBRATED_WEIGHT
    # The RAW capacity as published by that replica, pre-clamp — `None` when
    # none was published. Kept beside `weight` so an operator can see both the
    # number the peer claimed and the number this box actually used. NOTE it
    # is deliberately RETAINED when a pinned capacity is discarded on a
    # fingerprint change (that is what lets the next pass recognise the same
    # stale number and discard it again), so it is NOT the "is a capacity in
    # force?" signal — `calibrated`, below, is.
    capacity: float | None = None
    # Whether `weight` is a capacity actually IN FORCE — the signal
    # `_selection.is_calibrated` reads. False for: nothing published, a
    # refused (non-positive/non-finite) value, a capacity discarded on a
    # fingerprint change, and the fleet-wide kill switch. True for a published
    # capacity of exactly ONE SLOT, which is a legal measurement (a box whose
    # engine admits one request at a time) and must not be confused with the
    # `UNCALIBRATED_WEIGHT` fallback that happens to share its value.
    calibrated: bool = False
    # This box's OWN outstanding dispatches to this replica that the last probe
    # has not seen yet (see :meth:`ReplicaCache.in_flight`). Already folded
    # into `waiting` by :meth:`ReplicaCache.current`, and reported separately
    # so a consumer can recover the purely probed number by subtracting it.
    in_flight: int = 0
    # The fingerprint `capacity` was PINNED to: the live fingerprint under
    # which that number first arrived. A capacity is only valid for the
    # checkpoint/window/runtime it was measured on, so when the live
    # fingerprint stops matching this pin the capacity is discarded (`weight`
    # reverts to the sentinel) until a NEW number is published. `None` = no
    # capacity pinned. See :func:`_fingerprint_changed`.
    capacity_fingerprint: "Fingerprint | None" = None

    def evolve(self, **changes: object) -> "ReplicaState":
        """Typed wrapper over :func:`dataclasses.replace`.

        ``dataclasses.replace`` is typed as returning ``DataclassInstance`` in
        the stdlib stubs (Sonar S5886), which loses the concrete
        :class:`ReplicaState` type at every call site below. This method just
        narrows the return annotation back to ``ReplicaState`` — same runtime
        behaviour, precise type.
        """
        return replace(self, **changes)


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
    # THIS box's own declared capacity for the role — the operator-typed
    # ``<PREFIX>_MAX_ACTIVE`` that reaches the gateway as
    # ``ServerConfig.local_capacities`` (task t5 wires the call site).
    # ``None`` means UNDECLARED. It is `None` and not a sentinel number
    # precisely because ``<PREFIX>_MAX_ACTIVE=1`` is a legal declaration (an
    # engine that admits one request at a time), so no numeric value is free
    # to stand for "nothing here".
    weight: float | None = None


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
    # An operator-declared FALLBACK capacity for this peer, used only when the
    # peer publishes none of its own on ``/status`` (q1 makes the peer the
    # authority on its own capacity). ``None`` = undeclared — see
    # :class:`LocalLane`'s ``weight`` for why not a sentinel number.
    weight: float | None = None


# --- pure helpers ----------------------------------------------------------


def _as_int(value: object) -> int | None:
    """Best-effort positive int, else ``None`` (unknown — never a silent 0)."""
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def resolve_capacity(
    raw: object,
    *,
    capacity_max: float = CAPACITY_CLAMP_MAX,
    kill_switch: bool = False,
) -> tuple[float, str]:
    """Turn a raw published/declared capacity into ``(weight, note)``. PURE.

    The single ingest gate every capacity passes through, local and peer
    alike — so the clamp and the kill switch cannot apply to one side and not
    the other. ``note`` is empty when nothing worth telling the operator
    happened, and otherwise names exactly what this box did to the number; the
    caller folds it into :attr:`ReplicaState.reason`, which is what makes the
    clamp OBSERVABLE rather than silent (spec h13).

    Four outcomes:

    * kill switch engaged → the sentinel, silently. Pinning capacity fleet-wide
      is a deliberate operator action (``GATEWAY_CAPACITY_KILL_SWITCH``), not
      an anomaly to report per replica.
    * ``None`` (nothing published) → the sentinel, silently. An unpublished
      capacity is the normal state of an older lobes or a non-lobes replica
      and must never look like an error (spec h3).
    * present but not a positive finite number → the sentinel, WITH a note. A
      misdeclaration is not a capacity, and swallowing it silently would leave
      an operator staring at a knob that does nothing.
    * above *capacity_max* → clamped, WITH a note naming both numbers.
    """
    if kill_switch:
        return UNCALIBRATED_WEIGHT, ""
    if raw is None:
        return UNCALIBRATED_WEIGHT, ""
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = float("nan")
    # NaN fails every comparison, so this one check refuses NaN, inf, zero and
    # negatives together.
    if not (value > 0.0) or value == float("inf"):
        return UNCALIBRATED_WEIGHT, f"capacity ignored: {raw!r} is not a positive number"
    if value > capacity_max:
        return capacity_max, f"capacity clamped: {value:g} -> {capacity_max:g}"
    return value, ""


def _as_capacity(raw: object) -> float | None:
    """The raw published capacity as a float, or ``None`` when it is absent or
    not a positive finite number (the values :func:`resolve_capacity` refuses).

    Stored on :attr:`ReplicaState.capacity` PRE-clamp, so an operator can see
    what the peer claimed next to what this box used — and so the fingerprint
    keying below can tell "the same measurement again" from "a new one".
    """
    if raw is None:
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (value > 0.0) or value == float("inf"):
        return None
    return value


def capacity_is_in_force(raw: object, *, kill_switch: bool = False) -> bool:
    """Whether *raw* becomes a capacity this box will actually RANK by. PURE.

    The companion predicate to :func:`resolve_capacity`, and deliberately a
    separate answer from the weight it returns: `resolve_capacity` falls back
    to :data:`UNCALIBRATED_WEIGHT` for "nothing published", for a refused
    value and for the kill switch, and a legitimately published capacity of
    one slot resolves to that same number. Comparing the weight against the
    fallback therefore cannot tell the two apart — this can.

    Refuses exactly what :func:`resolve_capacity` refuses, by reusing
    :func:`_as_capacity` (the single definition of "a usable capacity"), so
    the two can never drift. A CLAMPED capacity is still in force: the clamp
    bounds the number, it does not reject it.
    """
    return not kill_switch and _as_capacity(raw) is not None


def _with_active(state: ReplicaState, active: int) -> ReplicaState:
    """*state* with its active count restated as *active*, recording the
    signed adjustment on ``in_flight``.

    ``running`` is the more literal of the two probed numbers, so it is kept
    intact wherever the new total still covers it and ``waiting`` absorbs the
    rest; a total BELOW the probed ``running`` (the stale-HIGH correction, where
    this box knows requests the probe counted have since finished) trims
    ``running`` down, since leaving it alone would leave the very load the
    correction just cancelled sitting in the field selection sums.
    """
    running = min(state.running, active)
    return state.evolve(
        in_flight=active - (state.running + state.waiting),
        running=running,
        waiting=active - running,
    )


def _with_note(reason: str, note: str) -> str:
    """Fold a capacity *note* into a compatibility *reason*, ``"; "``-joined.

    NOTE the deliberate renegotiation of :class:`ReplicaState`'s original
    invariant ("``reason`` is empty exactly when ``compatible`` is true"): a
    COMPATIBLE replica may now carry a capacity note. The spec requires the
    clamp to be recorded in the replica row's reason field, and a second field
    that only sometimes carries a message would just be a reason field with a
    different name.
    """
    return "; ".join(part for part in (reason, note) if part)


def _declared(declared: Mapping[str, str], key: str) -> str:
    value = str(declared.get(key, "") or "").strip()
    return value or UNKNOWN


_OWNED_BY_RUNTIME: Mapping[str, str] = {
    "vllm": "vllm",
    "llamacpp": "llamacpp",
    "llama.cpp": "llamacpp",
}


def _runtime_from(entry: Mapping[str, object], declared: Mapping[str, str]) -> str:
    """The lane's runtime: declared ``<PREFIX>_RUNTIME`` if set, else LIVE from
    ``/v1/models``' ``owned_by`` (vLLM reports ``"vllm"``), else ``unknown``.

    No lane declares a runtime knob today, and without this fallback every
    replica would carry ``runtime: unknown`` and never pool (the unknown-rule
    disqualifies it) — while the engine already says who it is on the wire.
    Only the two engines lobes serves are mapped; anything else stays unknown
    rather than being guessed (issue #199, t2/t6 reconcile).
    """
    value = _declared(declared, "runtime")
    if value != UNKNOWN:
        return value
    owned = str(entry.get("owned_by") or "").strip().lower()
    return _OWNED_BY_RUNTIME.get(owned, UNKNOWN)


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


def _fingerprint_changed(pin: Fingerprint | None, live: Fingerprint | None) -> str:
    """Has the fingerprint a capacity was measured under DEFINITELY changed?

    Returns ``""`` for "no evidence of a change" and otherwise a reason naming
    every changed field. Only :data:`DISQUALIFYING_FIELDS` are consulted —
    the informational fields (parsers, ``kv_cache_dtype``, drafter) differ
    across the Spark+Thor pair by explicit operator policy (spec c32) and must
    not throw a measured capacity away.

    Deliberately CONSERVATIVE where :func:`compare_fingerprints` is strict: an
    absent or ``unknown`` value on either side is NOT a change. The two
    functions answer different questions — "may these two replicas pool
    together?" (where silence must never pass) versus "did this one replica's
    serving identity change under a number we already hold?" (where silence is
    not evidence of a switch, and inventing one would throw away a good
    capacity every time a probe came back thin).
    """
    if pin is None or live is None:
        return ""
    changed = [
        f"{name}: {getattr(pin, name)} != {getattr(live, name)}"
        for name in DISQUALIFYING_FIELDS
        if _known(getattr(pin, name))
        and _known(getattr(live, name))
        and getattr(pin, name) != getattr(live, name)
    ]
    return "; ".join(changed)


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
        capacity_max: float = CAPACITY_CLAMP_MAX,
        capacity_kill_switch: bool = False,
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
        # Capacity ingest policy, applied identically to the local lane and to
        # every peer (see :func:`resolve_capacity`).
        self._capacity_max = capacity_max
        self._capacity_kill_switch = capacity_kill_switch
        self._urlopen: Opener = urlopen or _default_opener
        self._now: Clock = monotonic or _monotonic
        self._lock = threading.Lock()
        # Dispatch bookkeeping gets its OWN lock: a dispatch must never wait on
        # a probe pass writing the snapshot, and vice versa.
        self._inflight_lock = threading.Lock()
        self._inflight: dict[int, tuple[str, float]] = {}
        # (origin, start, end) for recently COMPLETED dispatches — the
        # evidence half of :meth:`_reconcile`'s stale-HIGH correction. Bounded
        # by both age (:data:`COMPLETION_RETENTION`) and length.
        self._completed: deque[tuple[str, float, float]] = deque(maxlen=_COMPLETION_LEDGER_MAX)
        self._inflight_tokens = itertools.count(1)
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

    def _seed(self, origin: str, local: bool, weight: float | None) -> ReplicaState:
        """The honest pre-probe state for one replica.

        *weight* is the OPERATOR-DECLARED capacity carried by the
        :class:`LocalLane` / :class:`PeerReplica` — this box's own
        ``<PREFIX>_MAX_ACTIVE`` for the local lane. It goes through the same
        :func:`resolve_capacity` gate a probed peer capacity does, so the
        clamp and the kill switch hold from the very first snapshot, before
        any probe has run.
        """
        resolved, note = self._resolve(weight)
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
            reason=note,
            last_seen=0.0,
            weight=resolved,
            capacity=weight,
            calibrated=self._in_force(weight),
        )

    def _resolve(self, raw: object) -> tuple[float, str]:
        """This cache's ingest gate — :func:`resolve_capacity` bound to its
        configured clamp and kill switch."""
        return resolve_capacity(
            raw,
            capacity_max=self._capacity_max,
            kill_switch=self._capacity_kill_switch,
        )

    def _in_force(self, raw: object) -> bool:
        """:func:`capacity_is_in_force` bound to this cache's kill switch."""
        return capacity_is_in_force(raw, kill_switch=self._capacity_kill_switch)

    def _keyed_capacity(
        self,
        previous: ReplicaState,
        raw: object,
        live: Fingerprint | None,
    ) -> tuple[float, float | None, bool, Fingerprint | None, str]:
        """Ingest *raw* against the live fingerprint → ``(weight, capacity,
        calibrated, pin, note)``.

        The pin travels with the NUMBER, not with the probe: as long as the
        replica keeps publishing the same capacity, the fingerprint it first
        arrived under is kept and re-validated every pass. A DEFINITE change in
        that fingerprint discards the capacity — and the pin is RETAINED
        alongside the discarded number so the very next pass, which sees the
        same stale number republished, discards it again. (Dropping the pin on
        discard would make the next pass read the stale number as brand new and
        re-pin it to the new fingerprint — the capacity would resurrect itself
        one refresh later, which is the exact failure this keying exists to
        prevent.)

        A capacity whose VALUE changes is a new measurement by definition, so
        it re-pins to whatever fingerprint is live now: after a `lobes switch`,
        routing falls back to the safe default until the operator recalibrates
        (spec h16).
        """
        capacity = _as_capacity(raw)
        if capacity is None:
            # Nothing usable published — resolve (for the "refused value" note)
            # and hold no pin.
            resolved, note = self._resolve(raw)
            return resolved, None, False, None, note
        republished = previous.capacity == capacity and previous.capacity_fingerprint is not None
        pin = previous.capacity_fingerprint if republished else live
        if republished:
            changed = _fingerprint_changed(pin, live)
            if changed:
                return (
                    UNCALIBRATED_WEIGHT,
                    capacity,
                    False,
                    pin,
                    f"capacity discarded: measured under a different fingerprint ({changed})",
                )
        resolved, note = self._resolve(raw)
        return resolved, capacity, self._in_force(raw), pin, note

    def current(self) -> tuple[ReplicaState, ...]:
        """The latest snapshot: local replica first, then peers in declared order.

        A pure read under the lock — never probes, never blocks. Every
        :class:`ReplicaState` is frozen, so the returned tuple is safe to hand
        straight to the selection policy on the request path.

        Each state's load is this box's own dispatch bookkeeping RECONCILED
        against the probed number (:meth:`_reconcile`), with the signed
        adjustment reported separately as ``in_flight`` (so the purely probed
        number is recoverable by subtraction). That reconciliation is what
        makes a burst self-correct: probed load alone is up to one refresh
        interval stale, so N concurrent arrivals would all read one idle
        snapshot and stampede a single replica — exactly the herd that an
        accurate capacity makes WORSE, since a genuinely least-full peer
        attracts the whole burst instead of ties resolving to local.
        """
        states = self._raw_states()
        if not self._inflight and not self._completed:
            return tuple(states)  # the common case: nothing to reconcile
        now = self._now()
        folded: list[ReplicaState] = []
        for state in states:
            probed = state.running + state.waiting
            adjusted = self._reconcile(state.origin, probed, state.last_seen, now)
            folded.append(state if adjusted == probed else _with_active(state, adjusted))
        return tuple(folded)

    # --- local in-flight accounting ---------------------------------------
    #
    # The API `server.py` (t5) drives. THREE entry points, in order of
    # preference:
    #
    #   with cache.dispatch(origin):      # leak-proof: releases on any exit
    #       ...
    #
    #   token = cache.begin_dispatch(origin)   # for a streamed response whose
    #   ...                                    # completion is not lexically
    #   cache.end_dispatch(token)              # scoped
    #
    # `end_dispatch` is idempotent and accepts ``None``/an unknown token, so a
    # double release (a retry path plus a finally, say) is a no-op rather than
    # a negative count, and every entry expires after
    # :data:`INFLIGHT_MAX_AGE` regardless — three independent guards against
    # the one failure mode that cannot self-heal, a leaked counter making a
    # healthy box look permanently full.

    def begin_dispatch(self, origin: str) -> int:
        """Record one outstanding dispatch to *origin*; returns its token."""
        now = self._now()
        token = next(self._inflight_tokens)
        with self._inflight_lock:
            # Opportunistic prune — bounded work on a path that already has the
            # lock, so a leak never accumulates unboundedly either.
            for stale in [
                tok for tok, (_o, start) in self._inflight.items() if now - start > INFLIGHT_MAX_AGE
            ]:
                self._inflight.pop(stale, None)
            self._inflight[token] = (origin, now)
        return token

    def end_dispatch(self, token: int | None) -> None:
        """Release the dispatch *token*, and remember that it FINISHED.

        Idempotent; ``None`` (and an unknown token) is a no-op, and only a
        token that was actually outstanding lands on the completion ledger —
        so a double release can neither go negative nor record the same
        completion twice.
        """
        if token is None:
            return
        now = self._now()
        with self._inflight_lock:
            entry = self._inflight.pop(token, None)
            if entry is None:
                return
            origin, start = entry
            self._completed.append((origin, start, now))
            self._prune_completed(now)

    def _prune_completed(self, now: float) -> None:
        """Drop ledger entries older than :data:`COMPLETION_RETENTION`.

        Called with ``_inflight_lock`` held. The deque is append-ordered by
        completion time, so this is a bounded walk from the left.
        """
        cutoff = now - COMPLETION_RETENTION
        while self._completed and self._completed[0][2] < cutoff:
            self._completed.popleft()

    @contextmanager
    def dispatch(self, origin: str) -> Iterator[int]:
        """Count one dispatch to *origin* for the duration of the block.

        The preferred call shape: the release is in a ``finally``, so an
        exception, an early ``return`` or a cancelled upstream cannot leak the
        counter.
        """
        token = self.begin_dispatch(origin)
        try:
            yield token
        finally:
            self.end_dispatch(token)

    def in_flight(self, origin: str, *, since: float | None = None) -> int:
        """The SIGNED adjustment this box's own bookkeeping makes to *origin*'s
        probed load — what :meth:`current` folds in. See :meth:`_reconcile`.

        Positive when the probe has not seen work this box knows is out there,
        NEGATIVE when the probe is still reporting work this box knows has
        finished, and zero when the two agree (the steady state). *since*
        overrides the probe stamp reconciled against; by default it is that
        replica's own ``last_seen``.
        """
        probed = 0
        for state in self._raw_states():
            if state.origin == origin:
                probed = state.running + state.waiting
                if since is None:
                    since = state.last_seen
                break
        return self._reconcile(origin, probed, since or 0.0, self._now()) - probed

    def _reconcile(self, origin: str, probed: int, since: float, now: float) -> int:
        """*origin*'s active count: the probe corrected by our own dispatches.

        ``since`` is the probe's own stamp. A dispatch is registered before the
        dial and released after the answer, so relative to that stamp each of
        this box's dispatches to *origin* falls into exactly one bucket:

        * **new** — started at/after the stamp. The probe cannot have seen it;
          it is added outright. (This is the whole of the pre-F6 behaviour.)
        * **outstanding-old** — started before the stamp and still running. The
          probe MAY have seen it, but only if the request had actually reached
          the engine when the sample was taken — the F6 race is precisely that
          it had not. The probe's own count bounds how many of these it can
          account for, so anything beyond ``probed`` is work the probe missed
          and is added.
        * **finished-old** — started before the stamp and finished after it.
          The probe may have counted it, and it is demonstrably gone now, so it
          is subtracted (bounded by ``probed``: we can only cancel load the
          probe actually reported).

        Both corrections are bounded by ``probed`` and therefore self-limiting:
        this box never claims a peer is running work it never reported, and
        never cancels more than it reported. Third-party load (another box
        dispatching to the same peer) is included in ``probed`` and can make
        the subtraction conservative for at most one refresh interval, after
        which a fresh probe is authoritative again.
        """
        with self._inflight_lock:
            outstanding = [
                start
                for entry_origin, start in self._inflight.values()
                if entry_origin == origin and now - start <= INFLIGHT_MAX_AGE
            ]
            finished_old = sum(
                1
                for entry_origin, start, end in self._completed
                if entry_origin == origin and start < since <= end
            )
        new = sum(1 for start in outstanding if start >= since)
        outstanding_old = len(outstanding) - new
        remaining = probed - min(probed, finished_old)
        return max(remaining, outstanding_old) + new

    def _count_inflight(self, origin: str, since: float, now: float) -> int:
        """Outstanding dispatches to *origin* started at/after *since*.

        The raw tally, with no probe reconciliation — :meth:`_reconcile` is
        what the snapshot uses.
        """
        with self._inflight_lock:
            entries = list(self._inflight.values())
        return sum(
            1
            for entry_origin, start in entries
            if entry_origin == origin and start >= since and now - start <= INFLIGHT_MAX_AGE
        )

    def _raw_states(self) -> tuple[ReplicaState, ...]:
        """The snapshot exactly as probed — no in-flight fold. The lock is held
        only for the dict reads; every state is frozen."""
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
        except BaseException as exc:  # noqa: BLE001
            # Every failure degrades, never crashes.
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
            runtime=_runtime_from(entry, lane.declared),
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
            return previous.evolve(
                ready=False, health=health, fingerprint=None, running=0, waiting=0
            )
        fingerprint = self._local_fingerprint(payload, lane)
        if fingerprint is None:
            return previous.evolve(ready=False, health="error", fingerprint=None)
        running, waiting = self._local_load(lane)
        raw = lane.weight
        resolved, capacity, calibrated, pin, note = self._keyed_capacity(previous, raw, fingerprint)
        return previous.evolve(
            ready=True,
            busy=False,  # local pressure is the caller's own signal, not a probe
            health="ok",
            running=running,
            waiting=waiting,
            fingerprint=fingerprint,
            compatible=True,
            reason=note,
            last_seen=self._now(),
            weight=resolved,
            capacity=capacity,
            calibrated=calibrated,
            capacity_fingerprint=pin,
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
            return previous.evolve(
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
            return previous.evolve(
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
        # The peer is the authority on its OWN capacity (q1): a value it
        # publishes wins over the operator's fallback declaration for it.
        raw = entry.get(PEER_CAPACITY_KEY)
        if raw is None:
            raw = peer.weight
        resolved, capacity, calibrated, pin, note = self._keyed_capacity(previous, raw, fingerprint)
        return previous.evolve(
            ready=backend_health == "ok",
            busy=busy,
            health=backend_health,
            running=int(metrics.get("running", 0) or 0),
            waiting=int(metrics.get("waiting", 0) or 0),
            fingerprint=fingerprint,
            compatible=compatible,
            reason=_with_note(reason, note),
            last_seen=last_seen,
            weight=resolved,
            capacity=capacity,
            calibrated=calibrated,
            capacity_fingerprint=pin,
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
            state = previous.evolve(ready=False, health="error", fingerprint=None)
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
                updated[peer.origin] = seed.evolve(
                    ready=False, health="error", compatible=False, reason="probe failed"
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
