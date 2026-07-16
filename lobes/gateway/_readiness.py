"""Background readiness cache for the fleet backends — "answered recently" vs "configured".

Why this module exists
----------------------
``RoleInfo.ready`` is an alias of ``loaded`` — a *config* fact (is this backend
wired in this deployment) that its own docstring admits is not a probe. That is
why ``GET /capabilities`` can advertise ``ready: true`` for an endpoint that
404s: a backend can be configured yet not answering. This cache supplies the
missing signal — a bounded, **background** probe of each backend's ``/health``
so the gateway can tell a backend that *answered recently* from one that is
*merely configured*. A later task wires :meth:`ReadinessCache.current` into
``GET /v1/models`` and ``GET /capabilities``; this module is that cache,
standalone and fully unit-tested.

Tri-state, never two-state
--------------------------
Readiness is **tri-state**, matching :func:`lobes.gateway.server.probe_audio_ready`
(issue #89):

* ``True``  — reached the backend and it answered HTTP 200 → a request will
  round-trip right now.
* ``False`` — reached the backend but it answered non-200 (e.g. 503 while the
  engine warms up) → advertised, reachable, but not yet consumable.
* ``None``  — could not reach the backend at all (connection refused / timeout /
  malformed URL) → readiness is *unknown*.

``False`` and ``None`` are deliberately **not** collapsed: "reachable but
warming" and "cannot reach at all" are different operational states and a caller
(or a later ``/capabilities`` overlay) needs to distinguish them.

Peer probing — proxied roles (task t4, issues #115/#127)
----------------------------------------------------------
Proxy-lobes t1 (already merged) added ``RoutingTable.peer_origins`` /
``peer_proxied`` / ``peer_api_keys`` as CONFIG ONLY — nothing dialed a peer.
This module now ALSO probes a proxied role's declared peer, via
:class:`PeerSpec` + :func:`probe_peer_ready`, so a later task can fold the
result into ``GET /v1/models`` / ``GET /capabilities``: the #92 "advertised
implies reachable" rule, extended across a box boundary. A proxied role is
honestly ready only when the PEER answers ``GET /v1/models`` with HTTP 200
**and** actually lists the served id this box would forward to it — not
merely when the peer's process answers HTTP at all. Every other outcome
(non-200, malformed body, id missing, connect refused, timeout) collapses to
``False``; unlike a local backend's tri-state, a peer probe OUTCOME is never
``None`` — ``None`` here is reserved for "never probed yet" (the same
seed-before-first-probe convention below), not for a failed probe.

Peer results are probed on a **separate background thread** from local
backends, with their own configurable timeout (never borrowing the local
probe's socket budget) — so a hanging or flapping peer can never delay or
fail local-backend probing (a different thread cannot share a deadline with
this one). Both land in the same ``.current()`` snapshot, keyed by backend/
role name, so a caller reads one uniform tri-state map regardless of whether
a name is served locally or proxied to a peer.

Threading discipline — mirror of :class:`lobes.gateway._tier_request.PressureCache`
-----------------------------------------------------------------------------------
This mirrors the shape, naming and threading discipline of ``PressureCache``: a
single background **daemon** thread refreshes an in-memory snapshot on an
interval, and :meth:`current` only ever returns a copy of that snapshot — it
**never probes**, so a request reads readiness in O(1) with no socket and no
blocking. Consistency with ``PressureCache`` matters more than novelty.

One deliberate divergence from ``PressureCache``: construction seeds every
backend to ``None`` (*unknown*) **without probing**, rather than sampling
synchronously. Probing at construction would open one socket per backend and
could block gateway startup for ``timeout × N`` if a backend is down; and
before the first probe completes, "unknown" is the honest readiness. The daemon
thread performs the first real probe immediately after it starts (off the
request path), so the snapshot populates promptly without ever blocking a
request or startup.

Stdlib only — this gateway is deliberately dependency-free.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlsplit

# Per-backend probe timeout: bounded so a slow/hung backend cannot stall the
# refresh thread indefinitely. Mirrors ``server._STATUS_PROBE_TIMEOUT``.
_READINESS_PROBE_TIMEOUT: float = 3.0

# How often the daemon thread re-probes every backend. Health changes on a
# human/warm-up timescale, so a few seconds is plenty; probing is off the
# request path, so this interval never affects request latency.
_DEFAULT_REFRESH_INTERVAL: float = 5.0

# vLLM serves ``/health`` unauthenticated and returns 200 only once the engine
# is live, so it is the correct readiness endpoint (same one ``_metrics`` GETs).
_HEALTH_PATH = "/health"

# Per-peer probe timeout: its OWN bound, deliberately a separate constant
# from ``_READINESS_PROBE_TIMEOUT`` rather than a shared default — a peer
# lives across a box boundary (possibly a slower/less trusted link than a
# co-resident local backend) and must never borrow the local probe's socket
# budget (see the module docstring's "Peer probing" section).
_PEER_PROBE_TIMEOUT: float = 3.0

# The OpenAI-shaped models list every backend (local or peer) serves — used
# to honestly verify a peer serves the id this box would forward to it.
_MODELS_PATH = "/v1/models"

# A probe maps a backend base URL to its tri-state readiness. Injectable so the
# cache is unit-testable without sockets (the ``.current()``-opens-no-socket and
# defensive-degradation properties are proved by driving this callable).
Probe = Callable[[str], "bool | None"]

# The raw opener maps a full URL to an HTTP status code (an int). It is the ONLY
# thing that opens a socket; injecting it keeps :func:`probe_backend_ready`
# unit-testable offline. Mirror of ``server._default_ready_probe``'s signature.
Opener = Callable[[str, float], int]


def _default_ready_opener(url: str, timeout: float) -> int:  # pragma: no cover - opens a socket
    """GET *url* over plain HTTP and return the response status code.

    Fleet backends are internal-only (``http://vllm-primary:8000``), so this is
    intentionally HTTP-only, matching ``server._default_ready_probe``. Accessing
    ``urlsplit(url).port`` on a non-numeric port raises ``ValueError`` here —
    caught by :func:`probe_backend_ready`, never by this opener.
    """
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        conn.request("GET", parts.path or "/")
        return conn.getresponse().status
    finally:
        conn.close()


def probe_backend_ready(
    base_url: str,
    *,
    timeout: float = _READINESS_PROBE_TIMEOUT,
    opener: Opener | None = None,
) -> bool | None:
    """Live-probe one backend's ``/health`` and map it to the readiness tri-state.

    * ``True``  — HTTP 200: the backend answered and is ready.
    * ``False`` — reached the backend but it answered non-200 (e.g. warming).
    * ``None``  — could not reach it at all, OR the URL was malformed → unknown.

    The ``opener`` is injected so this is unit-testable without sockets; the
    default opens a bounded ``http.client`` GET.

    The ``except`` clause catches ``OSError``, ``http.client.HTTPException`` AND
    ``ValueError`` and degrades to ``None``. ``ValueError`` is load-bearing: a
    malformed ``base_url`` with a non-numeric port makes ``urlsplit(...).port``
    raise ``ValueError`` (this exact bug was caught in review on PR #90, where
    ``probe_audio_ready`` originally caught only ``OSError``). A degrade-to-None
    here can never crash the caller that will later fold this into
    ``/capabilities``.
    """
    get_status = opener or _default_ready_opener
    try:
        return get_status(base_url.rstrip("/") + _HEALTH_PATH, timeout) == 200
    except (OSError, http.client.HTTPException, ValueError):
        return None


# --- Peer probing (proxied roles, task t4, issues #115/#127) ---------------

# The raw peer opener maps (full URL, timeout, optional API key) to a
# ``(status, body)`` pair. Unlike :data:`Opener` above it must return the
# BODY too (not just a status code): honestly verifying a peer requires
# reading its ``/v1/models`` payload, not just confirming it answered.
# Injecting it keeps :func:`probe_peer_ready` unit-testable offline.
PeerOpener = Callable[[str, float, "str | None"], "tuple[int, bytes]"]


def _default_peer_opener(
    url: str, timeout: float, api_key: str | None
) -> tuple[int, bytes]:  # pragma: no cover - opens a socket
    """GET *url* over plain HTTP, attach ``Authorization`` iff ``api_key`` is
    set, and return ``(status, body)``.

    Fleet peers are reached over their operator-declared origin (mirrors
    :func:`_default_ready_opener`'s HTTP-only contract for internal
    backends — a mesh link, not necessarily public). The header is built
    from ``api_key`` exactly once and never logged or otherwise persisted.
    """
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        conn.request("GET", parts.path or "/", headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def probe_peer_ready(
    origin: str,
    served_name: str,
    *,
    timeout: float = _PEER_PROBE_TIMEOUT,
    api_key: str | None = None,
    opener: PeerOpener | None = None,
) -> bool:
    """Live-probe one proxied role's PEER and decide whether it honestly serves ``served_name``.

    Returns a plain ``bool``, deliberately never ``None`` — unlike
    :func:`probe_backend_ready`'s tri-state, ``None`` here is reserved (see
    :class:`ReadinessCache`) for the seed-before-first-probe state, not for a
    probe OUTCOME:

    * ``True``  — the peer answered HTTP 200 to ``GET <origin>/v1/models``
      AND its ``data[]`` lists an entry whose ``id`` equals ``served_name``
      — i.e. the peer actually serves the model this box would forward to
      it (the #92 "advertised implies reachable" check, extended across a
      box boundary).
    * ``False`` — anything else: non-200, a malformed/non-JSON or
      unexpectedly-shaped body, a well-formed body whose ``data[]`` does not
      list ``served_name``, connection refused, or a timeout. A caller
      cannot distinguish these from the return value alone — nor does it
      need to: every one of them means "do not advertise this proxied role
      as ready right now".

    ``Authorization: Bearer <api_key>`` is attached only when ``api_key`` is
    non-``None`` (pairwise auth is opt-in per peer — see
    :attr:`lobes.gateway._routing.RoutingTable.peer_api_keys`); no header is
    sent otherwise. The key is passed straight through to the opener and
    never logged.

    The ``opener`` is injected so this is unit-testable without sockets,
    mirroring :func:`probe_backend_ready`'s ``opener`` parameter; the
    default opens a bounded ``http.client`` GET and reads the full body.
    ``timeout`` is this probe's OWN bound — callers (:class:`ReadinessCache`)
    must pass a peer-specific timeout, never the local backend probe's.
    """
    get_models = opener or _default_peer_opener
    try:
        status, body = get_models(origin.rstrip("/") + _MODELS_PATH, timeout, api_key)
    except (OSError, http.client.HTTPException, ValueError):
        return False
    if status != 200:
        return False
    try:
        payload = json.loads(body)
        ids = {entry.get("id") for entry in payload.get("data", []) if isinstance(entry, dict)}
    except (ValueError, TypeError, AttributeError):
        # Malformed JSON, a non-dict payload (``.get`` raises AttributeError),
        # or any other unexpected shape — degrade to False, never crash the
        # caller that will later fold this into /capabilities.
        return False
    return served_name in ids


@dataclass(frozen=True)
class PeerSpec:
    """One proxied role's peer probe target.

    Constructed by a later task from the routing table's peer-proxy config
    (``RoutingTable.peer_proxied`` / ``peer_origins`` / ``peer_api_keys`` /
    the owning backend's ``served_name``) — this module stays decoupled from
    :mod:`lobes.gateway._routing` (no import of it) and simply consumes
    whatever specs it is handed at :class:`ReadinessCache` construction.

    Attributes:
        name: the backend/role name this peer stands in for (e.g.
            ``"multimodal"``, ``"senses"``'s owning backend) — the SAME key
            space local backend names use, so the two land side by side in
            one ``.current()`` snapshot.
        origin: the operator-declared peer base URL (never derived — see the
            #92 lesson recorded on ``RoutingTable.peer_origins``).
        served_name: the model id this box would forward to the peer under
            this role. The honesty check in :func:`probe_peer_ready`
            succeeds only when the peer's OWN ``/v1/models`` actually lists
            this id.
        api_key: this box's outbound credential for the peer (``RoutingTable
            .peer_api_keys.get(name)``), sent as ``Authorization: Bearer
            <api_key>`` when declared, omitted entirely otherwise.
            ``repr=False`` — a SECRET must NEVER appear in ``repr``/``str``
            of this object (logs, tracebacks, ``--json`` debug output).
    """

    name: str
    origin: str
    served_name: str
    api_key: str | None = field(default=None, repr=False)


# A peer probe maps a PeerSpec to a plain bool (never None — see
# probe_peer_ready's docstring). Injectable so ReadinessCache is
# unit-testable without sockets, mirroring Probe above.
PeerProbe = Callable[[PeerSpec], bool]


class ReadinessCache:
    """A non-blocking, background readiness provider for the fleet's backends.

    Constructed with a mapping of ``backend name → base URL``. A single daemon
    thread probes each backend's ``/health`` every ``interval`` seconds and
    stores the tri-state verdicts; :meth:`current` returns a copy of the latest
    snapshot without ever probing, so a request reads readiness in O(1) with no
    socket and no blocking.

    The ``probe`` callable (``base_url → bool | None``) is injectable so tests
    can drive fixed verdicts and never touch a socket; it defaults to
    :func:`probe_backend_ready` bound to this cache's ``timeout``. A probe that
    raises is swallowed per-backend (that backend degrades to ``None``) so a
    transient failure can never kill the daemon thread or bubble out of a read.

    Mirrors :class:`lobes.gateway._tier_request.PressureCache` in shape, naming
    and threading discipline; see the module docstring for the one deliberate
    divergence (construction seeds ``None`` instead of probing).

    Optionally also constructed with ``peer_specs`` — an iterable of
    :class:`PeerSpec`, one per proxied role (task t4, issues #115/#127).
    Peer results are probed on a SEPARATE background thread, with their own
    ``peer_timeout``, and merged into the SAME :meth:`current` snapshot keyed
    by ``PeerSpec.name`` — but they live in an entirely independent internal
    store from local ``targets``, so a peer refresh cycle can never corrupt
    (or be corrupted by) a local one, and a hang in either probe path can
    never delay or fail the other (see the module docstring's "Peer probing"
    section). No ``peer_specs`` (the default) spawns no second thread at all
    — a deployment with no proxied roles is completely unaffected.
    """

    def __init__(
        self,
        targets: Mapping[str, str],
        *,
        probe: Probe | None = None,
        timeout: float = _READINESS_PROBE_TIMEOUT,
        interval: float = _DEFAULT_REFRESH_INTERVAL,
        peer_specs: Iterable[PeerSpec] | None = None,
        peer_probe: PeerProbe | None = None,
        peer_timeout: float = _PEER_PROBE_TIMEOUT,
        start: bool = True,
    ) -> None:
        # Copy the targets so a caller mutating theirs cannot change what we probe.
        self._targets: dict[str, str] = dict(targets)
        self._timeout = timeout
        self._probe: Probe = probe or self._default_probe
        self._interval = interval
        # Peer specs keyed by name (last-one-wins on a duplicate name, same
        # convention as dict.fromkeys below). A caller mutating their own
        # iterable after construction cannot change what we probe.
        self._peer_specs: dict[str, PeerSpec] = {spec.name: spec for spec in (peer_specs or ())}
        self._peer_timeout = peer_timeout
        self._peer_probe: PeerProbe = peer_probe or self._default_peer_probe
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peer_stop = threading.Event()
        self._peer_thread: threading.Thread | None = None
        # Seed every backend/peer to unknown WITHOUT probing: construction
        # opens no socket and never blocks startup on a down backend/peer
        # (see module docstring). Kept as TWO independent stores (never one
        # shared dict) so a local refresh pass — which replaces its store
        # wholesale — can never wipe out peer values, and vice versa; see
        # :meth:`current` for where they are merged for reading.
        self._value: dict[str, bool | None] = dict.fromkeys(self._targets, None)
        self._peer_value: dict[str, bool | None] = dict.fromkeys(self._peer_specs, None)
        if start:
            self.start()

    @classmethod
    def from_backends(cls, backends: Iterable[object], **kwargs) -> "ReadinessCache":
        """Build a cache from an iterable of backends (``.name`` + ``.base_url``).

        Convenience for the later wiring task: pass ``table.backends`` directly.
        Duck-typed (reads only ``.name`` / ``.base_url``) so this module needs no
        import of :class:`lobes.gateway._routing.Backend` and stays decoupled.
        """
        targets = {b.name: b.base_url for b in backends}  # type: ignore[attr-defined]
        return cls(targets, **kwargs)

    def _default_probe(self, base_url: str) -> bool | None:
        """The default probe: :func:`probe_backend_ready` bound to our timeout."""
        return probe_backend_ready(base_url, timeout=self._timeout)

    def _read(self) -> dict[str, bool | None]:
        """Probe every backend once, degrading a raising probe to ``None``.

        Per-backend ``try`` so one misbehaving probe cannot abort the whole pass
        or crash the daemon thread — the offending backend simply reads unknown.
        Runs on the background thread only, never the request path.
        """
        result: dict[str, bool | None] = {}
        for name, base_url in self._targets.items():
            try:
                result[name] = self._probe(base_url)
            except Exception:  # nosec B110 — readiness is best-effort; never crash the daemon
                result[name] = None
        return result

    def _refresh_once(self) -> None:
        value = self._read()
        with self._lock:
            self._value = value

    def _default_peer_probe(self, spec: PeerSpec) -> bool:
        """The default peer probe: :func:`probe_peer_ready` bound to OUR OWN
        ``peer_timeout`` — deliberately never ``self._timeout`` (the local
        backend probe's budget); see the module docstring's "Peer probing"
        section and :func:`probe_peer_ready`'s own docstring.
        """
        return probe_peer_ready(
            spec.origin, spec.served_name, timeout=self._peer_timeout, api_key=spec.api_key
        )

    def _read_peers(self) -> dict[str, bool]:
        """Probe every peer spec once, degrading a raising probe to ``False``.

        Per-peer ``try`` so one misbehaving/raising probe cannot abort the
        whole pass — the offending peer simply reads not-ready (``False``),
        never ``None``: once a probe has actually RUN, ``None`` no longer
        applies (see :func:`probe_peer_ready`'s tri-state note). Runs on the
        PEER background thread only, entirely independent of
        :meth:`_read`'s local-backend thread — so this method's runtime,
        however long a hung peer stretches it, can never delay a local
        refresh pass.
        """
        result: dict[str, bool] = {}
        for name, spec in self._peer_specs.items():
            try:
                result[name] = bool(self._peer_probe(spec))
            except Exception:  # nosec B110 — peer probing is best-effort; never crash the daemon
                result[name] = False
        return result

    def _refresh_once_peers(self) -> None:
        value = self._read_peers()
        with self._lock:
            self._peer_value = value

    def refresh(self) -> None:
        """Probe every backend AND every peer once, synchronously, and update
        the snapshot NOW.

        A public, blocking one-shot the gateway calls **once before it binds** so
        ``GET /v1/models`` and ``GET /capabilities`` are correct on the very first
        request: construction seeds every backend/peer to ``None`` (*unknown*)
        without probing, so without this a freshly-started cache would report
        everything unready until the daemon's first background pass lands (up to
        one ``interval``). This closes that startup window with a single bounded
        pass over each store (backend probed once ``timeout``-capped, peer probed
        once ``peer_timeout``-capped), then :meth:`start` hands subsequent
        refreshes to the daemon threads — off the request path. It is a thin
        public alias for the daemon loops' own :meth:`_refresh_once` /
        :meth:`_refresh_once_peers`; keeping the internal names private and
        exposing this one keeps the seed-before-bind intent legible at the one
        call site (``server.serve``) that needs it.
        """
        self._refresh_once()
        if self._peer_specs:
            self._refresh_once_peers()

    def current(self) -> dict[str, bool | None]:
        """Return a copy of the latest readiness snapshot. Never probes, never blocks.

        Values are ``True`` / ``False`` / ``None`` (the tri-state — a peer's
        own tri-state per :func:`probe_peer_ready` is ``True``/``False``
        post-probe, ``None`` only before its first probe). Local-backend and
        peer results are merged here — the ONLY place they meet — from their
        two independent internal stores; if a name were ever registered in
        both (not a configuration this module expects — see
        :class:`PeerSpec`), the peer value wins. The returned dict is a
        fresh copy, so a caller mutating it cannot corrupt the cache.
        """
        with self._lock:
            merged = dict(self._value)
            merged.update(self._peer_value)
            return merged

    def start(self) -> None:
        """Start the background refresh thread(s) (idempotent).

        Always (re)starts the local-backend refresh thread; additionally
        starts a SEPARATE peer-refresh thread iff any ``peer_specs`` were
        registered at construction — a deployment with no proxied roles
        spawns no second thread at all. See :meth:`_start_local` /
        :meth:`_start_peer` for the single-live-thread invariant each one
        preserves independently.
        """
        self._start_local()
        if self._peer_specs:
            self._start_peer()

    def _start_local(self) -> None:
        """Start the local-backend refresh thread (idempotent).

        Single-live-thread invariant: at most one refresh thread may be alive
        at any time. If ``self._thread`` is still genuinely alive, this is a
        no-op — it never spawns a second, overlapping refresh thread on top
        of one that is still probing. If ``self._thread`` is set but the
        thread has already exited (a stale reference left behind by
        :meth:`_stop_local` when its bounded join timed out before the thread
        noticed the stop flag — see that method's docstring), this clears
        the dead reference first so the cache restarts cleanly instead of
        refusing to run again.
        """
        if self._thread is not None:
            if self._thread.is_alive():
                return
            # Stale reference to an already-exited thread (left behind by an
            # incomplete stop() — see stop()'s docstring). Clear it so we can
            # restart; do NOT treat a dead Thread object as still "the" live
            # thread.
            self._thread = None
        # Clear the stop flag so a cache restarted after stop() runs again.
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="lobes-readiness-cache", daemon=True
        )
        self._thread.start()

    def _start_peer(self) -> None:
        """Start the peer-refresh thread (idempotent).

        Mirrors :meth:`_start_local`'s single-live-thread invariant exactly,
        but for ``self._peer_thread`` / ``self._peer_stop`` — a completely
        separate ``threading.Thread`` so a hang in ONE loop's probes can
        never share a deadline (or a lock-holding stretch) with the other's;
        see the module docstring's "Peer probing" section and
        :meth:`_stop_peer` for the shutdown half of this invariant.
        """
        if self._peer_thread is not None:
            if self._peer_thread.is_alive():
                return
            self._peer_thread = None
        self._peer_stop.clear()
        self._peer_thread = threading.Thread(
            target=self._peer_loop, name="lobes-readiness-peer-cache", daemon=True
        )
        self._peer_thread.start()

    def _loop(self) -> None:
        # Probe once immediately so the snapshot populates promptly after start()
        # (off the request path), then refresh every interval. Event.wait(interval)
        # returns True only when stop() is set, so it both paces the refresh and
        # exits promptly on shutdown.
        self._refresh_once()
        while not self._stop.wait(self._interval):
            self._refresh_once()

    def _peer_loop(self) -> None:
        # Mirrors _loop exactly, but reads self._peer_stop / self._interval
        # for ITS OWN pacing and calls _refresh_once_peers — entirely
        # independent of the local loop above (a different Thread, a
        # different Event), so neither can delay the other.
        self._refresh_once_peers()
        while not self._peer_stop.wait(self._interval):
            self._refresh_once_peers()

    def stop(self) -> None:
        """Signal both daemon threads to exit and join them (idempotent, clean shutdown).

        Stops the local-backend thread and, if one was ever started, the
        peer thread — see :meth:`_stop_local` / :meth:`_stop_peer` for the
        single-live-thread invariant each preserves independently. Safe to
        call before :meth:`start` (no threads yet). Idempotent.
        """
        self._stop_local()
        self._stop_peer()

    def _stop_local(self) -> None:
        """Signal the local-backend daemon thread to exit and join it.

        Single-live-thread invariant: this must never falsely report the
        thread gone while it is actually still running. A single refresh
        pass probes every target *sequentially* (:meth:`_read`), so its
        worst case is ``len(self._targets) * self._timeout`` — which can
        exceed a fixed, small join bound. The join below is sized to cover
        that worst case (plus a fixed margin) so a clean shutdown normally
        completes within this call. If the
        thread is STILL alive once the join returns (a pathologically slow
        probe outlasting even that bound), ``self._thread`` is left
        referring to it rather than cleared: clearing it here while the
        thread is still running would let a later :meth:`_start_local` spawn
        a SECOND, overlapping refresh thread while the first is still
        probing — exactly the bug this invariant guards against. The stop
        flag stays set, so the still-running thread exits on its own at its
        next ``Event.wait`` boundary; :meth:`_start_local` recognizes and
        clears that now-dead stale reference the next time it is called (see
        its docstring).

        Safe to call before :meth:`start` (no thread yet). Idempotent —
        calling this again while the thread is still alive (or after it has
        exited) returns promptly either way and never raises.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None:
            # Bound the join so shutdown cannot hang indefinitely, but size it
            # to cover one full sequential refresh pass across every target so
            # a clean shutdown normally completes within this call rather than
            # racing it. The thread is also a daemon, so a (pathological)
            # straggler that outlasts even this bound never blocks interpreter
            # exit anyway.
            join_bound = len(self._targets) * self._timeout + 1.0
            thread.join(timeout=join_bound)
            if not thread.is_alive():
                self._thread = None
            # else: the thread is still running — leave self._thread pointing
            # at it (see docstring above) rather than orphaning it.

    def _stop_peer(self) -> None:
        """Signal the peer daemon thread to exit and join it.

        Mirrors :meth:`_stop_local` exactly (same bounded-join, same
        stale-reference discipline), sized over ``self._peer_specs`` /
        ``self._peer_timeout`` instead. A no-op when no peer thread was ever
        started (``self._peer_thread is None`` — the common case for a
        deployment with no proxied roles).
        """
        self._peer_stop.set()
        thread = self._peer_thread
        if thread is not None:
            join_bound = len(self._peer_specs) * self._peer_timeout + 1.0
            thread.join(timeout=join_bound)
            if not thread.is_alive():
                self._peer_thread = None
            # else: still running — leave self._peer_thread pointing at it,
            # mirroring _stop_local's rationale.

    # Explicit alias — server shutdown code reads more naturally as close().
    close = stop

    def is_alive(self) -> bool:
        """True while the LOCAL-backend background refresh thread is running.

        Unchanged in scope by peer probing (still local-thread-only) so every
        existing caller's meaning is preserved byte-for-byte; check
        ``self._peer_thread`` directly for the peer thread's liveness.
        """
        return self._thread is not None and self._thread.is_alive()
