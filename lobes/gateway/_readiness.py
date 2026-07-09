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
import threading
from collections.abc import Iterable, Mapping
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
    """

    def __init__(
        self,
        targets: Mapping[str, str],
        *,
        probe: Probe | None = None,
        timeout: float = _READINESS_PROBE_TIMEOUT,
        interval: float = _DEFAULT_REFRESH_INTERVAL,
        start: bool = True,
    ) -> None:
        # Copy the targets so a caller mutating theirs cannot change what we probe.
        self._targets: dict[str, str] = dict(targets)
        self._timeout = timeout
        self._probe: Probe = probe or self._default_probe
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Seed every backend to unknown WITHOUT probing: construction opens no
        # socket and never blocks startup on a down backend (see module docstring).
        self._value: dict[str, bool | None] = {name: None for name in self._targets}
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

    def current(self) -> dict[str, bool | None]:
        """Return a copy of the latest readiness snapshot. Never probes, never blocks.

        Values are ``True`` / ``False`` / ``None`` (the tri-state). The returned
        dict is a fresh copy, so a caller mutating it cannot corrupt the cache.
        """
        with self._lock:
            return dict(self._value)

    def start(self) -> None:
        """Start the background refresh thread (idempotent)."""
        if self._thread is not None:
            return
        # Clear the stop flag so a cache restarted after stop() runs again.
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="lobes-readiness-cache", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        # Probe once immediately so the snapshot populates promptly after start()
        # (off the request path), then refresh every interval. Event.wait(interval)
        # returns True only when stop() is set, so it both paces the refresh and
        # exits promptly on shutdown.
        self._refresh_once()
        while not self._stop.wait(self._interval):
            self._refresh_once()

    def stop(self) -> None:
        """Signal the daemon thread to exit and join it (idempotent, clean shutdown).

        Safe to call before :meth:`start` (no thread yet). Joins with a bounded
        timeout so a caller (e.g. server shutdown) gets deterministic termination
        without hanging on a probe in flight.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None:
            # Bound the join so shutdown cannot hang on a probe still in flight;
            # the thread is a daemon, so a (pathological) straggler never blocks
            # interpreter exit anyway.
            thread.join(timeout=self._timeout + 1.0)
            self._thread = None

    # Explicit alias — server shutdown code reads more naturally as close().
    close = stop

    def is_alive(self) -> bool:
        """True while the background refresh thread is running."""
        return self._thread is not None and self._thread.is_alive()
