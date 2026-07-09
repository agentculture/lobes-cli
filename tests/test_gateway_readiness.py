"""Tests for the gateway's background readiness cache (:mod:`lobes.gateway._readiness`).

Task t3 of "advertised implies reachable": prove that ``ReadinessCache`` can tell
"this backend answered recently" from "this backend is merely configured", WITHOUT
ever probing on the request path.

Two properties are the most likely to be silently violated, so each has a test
that would actually catch a regression:

* ``.current()`` opens **no socket** — the probe callable is injected and its call
  count is asserted to stay flat across many reads (and zero at construction).
* the probe helper degrades a **non-numeric port** (``urlsplit(...).port`` raises
  ``ValueError``) to ``None`` instead of crashing — the exact bug caught in review
  on PR #90 for ``probe_audio_ready``.

Stdlib only, mirroring the gateway's dependency-free discipline.
"""

from __future__ import annotations

import http.client
import threading
import time

from lobes.gateway import _readiness as R

# --- probe_backend_ready: the tri-state helper ------------------------------


def test_probe_backend_ready_tristate() -> None:
    # 200 → True (reached, ready); non-200 → False (reached, warming);
    # OSError → None (could not reach at all). None must NOT collapse into False.
    assert R.probe_backend_ready("http://primary:8000", opener=lambda u, t: 200) is True
    assert R.probe_backend_ready("http://primary:8000", opener=lambda u, t: 503) is False

    def boom(_u, _t):
        raise OSError("connection refused")

    assert R.probe_backend_ready("http://primary:8000", opener=boom) is None


def test_probe_backend_ready_swallows_valueerror_and_httpexception() -> None:
    """A malformed base_url (``urlsplit(...).port`` raises ``ValueError``) or a
    broken HTTP exchange (``HTTPException``) must degrade to unknown (``None``),
    never bubble out — this is the PR #90 regression the ValueError guard exists
    for. ``probe_audio_ready`` originally caught only ``OSError``; do not repeat it.
    """

    def value_boom(_u, _t):
        raise ValueError("invalid literal for int() with base 10: 'abc'")

    def http_boom(_u, _t):
        raise http.client.BadStatusLine("garbage")

    assert R.probe_backend_ready("http://primary:8000", opener=value_boom) is None
    assert R.probe_backend_ready("http://primary:8000", opener=http_boom) is None


def test_probe_backend_ready_malformed_url_no_crash_no_socket() -> None:
    """The DEFAULT opener path: a non-numeric port makes ``urlsplit(...).port``
    raise ``ValueError`` *before any socket opens*, so the probe returns ``None``
    offline instead of crashing. This exercises the real opener, not an injected
    one, so a regression that dropped the ``ValueError`` guard would surface here.
    """
    assert R.probe_backend_ready("http://backend:notaport/") is None


def test_probe_backend_ready_hits_health_path() -> None:
    # The probe targets the vLLM ``/health`` endpoint, with base_url trailing
    # slash stripped so we never emit a double slash.
    seen = {}

    def opener(url, _timeout):
        seen["url"] = url
        return 200

    R.probe_backend_ready("http://primary:8000/", opener=opener)
    assert seen["url"] == "http://primary:8000/health"


# --- ReadinessCache: seed + socket-free reads -------------------------------


def test_seed_is_all_unknown_and_construction_opens_no_socket() -> None:
    # Before any probe completes, readiness is honestly UNKNOWN (None) for every
    # backend — and construction itself must not probe (no blocking N-socket
    # fan-out at gateway startup).
    calls = {"n": 0}

    def probe(_url):
        calls["n"] += 1
        return True

    cache = R.ReadinessCache(
        {"primary": "http://primary:8000", "minor": "http://minor:8000"},
        probe=probe,
        interval=1000,
        start=False,
    )
    try:
        assert cache.current() == {"primary": None, "minor": None}
        assert calls["n"] == 0  # construction probed nothing → opened no socket
    finally:
        cache.stop()


def test_current_probe_call_count_is_zero_across_reads() -> None:
    # The load-bearing property: reading the cache must never sample. Inject a
    # counting probe, read N times, assert the count does not move.
    calls = {"n": 0}

    def probe(_url):
        calls["n"] += 1
        return True

    cache = R.ReadinessCache(
        {"a": "http://a:8000", "b": "http://b:8000"},
        probe=probe,
        interval=1000,
        start=False,
    )
    try:
        before = calls["n"]
        for _ in range(2000):
            cache.current()
        assert calls["n"] == before  # reads add ZERO probe calls
        assert before == 0  # and the seed itself opened no socket
    finally:
        cache.stop()


def test_current_read_path_probe_never_invoked() -> None:
    # A second, independent guard: a probe that RAISES if ever called. With the
    # thread stopped, neither construction nor any read may invoke it.
    def probe(_url):
        raise AssertionError("current() must not probe on the read path")

    cache = R.ReadinessCache({"a": "http://a:8000"}, probe=probe, interval=1000, start=False)
    try:
        for _ in range(2000):
            assert cache.current() == {"a": None}
    finally:
        cache.stop()


def test_current_returns_a_copy_isolated_from_caller_mutation() -> None:
    cache = R.ReadinessCache({"a": "http://a:8000"}, probe=lambda u: True, start=False)
    try:
        snapshot = cache.current()
        snapshot["a"] = "corrupted"
        snapshot["injected"] = True
        assert cache.current() == {"a": None}
    finally:
        cache.stop()


def test_empty_targets_current_is_empty_dict() -> None:
    cache = R.ReadinessCache({}, probe=lambda u: True, start=False)
    try:
        assert cache.current() == {}
    finally:
        cache.stop()


# --- ReadinessCache: the background daemon thread ---------------------------


def test_background_thread_probes_each_backend_tristate() -> None:
    # Once the daemon thread completes a pass, .current() reports each backend's
    # own tri-state verdict — True, False and None must all survive side by side.
    targets = {
        "primary": "http://primary:8000",
        "minor": "http://minor:8000",
        "senses": "http://senses:8000",
    }
    verdicts = {
        "http://primary:8000": True,
        "http://minor:8000": False,
        "http://senses:8000": None,
    }
    cache = R.ReadinessCache(targets, probe=lambda u: verdicts[u], interval=0.01, start=True)
    try:
        expected = {"primary": True, "minor": False, "senses": None}
        deadline = time.time() + 2.0
        while time.time() < deadline and cache.current() != expected:
            time.sleep(0.005)
        assert cache.current() == expected
    finally:
        cache.stop()


def test_background_thread_refreshes_repeatedly() -> None:
    # The daemon re-probes every ``interval`` (tracks live health, not just a seed).
    counter = {"n": 0}
    twice = threading.Event()

    def probe(_url):
        counter["n"] += 1
        if counter["n"] >= 2:
            twice.set()
        return True

    cache = R.ReadinessCache({"a": "http://a:8000"}, probe=probe, interval=0.01, start=True)
    try:
        assert twice.wait(2.0), "background thread never re-probed"
    finally:
        cache.stop()


def test_one_probe_raising_does_not_kill_the_refresh() -> None:
    # criteria #6: a probe blowing up on one backend degrades THAT backend to
    # None and must not stop the others being probed nor kill the daemon thread.
    def probe(base_url):
        if base_url == "http://bad:8000":
            raise RuntimeError("boom")
        return True

    targets = {"bad": "http://bad:8000", "good": "http://good:8000"}
    cache = R.ReadinessCache(targets, probe=probe, interval=0.01, start=True)
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and cache.current().get("good") is not True:
            time.sleep(0.005)
        snap = cache.current()
        assert snap["good"] is True  # the pass ran past the raising probe
        assert snap["bad"] is None  # the raising probe degraded to unknown
        assert cache.is_alive()  # the daemon thread survived the exception
    finally:
        cache.stop()


def test_current_never_raises_and_thread_survives_chaotic_probe() -> None:
    # Every probe raises; .current() must still return a dict, and the thread
    # must keep looping (proved by a second refresh happening after the first raise).
    counter = {"n": 0}

    def probe(_url):
        counter["n"] += 1
        raise RuntimeError("chaos")

    cache = R.ReadinessCache({"a": "http://a:8000"}, probe=probe, interval=0.01, start=True)
    try:
        deadline = time.time() + 2.0
        while counter["n"] < 2 and time.time() < deadline:
            time.sleep(0.005)
        assert counter["n"] >= 2, "thread died after the first raising probe"
        result = cache.current()
        assert isinstance(result, dict)
        assert result.get("a") is None
        assert cache.is_alive()
    finally:
        cache.stop()


def test_refresh_thread_is_a_daemon() -> None:
    cache = R.ReadinessCache(
        {"primary": "http://primary:8000"}, probe=lambda u: True, interval=1000, start=True
    )
    try:
        assert cache.is_alive()
        # A daemon thread never blocks interpreter shutdown.
        assert cache._thread is not None and cache._thread.daemon is True
    finally:
        cache.stop()


# --- ReadinessCache: clean shutdown -----------------------------------------


def test_stop_terminates_the_thread() -> None:
    cache = R.ReadinessCache(
        {"primary": "http://primary:8000"}, probe=lambda u: True, interval=0.01, start=True
    )
    assert cache.is_alive()
    cache.stop()
    assert not cache.is_alive()  # stop() joined the thread → it is gone


def test_close_is_an_alias_for_stop() -> None:
    cache = R.ReadinessCache(
        {"primary": "http://primary:8000"}, probe=lambda u: True, interval=0.01, start=True
    )
    assert cache.is_alive()
    cache.close()
    assert not cache.is_alive()


def test_stop_is_idempotent_and_safe_before_start() -> None:
    cache = R.ReadinessCache({"a": "http://a:8000"}, probe=lambda u: True, start=False)
    cache.stop()  # no thread yet — must not raise
    cache.stop()  # idempotent
    assert not cache.is_alive()


# --- ReadinessCache: slow-probe / short-timeout race (Qodo finding, PR #102) --
#
# The bug: a single refresh pass probes every target SEQUENTIALLY, so its
# worst case is ``len(targets) * timeout``, which can exceed stop()'s bounded
# join. The old stop() unconditionally cleared ``self._thread`` after the
# join regardless of whether the thread had actually exited; a later start()
# then saw ``_thread is None`` and spawned a SECOND, overlapping refresh
# thread while the first was still probing. These tests force that race with
# an injected probe that blocks on a ``threading.Event`` (deterministic — no
# ``time.sleep`` guessing) until the test explicitly releases it.


def test_stop_incomplete_join_then_start_never_creates_second_live_thread() -> None:
    """Reproduces the Qodo finding directly: stop()'s bounded join can return
    while the refresh thread is still alive (blocked in a slow probe); a
    subsequent start() must recognize the thread is still live and refuse to
    spawn a second, concurrently-running refresh thread.

    Against the OLD ``stop()`` (which unconditionally set
    ``self._thread = None`` after the join, regardless of whether the thread
    had actually exited), this test FAILS: start() sees ``_thread is None``
    and spawns a brand-new thread while the first is still blocked inside the
    probe, so ``cache._thread`` ends up pointing at a second, distinct
    ``Thread`` object even though the first is still alive.
    """
    probe_entered = threading.Event()
    release_probe = threading.Event()

    def slow_probe(_url: str) -> bool | None:
        probe_entered.set()
        # Block until the test releases it. The wait() timeout is just a
        # safety net against a truly hung test process, not the mechanism
        # under test — stop()'s own join timeout is what must bound this.
        release_probe.wait(timeout=5.0)
        return True

    cache = R.ReadinessCache(
        {"slow": "http://slow:8000"},
        probe=slow_probe,
        timeout=0.05,
        interval=1000,
        start=True,
    )
    try:
        assert probe_entered.wait(2.0), "background thread never began probing"
        first_thread = cache._thread
        assert first_thread is not None and first_thread.is_alive()

        # stop()'s join is bounded well below how long the probe will block
        # (release_probe is not set yet), so stop() must return promptly
        # with the thread STILL alive — this is the race window.
        cache.stop()
        assert first_thread.is_alive(), (
            "test setup invalid: the probe returned before stop()'s join timed out, "
            "so this run never exercised the race"
        )

        # The invariant under test: start() after an incomplete stop() must
        # NOT spawn a second, concurrently-live refresh thread.
        cache.start()
        assert cache._thread is first_thread, (
            "start() spawned a second refresh thread while the first was still "
            "alive — overlapping refresh loops, the exact bug this test guards"
        )

        # Let the original probe return so the first thread can exit cleanly
        # and not leak past this test.
        release_probe.set()
        first_thread.join(timeout=5.0)
        assert not first_thread.is_alive()
    finally:
        release_probe.set()
        cache.stop()


def test_start_after_incomplete_stop_thread_exits_then_restart_replaces_stale_reference() -> None:
    """After a bounded join times out with the thread still alive, ``stop()``
    leaves ``self._thread`` referring to it (not orphaned, not nulled). Once
    that thread later actually exits on its own — it notices ``self._stop``
    at its next ``Event.wait`` boundary — a later ``start()`` must recognize
    the now-dead-but-still-referenced thread and replace it with a fresh one,
    rather than treating a dead ``Thread`` object as though it were still the
    live refresh thread forever.
    """
    probe_entered = threading.Event()
    release_probe = threading.Event()

    def slow_probe(_url: str) -> bool | None:
        probe_entered.set()
        release_probe.wait(timeout=5.0)
        return True

    cache = R.ReadinessCache(
        {"slow": "http://slow:8000"},
        probe=slow_probe,
        timeout=0.05,
        interval=1000,
        start=True,
    )
    try:
        assert probe_entered.wait(2.0)
        stale_thread = cache._thread

        cache.stop()  # bounded join times out; thread is still blocked in-probe
        assert cache._thread is stale_thread  # not cleared while still alive

        release_probe.set()  # let the blocked probe return so the thread can exit
        stale_thread.join(timeout=5.0)
        assert not stale_thread.is_alive()

        # stop() only clears the reference when ITS OWN join observes the
        # exit; the thread exiting later, on its own, does not retroactively
        # clear it — start() is responsible for noticing the stale reference.
        assert cache._thread is stale_thread

        cache.start()
        assert cache._thread is not stale_thread  # stale dead reference replaced
        assert cache.is_alive()
    finally:
        release_probe.set()
        cache.stop()


def test_start_after_clean_stop_restarts() -> None:
    # The ordinary path (no race): a fully-completed stop() clears the
    # reference outright, and start() spawns a normal fresh thread.
    cache = R.ReadinessCache(
        {"a": "http://a:8000"}, probe=lambda u: True, interval=0.01, start=True
    )
    try:
        assert cache.is_alive()
        cache.stop()
        assert not cache.is_alive()
        assert cache._thread is None  # a clean, fully-joined stop clears the reference

        cache.start()
        assert cache.is_alive()
    finally:
        cache.stop()


def test_stop_on_still_running_thread_does_not_hang_and_is_idempotent() -> None:
    """stop() must return promptly (its bounded join, not the full probe
    duration) even when the thread is still blocked in a slow probe, and it
    must not orphan that thread. A second stop() call afterward — while the
    thread is still alive — must also return promptly and not raise.
    """
    probe_entered = threading.Event()
    release_probe = threading.Event()

    def slow_probe(_url: str) -> bool | None:
        probe_entered.set()
        release_probe.wait(timeout=5.0)
        return True

    cache = R.ReadinessCache(
        {"slow": "http://slow:8000"},
        probe=slow_probe,
        timeout=0.05,
        interval=1000,
        start=True,
    )
    try:
        assert probe_entered.wait(2.0)

        started = time.time()
        cache.stop()
        elapsed = time.time() - started
        # Join bound is on the order of len(targets) * timeout + 1.0 (~1.05s
        # for one target here) — nowhere near the 5s the probe could block.
        assert elapsed < 3.0, f"stop() hung waiting on a still-running probe ({elapsed:.2f}s)"

        # Still alive (it deliberately outlasted the join) → not orphaned:
        # the reference must still point at the real, live thread.
        assert cache._thread is not None
        assert cache._thread.is_alive()

        # Idempotent: calling stop() again while the thread is STILL alive
        # must not hang or raise.
        started2 = time.time()
        cache.stop()
        assert time.time() - started2 < 3.0
    finally:
        release_probe.set()
        cache.stop()


# --- ReadinessCache.from_backends: ergonomic constructor --------------------


def test_from_backends_builds_targets_from_name_and_base_url() -> None:
    class _B:
        def __init__(self, name, base_url):
            self.name = name
            self.base_url = base_url

    backends = [_B("primary", "http://primary:8000"), _B("minor", "http://minor:8000")]
    cache = R.ReadinessCache.from_backends(backends, probe=lambda u: True, start=False)
    try:
        assert cache.current() == {"primary": None, "minor": None}
    finally:
        cache.stop()


def test_from_backends_accepts_routing_backend() -> None:
    from lobes.gateway._routing import Backend

    backends = [
        Backend(name="primary", base_url="http://primary:8000", served_name="qwen"),
        Backend(name="embed", base_url="http://embed:8000", served_name="emb", task="embed"),
    ]
    cache = R.ReadinessCache.from_backends(backends, probe=lambda u: True, start=False)
    try:
        assert set(cache.current()) == {"primary", "embed"}
    finally:
        cache.stop()
