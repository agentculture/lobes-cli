import pytest

from lobes.rollout._member_lock import LockHeld, MemberLock, member_lock


class FakeRemote:
    """Simulates the member's lock dir; interprets the script by marker."""

    def __init__(self):
        self.holder = None
        self.calls = []

    def __call__(self, argv, timeout):
        script = argv[2] if len(argv) > 2 else ""
        self.calls.append(argv)
        if "mkdir" in script and "rm -rf" not in script:
            info = argv[-1]
            if self.holder is not None:
                return 17, "HELD\n" + self.holder + "\n", ""
            self.holder = info
            return 0, "", ""
        if "rm -rf" in script:
            self.holder = None
            return 0, "", ""
        return 1, "", "?"


TARGET = {"local": True}


def mk(remote):
    return MemberLock(TARGET, runner=remote, host="driver", pid=42, now=lambda: "T0")


def test_acquire_takes_lock_and_stores_holder():
    r = FakeRemote()
    mk(r).acquire()
    assert r.holder is not None
    assert "driver" in r.holder and "42" in r.holder and "T0" in r.holder
    assert ".lobes/.rollout.lock" in r.calls[0][2]
    assert "mkdir" in r.calls[0][2]


def test_second_acquire_refused_naming_holder():
    r = FakeRemote()
    mk(r).acquire()
    with pytest.raises(LockHeld) as e:
        MemberLock(TARGET, runner=r, host="other", pid=7, now=lambda: "T1").acquire()
    assert "driver" in str(e.value) and "42" in str(e.value)


def test_released_on_success():
    r = FakeRemote()
    with member_lock(TARGET, runner=r, host="h", pid=1, now=lambda: "t"):
        assert r.holder
    assert r.holder is None


def test_released_on_failure():
    r = FakeRemote()
    with pytest.raises(RuntimeError):
        with member_lock(TARGET, runner=r, host="h", pid=1, now=lambda: "t"):
            raise RuntimeError("boom")
    assert r.holder is None


def test_released_on_keyboard_interrupt():
    r = FakeRemote()
    with pytest.raises(KeyboardInterrupt):
        with member_lock(TARGET, runner=r, host="h", pid=1, now=lambda: "t"):
            raise KeyboardInterrupt
    assert r.holder is None


def test_refused_acquire_does_not_release_others_lock():
    r = FakeRemote()
    mk(r).acquire()
    with pytest.raises(LockHeld):
        with member_lock(TARGET, runner=r, host="o", pid=2, now=lambda: "t"):
            pass
    assert r.holder is not None
