"""Member step (t11): idempotent upgrade of one member, all through fakes."""

import json

import pytest

from lobes.rollout import _member_step as ms
from lobes.rollout._journal import Journal
from lobes.rollout._rollback import backup_from_journal
from lobes.rollout._transport import Target

T = Target(ssh_target="h", user="u", lobes_path="/home/u/.local/bin/lobes")
OLD, NEW = "0.94.0", "0.95.0"
IMAGE_ID = "sha256:abc123"
KEY = "sk-very-secret-gateway-key"


class FakeMember:
    """A fake member box. Dispatches on the remote command string (argv[-1])."""

    def __init__(
        self,
        installed=OLD,
        pinned=OLD,
        running=OLD,
        window=None,
        fail_build=False,
        repin_exit=1,
        install_lags=False,
        lock_held=False,
        health_up_after=0,
    ):
        self.installed = installed
        self.pinned = pinned
        self.running = running
        self.window = window  # None = no machine CLI; dict = window show payload
        self.fail_build = fail_build
        self.repin_exit = repin_exit
        self.install_lags = install_lags
        self.lock_held = lock_held
        self.health_up_after = health_up_after
        self.health_calls_after_build = 0
        self.built = False
        self.tags = {}
        self.backups = []
        self.calls = []
        self.lock_taken = False

    # classification -----------------------------------------------------
    def kind(self, cmd):
        if "rollout.lock" in cmd:
            return "lock_release" if cmd.startswith("sh -c 'rm") else "lock_acquire"
        if "window show" in cmd:
            return "window"
        if "uv tool install" in cmd or "/uv tool install" in cmd:
            return "install"
        if "doctor --repin-version" in cmd:
            return "repin"
        if "up gateway --build" in cmd:
            return "build"
        if "docker tag" in cmd:
            return "tag"
        if "docker inspect" in cmd:
            return "inspect"
        if "/health" in cmd:
            return "health"
        if ".env.bak" in cmd:
            return "backup"
        if "MODEL_GEAR_VERSION" in cmd:
            return "read_pin"
        if "--version" in cmd:
            return "version"
        return "other"

    def __call__(self, argv, timeout):
        cmd = argv[-1]
        k = self.kind(cmd)
        self.calls.append((k, cmd))
        if k == "lock_acquire":
            if self.lock_held:
                return 17, "HELD\nhost=other pid=9 started=x\n", ""
            self.lock_taken = True
            return 0, "", ""
        if k == "lock_release":
            self.lock_taken = False
            return 0, "", ""
        if k == "version":
            return 0, f"lobes {self.installed}\n", ""
        if k == "read_pin":
            return 0, (self.pinned or "") + "\n", ""
        if k == "health":
            if self.built:
                self.health_calls_after_build += 1
                if self.health_calls_after_build <= self.health_up_after:
                    return 7, "", "connection refused"
            if self.running is None:
                return 7, "", "connection refused"
            return 0, json.dumps({"status": "ok", "version": self.running}), ""
        if k == "window":
            if self.window is None:
                return 3, "NO_MACHINE_CLI\n", ""
            return 0, json.dumps(self.window), ""
        if k == "backup":
            path = "/home/u/.lobes/.env.bak-STAMP-rollout"
            self.backups.append(path)
            return 0, path + "\n", ""
        if k == "install":
            if not self.install_lags:
                self.installed = NEW
            return 0, "Installed 1 executable: lobes\n", ""
        if k == "repin":
            self.pinned = self.installed
            # the 2026-11-09 trap: doctor exits non-zero on the skew it fixes
            return self.repin_exit, f"re-pinned MODEL_GEAR_VERSION={self.installed}\n", "FAIL"
        if k == "inspect":
            return 0, IMAGE_ID + "\n", ""
        if k == "tag":
            parts = cmd.split()
            self.tags[parts[-1]] = parts[-2]
            return 0, "", ""
        if k == "build":
            if self.fail_build:
                return 1, "", f"build failed GATEWAY_API_KEY={KEY}"
            self.built = True
            self.running = self.pinned
            return 0, "", ""
        return 0, "", ""

    def kinds(self):
        return [k for k, _ in self.calls]

    def writes(self):
        mutating = {"backup", "install", "repin", "tag", "build"}
        return [k for k in self.kinds() if k in mutating]


def step(fake, **kw):
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("stamp", "STAMP")
    return ms.run_member_step("spark2", T, NEW, runner=fake, **kw)


def test_full_upgrade_runs_steps_in_order():
    f = FakeMember()
    r = step(f)
    assert r.ok and r.changed, r.error
    order = [
        k
        for k in f.kinds()
        if k in ("lock_acquire", "window", "backup", "install", "repin", "tag", "build")
    ]
    assert order == ["lock_acquire", "window", "backup", "install", "repin", "tag", "build"]
    # health wait comes after the build, lock release is last
    assert "health" in f.kinds()[f.kinds().index("build") :]
    assert f.kinds()[-1] == "lock_release" and not f.lock_taken
    assert r.old_version == OLD and r.target_version == NEW
    assert f.tags == {f"lobes-gateway:{OLD}": IMAGE_ID}


def test_install_uses_absolute_uv_and_pins_exact_version():
    f = FakeMember()
    step(f)
    cmd = next(c for k, c in f.calls if k == "install")
    assert cmd.startswith("/home/u/.local/bin/uv tool install")
    assert f"lobes-cli=={NEW}" in cmd


def test_explicit_uv_path_must_be_absolute():
    with pytest.raises(ValueError):
        step(FakeMember(), uv_path="uv")


def test_lobes_invoked_by_absolute_path():
    f = FakeMember()
    step(f)
    for k, c in f.calls:
        if k in ("repin", "build"):
            assert c.startswith("/home/u/.local/bin/lobes ")
    build = next(c for k, c in f.calls if k == "build")
    assert build.endswith("up gateway --build --apply")
    repin = next(c for k, c in f.calls if k == "repin")
    assert repin.endswith("doctor --repin-version --apply")


def test_compose_dir_threaded_through_when_given():
    f = FakeMember()
    r = step(f, deploy_dir="/srv/lobes")
    assert r.ok
    for k, c in f.calls:
        if k in ("repin", "build"):
            assert "--compose-dir /srv/lobes" in c


def test_second_run_on_upgraded_member_changes_nothing_and_succeeds():
    f = FakeMember()
    assert step(f).ok
    f.calls.clear()
    r = step(f)
    assert r.ok and not r.changed
    assert f.writes() == []
    assert not f.lock_taken


def test_already_current_member_is_a_noop():
    f = FakeMember(installed=NEW, pinned=NEW, running=NEW)
    r = step(f)
    assert r.ok and not r.changed and f.writes() == []


def test_repin_nonzero_exit_does_not_abort_when_env_is_repinned():
    f = FakeMember(repin_exit=1)
    r = step(f)
    assert r.ok, r.error
    assert f.pinned == NEW and "build" in f.kinds()


def test_repin_that_did_not_change_env_fails_even_with_exit_zero():
    f = FakeMember(repin_exit=0)
    f_repin = f.__call__

    def runner(argv, timeout):
        if "doctor --repin-version" in argv[-1]:
            f.calls.append(("repin", argv[-1]))
            return 0, "", ""  # claims success, writes nothing
        return f_repin(argv, timeout)

    r = ms.run_member_step("spark2", T, NEW, runner=runner, sleep=lambda s: None, stamp="S")
    assert not r.ok and r.failed_step == "repin"
    assert "build" not in f.kinds() and "tag" not in f.kinds()


def test_install_that_did_not_land_target_fails_before_repin():
    f = FakeMember(install_lags=True)
    r = step(f)
    assert not r.ok and r.failed_step == "install_cli"
    assert "repin" not in f.kinds() and "build" not in f.kinds()
    assert not f.lock_taken


def test_failed_build_leaves_old_gateway_running():
    f = FakeMember(fail_build=True)
    r = step(f)
    assert not r.ok and r.failed_step == "rebuild_gateway"
    assert f.running == OLD
    # nothing destructive after the failed build; old gateway checked healthy
    after = f.kinds()[f.kinds().index("build") + 1 :]
    assert set(after) <= {"health", "lock_release"}
    assert r.old_gateway_version == OLD
    assert "lobes rollout rollback --member spark2 --apply" in r.error
    assert KEY not in r.error
    assert not f.lock_taken


def test_active_window_refuses_before_any_write():
    f = FakeMember(window={"ok": True, "reason": None, "active": True})
    r = step(f)
    assert not r.ok and r.skipped and "window" in r.reason
    assert f.writes() == []
    assert not f.lock_taken


def test_active_window_overridable():
    f = FakeMember(window={"ok": True, "reason": None, "active": True})
    r = step(f, allow_window=True)
    assert r.ok and "build" in f.kinds()


def test_inactive_window_proceeds():
    f = FakeMember(window={"ok": True, "reason": None, "active": False})
    assert step(f).ok


def test_unreadable_window_state_refuses_unless_overridden():
    f = FakeMember()

    def runner(argv, timeout):
        if "window show" in argv[-1]:
            f.calls.append(("window", argv[-1]))
            return 0, "not json", ""
        return f(argv, timeout)

    r = ms.run_member_step("spark2", T, NEW, runner=runner, sleep=lambda s: None, stamp="S")
    assert not r.ok and r.skipped and f.writes() == []


def test_explicit_machine_cli_is_used():
    f = FakeMember(window={"ok": True, "reason": None, "active": False})
    step(f, machine_cli="/home/u/.local/bin/thor")
    cmd = next(c for k, c in f.calls if k == "window")
    assert cmd == "/home/u/.local/bin/thor window show --json"


def test_lock_held_refuses_naming_holder():
    f = FakeMember(lock_held=True)
    r = step(f)
    assert not r.ok and r.failed_step == "lock"
    assert "host=other" in r.error
    assert f.writes() == []


def test_health_wait_polls_until_target():
    f = FakeMember(health_up_after=2)
    sleeps = []
    r = step(f, sleep=sleeps.append, health_interval=1.0)
    assert r.ok and len(sleeps) >= 2


def test_health_wait_times_out():
    f = FakeMember(health_up_after=100)
    r = step(f, health_attempts=3)
    assert not r.ok and r.failed_step == "wait_health"
    assert not f.lock_taken


def test_interrupted_after_repin_resumes_without_new_backup_or_reinstall():
    # previous run installed + repinned, then died before the rebuild
    f = FakeMember(installed=NEW, pinned=NEW, running=OLD)
    r = step(f)
    assert r.ok and r.changed
    assert "backup" not in f.kinds() and "install" not in f.kinds()
    assert f.tags == {f"lobes-gateway:{OLD}": IMAGE_ID}
    assert "build" in f.kinds()


def test_journal_records_each_step_and_rollback_can_find_backup(tmp_path):
    j = Journal(tmp_path)
    f = FakeMember()
    r = step(f, journal=j.append)
    assert r.ok
    steps = [rec["step"] for rec in j.records()]
    assert steps == [
        "lock",
        "window_check",
        "backup_env",
        "install_cli",
        "repin",
        "tag_image",
        "rebuild_gateway",
        "wait_health",
    ]
    assert all(rec["member"] == "spark2" for rec in j.records())
    assert backup_from_journal(j, "spark2") == "/home/u/.lobes/.env.bak-STAMP-rollout"
    # a resumed run must not shadow the pre-apply backup
    step(FakeMember(installed=NEW, pinned=NEW, running=OLD), journal=j.append)
    assert backup_from_journal(j, "spark2") == "/home/u/.lobes/.env.bak-STAMP-rollout"


def test_no_secret_in_journal_or_result(tmp_path):
    j = Journal(tmp_path)
    f = FakeMember(fail_build=True)
    r = step(f, journal=j.append, secrets=(KEY,))
    text = (tmp_path / "rollout-journal.jsonl").read_text()
    assert KEY not in text
    assert KEY not in repr(r)


def test_env_value_read_never_dumps_whole_env():
    f = FakeMember()
    step(f)
    for k, c in f.calls:
        assert "cat " not in c or ".env" not in c
        if k == "read_pin":
            assert "MODEL_GEAR_VERSION" in c


def test_target_version_validated():
    with pytest.raises(ValueError):
        ms.run_member_step("m", T, "1.0; rm -rf /", runner=FakeMember())


def test_install_adds_dev_index_when_set():
    f = FakeMember()
    step(f, index_url="https://test.pypi.org/simple/")
    cmd = next(c for k, c in f.calls if k == "install")
    assert "--extra-index-url https://test.pypi.org/simple/" in cmd
    assert "--index-strategy unsafe-best-match" in cmd


def test_install_without_dev_index_is_plain():
    f = FakeMember()
    step(f)
    cmd = next(c for k, c in f.calls if k == "install")
    assert "index" not in cmd
