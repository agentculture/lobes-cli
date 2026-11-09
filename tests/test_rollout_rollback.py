from lobes.rollout import _rollback as rb
from lobes.rollout._transport import Target

T = Target(ssh_target="h", user="u", lobes_path="/home/u/.local/bin/lobes")
BAK = "/home/u/.lobes/.env.bak-20261109-rollout"


class Fake:
    def __init__(self, fail_on=None, inspect_out="lobes-gateway:latest\n"):
        self.calls = []
        self.fail_on = fail_on
        self.inspect_out = inspect_out

    def __call__(self, argv, timeout):
        cmd = argv[-1]
        self.calls.append(cmd)
        if self.fail_on and self.fail_on in cmd:
            return 1, "", "boom"
        if "inspect" in cmd:
            return 0, self.inspect_out, ""
        return 0, "", ""


def test_command_string():
    assert rb.rollback_command("spark2") == "lobes rollout rollback --member spark2 --apply"


def test_dry_run_runs_nothing_and_returns_plan():
    f = Fake()
    r = rb.rollback("m", T, "1.2.3", BAK, runner=f)
    assert f.calls == [] and not r.applied and r.ok
    text = " ".join(a.command for a in r.actions)
    assert BAK in text and "lobes-gateway:1.2.3" in text and "up gateway --apply" in text
    assert "--build" not in text


def test_apply_restores_env_tags_and_recreates_without_index():
    f = Fake()
    r = rb.rollback("m", T, "1.2.3", BAK, apply=True, runner=f)
    assert r.ok and r.applied
    joined = "\n".join(f.calls)
    assert f"cp -p {BAK} /home/u/.lobes/.env" in joined
    assert "docker tag lobes-gateway:1.2.3 lobes-gateway:latest" in joined
    assert f.calls[-1].endswith("/home/u/.local/bin/lobes up gateway --apply")
    for banned in ("pip", "uv ", "--build", "install"):
        assert banned not in joined


def test_apply_stops_on_failure():
    f = Fake(fail_on="docker tag")
    r = rb.rollback("m", T, "1.2.3", BAK, apply=True, runner=f)
    assert not r.ok
    assert not any("up gateway" in c for c in f.calls)


def test_backup_from_journal(tmp_path):
    from lobes.rollout._journal import Journal

    j = Journal(tmp_path)
    j.append("m", "backup_env", "ok", detail={"path": BAK})
    assert rb.backup_from_journal(j, "m") == BAK
    assert rb.backup_from_journal(j, "other") is None


def test_missing_backup_refused():
    r = rb.rollback("m", T, "1.2.3", None, apply=True, runner=Fake())
    assert not r.ok and "backup" in r.error
