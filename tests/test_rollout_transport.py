import pytest

from lobes.rollout._redact import redact
from lobes.rollout._transport import read_remote_value, run, run_lobes

R = {"ssh_target": "spark2", "user": "ori", "lobes_path": "/home/ori/.local/bin/lobes"}


def fake(calls, out="", err="", rc=0):
    def runner(argv, timeout):
        calls.append((list(argv), timeout))
        return rc, out, err

    return runner


def test_ssh_uses_target_user_batchmode_timeout():
    calls = []
    run(R, ["echo", "a b"], timeout=7, runner=fake(calls))
    argv, to = calls[0]
    assert argv[0] == "ssh" and "BatchMode=yes" in argv and "ConnectTimeout=7" in argv
    assert "ori@spark2" in argv and argv[-1] == "echo 'a b'" and to == 7


def test_lobes_absolute_path():
    calls = []
    run_lobes(R, ["status"], runner=fake(calls))
    assert calls[0][0][-1] == "/home/ori/.local/bin/lobes status"
    with pytest.raises(ValueError):
        run_lobes({**R, "lobes_path": "lobes"}, ["x"], runner=fake([]))


def test_local_no_ssh():
    calls = []
    run_lobes({"local": True, "lobes_path": "/x/lobes"}, ["v"], runner=fake(calls))
    assert calls[0][0] == ["/x/lobes", "v"]


def test_timeout_flag():
    assert run(R, ["x"], runner=fake([], rc=124)).timed_out


@pytest.mark.parametrize(
    "text",
    [
        "GATEWAY_API_KEY=sekret1",
        "LOBES_MESH_KEY: sekret1",
        "FOO_API_KEY='sekret1'",
        "HF_TOKEN=sekret1",
        '{"GATEWAY_API_KEY": "sekret1"}',
        "Authorization: Bearer sekret1",
    ],
)
def test_redact_patterns(text):
    assert "sekret1" not in redact(text)


def test_fixture_run_has_no_secrets():
    calls = []
    key = read_remote_value(R, ["cat", "k"], runner=fake(calls, out="sekretvalue99\n"))
    assert key == "sekretvalue99"
    res = run(
        R,
        ["x"],
        runner=fake([], out="using sekretvalue99 now", err="GATEWAY_API_KEY=other77"),
        secrets=[key],
    )
    blob = res.stdout + res.stderr
    assert "sekretvalue99" not in blob and "other77" not in blob
