"""Release preflight: resolvability of lobes-cli==<v> from the gateway base image."""

import inspect

from lobes.rollout import _preflight as pf


class FakeRunner:
    def __init__(self, fail_first=0):
        self.fail_first = fail_first
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if len(self.calls) <= self.fail_first:
            return 1, "ERROR: No matching distribution found for lobes-cli (404)"
        return 0, "Saved lobes_cli.whl"


class Clock:
    def __init__(self):
        self.sleeps = []

    def __call__(self, s):
        self.sleeps.append(s)


def test_resolvable_passes_first_try():
    r, c = FakeRunner(), Clock()
    res = pf.check_release("1.2.3", runner=r, sleep=c, environ={})
    assert res.ok and res.attempts == 1 and c.sleeps == []
    assert r.calls[0][:4] == ["docker", "run", "--rm", pf.gateway_base_image()]
    assert "lobes-cli==1.2.3" in r.calls[0] and "--no-deps" in r.calls[0]
    assert "pip" in r.calls[0] and "download" in r.calls[0]


def test_404_then_resolves_passes_with_backoff():
    r, c = FakeRunner(fail_first=3), Clock()
    res = pf.check_release("1.2.3", runner=r, sleep=c, environ={})
    assert res.ok and res.attempts == 4
    assert c.sleeps == [5.0, 10.0, 20.0]


def test_never_resolving_fails_with_hint_bounded():
    r, c = FakeRunner(fail_first=10**6), Clock()
    res = pf.check_release("9.9.9", runner=r, sleep=c, environ={}, total_s=300)
    assert not res.ok
    assert "lobes-cli==9.9.9" in res.hint and "No member was contacted" in res.hint
    assert sum(c.sleeps) <= 300 and res.waited_s == sum(c.sleeps)
    assert max(c.sleeps) <= 60


def test_no_member_access_in_api():
    params = inspect.signature(pf.check_release).parameters
    assert not [p for p in params if "member" in p or "transport" in p]
    src = inspect.getsource(pf)
    assert "urllib" not in src and "http.client" not in src and "socket" not in src


def test_base_image_read_from_dockerfile(tmp_path):
    df = tmp_path / "Dockerfile.gateway"
    df.write_text("# c\nFROM example/base:9\nRUN true\n")
    r = FakeRunner()
    pf.check_release("1.0.0", runner=r, sleep=Clock(), dockerfile=df, environ={})
    assert r.calls[0][3] == "example/base:9"
    assert pf.gateway_base_image().startswith("python:")


def test_dev_index_url_honoured():
    r = FakeRunner()
    env = {"LOBES_DEV_INDEX_URL": "https://test.pypi.org/simple/"}
    pf.check_release("1.2.3.dev4", runner=r, sleep=Clock(), environ=env)
    argv = r.calls[0]
    assert argv[argv.index("--index-url") + 1] == "https://test.pypi.org/simple/"
    r2 = FakeRunner()
    pf.check_release("1.2.3", runner=r2, sleep=Clock(), environ={})
    assert "--index-url" not in r2.calls[0]
