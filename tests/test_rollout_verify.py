"""Post-member verification: health target, capability diff, one real request per role."""

from __future__ import annotations

import json

from lobes.rollout import _observe, _verify

ORIGIN = "http://m:8000"


class _Resp:
    def __init__(self, status, body):
        self.status = status
        self._b = body if isinstance(body, bytes) else json.dumps(body).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeNet:
    def __init__(self, version="1.2.3", caps=None, overrides=None):
        self.version = version
        self.caps = caps
        self.overrides = overrides or {}
        self.calls = []

    def __call__(self, req, timeout=None):
        path = req.full_url.replace(ORIGIN, "")
        body = req.data
        self.calls.append((req.get_method(), path, dict(req.header_items()), body))
        if path in self.overrides:
            r = self.overrides[path]
            if isinstance(r, Exception):
                raise r
            return _Resp(*r)
        if path == "/health":
            return _Resp(200, {"version": self.version})
        if path == "/capabilities":
            return _Resp(200, {"roles": self.caps})
        if path == "/mesh/roster":
            return _Resp(200, {"members": []})
        if path == "/v1/chat/completions":
            return _Resp(200, {"choices": [{"message": {"content": "hi"}}]})
        if path == "/v1/embeddings":
            return _Resp(200, {"data": [{"embedding": [0.1]}]})
        if path == "/v1/rerank":
            return _Resp(200, {"results": [{"index": 0, "relevance_score": 1.0}]})
        if path == "/v1/audio/transcriptions":
            return _Resp(200, {"text": ""})
        return _Resp(404, {})

    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]


UP = {"feasible": True, "ready": True}
CAPS = {
    "cortex": dict(UP),
    "embedder": dict(UP),
    "reranker": dict(UP),
    "stt-en": {**UP, "lane": True},
    "worker": {**UP, "hosted_by": "http://peer"},
    "muse": {"feasible": False, "ready": False},
    "gemma2-embed": {**UP, "lane": True},
}


def _run(net, before=None, **kw):
    return _verify.verify_member(
        ORIGIN, target_version="1.2.3", before=before, gateway_key="k", opener=net, **kw
    )


def test_all_pass_one_request_per_up_role():
    net = FakeNet(caps=CAPS)
    res = _run(net)
    assert res.ok and not res.failures
    paths = sorted(p for _, p, _, _ in net.posts())
    assert paths == sorted(
        [
            "/v1/chat/completions",
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/embeddings",
            "/v1/rerank",
            "/v1/audio/transcriptions",
        ]
    )
    assert set(res.probed) == {"cortex", "worker", "embedder", "gemma2-embed", "reranker", "stt-en"}
    for _, _, h, _ in net.posts():
        assert h["Authorization"] == "Bearer k"


def test_request_bodies_use_role_model():
    net = FakeNet(caps={"cortex": dict(UP), "stt-he": {**UP, "lane": True}})
    _run(net)
    chat = [c for c in net.posts() if c[1] == "/v1/chat/completions"][0]
    assert json.loads(chat[3])["model"] == "cortex"
    stt = [c for c in net.posts() if c[1] == "/v1/audio/transcriptions"][0]
    assert b'name="model"' in stt[3] and b"stt-he" in stt[3] and b"RIFF" in stt[3]
    assert "multipart/form-data" in stt[2]["Content-type"]


def test_version_mismatch_fails_health():
    res = _run(FakeNet(version="9.9", caps=CAPS))
    assert not res.ok
    assert res.failures[0].check == "health" and res.failures[0].member == ORIGIN


def test_unreachable_fails_health_and_skips_probes():
    net = FakeNet(caps=CAPS, overrides={"/health": (500, {})})
    res = _run(net)
    assert not res.ok and res.failures[0].check == "health"
    assert not net.posts()


def test_capability_regression_reported_with_role():
    before = _observe.MemberObservation(
        ORIGIN, "1.0", {"cortex": dict(UP), "embedder": dict(UP)}, [], True
    )
    net = FakeNet(caps={"cortex": dict(UP), "embedder": {"feasible": True, "ready": False}})
    res = _run(net, before=before)
    assert not res.ok
    f = [f for f in res.failures if f.check == "capability_regression"]
    assert f and f[0].role == "embedder"


def test_probe_failure_names_member_check_role():
    net = FakeNet(caps=CAPS, overrides={"/v1/rerank": (503, {"error": "x"})})
    res = _run(net)
    assert not res.ok
    f = res.failures[0]
    assert (f.member, f.check, f.role) == (ORIGIN, "probe", "reranker")


def test_malformed_probe_body_fails_and_exception_fails():
    net = FakeNet(
        caps={"cortex": dict(UP), "embedder": dict(UP)},
        overrides={
            "/v1/chat/completions": (200, {"choices": []}),
            "/v1/embeddings": OSError("boom"),
        },
    )
    res = _run(net)
    assert sorted(f.role for f in res.failures) == ["cortex", "embedder"]


def test_unknown_family_not_probed_not_passed():
    caps = {"cortex": dict(UP), "decider": dict(UP), "weird-lane": {**UP, "lane": True}}
    res = _run(FakeNet(caps=caps))
    assert res.ok
    assert set(res.not_probed) == {"decider", "weird-lane"}
    assert res.probed == ["cortex"]


def test_error_does_not_leak_key():
    net = FakeNet(
        caps={"cortex": dict(UP)}, overrides={"/v1/chat/completions": OSError("k secret")}
    )
    res = _run(net)
    assert "Bearer" not in repr(res)
