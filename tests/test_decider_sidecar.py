"""Decider sidecar: backend-agnostic HTTP core, tested with a fake backend (no torch)."""

import json
import sys

import pytest

from lobes.decider_sidecar import server as ds


class FakeBackend:
    def __init__(self, response=b'{"model":"fake","answers":{}}'):
        self.loaded = False
        self.seen = []
        self.response = response

    def load(self):
        self.loaded = True

    def decide(self, body):
        self.seen.append(body)
        return self.response


def test_health_503_until_loaded_then_200():
    sidecar = ds.Sidecar(FakeBackend())
    status, body, _ = sidecar.handle("GET", "/health", b"")
    assert status == 503 and json.loads(body)["status"] == "loading"
    sidecar.load()
    status, body, _ = sidecar.handle("GET", "/health", b"")
    assert status == 200 and json.loads(body)["status"] == "ok"


def test_post_before_load_is_503_and_backend_untouched():
    backend = FakeBackend()
    status, _, _ = ds.Sidecar(backend).handle("POST", "/v1/systemone", b"{}")
    assert status == 503 and backend.seen == []


def test_body_passthrough_byte_identical_both_ways():
    native = b'{"model":"clef","answers":{"q":{"choice":"a"}},  "usage":{}}'
    backend = FakeBackend(native)
    sidecar = ds.Sidecar(backend)
    sidecar.load()
    raw = b'{"state": "x",  "questions":{"q":{"type":"choice"}}}'
    status, out, ctype = sidecar.handle("POST", "/v1/systemone", raw)
    assert status == 200 and out == native and ctype == "application/json"
    assert backend.seen == [raw]


def test_dict_response_is_serialised_not_reshaped():
    native = {"model": "m", "answers": {"q": {"choice": "a"}}, "latency_ms": 1.0}
    sidecar = ds.Sidecar(FakeBackend(native))
    sidecar.load()
    _, out, _ = sidecar.handle("POST", "/v1/systemone", b"{}")
    assert json.loads(out) == native


def test_unknown_path_404_and_wrong_method_405():
    sidecar = ds.Sidecar(FakeBackend())
    sidecar.load()
    assert sidecar.handle("GET", "/nope", b"")[0] == 404
    assert sidecar.handle("POST", "/health", b"")[0] == 405
    assert sidecar.handle("GET", "/v1/systemone", b"")[0] == 405


def test_backend_value_error_is_422():
    class Bad(FakeBackend):
        def decide(self, body):
            raise ValueError("too many options")

    sidecar = ds.Sidecar(Bad())
    sidecar.load()
    status, out, _ = sidecar.handle("POST", "/v1/systemone", b"{}")
    assert status == 422 and "too many options" in json.loads(out)["detail"]


def test_select_backend_rejects_invalid_value():
    with pytest.raises(ds.DeciderConfigError, match="DECIDER_BACKEND"):
        ds.select_backend("vllm", {})
    with pytest.raises(ds.DeciderConfigError, match="DECIDER_BACKEND"):
        ds.select_backend("", {})


def test_select_backend_requires_model_and_names_both_backends():
    with pytest.raises(ds.DeciderConfigError, match="DECIDER_MODEL"):
        ds.select_backend("clef", {})
    assert type(ds.select_backend("clef", {"DECIDER_MODEL": "a/b"})).__name__ == "ClefBackend"
    assert (
        type(ds.select_backend("strands", {"DECIDER_MODEL": "/ckpt"})).__name__ == "StrandsBackend"
    )


def test_constructing_backends_does_not_import_torch():
    ds.select_backend("clef", {"DECIDER_MODEL": "a/b", "DECIDER_REVISION": "abc"})
    ds.select_backend("strands", {"DECIDER_MODEL": "/ckpt"})
    assert "torch" not in sys.modules
