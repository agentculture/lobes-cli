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
    assert status == 503
    assert json.loads(body)["status"] == "loading"
    sidecar.load()
    status, body, _ = sidecar.handle("GET", "/health", b"")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


def test_post_before_load_is_503_and_backend_untouched():
    backend = FakeBackend()
    status, _, _ = ds.Sidecar(backend).handle("POST", "/v1/systemone", b"{}")
    assert status == 503
    assert backend.seen == []


def test_body_passthrough_byte_identical_both_ways():
    native = b'{"model":"clef","answers":{"q":{"choice":"a"}},  "usage":{}}'
    backend = FakeBackend(native)
    sidecar = ds.Sidecar(backend)
    sidecar.load()
    raw = b'{"state": "x",  "questions":{"q":{"type":"choice"}}}'
    status, out, ctype = sidecar.handle("POST", "/v1/systemone", raw)
    assert status == 200
    assert out == native
    assert ctype == "application/json"
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
    assert status == 422
    assert "too many options" in json.loads(out)["detail"]


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


def test_the_strands_backend_keeps_the_pinned_revision():
    # strands-decider's StrandsDeciderModel.load(path) resolves a hub id WITHOUT a
    # revision (always the repo HEAD), so the sidecar must carry DECIDER_REVISION
    # itself and resolve the pinned snapshot before loading (t12).
    pinned = ds.select_backend(
        "strands", {"DECIDER_MODEL": "org/strands-2b", "DECIDER_REVISION": "216d1ccb"}
    )
    assert pinned.revision == "216d1ccb"
    assert ds.select_backend("strands", {"DECIDER_MODEL": "/ckpt"}).revision is None


def test_constructing_backends_does_not_import_torch():
    ds.select_backend("clef", {"DECIDER_MODEL": "a/b", "DECIDER_REVISION": "abc"})
    ds.select_backend("strands", {"DECIDER_MODEL": "/ckpt"})
    assert "torch" not in sys.modules


def _decider_dockerfile() -> str:
    from importlib.resources import files

    return files("lobes.templates").joinpath("fleet", "Dockerfile.decider").read_text()


def test_decider_image_needs_no_git():
    # the vLLM base image ships no git binary (measured on the Spark, 2026-10-14)
    text = _decider_dockerfile()
    assert "git+" not in text
    assert "strands-decider/archive/${STRANDS_DECIDER_VERSION}.tar.gz" in text


def test_decider_image_pins_transformers_in_one_resolve():
    # a second resolve silently moved a --no-deps pin (5.19.0 -> 5.17.0)
    text = _decider_dockerfile()
    assert text.count("transformers==") == 1
    assert '"transformers==5.17.0" pillow' in text


def test_load_failure_reports_failed_never_loading():
    # A load that raises in the daemon thread used to leave /health at 503
    # "loading" forever, so the restart policy never fired.
    class Broken(FakeBackend):
        def load(self):
            raise RuntimeError("CUDA out of memory")

    backend = Broken()
    sidecar = ds.Sidecar(backend)
    with pytest.raises(RuntimeError):
        sidecar.load()
    status, body, _ = sidecar.handle("GET", "/health", b"")
    assert status == 500
    payload = json.loads(body)
    assert payload["status"] == "failed"
    assert "CUDA out of memory" in payload["error"]
    status, body, _ = sidecar.handle("POST", "/v1/systemone", b"{}")
    assert status == 500
    assert json.loads(body)["status"] == "failed"
    assert backend.seen == []


def test_load_or_exit_exits_nonzero_on_a_failed_load():
    class Broken(FakeBackend):
        def load(self):
            raise RuntimeError("bad checkpoint")

    exits = []
    sidecar = ds.Sidecar(Broken())
    ds.load_or_exit(sidecar, exit_=exits.append)
    assert exits == [1]
    assert sidecar.handle("GET", "/health", b"")[0] == 500


def test_load_or_exit_does_not_exit_on_success():
    exits = []
    sidecar = ds.Sidecar(FakeBackend())
    ds.load_or_exit(sidecar, exit_=exits.append)
    assert exits == []
    assert sidecar.handle("GET", "/health", b"")[0] == 200


def test_health_answers_while_a_decision_is_running():
    # The FastAPI shell's route is async: a blocking decide() on the event loop
    # would stall /health for the whole forward pass.
    import asyncio
    import threading

    started, release = threading.Event(), threading.Event()

    class Slow(FakeBackend):
        def decide(self, body):
            started.set()
            assert release.wait(5)
            return super().decide(body)

    sidecar = ds.Sidecar(Slow())
    sidecar.load()

    async def scenario():
        decision = asyncio.ensure_future(ds.respond(sidecar, "POST", "/v1/systemone", b"{}"))
        while not started.is_set():
            await asyncio.sleep(0.01)
        health = await asyncio.wait_for(ds.respond(sidecar, "GET", "/health", b""), 1)
        assert not decision.done()
        release.set()
        return health, await decision

    health, decision = asyncio.run(scenario())
    assert health[0] == 200
    assert decision[0] == 200
