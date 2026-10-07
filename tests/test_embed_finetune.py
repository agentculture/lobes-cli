"""Fine-tune lane mechanism (orin-embedding-specialist t11, o11)."""

import math

import pytest

from lobes.embed_lanes import EMBED_LANES, all_lanes, parse_finetune_lanes
from lobes.embed_sidecar.server import handle_embeddings, load_settings

PROBES = ["alpha", "beta gamma", "def f(x): return x", "the quick brown fox"]
TUNE_ENV = {"EMBED_FINETUNE_LANES": "my-tune=/models/my-tune"}
TUNE_SIDECAR_ENV = {"EMBED_MODEL_ID": "/models/my-tune", "EMBED_SERVED_NAME": "local:my-tune"}


class FakeEncoder:
    prompts: dict = {}

    def __init__(self, salt: int) -> None:
        self.salt = salt

    def encode(self, inputs, prompt_name):
        out = []
        for item in inputs:
            text = "".join(str(p.value) for p in item.parts)
            seed = sum(map(ord, text)) + self.salt
            out.append([math.sin(seed * (i + 1)) + 2.0 for i in range(768)])
        return out


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _vectors(encoder, settings):
    code, body = handle_embeddings({"input": PROBES}, encoder, settings)
    assert code == 200
    return body, [d["embedding"] for d in body["data"]]


def test_declared_lane_has_own_identity_and_base_shape():
    (lane,) = parse_finetune_lanes(TUNE_ENV)
    base = next(x for x in EMBED_LANES if x.name == "gemma2-embed")
    assert (lane.name, lane.catalog_id) == ("my-tune", "local:my-tune")
    assert lane.base_url_env == "MY_TUNE_BASE_URL"
    assert lane.checkpoint_path == "/models/my-tune"
    for attr in ("engine", "task", "modalities", "dim", "mrl_dims", "normalization"):
        assert getattr(lane, attr) == getattr(base, attr)


def test_all_lanes_does_not_mutate_registry():
    before = tuple(EMBED_LANES)
    assert all_lanes({}) == before
    assert all_lanes(TUNE_ENV)[: len(before)] == before
    assert len(all_lanes(TUNE_ENV)) == len(before) + 1
    assert EMBED_LANES == before


@pytest.mark.parametrize(
    "value",
    [
        "gemma2-embed=/p",  # collides with a lane
        "cortex=/p",  # role
        "cortex-x=/p",  # member lane
        "BAD_Name=/p",
        "a=relative/p",
        "a=",
        "novalue",
        "a=/p,a=/q",  # duplicate
    ],
)
def test_bad_declarations_refused(value):
    with pytest.raises(ValueError):
        parse_finetune_lanes({"EMBED_FINETUNE_LANES": value})


def test_sidecar_reports_served_name_not_path():
    body, _ = _vectors(FakeEncoder(1), load_settings(TUNE_SIDECAR_ENV))
    assert body["model"] == "local:my-tune"
    assert load_settings({}).reported_model == "google/embeddinggemma-2"


def test_finetune_instance_leaves_base_vectors_identical():
    base_settings = load_settings({})
    base_body, before = _vectors(FakeEncoder(0), base_settings)
    tune_body, tuned = _vectors(FakeEncoder(7), load_settings(TUNE_SIDECAR_ENV))
    base_body2, after = _vectors(FakeEncoder(0), base_settings)
    assert base_body["model"] == base_body2["model"] == "google/embeddinggemma-2"
    assert tune_body["model"] == "local:my-tune"
    assert all(_cos(a, b) == pytest.approx(1.0, abs=1e-12) for a, b in zip(before, after))
    assert any(_cos(a, b) < 1.0 - 1e-6 for a, b in zip(before, tuned))
