"""Tests for the EmbeddingGemma 2 sidecar's pure request/response layer.

Stdlib + pytest only: every test drives ``handle_embeddings`` (or one of its
helpers) with an injected fake encoder, so no torch / sentence-transformers /
fastapi is needed — the same split as the realtime TTS sidecars.
"""

from __future__ import annotations

import base64
import math
import struct

import pytest

from lobes.embed_sidecar.server import (
    MRL_DIMENSIONS,
    EmbedError,
    EmbedInput,
    Part,
    Settings,
    guard_vector,
    handle_embeddings,
    health_status,
    load_settings,
    parse_dimensions,
    parse_request,
    resolve_dtype,
    resolve_prompt,
    truncate_and_normalize,
)

PROMPTS = {
    "query": "task: search result | query: ",
    "Document": "title: none | text: ",
    "SearchQuery": "task: search result | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
}

ALL = frozenset({"text", "image", "video", "audio"})
TEXT_ONLY = frozenset({"text"})

PNG_URL = "data:image/png;base64," + base64.b64encode(b"\x89PNG-fake").decode()
MP4_URL = "data:video/mp4;base64," + base64.b64encode(b"fake-mp4").decode()
WAV_B64 = base64.b64encode(b"RIFF-fake-wav").decode()


def _norm(vec):
    return math.sqrt(sum(x * x for x in vec))


class FakeEncoder:
    """Records what it was asked to encode; returns a fixed (or given) vector."""

    def __init__(self, vector=None, prompts=None):
        self.prompts = dict(PROMPTS if prompts is None else prompts)
        self.vector = vector if vector is not None else [float(i + 1) for i in range(768)]
        self.calls: list[tuple[list[EmbedInput], str | None]] = []

    def encode(self, inputs, prompt_name):
        self.calls.append((list(inputs), prompt_name))
        return [list(self.vector) for _ in inputs]


def _settings(modalities=ALL):
    return Settings(model_id="google/embeddinggemma-2", modalities=modalities, dtype="bfloat16")


def _ok(body, encoder=None, settings=None):
    enc = encoder or FakeEncoder()
    status, payload = handle_embeddings(body, enc, settings or _settings())
    assert status == 200, payload
    return payload, enc


def _err(body, encoder=None, settings=None):
    status, payload = handle_embeddings(body, encoder or FakeEncoder(), settings or _settings())
    assert status != 200, payload
    assert "error" in payload
    return status, payload["error"]


# --------------------------------------------------------------------------
# settings / load (o2: float16 refused at load)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["fp16", "float16", "FP16", "half", "torch.float16"])
def test_float16_is_refused_at_load(value):
    with pytest.raises(EmbedError) as exc:
        resolve_dtype(value)
    assert "float16" in str(exc.value)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "bfloat16"),
        ("", "bfloat16"),
        ("bf16", "bfloat16"),
        ("bfloat16", "bfloat16"),
        ("fp32", "float32"),
        ("float32", "float32"),
    ],
)
def test_supported_dtypes(value, expected):
    assert resolve_dtype(value) == expected


def test_unknown_dtype_is_refused():
    with pytest.raises(EmbedError):
        resolve_dtype("int8")


def test_load_settings_defaults():
    s = load_settings({})
    assert s.model_id == "google/embeddinggemma-2"
    assert s.dtype == "bfloat16"
    assert s.modalities == TEXT_ONLY


def test_load_settings_full_modalities_and_fp16_refusal():
    s = load_settings({"EMBED_MODALITIES": "text, image,video ,audio", "EMBED_DTYPE": "fp32"})
    assert s.modalities == ALL
    assert s.dtype == "float32"
    with pytest.raises(EmbedError):
        load_settings({"EMBED_DTYPE": "float16"})


def test_load_settings_refuses_unknown_modality_and_missing_text():
    with pytest.raises(EmbedError):
        load_settings({"EMBED_MODALITIES": "text,smell"})
    with pytest.raises(EmbedError):
        load_settings({"EMBED_MODALITIES": "image"})


def test_health_status():
    s = _settings(TEXT_ONLY)
    assert health_status(False, s)[0] == 503
    code, body = health_status(True, s)
    assert code == 200
    assert body["status"] == "ok"
    assert body["modalities_loaded"] == ["text"]
    assert body["model"] == "google/embeddinggemma-2"
    assert body["dtype"] == "bfloat16"


# --------------------------------------------------------------------------
# vector guard (o2) and MRL truncation (o3)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vec", [[float("nan"), 1.0], [float("inf"), 1.0], [float("-inf"), 0.0], [0.0, 0.0, 0.0]]
)
def test_guard_vector_refuses_degenerate(vec):
    with pytest.raises(EmbedError):
        guard_vector(vec)


@pytest.mark.parametrize("vec", [[float("nan")] * 768, [0.0] * 768, [1.0] * 767 + [float("inf")]])
def test_degenerate_vector_is_an_error_never_200(vec):
    status, err = _err({"input": "hello"}, FakeEncoder(vector=vec))
    assert status >= 500
    assert err["code"] == "degenerate_embedding"


def test_truncated_slice_all_zero_is_an_error():
    vec = [0.0] * 128 + [1.0] * 640
    status, err = _err({"input": "x", "dimensions": 128}, FakeEncoder(vector=vec))
    assert status >= 500
    assert err["code"] == "degenerate_embedding"


@pytest.mark.parametrize("dims", MRL_DIMENSIONS)
def test_truncate_and_normalize_unit_length(dims):
    vec = [float(i + 1) for i in range(768)]
    out = truncate_and_normalize(vec, dims)
    assert len(out) == dims
    assert math.isclose(_norm(out), 1.0, rel_tol=1e-9)
    # direction preserved: proportional to the leading slice
    assert math.isclose(out[1] / out[0], 2.0, rel_tol=1e-9)


@pytest.mark.parametrize("dims", [128, 256, 512, 768])
def test_response_vectors_are_unit_length_at_every_mrl_dim(dims):
    payload, _ = _ok({"input": ["a", "b"], "dimensions": dims})
    assert payload["dimensions"] == dims
    for item in payload["data"]:
        assert len(item["embedding"]) == dims
        assert math.isclose(_norm(item["embedding"]), 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("dims", [0, 64, 100, 129, 1024, -128, "256", 256.0, True])
def test_unsupported_dimensions_are_refused(dims):
    status, err = _err({"input": "x", "dimensions": dims})
    assert status == 400
    assert err["code"] == "unsupported_dimensions"


def test_default_dimensions_is_768():
    assert parse_dimensions(None) == 768
    payload, _ = _ok({"input": "x"})
    assert payload["dimensions"] == 768


def test_encoder_wrong_width_is_an_error():
    status, err = _err({"input": "x", "dimensions": 768}, FakeEncoder(vector=[1.0] * 300))
    assert status >= 500


# --------------------------------------------------------------------------
# request parsing / response shape
# --------------------------------------------------------------------------


def test_text_string_input_openai_shape():
    payload, enc = _ok({"input": "hello", "model": "whatever-the-client-sent"})
    assert payload["object"] == "list"
    assert payload["model"] == "google/embeddinggemma-2"
    assert payload["prompt_name"] is None
    assert payload["modalities_loaded"] == ["audio", "image", "text", "video"]
    assert payload["data"][0]["object"] == "embedding"
    assert payload["data"][0]["index"] == 0
    assert "usage" in payload
    inputs, prompt_name = enc.calls[0]
    assert prompt_name is None
    assert inputs == [EmbedInput(parts=(Part("text", "hello"),))]


def test_text_list_input_indices():
    payload, enc = _ok({"input": ["a", "b", "c"]})
    assert [d["index"] for d in payload["data"]] == [0, 1, 2]
    assert len(enc.calls[0][0]) == 3


@pytest.mark.parametrize("bad", [[], "", [""], [1, 2], 42, None, [["nested"]]])
def test_bad_text_input_refused(bad):
    status, _ = _err({"input": bad})
    assert status == 400


def test_input_and_messages_are_exclusive_and_one_required():
    assert _err({"input": "x", "messages": [{"role": "user", "content": "y"}]})[0] == 400
    assert _err({})[0] == 400
    assert _err("not a dict")[0] == 400


def test_prompt_name_applied_and_echoed():
    payload, enc = _ok({"input": "q", "prompt_name": "SearchQuery"})
    assert payload["prompt_name"] == "SearchQuery"
    assert payload["prompt"] == PROMPTS["SearchQuery"]
    assert enc.calls[0][1] == "SearchQuery"


def test_unknown_prompt_name_refused():
    status, err = _err({"input": "q", "prompt_name": "Nope"})
    assert status == 400
    assert err["code"] == "unknown_prompt_name"


def test_resolve_prompt():
    assert resolve_prompt(None, PROMPTS) == (None, None)
    assert resolve_prompt("Document", PROMPTS) == ("Document", PROMPTS["Document"])
    with pytest.raises(EmbedError):
        resolve_prompt(3, PROMPTS)


def test_base64_encoding_format():
    payload, _ = _ok({"input": "x", "dimensions": 128, "encoding_format": "base64"})
    raw = base64.b64decode(payload["data"][0]["embedding"])
    floats = struct.unpack("<128f", raw)
    assert math.isclose(_norm(floats), 1.0, rel_tol=1e-5)


def test_unknown_encoding_format_refused():
    assert _err({"input": "x", "encoding_format": "binary"})[0] == 400


# --------------------------------------------------------------------------
# multimodal messages (o4: never converted to text)
# --------------------------------------------------------------------------


def _msg(*parts):
    return {"messages": [{"role": "user", "content": list(parts)}]}


def test_messages_with_every_modality_reach_their_own_encoder():
    body = _msg(
        {"type": "text", "text": "a red square"},
        {"type": "image_url", "image_url": {"url": PNG_URL}},
        {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}},
        {"type": "video_url", "video_url": {"url": MP4_URL}},
    )
    payload, enc = _ok(body)
    assert len(payload["data"]) == 1
    inputs, _ = enc.calls[0]
    parts = inputs[0].parts
    assert [p.modality for p in parts] == ["text", "image", "audio", "video"]
    assert parts[0].value == "a red square"
    assert parts[1].value == b"\x89PNG-fake"
    assert parts[1].mime == "image/png"
    assert parts[2].value == b"RIFF-fake-wav"
    assert parts[2].mime == "audio/wav"
    assert parts[3].value == b"fake-mp4"
    # never captioned: no non-text part is a str, and no text part carries media
    for p in parts:
        if p.modality != "text":
            assert isinstance(p.value, bytes)
    assert payload["modalities"] == ["text", "image", "audio", "video"]


def test_image_part_with_text_like_payload_stays_image():
    url = "data:text/plain;base64," + base64.b64encode(b"hello").decode()
    status, err = _err(_msg({"type": "image_url", "image_url": {"url": url}}))
    assert status == 400
    assert err["code"] == "invalid_media"


@pytest.mark.parametrize(
    "part,modality",
    [
        ({"type": "image_url", "image_url": {"url": PNG_URL}}, "image"),
        ({"type": "video_url", "video_url": {"url": MP4_URL}}, "video"),
        ({"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}}, "audio"),
    ],
)
def test_unloaded_modality_refused_never_dropped(part, modality):
    enc = FakeEncoder()
    status, err = _err(_msg({"type": "text", "text": "x"}, part), enc, _settings(TEXT_ONLY))
    assert status == 400
    assert err["code"] == "modality_not_loaded"
    assert modality in err["message"]
    assert enc.calls == []  # nothing encoded — not even the text part


def test_http_urls_refused_not_fetched():
    status, err = _err(_msg({"type": "image_url", "image_url": {"url": "https://x/y.png"}}))
    assert status == 400
    assert err["code"] == "remote_url_unsupported"


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!notb64"}},
        {"type": "image_url", "image_url": {}},
        {"type": "image_url", "image_url": "data:image/png;base64,AAAA"},
        {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "ogg-weird"}},
        {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
        {"type": "video_url", "video_url": {"url": "data:image/png;base64,AAAA"}},
    ],
)
def test_malformed_media_parts_refused(part):
    assert _err(_msg(part))[0] == 400


def test_unknown_part_type_refused():
    status, err = _err(_msg({"type": "file", "file": {}}))
    assert status == 400
    assert err["code"] == "unsupported_part_type"


def test_duplicate_modality_in_one_message_refused():
    body = _msg(
        {"type": "image_url", "image_url": {"url": PNG_URL}},
        {"type": "image_url", "image_url": {"url": PNG_URL}},
    )
    assert _err(body)[0] == 400


def test_string_content_is_a_text_part():
    payload, enc = _ok({"messages": [{"role": "user", "content": "plain"}]})
    assert enc.calls[0][0][0].parts == (Part("text", "plain"),)
    assert payload["modalities"] == ["text"]


@pytest.mark.parametrize(
    "messages",
    [[], "x", [{"role": "user"}], [{"role": "user", "content": []}], [{"content": [42]}]],
)
def test_bad_messages_refused(messages):
    assert _err({"messages": messages})[0] == 400


def test_parse_request_returns_structured_request():
    req = parse_request({"input": "x", "prompt_name": "query", "dimensions": 256}, ALL)
    assert req.dimensions == 256
    assert req.prompt_name == "query"
    assert req.inputs == (EmbedInput(parts=(Part("text", "x"),)),)


def test_encoder_exception_is_500_not_200():
    class Boom(FakeEncoder):
        def encode(self, inputs, prompt_name):
            raise RuntimeError("cuda died")

    status, err = _err({"input": "x"}, Boom())
    assert status == 500
    assert err["code"] == "encode_failed"


def test_encoder_returning_wrong_count_is_an_error():
    class Short(FakeEncoder):
        def encode(self, inputs, prompt_name):
            return []

    assert _err({"input": ["a", "b"]}, Short())[0] == 500
