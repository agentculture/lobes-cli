"""The gateway's multipart helpers (stt-lanes): read the small form fields of a
``/v1/audio/transcriptions`` upload and append a ``language`` part.

Every injected body is re-parsed with the stdlib MIME parser (``email``), an
independent implementation, so the boundary/CRLF framing is proven rather than
asserted against this module's own reader.
"""

from __future__ import annotations

from email import policy
from email.parser import BytesParser

import pytest

from lobes.gateway._multipart import append_form_field, form_boundary, read_form_fields

CT = "multipart/form-data; boundary=----lobesBoundary7MA4YWxk"
BOUNDARY = b"----lobesBoundary7MA4YWxk"
AUDIO = bytes(range(256)) * 64 + b"\r\n--not-a-boundary\r\n" + bytes(range(256))


def _part(name: str, value: bytes, filename: str | None = None, ctype: str | None = None) -> bytes:
    disp = f'form-data; name="{name}"'
    if filename is not None:
        disp += f'; filename="{filename}"'
    head = f"Content-Disposition: {disp}\r\n"
    if ctype:
        head += f"Content-Type: {ctype}\r\n"
    return b"--" + BOUNDARY + b"\r\n" + head.encode() + b"\r\n" + value + b"\r\n"


def _body(*parts: bytes) -> bytes:
    return b"".join(parts) + b"--" + BOUNDARY + b"--\r\n"


def _upload(**fields: str) -> bytes:
    parts = [_part("file", AUDIO, filename="clip.wav", ctype="audio/wav")]
    parts += [_part(k, v.encode()) for k, v in fields.items()]
    return _body(*parts)


def _parse(body: bytes, content_type: str = CT) -> dict[str, bytes]:
    """Parse with the stdlib MIME parser — independent of the module under test."""
    msg = BytesParser(policy=policy.HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body
    )
    assert msg.is_multipart()
    out: dict[str, bytes] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        out[name] = part.get_payload(decode=True)
    return out


def test_boundary_parsed_plain_and_quoted():
    assert form_boundary(CT) == BOUNDARY
    assert form_boundary('multipart/form-data; boundary="a b:c"') == b"a b:c"
    assert form_boundary("Multipart/Form-Data; charset=utf-8; Boundary=xyz") == b"xyz"


@pytest.mark.parametrize(
    "ctype", [None, "", "application/json", "multipart/form-data", "text/plain; boundary=x"]
)
def test_no_boundary_for_non_multipart(ctype):
    assert form_boundary(ctype) is None


def test_reads_model_and_language_fields():
    body = _upload(model="stt-he", language="he", response_format="json")
    assert read_form_fields(body, CT, ("model", "language")) == {
        "model": "stt-he",
        "language": "he",
    }


def test_never_reads_a_file_part_as_a_field():
    body = _body(_part("model", b"stt-he", filename="model.txt"))
    assert read_form_fields(body, CT, ("model",)) == {}


def test_first_occurrence_wins():
    body = _body(_part("model", b"stt-en"), _part("model", b"stt-he"))
    assert read_form_fields(body, CT, ("model",)) == {"model": "stt-en"}


def test_absent_fields_are_absent():
    assert read_form_fields(_upload(), CT, ("model", "language")) == {}


@pytest.mark.parametrize(
    "body",
    [b"", b"garbage", b'{"model": "stt-he"}', b"--" + BOUNDARY + b"\r\nno-headers-end"],
)
def test_malformed_bodies_yield_nothing(body):
    assert read_form_fields(body, CT, ("model",)) == {}


def test_non_multipart_content_type_yields_nothing():
    assert read_form_fields(_upload(model="stt-he"), "application/json", ("model",)) == {}


def test_unquoted_and_uppercase_disposition_header():
    raw = (
        b"--" + BOUNDARY + b"\r\nCONTENT-DISPOSITION: form-data; name=model\r\n\r\nstt-he\r\n"
        b"--" + BOUNDARY + b"--\r\n"
    )
    assert read_form_fields(raw, CT, ("model",)) == {"model": "stt-he"}


def test_appended_language_is_a_valid_part_and_the_rest_is_untouched():
    body = _upload(model="stt-he", response_format="json")
    out = append_form_field(body, CT, "language", "he")
    assert out is not None
    parsed = _parse(out)
    assert parsed["language"] == b"he"
    assert parsed["model"] == b"stt-he"
    assert parsed["response_format"] == b"json"
    assert parsed["file"] == AUDIO  # the audio bytes survive verbatim
    assert read_form_fields(out, CT, ("language",)) == {"language": "he"}
    # Only bytes are inserted, before the close delimiter; nothing is rewritten.
    close = b"\r\n--" + BOUNDARY + b"--"
    idx = body.rindex(close)
    assert out[:idx] == body[:idx]
    assert out.endswith(body[idx:])


def test_append_with_a_quoted_boundary_and_epilogue():
    ct = 'multipart/form-data; boundary="q-b"'
    body = (
        b"preamble\r\n--q-b\r\n"
        b'Content-Disposition: form-data; name="model"\r\n\r\nstt-en\r\n'
        b"--q-b--\r\nepilogue"
    )
    out = append_form_field(body, ct, "language", "en")
    assert _parse(out, ct) == {"model": b"stt-en", "language": b"en"}
    assert out.endswith(b"--q-b--\r\nepilogue")


@pytest.mark.parametrize("ctype", ["application/json", None])
def test_append_refuses_a_non_multipart_body(ctype):
    assert append_form_field(b'{"a": 1}', ctype, "language", "he") is None


def test_append_refuses_a_body_with_no_close_delimiter():
    truncated = _upload(model="stt-he")[:-10]
    assert append_form_field(truncated, CT, "language", "he") is None


def test_append_refuses_a_value_that_could_break_framing():
    with pytest.raises(ValueError):
        append_form_field(_upload(), CT, "language", "he\r\n--x")
    with pytest.raises(ValueError):
        append_form_field(_upload(), CT, 'lang"uage', "he")
