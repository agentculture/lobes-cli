"""Minimal ``multipart/form-data`` helpers for the stt lanes (stdlib only, pure).

The gateway forwards ``/v1/audio/transcriptions`` bodies verbatim. Two small
needs break that rule, both for the speech-to-text LANES only
(:mod:`lobes.stt_lanes`):

* reading the short text fields ``model`` (the lane selector) and
  ``language`` (to know whether the caller set one);
* appending ONE ``language`` part when the caller sent none.

Neither re-encodes the body: :func:`read_form_fields` only scans it, and
:func:`append_form_field` inserts bytes immediately before the close
delimiter, so every existing byte — the audio file included — reaches the
engine unchanged. File parts (a ``filename`` parameter) are never read as
fields, and a field value is read only when short, so the scan never copies
the upload.

Framing follows RFC 7578 / RFC 2046: a part starts after ``--<boundary>``
plus a line break, and ends at the next ``CRLF--<boundary>``; the close
delimiter is ``--<boundary>--``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

_MULTIPART = "multipart/form-data"
# A form field the gateway reads is a short token (a lane name, a language
# code); anything longer is not one of those, so it is never copied out.
_MAX_FIELD_BYTES = 256
_PARAM_RE = re.compile(r';\s*([A-Za-z0-9_.-]+)\s*=\s*("(?:[^"\\]|\\.)*"|[^;\s]*)')
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:+-]*$")


def _params(header_value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, raw in _PARAM_RE.findall(";" + header_value.split(";", 1)[-1]):
        value = raw[1:-1].replace('\\"', '"') if raw.startswith('"') else raw
        out.setdefault(key.lower(), value)
    return out


def form_boundary(content_type: str | None) -> bytes | None:
    """The boundary of a ``multipart/form-data`` Content-Type, else ``None``."""
    if not content_type:
        return None
    if content_type.split(";", 1)[0].strip().lower() != _MULTIPART:
        return None
    boundary = _params(content_type).get("boundary")
    return boundary.encode("latin-1") if boundary else None


def _iter_parts(body: bytes, boundary: bytes) -> Iterator[tuple[bytes, int, int]]:
    """``(header_bytes, content_start, content_end)`` per part, by index."""
    delim = b"--" + boundary
    pos = body.find(delim)
    while pos >= 0:
        pos += len(delim)
        if body[pos : pos + 2] == b"--":
            return  # the close delimiter
        eol = body.find(b"\r\n", pos)
        if eol < 0:
            return
        start = eol + 2
        end = body.find(b"\r\n" + delim, start)
        if end < 0:
            return  # no next delimiter: truncated, take nothing from it
        head_end = body.find(b"\r\n\r\n", start, end + 2)
        if head_end < 0:
            return
        yield body[start:head_end], min(head_end + 4, end), end
        pos = end + 2


def _disposition(headers: bytes) -> dict[str, str] | None:
    for line in headers.decode("latin-1").split("\r\n"):
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "content-disposition":
            if value.strip().split(";", 1)[0].strip().lower() != "form-data":
                return None
            return _params(value)
    return None


def read_form_fields(body: bytes, content_type: str | None, names: Sequence[str]) -> dict[str, str]:
    """The first value of each short, non-file field in *names* that is present.

    Anything that is not well-formed ``multipart/form-data`` yields ``{}`` —
    the caller then behaves exactly as if no field had been sent.
    """
    boundary = form_boundary(content_type)
    if boundary is None:
        return {}
    wanted = set(names)
    out: dict[str, str] = {}
    for headers, start, end in _iter_parts(body, boundary):
        disp = _disposition(headers)
        name = disp.get("name") if disp else None
        if name not in wanted or name in out or "filename" in disp:
            continue
        if end - start > _MAX_FIELD_BYTES:
            continue
        out[name] = body[start:end].decode("utf-8", "replace").strip()
    return out


def append_form_field(body: bytes, content_type: str | None, name: str, value: str) -> bytes | None:
    """*body* with one ``name=value`` text part appended, or ``None``.

    ``None`` when *body* is not a well-formed multipart body with a close
    delimiter (nothing can be inserted safely; the caller forwards it as is).
    The new part goes immediately before the close delimiter, so the bytes
    before and after it are untouched. *name*/*value* must be plain tokens:
    anything that could forge a delimiter or a header is refused.
    """
    if not _SAFE_NAME_RE.match(name) or not _SAFE_VALUE_RE.match(value):
        raise ValueError("form field name/value must be a plain token")
    boundary = form_boundary(content_type)
    if boundary is None:
        return None
    idx = body.rfind(b"\r\n--" + boundary + b"--")
    if idx < 0 or next(_iter_parts(body, boundary), None) is None:
        return None
    part = (
        b"\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="'
        + name.encode("ascii")
        + b'"\r\n\r\n'
        + value.encode("ascii")
    )
    return body[:idx] + part + body[idx:]
