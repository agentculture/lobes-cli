"""A streamed relay ALWAYS ends the client's stream (issue #220).

The defect this pins: :meth:`lobes.gateway.server._Handler._relay_streaming`
used to loop ``upstream.read`` → ``wfile.write`` with no exception handling, so
any mid-stream failure unwound out of the handler having sent neither a terminal
SSE event nor the zero-length chunk that closes HTTP chunked framing. A client
parked in a blocking read on a still-ESTABLISHED socket had nothing to tell it
apart from a slow model — observed on the DGX Spark 2026-08-27/28 as 17-24
minute hangs against a vLLM reporting ``num_requests_running 0``.

These are pure unit tests over the handler's relay methods: the handler is
constructed with ``__new__`` and handed doubles for the two sockets it touches,
so nothing here binds a port or opens a connection.
"""

from __future__ import annotations

import http.client
import io

import pytest

from lobes.gateway import server as S


class _FakeUpstream:
    """Yields ``chunks``, then either EOF or ``raises``."""

    def __init__(self, chunks: list[bytes], raises: BaseException | None = None) -> None:
        self._chunks = list(chunks)
        self._raises = raises
        self.closed = False

    def read(self, _n: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        if self._raises is not None:
            raise self._raises
        return b""

    def close(self) -> None:
        self.closed = True


class _FakeWFile(io.BytesIO):
    """A client socket that optionally dies after ``fail_after`` writes."""

    def __init__(self, fail_after: int | None = None) -> None:
        super().__init__()
        self._fail_after = fail_after
        self.writes = 0

    def write(self, data: bytes) -> int:  # type: ignore[override]
        self.writes += 1
        if self._fail_after is not None and self.writes > self._fail_after:
            raise BrokenPipeError(32, "Broken pipe")
        return super().write(data)

    def flush(self) -> None:  # BytesIO.flush is a no-op; keep it explicit
        return None


def _handler(wfile: _FakeWFile) -> S._Handler:
    """A handler with only what the relay methods touch — no socket, no server."""
    handler = S._Handler.__new__(S._Handler)
    handler.wfile = wfile
    handler.close_connection = False
    handler.send_response = lambda *_a, **_k: None  # type: ignore[method-assign]
    handler.send_header = lambda *_a, **_k: None  # type: ignore[method-assign]
    handler.end_headers = lambda: None  # type: ignore[method-assign]
    return handler


def _response(upstream: _FakeUpstream, *, attempts: list[str] | None = None) -> S.GatewayResponse:
    return S.GatewayResponse(
        status=200,
        headers=[("Content-Type", "text/event-stream")],
        upstream=upstream,
        streaming=True,
        attempts=attempts if attempts is not None else ["primary"],
    )


# --- the clean path is unchanged -------------------------------------------


def test_clean_stream_relays_chunks_then_the_terminator() -> None:
    wfile = _FakeWFile()
    handler = _handler(wfile)
    handler._relay_streaming(_response(_FakeUpstream([b"data: a\n\n", b"data: [DONE]\n\n"])))
    body = wfile.getvalue()
    assert body == (
        S.frame_chunk(b"data: a\n\n") + S.frame_chunk(b"data: [DONE]\n\n") + S.CHUNK_TERMINATOR
    )
    # A cleanly finished response is still keep-alive eligible.
    assert handler.close_connection is False


# --- the upstream dies mid-stream ------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionResetError(104, "Connection reset by peer"),
        TimeoutError("timed out"),
        http.client.IncompleteRead(b"partial"),
    ],
)
def test_upstream_failure_sends_an_error_frame_then_done_then_the_terminator(
    failure: BaseException, capsys: pytest.CaptureFixture[str]
) -> None:
    wfile = _FakeWFile()
    handler = _handler(wfile)
    handler._relay_streaming(_response(_FakeUpstream([b"data: a\n\n"], raises=failure)))
    body = wfile.getvalue()
    # What the client actually received, in order: the one good chunk, an error
    # event, the [DONE] sentinel, and the end of the chunked body.
    assert body.startswith(S.frame_chunk(b"data: a\n\n"))
    assert body.endswith(S.frame_chunk(S.SSE_DONE) + S.CHUNK_TERMINATOR)
    assert b'"type": "upstream_error"' in body
    assert b"stopped sending mid-stream" in body
    # ...and the socket is NOT handed back to keep-alive.
    assert handler.close_connection is True
    # The upstream status is logged, as the issue asked.
    assert "stream aborted (upstream status 200" in capsys.readouterr().err


def test_upstream_failure_before_any_chunk_still_terminates() -> None:
    """A backend that dies before its first token is the worst case for a
    client: nothing at all has arrived, so silence is indistinguishable from a
    slow prefill."""
    wfile = _FakeWFile()
    handler = _handler(wfile)
    handler._relay_streaming(_response(_FakeUpstream([], raises=ConnectionResetError())))
    assert wfile.getvalue().endswith(S.frame_chunk(S.SSE_DONE) + S.CHUNK_TERMINATOR)


def test_error_frame_is_parseable_openai_shaped_json() -> None:
    import json

    raw = S.sse_error_frame("upstream went away")
    assert raw.startswith(b"data: ") and raw.endswith(b"\n\n")
    payload = json.loads(raw[len(b"data: ") :].decode())
    assert payload["error"]["message"] == "upstream went away"
    assert payload["error"]["type"] == "upstream_error"
    assert payload["error"]["code"] == "upstream_error"


# --- the client hangs up ----------------------------------------------------


def test_client_disconnect_mid_stream_is_logged_and_never_raises(
    capsys: pytest.CaptureFixture[str],
) -> None:
    wfile = _FakeWFile(fail_after=1)
    handler = _handler(wfile)
    upstream = _FakeUpstream([b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"])
    handler._relay_streaming(_response(upstream, attempts=["primary", "thor"]))
    assert handler.close_connection is True
    err = capsys.readouterr().err
    assert "client disconnected" in err
    assert "primary>thor" in err  # the attempt chain is named, for triage


def test_client_disconnect_stops_draining_the_upstream() -> None:
    """Once the client is gone there is nothing to deliver, so the relay must
    not keep pulling a whole turn out of the backend into a dead socket."""
    wfile = _FakeWFile(fail_after=1)
    upstream = _FakeUpstream([b"a", b"b", b"c", b"d"])
    _handler(wfile)._relay_streaming(_response(upstream))
    # One chunk written, one more read attempted at most — not all four drained.
    assert len(upstream._chunks) >= 2


def test_client_gone_before_the_terminator_is_survivable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The upstream finished cleanly but the client vanished in the gap before
    the terminator — there is nothing left to deliver and nothing to fix."""
    wfile = _FakeWFile(fail_after=1)
    handler = _handler(wfile)
    handler._relay_streaming(_response(_FakeUpstream([b"data: a\n\n"])))
    assert handler.close_connection is True
    assert "client gone before terminator" in capsys.readouterr().err
