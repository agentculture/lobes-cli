"""How a failed generate call surfaces to the client (hebrew-realtime t8, c3).

The acceptance criterion this file proves (or, where the code does not yet
hold it, names as a gap): a generate failure for `model=<any>` — a gateway
404 `role_infeasible` (with `hosted_by`), a 429 pressure shed, or a 503
`role_unverified` (mesh boot window, also carrying `hosted_by`) — surfaces to
the WebSocket client as a single named `generate_failed` error event,
carrying enough of the gateway's own error body to be actionable, and NEVER
as if the reply text came back from a different, substituted model. This
module owns no production code — every name it imports belongs to
:mod:`lobes.realtime._turn` and :mod:`lobes.realtime._conversation`, task t6's
(already-merged) convergence modules — this file only writes NEW tests
proving what that code already does, and marks the two shapes it does not yet
carry through as ``xfail(strict=True)`` rather than papering over them.

Three response shapes, modeled on the gateway's own OpenAI-shaped error
bodies (`lobes/gateway/server.py`):

* ``_role_infeasible_body`` — 404, ``code: "role_infeasible"``, optional
  ``hosted_by``.
* ``_busy_body`` — 429, ``code: "busy"``, ``type: "server_busy"``, NO
  ``hosted_by`` (a pressure shed names no peer).
* ``_role_unverified_body`` — 503, ``code: "role_unverified"``, a
  ``hosted_by`` naming the pending mesh member (the boot-window "not yet",
  distinct from the 404's terminal "never").

GAPS FOUND by t8 — BOTH CLOSED by hebrew-realtime t7, whose brief owned the
two files these needed (``_conversation.py``, plus a minimal additive change
to ``_turn.py``). The descriptions below are kept for the record; the two
tests that named them are no longer ``xfail``:

1. **429 status is not carried through the message text.** ``_busy_body``
   supplies its own ``error.message`` ("<lane> is under pressure; retry
   shortly"), so :func:`_turn._raise_for_error_status`'s HTTP-status fallback
   text never fires and the numeric ``429``/the ``Retry-After`` seconds are
   nowhere in the resulting :class:`~lobes.realtime._session.ErrorEvent`.
   Even the bare status code, when a body carries no message at all, is
   accessible only via ``TurnResponseError.status_code`` — a Python
   attribute the exception chain does not thread into
   ``describe_failure``'s message text, and ``on_generate_response``'s
   signature (``status_code: int, body: bytes``) has no ``headers``
   parameter at all, so a ``Retry-After`` header could not reach this layer
   even if a caller wanted to forward it. See
   ``test_429_busy_status_and_retry_after_are_not_surfaced`` (xfail).
2. **503 role_unverified's `hosted_by` is not a STRUCTURED field on the
   raised exception.** :func:`_turn._raise_for_error_status` special-cases
   reading the error object's ``hosted_by`` key ONLY for the ``(404,
   role_infeasible)`` shape — a 503 whose ``code`` is ``role_unverified``
   (also carrying its own ``hosted_by``, per
   ``lobes/gateway/server.py``'s ``_role_unverified_body``) falls through to
   the generic :class:`~lobes.realtime._turn.TurnResponseError`, which has
   no ``hosted_by`` attribute at all and is not even distinguishable BY TYPE
   from an ordinary 5xx or a terminal ``role_infeasible`` 404's sibling
   failures — a caller can only recover the pending peer's origin by
   pattern-matching the gateway's own message TEXT (which happens to name
   it), never by reading a field. See
   ``test_503_role_unverified_exception_has_no_structured_hosted_by``
   (xfail).

Both gaps were closed by t7, in exactly the two shapes sketched here:
generalizing :func:`_turn._raise_for_error_status`'s hosted_by
extraction to also cover ``role_unverified`` (and minting a distinct
exception type for it, mirroring ``RoleInfeasibleError``), and by adding a
``headers`` parameter to
:func:`~lobes.realtime._conversation.ConversationBridge.on_generate_response`
so 429's ``Retry-After`` has somewhere to go. t7 chose the plainer half of
the first: ``TurnResponseError`` gained a ``hosted_by`` attribute rather than
a new exception TYPE, because ``role_unverified`` means "not yet" — a
transient failure a caller handles like any other — while
``role_infeasible``'s terminal "never" is the one that earns its own type.
"""

from __future__ import annotations

import json

import pytest

import lobes.realtime._conversation as C
import lobes.realtime._floor as F
import lobes.realtime._session as S
import lobes.realtime._turn as T

# ---------------------------------------------------------------------------
# fixtures — a minimal bridge driven to a pending generate call, mirroring
# tests/test_realtime_conversation.py's own make_bridge/commit_turn helpers
# (duplicated narrowly here rather than imported, since that module is
# owned by task t6 and this file must not create a cross-task import
# dependency on its private test helpers).
# ---------------------------------------------------------------------------


def _make_bridge(model: str = "multimodal"):
    config = S.parse_session_config({})
    session, _created = S.Session.create(config)
    return C.ConversationBridge(
        session,
        cancel_generate=lambda: None,
        cancel_tts=lambda: None,
        generate=C.GenerateConfig(base_url="http://gateway:8000", model=model),
        chunk_bytes=480,
    )


def _commit_and_take_turn(bridge) -> int:
    bridge.arm()
    bridge.on_speech_started(at_ms=100)
    bridge.on_speech_stopped(at_ms=2000, reason="silence")
    bridge.on_transcript("what is on the machine right now")
    turn_id = bridge.take_pending_response()
    assert turn_id is not None
    assert bridge.build_generate_request(turn_id) is not None
    return turn_id


def _role_infeasible_body(*, hosted_by: str | None) -> bytes:
    error = {
        "message": "The model `multimodal` is not feasible on this machine.",
        "type": "role_infeasible",
        "code": "role_infeasible",
    }
    if hosted_by:
        error["hosted_by"] = hosted_by
        error["message"] += f" It is hosted by the peer at `{hosted_by}`."
    return json.dumps({"error": error}).encode("utf-8")


def _busy_body(model: str = "multimodal") -> bytes:
    # Mirrors lobes/gateway/server.py's _busy_body exactly — no hosted_by key
    # at all, a pressure shed names no peer.
    return json.dumps(
        {
            "error": {
                "message": f"{model} is under pressure; retry shortly",
                "type": "server_busy",
                "code": "busy",
            }
        }
    ).encode("utf-8")


def _role_unverified_body(*, hosted_by: str) -> bytes:
    # Mirrors lobes/gateway/server.py's _role_unverified_body exactly.
    return json.dumps(
        {
            "error": {
                "message": (
                    "The model `multimodal` is not served on this machine yet — "
                    f"announced by the mesh member at `{hosted_by}`, unverified. "
                    "Retry shortly."
                ),
                "type": "role_unverified",
                "code": "role_unverified",
                "hosted_by": hosted_by,
            }
        }
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# 404 role_infeasible — HOLDS today, proven end to end through the bridge.
# ---------------------------------------------------------------------------


def test_404_role_infeasible_surfaces_as_generate_failed_carrying_hosted_by():
    bridge = _make_bridge(model="multimodal")
    turn_id = _commit_and_take_turn(bridge)

    bridge.on_generate_response(
        404, _role_infeasible_body(hosted_by="http://thor:8000"), turn_id=turn_id
    )

    events = bridge.drain()
    error = events[-1]
    assert error["code"] is S.ErrorCode.GENERATE_FAILED
    assert "http://thor:8000" in str(error["message"])
    # Never a fallback to a placeholder/other-model reply: no synthesis was
    # queued, and the floor released back to listening rather than speaking.
    assert bridge.take_pending_synthesis() is None
    assert bridge.floor.state is F.FloorState.LISTENING


def test_404_role_infeasible_without_hosted_by_still_names_the_failure_only():
    # No peer declared: no hosted_by to carry, but still a clean named
    # failure, never text presented as though a reply arrived.
    bridge = _make_bridge()
    turn_id = _commit_and_take_turn(bridge)

    bridge.on_generate_response(404, _role_infeasible_body(hosted_by=None), turn_id=turn_id)

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.GENERATE_FAILED
    assert "hosted_by" not in str(error["message"])
    assert bridge.take_pending_synthesis() is None


def test_parse_turn_response_role_infeasible_exception_carries_hosted_by_and_status_shape():
    # Unit-level proof at _turn.py's own boundary: RoleInfeasibleError is a
    # DISTINCT, named exception (never TurnResponseError, never a bare
    # string) and it is what carries hosted_by — status is implicit in the
    # exception TYPE for this one shape (RoleInfeasibleError only ever means
    # 404), unlike the 429/503 gaps below where status has nowhere to go.
    with pytest.raises(T.RoleInfeasibleError) as excinfo:
        T.parse_turn_response(404, _role_infeasible_body(hosted_by="http://thor:8000"))
    assert excinfo.value.hosted_by == "http://thor:8000"


# ---------------------------------------------------------------------------
# 429 busy (pressure shed) — HOLDS as a named generate_failed, but the
# status code / Retry-After are NOT carried through. Gap 1 above.
# ---------------------------------------------------------------------------


def test_429_busy_surfaces_as_a_named_generate_failed_never_reply_text():
    bridge = _make_bridge()
    turn_id = _commit_and_take_turn(bridge)

    bridge.on_generate_response(429, _busy_body(), turn_id=turn_id)

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.GENERATE_FAILED
    assert "under pressure" in str(error["message"])
    assert bridge.take_pending_synthesis() is None
    assert bridge.floor.state is F.FloorState.LISTENING


def test_429_busy_status_and_retry_after_are_surfaced():
    # CLOSED by hebrew-realtime t7: on_generate_response now appends a
    # machine-readable `(status=..., retry_after=...)` detail, and takes an
    # optional `headers` mapping so a real Retry-After can travel. With no
    # header the retryable status still says so explicitly
    # (retry_after=unspecified) rather than staying silent.
    bridge = _make_bridge()
    turn_id = _commit_and_take_turn(bridge)

    bridge.on_generate_response(429, _busy_body(), turn_id=turn_id)

    error = bridge.drain()[-1]
    message = str(error["message"])
    assert "429" in message
    assert "retry_after" in message or "Retry-After" in message


# ---------------------------------------------------------------------------
# 503 role_unverified (mesh boot window) — HOLDS as a named generate_failed,
# but hosted_by is silently dropped. Gap 2 above.
# ---------------------------------------------------------------------------


def test_503_role_unverified_surfaces_as_a_named_generate_failed_never_reply_text():
    bridge = _make_bridge()
    turn_id = _commit_and_take_turn(bridge)

    bridge.on_generate_response(
        503, _role_unverified_body(hosted_by="http://spark:8000"), turn_id=turn_id
    )

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.GENERATE_FAILED
    assert bridge.take_pending_synthesis() is None
    assert bridge.floor.state is F.FloorState.LISTENING


def test_503_role_unverified_message_text_still_names_the_pending_peer():
    # HOLDS today, but only INCIDENTALLY: the gateway's own
    # _role_unverified_body embeds the pending origin in its `message`
    # text ("...announced by the mesh member at `<origin>`..."), and
    # _turn's generic TurnResponseError branch relays that message
    # unchanged — so the origin reaches the client as free text, not
    # because _turn.py extracted a structured hosted_by. See the next test
    # for the gap that incidental-ness leaves behind.
    bridge = _make_bridge()
    turn_id = _commit_and_take_turn(bridge)

    bridge.on_generate_response(
        503, _role_unverified_body(hosted_by="http://spark:8000"), turn_id=turn_id
    )

    error = bridge.drain()[-1]
    assert "http://spark:8000" in str(error["message"])


def test_503_role_unverified_exception_has_a_structured_hosted_by():
    # CLOSED by hebrew-realtime t7: TurnResponseError carries `hosted_by`
    # whenever the gateway's error body declared one, so a 503's pending peer
    # is a field rather than something to grep out of English. It stays the
    # generic type on purpose — role_unverified means "not yet", which a
    # caller handles like any other transient failure.
    with pytest.raises(T.TurnResponseError) as excinfo:
        T.parse_turn_response(503, _role_unverified_body(hosted_by="http://spark:8000"))
    assert not isinstance(excinfo.value, T.RoleInfeasibleError)
    assert hasattr(excinfo.value, "hosted_by")
    assert excinfo.value.hosted_by == "http://spark:8000"
