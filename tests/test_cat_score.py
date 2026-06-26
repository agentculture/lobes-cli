"""Tests for lobes.bench.cat_score (t5 — logprobs cat scorer).

Covers three acceptance criteria:

* AC1 — headline = softmax over echo logprobs; ≈ 1.0 when all mass is on
  the correct candidate.
* AC2 — first_token_mass is always a float in [0, 1] (even on the echo path).
* AC3 — soft_score ∈ [0, 1] and per_candidate sums to 1.0; headline
  records ``"unavailable"`` and soft_score falls back to first_token_mass
  when echo is unavailable.

Hermetic: all server-based tests use http.server on 127.0.0.1:0 (OS-assigned
port).  No network traffic outside localhost; no GPU required.
"""

from __future__ import annotations

import http.server
import json
import math
import threading
from typing import Any

from lobes.bench.cat_probe import CatCase
from lobes.bench.cat_score import (
    _first_token_mass,
    _sequence_logprob,
    _softmax,
    score_case,
)

# ---------------------------------------------------------------------------
# Fake CatCase with known, short prompt so text_offsets are easy to compute
# ---------------------------------------------------------------------------

_FAKE_CASE = CatCase(
    prompt="x",  # length 1 — continuation tokens start at offset 1
    answer="kitchen",
    candidates=("kitchen", "garden"),
    events=(("A", "kitchen", "12:00"), ("B", "garden", "11:00")),
    mode="closed",
)

# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------

# Probe response for gateway_supports_echo (path: /v1/completions, "Ping pong")
_PROBE_RESP: dict[str, Any] = {
    "choices": [
        {
            "logprobs": {
                "tokens": ["Ping", " pong"],
                "token_logprobs": [-0.3, -0.7],
            }
        }
    ]
}

# Chat response with top_logprobs matching our two candidates
_CHAT_RESP: dict[str, Any] = {
    "choices": [
        {
            "message": {"role": "assistant", "content": " kitchen"},
            "logprobs": {
                "content": [
                    {
                        "token": " kitchen",
                        "logprob": -0.5,
                        "top_logprobs": [
                            {"token": " kitchen", "logprob": -0.5},
                            {"token": " garden", "logprob": -1.5},
                            {"token": " noise", "logprob": -3.0},
                        ],
                    }
                ]
            },
        }
    ]
}

# Expected first_token_mass for "kitchen" given _CHAT_RESP
_EXPECTED_FTM = math.exp(-0.5) / (math.exp(-0.5) + math.exp(-1.5))


def _make_echo_resp(continuation_lp: float) -> dict[str, Any]:
    """Canned echo response for fake_case with the given continuation logprob.

    The fake prompt ``"x"`` has length 1.  We provide ``text_offset=[0, 1]``
    so ``_sequence_logprob`` correctly isolates the continuation token at
    offset 1.
    """
    return {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["[prompt]", "[continuation]"],
                    "token_logprobs": [None, continuation_lp],
                    # text_offset[1] = 1 = len("x") → continuation token
                    "text_offset": [0, 1],
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------


def _make_server(handler_cls: type) -> tuple[http.server.HTTPServer, str]:
    """Create an HTTPServer on an ephemeral port; return (server, base_url)."""
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}/v1"


def _serve_n(server: http.server.HTTPServer, n: int) -> threading.Thread:
    """Serve exactly *n* requests in a daemon thread; return the thread."""

    def _run() -> None:
        for _ in range(n):
            server.handle_request()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Parameterised handler factory for echo-path tests
# ---------------------------------------------------------------------------


def _make_echo_handler(logprobs_map: dict[str, float]) -> type:
    """Build a handler class serving canned echo + chat responses.

    Routes:
    * ``/v1/completions`` (prompt contains "Ping") → ``_PROBE_RESP``
    * ``/v1/completions`` (prompt ends with " <candidate>") → echo response
      with the logprob from *logprobs_map*
    * ``/v1/chat/completions`` → ``_CHAT_RESP``
    """

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            body = json.loads(raw)

            if self.path == "/v1/completions":
                prompt_text = body.get("prompt", "")
                if "Ping" in prompt_text:
                    resp: dict[str, Any] = _PROBE_RESP
                else:
                    resp = None  # type: ignore[assignment]
                    for cand, lp in logprobs_map.items():
                        if prompt_text.endswith(" " + cand):
                            resp = _make_echo_resp(lp)
                            break
                    if resp is None:
                        self.send_response(400)
                        self.end_headers()
                        return
            elif self.path == "/v1/chat/completions":
                resp = _CHAT_RESP
            else:
                self.send_response(404)
                self.end_headers()
                return

            body_bytes = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
            pass  # silence test noise

    return _Handler


# ---------------------------------------------------------------------------
# Fallback handler: 404 for /v1/completions, chat response for /v1/chat/...
# ---------------------------------------------------------------------------


class _FallbackHandler(http.server.BaseHTTPRequestHandler):
    """Simulates a gateway that has no /v1/completions echo route."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain body
        if self.path == "/v1/chat/completions":
            body_bytes = json.dumps(_CHAT_RESP).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)
        else:
            # 404 for /v1/completions — gateway_supports_echo will return False
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        pass


# ===========================================================================
# Pure helper tests (no server, no network)
# ===========================================================================


def test_softmax_empty_returns_empty() -> None:
    """_softmax([]) returns [] — no division-by-zero or IndexError."""
    assert _softmax([]) == []


def test_softmax_all_neg_inf_returns_uniform() -> None:
    """All-inf input returns a finite uniform distribution, never NaN/Inf."""
    probs = _softmax([float("-inf"), float("-inf")])
    assert len(probs) == 2
    assert all(math.isfinite(p) for p in probs), f"non-finite value in {probs}"
    assert abs(sum(probs) - 1.0) < 1e-9, f"sum={sum(probs)}"
    # Uniform → each entry == 0.5
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[1] - 0.5) < 1e-9


def test_softmax_all_mass_on_max() -> None:
    """All probability mass lands on the max logprob when others are ~ -inf."""
    probs = _softmax([0.0, -1e9, -1e9])
    assert abs(probs[0] - 1.0) < 1e-6
    assert probs[1] < 1e-6
    assert probs[2] < 1e-6


def test_softmax_uniform_distribution() -> None:
    """Equal logprobs produce a uniform distribution."""
    probs = _softmax([0.0, 0.0, 0.0])
    for p in probs:
        assert abs(p - 1 / 3) < 1e-9


def test_softmax_sums_to_one() -> None:
    """Softmax over arbitrary values sums to 1.0 within floating-point tolerance."""
    probs = _softmax([-1.0, -2.0, -0.5, -3.5])
    assert abs(sum(probs) - 1.0) < 1e-9


def test_softmax_single_element() -> None:
    """Single-element softmax returns [1.0]."""
    assert _softmax([-5.0]) == [1.0]


def test_sequence_logprob_uses_text_offset() -> None:
    """_sequence_logprob sums only tokens whose text_offset >= len(prefix)."""
    echo_resp = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["prefix_tok", " cont"],
                    "token_logprobs": [-0.3, -0.7],
                    "text_offset": [0, 6],  # " cont" starts at offset 6
                }
            }
        ]
    }
    # prefix = "prefix" (length 6); " cont" at offset 6 is the continuation
    result = _sequence_logprob(echo_resp, "prefix")
    assert abs(result - (-0.7)) < 1e-9


def test_sequence_logprob_skips_prefix_tokens() -> None:
    """Tokens whose text_offset < len(prefix) are excluded from the sum."""
    echo_resp = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["ab", "cd", " ef"],
                    "token_logprobs": [-0.1, -0.2, -0.9],
                    "text_offset": [0, 2, 4],
                }
            }
        ]
    }
    # prefix = "abcd" (length 4): "ab" at 0, "cd" at 2 are prefix; " ef" at 4 is cont
    result = _sequence_logprob(echo_resp, "abcd")
    assert abs(result - (-0.9)) < 1e-9


def test_sequence_logprob_null_treated_as_zero() -> None:
    """None/null token_logprob entries are treated as 0.0."""
    echo_resp = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["x", " y", " z"],
                    "token_logprobs": [None, -0.5, -0.3],
                    "text_offset": [0, 1, 3],
                }
            }
        ]
    }
    # prefix = "x" (length 1); " y" and " z" are continuation tokens
    result = _sequence_logprob(echo_resp, "x")
    assert abs(result - (-0.8)) < 1e-9


def test_sequence_logprob_fallback_no_text_offset() -> None:
    """Falls back to token-length accumulation when text_offset is absent."""
    echo_resp = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["prefix", " cont"],
                    "token_logprobs": [None, -0.5],
                    # no "text_offset" key
                }
            }
        ]
    }
    # "prefix" (6 chars) covers the prefix "prefix" (length 6) exactly
    result = _sequence_logprob(echo_resp, "prefix")
    assert abs(result - (-0.5)) < 1e-9


def test_first_token_mass_renormalized() -> None:
    """_first_token_mass returns mass renormalised over the candidate set."""
    chat_resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": " kitchen",
                            "logprob": -0.5,
                            "top_logprobs": [
                                {"token": " kitchen", "logprob": -0.5},
                                {"token": " garden", "logprob": -1.5},
                                {"token": " noise", "logprob": -3.0},
                            ],
                        }
                    ]
                }
            }
        ]
    }
    mass = _first_token_mass(chat_resp, ("kitchen", "garden"), "kitchen")
    expected = math.exp(-0.5) / (math.exp(-0.5) + math.exp(-1.5))
    assert abs(mass - expected) < 1e-9
    assert 0.0 <= mass <= 1.0


def test_first_token_mass_returns_zero_when_no_match() -> None:
    """Returns 0.0 when no top_logprob token matches any candidate."""
    chat_resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": " xyz",
                            "logprob": -0.5,
                            "top_logprobs": [
                                {"token": " xyz", "logprob": -0.5},
                            ],
                        }
                    ]
                }
            }
        ]
    }
    mass = _first_token_mass(chat_resp, ("kitchen", "garden"), "kitchen")
    assert mass == 0.0


def test_first_token_mass_sums_to_one_over_all_candidates() -> None:
    """Sum of _first_token_mass over all candidates equals 1.0."""
    chat_resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": " kitchen",
                            "logprob": -0.5,
                            "top_logprobs": [
                                {"token": " kitchen", "logprob": -0.5},
                                {"token": " garden", "logprob": -1.5},
                            ],
                        }
                    ]
                }
            }
        ]
    }
    candidates = ("kitchen", "garden")
    total = sum(_first_token_mass(chat_resp, candidates, c) for c in candidates)
    assert abs(total - 1.0) < 1e-9


# ===========================================================================
# AC1: headline soft-score — server-based
# Request budget for a 2-candidate case on the echo path:
#   1 probe (/v1/completions via gateway_supports_echo)
#   1 chat  (/v1/chat/completions)
#   2 echo  (/v1/completions, one per candidate)
#   = 4 requests total
# ===========================================================================


def test_score_case_headline_all_mass_on_answer() -> None:
    """AC1: headline ≈ 1.0 when correct candidate's sequence logprob >> others."""
    handler = _make_echo_handler({"kitchen": -0.01, "garden": -1000.0})
    server, base_url = _make_server(handler)
    t = _serve_n(server, 4)
    try:
        result = score_case(_FAKE_CASE, base_url=base_url, model="m", timeout=10)
        t.join(timeout=5)
    finally:
        server.server_close()

    assert result["echo_available"] is True
    assert isinstance(result["headline"], float)
    assert abs(result["headline"] - 1.0) < 1e-4, f"headline={result['headline']}"


def test_score_case_headline_balanced() -> None:
    """AC1 (complement): equal logprobs for both candidates → headline in (0, 1)."""
    handler = _make_echo_handler({"kitchen": -0.5, "garden": -0.5})
    server, base_url = _make_server(handler)
    t = _serve_n(server, 4)
    try:
        result = score_case(_FAKE_CASE, base_url=base_url, model="m", timeout=10)
        t.join(timeout=5)
    finally:
        server.server_close()

    assert result["echo_available"] is True
    assert 0.0 < result["headline"] < 1.0


# ===========================================================================
# AC2: first_token_mass always present as float in [0, 1]
# ===========================================================================


def test_first_token_mass_present_on_echo_path() -> None:
    """AC2: first_token_mass is a float in [0, 1] even on the echo (non-fallback) path."""
    handler = _make_echo_handler({"kitchen": -0.01, "garden": -1000.0})
    server, base_url = _make_server(handler)
    t = _serve_n(server, 4)
    try:
        result = score_case(_FAKE_CASE, base_url=base_url, model="m", timeout=10)
        t.join(timeout=5)
    finally:
        server.server_close()

    ftm = result["first_token_mass"]
    assert isinstance(ftm, float)
    assert 0.0 <= ftm <= 1.0
    # Confirm value matches expected renormalised mass
    assert abs(ftm - _EXPECTED_FTM) < 1e-9


# ===========================================================================
# AC3: renormalisation on the echo path
# ===========================================================================


def test_per_candidate_sums_to_one_echo_path() -> None:
    """AC3: per_candidate values sum to 1.0 (within 1e-6) on the echo path."""
    handler = _make_echo_handler({"kitchen": -0.3, "garden": -0.9})
    server, base_url = _make_server(handler)
    t = _serve_n(server, 4)
    try:
        result = score_case(_FAKE_CASE, base_url=base_url, model="m", timeout=10)
        t.join(timeout=5)
    finally:
        server.server_close()

    assert abs(sum(result["per_candidate"].values()) - 1.0) < 1e-6


def test_soft_score_in_range_echo_path() -> None:
    """AC3: soft_score ∈ [0, 1] on the echo path."""
    handler = _make_echo_handler({"kitchen": -0.3, "garden": -0.9})
    server, base_url = _make_server(handler)
    t = _serve_n(server, 4)
    try:
        result = score_case(_FAKE_CASE, base_url=base_url, model="m", timeout=10)
        t.join(timeout=5)
    finally:
        server.server_close()

    assert 0.0 <= result["soft_score"] <= 1.0


# ===========================================================================
# AC3: fallback path (echo unavailable)
# Request budget:
#   1 probe (/v1/completions → 404, so gateway_supports_echo returns False)
#   1 chat  (/v1/chat/completions → 200)
#   = 2 requests total
# ===========================================================================


def test_score_case_fallback_on_no_echo() -> None:
    """AC3 fallback: echo_available=False, headline='unavailable', soft_score=first_token_mass."""
    server, base_url = _make_server(_FallbackHandler)
    t = _serve_n(server, 2)
    try:
        result = score_case(_FAKE_CASE, base_url=base_url, model="m", timeout=10)
        t.join(timeout=5)
    finally:
        server.server_close()

    assert result["echo_available"] is False
    assert result["headline"] == "unavailable"
    assert result["soft_score"] == result["first_token_mass"]
    assert 0.0 <= result["soft_score"] <= 1.0


def test_fallback_per_candidate_sums_to_one() -> None:
    """AC3 fallback: per_candidate values sum to 1.0 even when echo is unavailable."""
    server, base_url = _make_server(_FallbackHandler)
    t = _serve_n(server, 2)
    try:
        result = score_case(_FAKE_CASE, base_url=base_url, model="m", timeout=10)
        t.join(timeout=5)
    finally:
        server.server_close()

    assert result["echo_available"] is False
    assert abs(sum(result["per_candidate"].values()) - 1.0) < 1e-6


def test_fallback_first_token_mass_in_range() -> None:
    """AC3 fallback: first_token_mass is a float in [0, 1] on the fallback path."""
    server, base_url = _make_server(_FallbackHandler)
    t = _serve_n(server, 2)
    try:
        result = score_case(_FAKE_CASE, base_url=base_url, model="m", timeout=10)
        t.join(timeout=5)
    finally:
        server.server_close()

    ftm = result["first_token_mass"]
    assert isinstance(ftm, float)
    assert 0.0 <= ftm <= 1.0
    assert abs(ftm - _EXPECTED_FTM) < 1e-9


# ===========================================================================
# Finding A: all-candidate echo logprobs are -inf → must force fallback
# Handler: probe succeeds (echo_available=True) but every candidate echo
# returns 400 → completions_echo raises → lp=−inf for all candidates.
# Request budget:
#   1 probe  (/v1/completions, "Ping")
#   1 chat   (/v1/chat/completions)
#   2 echo   (/v1/completions, one per candidate → 400 → exception → -inf)
#   = 4 requests total
# ===========================================================================


def test_score_case_all_inf_logprobs_forces_fallback() -> None:
    """Finding A: when all candidate echo logprobs are -inf, score_case falls back.

    The echo path is probed successfully (gateway_supports_echo returns True),
    but every completions_echo call for a candidate returns HTTP 400, causing
    all sequence logprobs to be -inf.  score_case must force the fallback path
    rather than emitting NaN headline/soft_score.
    """
    # Empty logprobs_map → handler returns 400 for any non-Ping echo request.
    handler = _make_echo_handler({})
    server, base_url = _make_server(handler)
    t = _serve_n(server, 4)
    try:
        result = score_case(_FAKE_CASE, base_url=base_url, model="m", timeout=10)
        t.join(timeout=5)
    finally:
        server.server_close()

    # Must have forced the fallback path.
    assert result["echo_available"] is False, "expected echo_available=False"
    assert result["headline"] == "unavailable", f"headline={result['headline']!r}"
    assert (
        result["soft_score"] == result["first_token_mass"]
    ), f"soft_score={result['soft_score']} != first_token_mass={result['first_token_mass']}"
    # No value in the result dict may be NaN or Inf.
    numeric_fields = ["first_token_mass", "soft_score"]
    for field in numeric_fields:
        v = result[field]
        assert math.isfinite(v), f"result[{field!r}] = {v} is not finite"
    for cand, mass in result["per_candidate"].items():
        assert math.isfinite(mass), f"per_candidate[{cand!r}] = {mass} is not finite"


# ===========================================================================
# Finding B: whitespace-only token must not inflate every candidate's mass
# ===========================================================================


def test_first_token_mass_whitespace_token_skipped() -> None:
    """Finding B: a whitespace-only top_logprob token is skipped, not broadcast.

    Without the fix, token " " strips to "" and `first_word.startswith("")`
    is always True, so exp(logprob(" ")) is added to every candidate's mass,
    distorting the renormalised distribution.  With the fix, the token is
    skipped; only the real tokens contribute.
    """
    whitespace_lp = -2.0
    chat_resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": " kitchen",
                            "logprob": -0.5,
                            "top_logprobs": [
                                {"token": " kitchen", "logprob": -0.5},
                                {"token": " garden", "logprob": -1.5},
                                # Whitespace-only token — must be ignored.
                                {"token": " ", "logprob": whitespace_lp},
                            ],
                        }
                    ]
                }
            }
        ]
    }
    candidates = ("kitchen", "garden")
    # Expected: whitespace token absent → only kitchen and garden contribute.
    kitchen_expected = math.exp(-0.5) / (math.exp(-0.5) + math.exp(-1.5))

    mass = _first_token_mass(chat_resp, candidates, "kitchen")
    assert abs(mass - kitchen_expected) < 1e-9, (
        f"mass={mass}, expected={kitchen_expected} " f"(whitespace token leaked into distribution)"
    )
    # Cross-check: if the bug were present, the whitespace token would inflate
    # both numerator and denominator — the result would differ from kitchen_expected.
    ws_prob = math.exp(whitespace_lp)
    kitchen_prob = math.exp(-0.5)
    garden_prob = math.exp(-1.5)
    buggy_kitchen = (kitchen_prob + ws_prob) / (kitchen_prob + garden_prob + 2 * ws_prob)
    # Ensure the expected value differs from the buggy value (so the test is meaningful).
    assert (
        abs(kitchen_expected - buggy_kitchen) > 1e-6
    ), "Test setup flaw: expected and buggy values are indistinguishable"
