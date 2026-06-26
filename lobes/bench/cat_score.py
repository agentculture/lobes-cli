"""Logprobs cat scorer — echo softmax headline + first-token cross-check + fallback.

Implements three pure helpers (fully exercisable without a network) and one
orchestrating function that drives the network calls:

Public API
----------
score_case(case, *, base_url, model, top_logprobs=20, timeout=60) -> dict
    Score one CatCase.  Returns ``headline`` (softmax over per-candidate echo
    logprobs, or ``"unavailable"``), a ``first_token_mass`` cross-check, and the
    renormalised ``per_candidate`` distribution.

Pure helpers (no network)
-----------
_softmax(logprobs) -> list[float]
_sequence_logprob(echo_resp, prefix) -> float
_first_token_mass(chat_resp, candidates, answer) -> float
"""

from __future__ import annotations

import math

from lobes.bench.cat_probe import CatCase
from lobes.minor import chat_completion
from lobes.minor._client import completions_echo, gateway_supports_echo

__all__ = [
    "_softmax",
    "_sequence_logprob",
    "_first_token_mass",
    "score_case",
]


def _softmax(logprobs: list[float]) -> list[float]:
    """Numerically stable softmax over log-probabilities.

    Subtracts the maximum before exponentiation so very-negative entries
    (e.g. ``-1e9``) contribute ~0 probability without overflow.

    Degenerate cases are handled safely:
    * Empty input → ``[]``.
    * All entries ``-inf`` (or max not finite) → uniform distribution ``[1/n]*n``.
    * ``total`` is zero or non-finite after exponentiation → uniform distribution.

    Args:
        logprobs: List of log-probabilities (may contain very negative floats).

    Returns:
        A probability distribution whose elements sum to 1.0.  Never contains
        NaN or Inf.
    """
    if not logprobs:
        return []
    n = len(logprobs)
    max_lp = max(logprobs)
    if not math.isfinite(max_lp):
        # All entries are -inf (or the max is otherwise non-finite).
        # Return a uniform distribution so callers always get finite values.
        return [1.0 / n] * n
    exps = [math.exp(lp - max_lp) for lp in logprobs]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / n] * n
    return [e / total for e in exps]


def _sequence_logprob(echo_resp: dict, prefix: str) -> float:
    """Sum ``token_logprobs`` for continuation tokens only.

    Continuation tokens are those whose ``text_offset`` is >=
    ``len(prefix)``.  When ``text_offset`` is absent the function falls back
    to accumulating token character lengths until the prefix boundary is
    reached and then summing the remaining entries.

    Treats ``None`` / ``null`` token_logprob entries as ``0.0``.

    Args:
        echo_resp: Full response dict from :func:`~lobes.minor.completions_echo`.
        prefix: The prompt prefix (not the continuation) whose length sets the
            offset cutoff.

    Returns:
        Sum of log-probabilities over the continuation tokens.
    """
    logprobs_block = echo_resp["choices"][0]["logprobs"]
    token_logprobs: list = logprobs_block["token_logprobs"]
    tokens: list[str] = logprobs_block["tokens"]
    text_offset: list[int] | None = logprobs_block.get("text_offset")
    prefix_len = len(prefix)

    if text_offset is not None:
        # Primary path: text_offset identifies continuation tokens directly.
        total = 0.0
        for offset, lp in zip(text_offset, token_logprobs):
            if offset >= prefix_len:
                total += lp if lp is not None else 0.0
        return total

    # Fallback: walk token lengths to find the prefix/continuation boundary.
    pos = 0
    prefix_token_count = 0
    for tok in tokens:
        if pos < prefix_len:
            prefix_token_count += 1
            pos += len(tok)
        else:
            break
    return sum(lp for lp in token_logprobs[prefix_token_count:] if lp is not None)


def _first_token_mass(
    chat_resp: dict,
    candidates: tuple[str, ...],
    answer: str,
) -> float:
    """Renormalised first-token probability mass on *answer*.

    Reads the first generated token's ``top_logprobs`` from a chat-completions
    response and maps each candidate to a probability by matching top tokens
    whose stripped, lower-cased text equals or is a leading chunk of the
    candidate's first word.  The result is renormalised over the candidate set.

    Args:
        chat_resp: Full response from :func:`~lobes.minor.chat_completion` with
            ``logprobs=True``.
        candidates: The full candidate set (all locations).
        answer: The candidate whose mass to return.

    Returns:
        Renormalised share of first-token probability mass on *answer* in
        ``[0, 1]``.  Returns ``0.0`` when no candidate matched any top token.
    """
    top_lp_list = chat_resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    # Build a normalised token → probability map.
    # Skip whitespace-only tokens: after strip() they become "" and would
    # match every candidate via `first_word.startswith("")`, inflating all
    # candidate masses and distorting the renormalised distribution.
    token_probs: dict[str, float] = {}
    for entry in top_lp_list:
        tok = entry["token"].strip().lower()
        if not tok:
            continue
        prob = math.exp(entry["logprob"])
        token_probs[tok] = token_probs.get(tok, 0.0) + prob

    def _candidate_mass(candidate: str) -> float:
        """Sum probabilities for tokens matching the candidate's first word."""
        first_word = candidate.strip().split()[0].lower()
        total = 0.0
        for tok, prob in token_probs.items():
            # Match: token equals the first word, or the first word starts
            # with the token (handles sub-word chunks of the first word).
            if tok == first_word or first_word.startswith(tok):
                total += prob
        return total

    candidate_masses = {c: _candidate_mass(c) for c in candidates}
    denom = sum(candidate_masses.values())
    if denom <= 0.0:
        return 0.0
    return candidate_masses.get(answer, 0.0) / denom


def score_case(
    case: CatCase,
    *,
    base_url: str,
    model: str,
    top_logprobs: int = 20,
    timeout: int = 60,
) -> dict:
    """Score one :class:`~lobes.bench.cat_probe.CatCase` via logprobs.

    **Echo path** (when ``/v1/completions`` is available): calls
    :func:`~lobes.minor.completions_echo` for each candidate, computes a
    per-candidate sequence log-probability, and softmax-normalises the result.
    ``headline`` is the softmax mass on ``case.answer``.

    **Fallback path** (echo unavailable): ``headline = "unavailable"``,
    ``soft_score = first_token_mass``, ``per_candidate`` is the first-token
    distribution.

    The chat first-token cross-check is always computed via
    :func:`~lobes.minor.chat_completion` with ``logprobs=True``.

    Args:
        case: A :class:`~lobes.bench.cat_probe.CatCase` to score.
        base_url: OpenAI-compatible base URL (e.g. ``"http://localhost:8000/v1"``).
        model: Model identifier to pass in requests.
        top_logprobs: How many top logprobs to request in the chat cross-check.
        timeout: Socket timeout in seconds.

    Returns:
        A dict with keys:

        * ``"answer"`` — the correct candidate string.
        * ``"echo_available"`` — whether echo scoring was used.
        * ``"headline"`` — ``float`` (softmax mass on answer) or
          ``"unavailable"``.
        * ``"first_token_mass"`` — ``float`` in ``[0, 1]``, chat cross-check.
        * ``"soft_score"`` — ``float``; equals ``headline`` when echo is
          available, else ``first_token_mass``.
        * ``"per_candidate"`` — ``dict[str, float]`` mapping each candidate to
          its probability in the headline distribution (or first-token
          distribution on fallback).
    """
    echo_available = gateway_supports_echo(base_url=base_url, model=model, timeout=timeout)

    # Always compute the chat first-token cross-check.
    chat_resp = chat_completion(
        case.prompt,
        base_url=base_url,
        model=model,
        logprobs=True,
        top_logprobs=top_logprobs,
        max_tokens=1,
        temperature=0,
        timeout=timeout,
    )
    first_token_mass = _first_token_mass(chat_resp, case.candidates, case.answer)

    if echo_available:
        # Score each candidate via full-sequence echo logprobs.
        candidate_seq_logprobs: list[float] = []
        for candidate in case.candidates:
            try:
                resp = completions_echo(
                    case.prompt,
                    " " + candidate,
                    base_url=base_url,
                    model=model,
                    timeout=timeout,
                )
                lp = _sequence_logprob(resp, case.prompt)
            except Exception:
                lp = float("-inf")
            candidate_seq_logprobs.append(lp)

        # If every candidate's logprob is non-finite (-inf), softmax would
        # return a uniform distribution but the headline would be meaningless.
        # Force the fallback path so callers get a semantically valid result.
        if all(not math.isfinite(lp) for lp in candidate_seq_logprobs):
            echo_available = False

    if echo_available:
        distribution = _softmax(candidate_seq_logprobs)
        per_candidate = dict(zip(case.candidates, distribution))
        answer_idx = list(case.candidates).index(case.answer)
        headline: float | str = distribution[answer_idx]
        soft_score: float = float(headline)
    else:
        # Fallback: headline is unavailable; use first-token distribution.
        headline = "unavailable"
        soft_score = first_token_mass
        per_candidate = {
            c: _first_token_mass(chat_resp, case.candidates, c) for c in case.candidates
        }

    return {
        "answer": case.answer,
        "echo_available": echo_available,
        "headline": headline,
        "first_token_mass": first_token_mass,
        "soft_score": soft_score,
        "per_candidate": per_candidate,
    }
