"""Tests for lobes.bench.cat_probe — the timestamped 'Where is the cat?' generator.

The load-bearing invariant under test: every generated case has EXACTLY ONE
current location, entailed *purely* by the event timestamps. We prove it with an
INDEPENDENT, test-owned deterministic solver (re-derived here, never imported
from the module under test) over the case's exposed ``events`` — the test must
agree with the module's declared ``answer``, and no second location may tie the
latest timestamp.
"""

from __future__ import annotations

import pytest

from lobes.bench.cat_probe import CatCase, generate_case

_CUE = "And the cat is at:"


# ---------------------------------------------------------------------------
# Independent solver (deliberately NOT importing the module's own logic)
# ---------------------------------------------------------------------------


def _minutes(ts: str) -> int:
    """Parse an ``HH:MM`` 24-hour stamp into minutes since midnight."""
    hh, mm = ts.split(":")
    return int(hh) * 60 + int(mm)


def _solve(events: tuple[tuple[str, str, str], ...]) -> tuple[str, int]:
    """Re-derive the current location: the location of the latest-timestamp event.

    Returns ``(answer_location, n_holders)`` where ``n_holders`` is how many
    events hold the maximum timestamp. The unambiguity invariant requires
    ``n_holders == 1``.
    """
    max_min = max(_minutes(ts) for _, _, ts in events)
    holders = [loc for _, loc, ts in events if _minutes(ts) == max_min]
    return holders[0], len(holders)


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — exactly one current location, proven by the solver
# ---------------------------------------------------------------------------


def test_single_case_unambiguous_via_independent_solver() -> None:
    case = generate_case(seed=0)
    answer, n_holders = _solve(case.events)
    assert n_holders == 1, "the latest timestamp must be held by exactly one event"
    assert answer == case.answer


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_invariant_holds_for_many_seeds(mode: str) -> None:
    """Loop 50 seeds in both modes: the unambiguity invariant must never break."""
    for seed in range(50):
        case = generate_case(seed=seed, mode=mode)

        # Distinct timestamps by construction => the maximum is unique.
        stamps = [ts for _, _, ts in case.events]
        assert len(set(stamps)) == len(stamps), f"tie in timestamps at seed={seed}"

        answer, n_holders = _solve(case.events)
        assert n_holders == 1, f"latest timestamp not unique at seed={seed}"
        assert answer == case.answer, f"solver disagrees at seed={seed}"

        assert case.answer in case.candidates
        assert len(case.candidates) >= 2


# ---------------------------------------------------------------------------
# Determinism — same seed reproduces the identical case
# ---------------------------------------------------------------------------


def test_determinism_same_seed_same_case() -> None:
    a = generate_case(seed=7)
    b = generate_case(seed=7)
    assert a == b
    assert a.prompt == b.prompt
    assert a.answer == b.answer
    assert a.candidates == b.candidates
    assert a.events == b.events


def test_different_seeds_generally_differ() -> None:
    # Not strictly guaranteed for every pair, but the prompts across a spread of
    # seeds must not collapse to a single constant string.
    prompts = {generate_case(seed=s).prompt for s in range(20)}
    assert len(prompts) > 1


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — closed lists candidates, open omits them
# ---------------------------------------------------------------------------


def test_closed_mode_lists_options_open_mode_omits() -> None:
    closed = generate_case(seed=3, mode="closed")
    assert "Options:" in closed.prompt
    for loc in closed.candidates:
        assert loc in closed.prompt

    opens = generate_case(seed=3, mode="open")
    assert "Options:" not in opens.prompt
    # Even without the explicit options list, every candidate is recoverable.
    for loc in opens.candidates:
        assert loc in opens.prompt


def test_open_and_closed_share_world_differ_only_in_options() -> None:
    opens = generate_case(seed=11, mode="open")
    closed = generate_case(seed=11, mode="closed")
    assert opens.events == closed.events
    assert opens.answer == closed.answer
    assert opens.candidates == closed.candidates
    assert opens.prompt != closed.prompt


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — candidate set exposed AND recoverable from text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_every_candidate_appears_verbatim_in_narrative(mode: str) -> None:
    for seed in range(25):
        case = generate_case(seed=seed, mode=mode)
        for loc in case.candidates:
            assert loc in case.prompt, f"{loc!r} not recoverable at seed={seed}"


def test_candidates_exposed_and_distinct() -> None:
    case = generate_case(seed=21, n_characters=5)
    assert len(case.candidates) == 5
    assert len(set(case.candidates)) == len(case.candidates)
    assert case.mode == "closed"


# ---------------------------------------------------------------------------
# Prompt shape and type contract
# ---------------------------------------------------------------------------


def test_prompt_ends_with_cue() -> None:
    for seed in range(10):
        for mode in ("open", "closed"):
            case = generate_case(seed=seed, mode=mode)
            assert case.prompt.endswith(_CUE)


def test_is_catcase_instance() -> None:
    case = generate_case(seed=1)
    assert isinstance(case, CatCase)
    assert case.mode in ("open", "closed")
    assert isinstance(case.candidates, tuple)
    assert isinstance(case.events, tuple)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        generate_case(seed=0, mode="sideways")
    with pytest.raises(ValueError):
        generate_case(seed=0, n_characters=1)
    with pytest.raises(ValueError):
        generate_case(seed=0, n_characters=9999)
