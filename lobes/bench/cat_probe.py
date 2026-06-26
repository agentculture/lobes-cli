"""Timestamped 'Where is the cat?' probe generator (stdlib-only, deterministic).

A cat-probe is a tiny temporal-reasoning case: several characters each report
*when* (a timestamp) and *where* (a location) they last saw the family cat, and
an investigating kid asks where the cat is **right now**. The puzzle has a single
provably-correct answer: the cat is wherever the *latest* report places it.

The generator guarantees the answer is unambiguous **by construction**:

* every event gets a *distinct* ``HH:MM`` timestamp (sampled without
  replacement), so the maximum timestamp is held by exactly one event;
* every character gets a *distinct* location, so the candidate set is the set of
  introduced locations and the latest event's location is unique among them.

The result is therefore decidable by a trivial independent solver
(``location of max-timestamp event``) — which is exactly how the test suite
re-derives and checks the answer.

Two prompt modes:

* ``"closed"`` — the prompt enumerates the candidate locations in an
  ``Options:`` list (a multiple-choice probe).
* ``"open"`` — no options list; every candidate still appears verbatim inside a
  character's report sentence, so an open-mode scorer can scan them out of the
  narrative text.

Nothing here touches the network, the clock, or the filesystem. All randomness
flows through a seeded :class:`random.Random`, so ``generate_case(seed=N)`` is
fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

__all__ = ["CatCase", "generate_case"]


# ---------------------------------------------------------------------------
# Sampling pools (all entries verbatim-stable; locations are short noun phrases)
# ---------------------------------------------------------------------------

_CHARACTERS: tuple[str, ...] = (
    "Mia",
    "Leo",
    "Ava",
    "Noah",
    "Zoe",
    "Eli",
    "Ivy",
    "Max",
    "Nora",
    "Theo",
    "Lily",
    "Owen",
)

_INVESTIGATORS: tuple[str, ...] = (
    "Detective Sam",
    "Inspector Remy",
    "Sleuth Nina",
    "Captain Bo",
)

_LOCATIONS: tuple[str, ...] = (
    "the kitchen",
    "the garden",
    "the attic",
    "the garage",
    "the living room",
    "the basement",
    "the back porch",
    "the laundry room",
    "the garden shed",
    "the tool shed",
    "the front hall",
    "the sunroom",
    "the pantry",
    "the hallway closet",
)

# A pool of distinct, zero-padded 24-hour stamps. Zero-padding means lexical and
# chronological ordering coincide, but the generator compares by minutes anyway.
_TIMES: tuple[str, ...] = (
    "07:05",
    "07:50",
    "08:20",
    "09:15",
    "09:40",
    "10:25",
    "11:10",
    "12:35",
    "13:05",
    "13:45",
    "14:20",
    "15:10",
    "15:55",
    "16:30",
    "17:15",
    "18:00",
)

# The largest n_characters the pools can satisfy without replacement.
_MAX_CHARACTERS = min(len(_CHARACTERS), len(_LOCATIONS), len(_TIMES))

_CUE = "And the cat is at:"


# ---------------------------------------------------------------------------
# Case dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatCase:
    """One generated cat-probe.

    Attributes:
        prompt: The full narrative the model reads. It always ends with the
            exact cue line ``And the cat is at:``. In ``closed`` mode an
            ``Options:`` line listing :attr:`candidates` appears just before the
            closing question.
        answer: The single current location — the location of the event with the
            latest timestamp.
        candidates: The full set of distinct locations introduced, in narrative
            order. ``answer`` is always a member and ``len(candidates) >= 2``.
        events: The ``(character, location, timestamp)`` triples in narrative
            order, so an independent solver can re-derive the answer.
        mode: ``"open"`` or ``"closed"``.
    """

    prompt: str
    answer: str
    candidates: tuple[str, ...]
    events: tuple[tuple[str, str, str], ...]
    mode: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minutes(timestamp: str) -> int:
    """Convert an ``HH:MM`` 24-hour stamp into minutes since midnight."""
    hours, mins = timestamp.split(":")
    return int(hours) * 60 + int(mins)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_case(*, seed: int, mode: str = "closed", n_characters: int = 4) -> CatCase:
    """Generate one deterministic cat-probe case.

    Args:
        seed: Seeds a private :class:`random.Random`; identical ``seed`` (with the
            same ``mode`` and ``n_characters``) yields an identical case.
        mode: ``"closed"`` to enumerate the candidate locations in the prompt, or
            ``"open"`` to omit the options list (candidates stay recoverable from
            the narrative text).
        n_characters: How many reporters (hence events and distinct candidate
            locations) to include. Must be in ``2..{}`` so the candidate set has
            at least two members and the pools can supply distinct draws.

    Returns:
        A :class:`CatCase` whose answer is unambiguous by construction.

    Raises:
        ValueError: If ``mode`` is not ``"open"``/``"closed"`` or ``n_characters``
            is outside the valid range.
    """
    if mode not in ("open", "closed"):
        raise ValueError(f"mode must be 'open' or 'closed', got {mode!r}")
    if n_characters < 2:
        raise ValueError(
            f"n_characters must be >= 2 (need >= 2 candidate locations), got {n_characters}"
        )
    if n_characters > _MAX_CHARACTERS:
        raise ValueError(
            f"n_characters must be <= {_MAX_CHARACTERS} (pool limit), got {n_characters}"
        )

    rng = Random(seed)

    # Distinct draws without replacement: distinct names, distinct locations, and
    # — crucially — distinct timestamps, which makes the latest event unique.
    names = rng.sample(_CHARACTERS, n_characters)
    locations = rng.sample(_LOCATIONS, n_characters)
    times = rng.sample(_TIMES, n_characters)
    investigator = rng.choice(_INVESTIGATORS)

    # Pair them, then shuffle presentation order so the latest report is not
    # trivially the last line — the reader must actually compare timestamps.
    events = list(zip(names, locations, times))
    rng.shuffle(events)
    events_tuple: tuple[tuple[str, str, str], ...] = tuple(events)

    # The answer is the location of the latest-timestamp event. Distinct
    # timestamps guarantee this max is held by exactly one event.
    answer = max(events_tuple, key=lambda ev: _minutes(ev[2]))[1]

    # Each character owns a distinct location, so candidates == the introduced
    # locations, in narrative order, with no duplicates.
    candidates = tuple(loc for _, loc, _ in events_tuple)

    prompt = _render_prompt(
        investigator=investigator,
        events=events_tuple,
        candidates=candidates,
        n_characters=n_characters,
        mode=mode,
    )

    return CatCase(
        prompt=prompt,
        answer=answer,
        candidates=candidates,
        events=events_tuple,
        mode=mode,
    )


# Inject the live pool ceiling into the docstring so it never drifts.
generate_case.__doc__ = generate_case.__doc__.format(_MAX_CHARACTERS)  # type: ignore[union-attr]


def _render_prompt(
    *,
    investigator: str,
    events: tuple[tuple[str, str, str], ...],
    candidates: tuple[str, ...],
    n_characters: int,
    mode: str,
) -> str:
    """Assemble the narrative; the final line is always the exact cue."""
    lines: list[str] = [
        f"{investigator} wants to know where the family cat is right now.",
        f"{n_characters} people each remember the last time they saw the cat:",
        "",
    ]
    for char, loc, timestamp in events:
        # The location appears verbatim here, so open-mode scorers can recover it.
        lines.append(f"At {timestamp}, {char} spotted the cat in {loc}.")
    lines.append("")
    lines.append("Whoever saw the cat most recently knows where it is now.")
    if mode == "closed":
        lines.append("Options: " + ", ".join(candidates) + ".")
    lines.append("Based on the times above, where is the cat right now?")
    lines.append(_CUE)
    return "\n".join(lines)
