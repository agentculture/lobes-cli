"""The sentence chunker that makes sentence-level streaming possible.

``lobes.realtime._sentences`` is the pure, stdlib-only module that turns a
stream of generate deltas into COMPLETE sentences the TTS stage can start on
before the reply is finished (approved deviation d7, 2026-09-18). Every rule
it applies is a latency/quality trade, so every rule is pinned here:

- a sentence is released only when it is genuinely complete (a terminator
  followed by whitespace, or a newline) — never on a bare terminator at the
  end of the buffer, because the next delta may turn ``3.`` into ``3.5``;
- a crumb shorter than ``min_sentence_chars`` is MERGED FORWARD rather than
  handed to TTS on its own (a one-word synthesis costs a whole request for a
  fraction of a second of audio) — unless it is the whole reply;
- the FIRST sentence may also break at a comma once it is long enough
  (``eager_first_min_chars``), because time-to-first-audio is the number this
  whole change exists to move;
- a run-on longer than ``max_sentence_chars`` is cut anyway, at the last
  ``", "``/space, sized by BASE characters so Hebrew niqqud never counts.

Hebrew-first by construction: the deployment this ships for is the Hebrew
voice lane, so the fixtures are Hebrew (RTL, gershayim, niqqud) with English
kept only where it proves the same rule from the other side.
"""

from __future__ import annotations

import pytest

from lobes.realtime._sentences import (
    DEFAULT_EAGER_FIRST_MIN_CHARS,
    DEFAULT_MAX_SENTENCE_CHARS,
    DEFAULT_MIN_SENTENCE_CHARS,
    SentenceChunker,
)

# A Hebrew sentence comfortably over DEFAULT_MIN_SENTENCE_CHARS.
HE_ONE = "השעה עכשיו ארבע וחצי אחר הצהריים."
HE_TWO = "מזג האוויר נעים בחוץ."


def chunk_all(chunker: SentenceChunker, *deltas: str) -> list[str]:
    """Feed every delta, then flush — the full lifetime of one reply."""
    out: list[str] = []
    for delta in deltas:
        out.extend(chunker.feed(delta))
    out.extend(chunker.flush())
    return out


# --- the core release rule ---------------------------------------------------


def test_a_sentence_is_released_as_soon_as_its_terminator_is_confirmed() -> None:
    chunker = SentenceChunker()
    # The terminator alone is not enough: the next delta could continue it.
    assert chunker.feed(HE_ONE) == []
    # A following space confirms the boundary — and the sentence goes out
    # while the model is still generating the rest of the reply.
    assert chunker.feed(" ") == [HE_ONE]
    assert chunker.feed(HE_TWO + " ") == [HE_TWO]
    assert chunker.flush() == []


def test_deltas_that_split_a_word_still_produce_whole_sentences() -> None:
    # vLLM streams token fragments, not words: the chunker must reassemble.
    chunker = SentenceChunker()
    released: list[str] = []
    for i in range(0, len(HE_ONE), 3):
        released.extend(chunker.feed(HE_ONE[i : i + 3]))
    released.extend(chunker.flush())
    assert released == [HE_ONE]


def test_the_remainder_is_released_by_flush_when_the_reply_has_no_terminator() -> None:
    chunker = SentenceChunker()
    text = "אין כאן שום סימן פיסוק בכלל"
    assert chunker.feed(text) == []
    assert chunker.flush() == [text]
    # flush is idempotent — a second one has nothing left to give.
    assert chunker.flush() == []


def test_a_newline_is_a_boundary_on_its_own() -> None:
    chunker = SentenceChunker()
    assert chunker.feed("שורה ראשונה כאן\nשורה שנייה") == ["שורה ראשונה כאן"]
    assert chunker.flush() == ["שורה שנייה"]


@pytest.mark.parametrize("terminator", (".", "!", "?", "…", ":"))
def test_every_declared_terminator_closes_a_sentence(terminator: str) -> None:
    chunker = SentenceChunker()
    sentence = "זאת בדיקה של סימן הסיום" + terminator
    assert chunker.feed(sentence + " ") == [sentence]


# --- the traps: numbers, ellipses, abbreviations -----------------------------


def test_a_decimal_number_is_never_split() -> None:
    chunker = SentenceChunker()
    assert chunk_all(chunker, "המחיר הוא 3.5 שקלים לקילו.") == ["המחיר הוא 3.5 שקלים לקילו."]


def test_an_ellipsis_run_splits_once_at_its_end_not_inside_it() -> None:
    chunker = SentenceChunker()
    assert chunk_all(chunker, "רגע אחד בבקשה... אני בודק עכשיו.") == [
        "רגע אחד בבקשה...",
        "אני בודק עכשיו.",
    ]


def test_a_gershayim_abbreviation_is_never_split_inside() -> None:
    chunker = SentenceChunker()
    released = chunk_all(chunker, 'דוח צה"ל פורסם אתמול בבוקר. הנה הפרטים.')
    assert released == ['דוח צה"ל פורסם אתמול בבוקר.', "הנה הפרטים."]
    assert all('צה"ל' in r or 'צה"ל' not in r for r in released)
    assert released[0].count('"') == 1  # the acronym survived whole


def test_a_terminator_directly_after_a_gershayim_is_not_a_boundary() -> None:
    # An abbreviation's own mark is not the end of a thought: releasing here
    # would hand TTS a fragment, so the chunker waits for the real boundary.
    chunker = SentenceChunker()
    assert chunk_all(chunker, "הוא אמר וכו׳: ואז המשיך הלאה בשקט.") == [
        "הוא אמר וכו׳: ואז המשיך הלאה בשקט."
    ]


# --- the minimum: crumbs merge forward --------------------------------------


def test_a_crumb_is_merged_into_the_next_sentence_not_synthesized_alone() -> None:
    chunker = SentenceChunker()
    # "כן." is 3 base chars — far below the 12-char floor, and a TTS request
    # of its own would cost a whole round trip for a fraction of a second.
    assert chunker.feed("כן. ") == []
    assert chunker.feed(HE_ONE + " ") == ["כן. " + HE_ONE]


def test_a_short_reply_is_still_spoken_when_it_is_the_whole_reply() -> None:
    chunker = SentenceChunker()
    assert chunker.feed("כן.") == []
    assert chunker.flush() == ["כן."]


def test_the_minimum_counts_base_characters_not_code_points() -> None:
    # Vocalized Hebrew doubles the code points without adding a syllable, so
    # sizing by len() would let a niqqud crumb through as if it were long.
    chunker = SentenceChunker(min_sentence_chars=12)
    vocalized = "כֵּן."  # 3 base chars, 5 code points
    assert len(vocalized) > 3
    assert chunker.feed(vocalized + " ") == []


# --- the maximum: a run-on is cut anyway ------------------------------------


def test_a_run_on_longer_than_the_cap_is_cut_at_a_comma() -> None:
    chunker = SentenceChunker(max_sentence_chars=40, eager_first_min_chars=1000)
    text = "אחת שתיים שלוש, ארבע חמש שש שבע, שמונה תשע עשר אחת עשרה שתים עשרה"
    released = chunk_all(chunker, text)
    assert len(released) > 1
    assert all(len(piece) <= 40 for piece in released)
    # Nothing but whitespace is lost when the pieces are put back together.
    assert "".join(released).replace(" ", "") == text.replace(" ", "")


def test_the_cap_fires_mid_stream_without_waiting_for_a_terminator() -> None:
    chunker = SentenceChunker(max_sentence_chars=30, eager_first_min_chars=1000)
    # No terminator anywhere — yet the chunker must not hoard the whole reply.
    assert chunker.feed("מילה " * 20) != []


# --- the eager first sentence: time to first audio ---------------------------


def test_the_first_sentence_may_break_at_a_comma_once_it_is_long_enough() -> None:
    chunker = SentenceChunker(eager_first_min_chars=10)
    assert chunker.feed("בוקר טוב לכולם, ") == ["בוקר טוב לכולם,"]
    # …and only the FIRST: a later comma waits for a real terminator.
    assert chunker.feed("היום נעים בחוץ, ואפשר לצאת לטייל.") == []
    assert chunker.flush() == ["היום נעים בחוץ, ואפשר לצאת לטייל."]


def test_the_eager_break_waits_for_its_own_minimum() -> None:
    chunker = SentenceChunker(eager_first_min_chars=30)
    assert chunker.feed("כן, ") == []
    assert chunker.flush() == ["כן,"]


def test_a_real_terminator_still_wins_over_an_eager_comma() -> None:
    chunker = SentenceChunker(eager_first_min_chars=5, min_sentence_chars=5)
    assert chunker.feed("שלום לך. מה שלומך, ידידי?") == ["שלום לך."]


def test_reset_re_arms_the_eager_first_sentence_for_the_next_reply() -> None:
    chunker = SentenceChunker(eager_first_min_chars=10)
    assert chunker.feed("בוקר טוב לכולם, ") == ["בוקר טוב לכולם,"]
    chunker.reset()
    assert chunker.feed("ערב טוב לכולם, ") == ["ערב טוב לכולם,"]


# --- thresholds are constructor arguments, never module policy --------------


def test_every_threshold_is_a_constructor_argument_with_a_declared_default() -> None:
    chunker = SentenceChunker()
    assert chunker.min_sentence_chars == DEFAULT_MIN_SENTENCE_CHARS
    assert chunker.max_sentence_chars == DEFAULT_MAX_SENTENCE_CHARS
    assert chunker.eager_first_min_chars == DEFAULT_EAGER_FIRST_MIN_CHARS
    tuned = SentenceChunker(min_sentence_chars=1, max_sentence_chars=9, eager_first_min_chars=2)
    assert (tuned.min_sentence_chars, tuned.max_sentence_chars, tuned.eager_first_min_chars) == (
        1,
        9,
        2,
    )


def test_english_behaves_the_same_way_as_hebrew() -> None:
    chunker = SentenceChunker()
    assert chunk_all(chunker, "The time is half past four. It is a nice day outside.") == [
        "The time is half past four.",
        "It is a nice day outside.",
    ]


def test_nothing_is_ever_lost_between_feed_and_flush() -> None:
    chunker = SentenceChunker()
    text = "ראשית, הנה הסבר קצר. שנית, יש עוד פרט אחד חשוב! ולבסוף, זהו."
    released = chunk_all(chunker, *[text[i : i + 5] for i in range(0, len(text), 5)])
    assert "".join(released).replace(" ", "") == text.replace(" ", "")


# --- a tunable first-clause break (2026-09-18 latency tuning) ---------------


def test_default_first_clause_still_waits_for_24_chars():
    chunker = SentenceChunker()
    assert chunker.feed("מצטער, אני לא ") == []


def test_a_lowered_first_clause_threshold_releases_a_short_opening_clause():
    chunker = SentenceChunker(eager_first_min_chars=5)
    assert chunker.feed("מצטער, אני לא ") == ["מצטער,"]
    # only the FIRST piece gets the allowance: later commas wait for a terminator
    assert chunker.feed("מוצא תיקייה, בשם ") == []
    assert chunker.feed("מסמכים. ") == ["אני לא מוצא תיקייה, בשם מסמכים."]


def test_a_lowered_threshold_never_releases_below_itself():
    chunker = SentenceChunker(eager_first_min_chars=5)
    assert chunker.feed("כן, בטח ") == []
