"""Offline tests for ``scripts/spec-arms.py``'s pure logic.

The script is standalone (not a package module), so it is loaded by file path
(same pattern as ``tests/test_gen_api_key.py``). These tests never touch the
network or a real Docker daemon — they exercise the parsing and arm/shape
bookkeeping helpers directly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "spec-arms.py"
_ACCEPTED_KEY = "accepted_tokens"
_DRAFTED_KEY = "drafted_tokens"
_RATE_KEY = "acceptance_rate"


def _load():
    spec = importlib.util.spec_from_file_location("spec_arms", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sa = _load()

_METRICS_TEXT_BEFORE = """\
# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="x"} 1000.0
# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="x"} 1200.0
"""

_METRICS_TEXT_AFTER = """\
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="x"} 1090.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="x"} 1300.0
"""

_REQ_METRIC = "vllm:request_success_total"
_SCOPE_KEY = "scope"
_CONTAMINATED_KEY = "contaminated"
_TOK_S_KEY = "decode_tok_s"
_ESTIMATED_KEY = "decode_tok_s_estimated"
_CHUNKS_KEY = "sse_content_chunks"
_COMPLETION_KEY = "completion_tokens"


def _metrics(accepted: float, drafted: float, requests: float | None = None) -> dict:
    snap = {sa.ACCEPTED_METRIC: accepted, sa.DRAFTED_METRIC: drafted}
    if requests is not None:
        snap[sa.REQUESTS_METRIC] = requests
    return snap


_LOG_LINE = (
    "(APIServer pid=1) INFO 08-24 16:22:46 [metrics.py:120] SpecDecoding metrics: "
    "Mean acceptance length: 2.77, Accepted throughput: 5.50 tokens/s, Drafted "
    "throughput: 6.20 tokens/s, Accepted: 55 tokens, Drafted: 62 tokens, "
    "Per-position acceptance rate: 0.903, 0.871, Avg Draft acceptance rate: 88.7%"
)


def test_parse_metrics_text_extracts_wanted_counters() -> None:
    parsed = sa.parse_metrics_text(_METRICS_TEXT_BEFORE)
    assert parsed["vllm:spec_decode_num_accepted_tokens_total"] == 1000.0
    assert parsed["vllm:spec_decode_num_draft_tokens_total"] == 1200.0


def test_parse_metrics_text_ignores_unrelated_lines() -> None:
    text = 'vllm:cache_config_info{block_size="1600"} 1.0\n'
    assert sa.parse_metrics_text(text) == {}


def test_acceptance_delta_computes_rate_over_window() -> None:
    before = sa.parse_metrics_text(_METRICS_TEXT_BEFORE)
    after = sa.parse_metrics_text(_METRICS_TEXT_AFTER)
    delta = sa.acceptance_delta(before, after)
    assert delta[_ACCEPTED_KEY] == 90.0
    assert delta[_DRAFTED_KEY] == 100.0
    assert delta[_RATE_KEY] == 0.9
    assert delta["surface"] == "vllm_metrics_http"


def test_acceptance_delta_none_when_no_draft_tokens_in_window() -> None:
    before = sa.parse_metrics_text(_METRICS_TEXT_BEFORE)
    same = sa.parse_metrics_text(_METRICS_TEXT_BEFORE)
    assert sa.acceptance_delta(before, same) is None


def test_acceptance_delta_none_when_counters_absent() -> None:
    assert sa.acceptance_delta({}, {}) is None


def test_parse_acceptance_log_line_matches_real_vllm_line() -> None:
    parsed = sa.parse_acceptance_log_line(_LOG_LINE)
    assert parsed["mean_acceptance_length"] == 2.77
    assert parsed[_ACCEPTED_KEY] == 55
    assert parsed[_DRAFTED_KEY] == 62
    assert parsed[_RATE_KEY] == 0.887


def test_parse_acceptance_log_line_none_on_unrelated_line() -> None:
    assert sa.parse_acceptance_log_line("some unrelated INFO line") is None


def test_summarize_log_lines_aggregates_across_window() -> None:
    line2 = _LOG_LINE.replace(
        "Accepted: 55 tokens, Drafted: 62 tokens", "Accepted: 45 tokens, Drafted: 50 tokens"
    )
    summary = sa.summarize_log_lines([_LOG_LINE, line2, "unrelated"])
    assert summary[_ACCEPTED_KEY] == 100
    assert summary[_DRAFTED_KEY] == 112
    assert summary["sample_count"] == 2
    assert summary["surface"] == "docker_logs"


def test_summarize_log_lines_none_on_empty_window() -> None:
    assert sa.summarize_log_lines(["no spec decoding lines here"]) is None


def _fake_transcript(arm: str, shape_names=("code", "reasoning", "prose")) -> dict:
    return {
        "arm": arm,
        "shapes": {
            name: {
                "ttft_ms": 100.0,
                "decode_tok_s": 20.0,
                "max_model_len": 1048576,
                "acceptance": {"acceptance_rate": 0.6, "surface": "vllm_metrics_http"},
            }
            for name in shape_names
        },
    }


def test_build_comparison_all_arms_present() -> None:
    transcripts = {arm: _fake_transcript(arm) for arm in sa.ARMS}
    rows, missing = sa.build_comparison(transcripts)
    assert missing == []
    # 3 shapes * 3 arms
    assert len(rows) == len(sa.SHAPES) * len(sa.ARMS)
    assert all(r["status"] == "ok" for r in rows)


def test_build_comparison_reports_missing_arm_without_fabricating_data() -> None:
    transcripts = {"mtp-n2": _fake_transcript("mtp-n2"), "none": _fake_transcript("none")}
    rows, missing = sa.build_comparison(transcripts)
    assert missing == ["dspark"]
    dspark_rows = [r for r in rows if r["arm"] == "dspark"]
    assert len(dspark_rows) == len(sa.SHAPES)
    assert all(r["status"] == "MISSING" for r in dspark_rows)
    # present arms are untouched, never backfilled from another arm's data
    present_rows = [r for r in rows if r["arm"] != "dspark"]
    assert all(r["status"] == "ok" for r in present_rows)


def test_build_comparison_reports_missing_shape_within_present_arm() -> None:
    partial = _fake_transcript("mtp-n2", shape_names=("code",))
    transcripts = {
        "mtp-n2": partial,
        "dspark": _fake_transcript("dspark"),
        "none": _fake_transcript("none"),
    }
    rows, missing = sa.build_comparison(transcripts)
    assert missing == []  # the arm itself is present
    mtp_reasoning = next(r for r in rows if r["arm"] == "mtp-n2" and r["shape"] == "reasoning")
    assert mtp_reasoning["status"] == "MISSING"


def test_combine_cli_exits_nonzero_on_missing_arm(tmp_path, capsys) -> None:
    (tmp_path / "mtp.json").write_text(
        __import__("json").dumps(_fake_transcript("mtp-n2")), encoding="utf-8"
    )
    (tmp_path / "none.json").write_text(
        __import__("json").dumps(_fake_transcript("none")), encoding="utf-8"
    )
    rc = sa.main(["--combine", str(tmp_path / "mtp.json"), str(tmp_path / "none.json")])
    out = capsys.readouterr()
    assert rc == 1
    assert "MISSING" in out.out or "MISSING" in out.err


def test_combine_cli_allow_partial_exits_zero(tmp_path) -> None:
    (tmp_path / "mtp.json").write_text(
        __import__("json").dumps(_fake_transcript("mtp-n2")), encoding="utf-8"
    )
    rc = sa.main(["--combine", str(tmp_path / "mtp.json"), "--allow-partial"])
    assert rc == 0


def test_combine_cli_refuses_duplicate_arm_files(tmp_path, capsys) -> None:
    (tmp_path / "a.json").write_text(
        __import__("json").dumps(_fake_transcript("mtp-n2")), encoding="utf-8"
    )
    (tmp_path / "b.json").write_text(
        __import__("json").dumps(_fake_transcript("mtp-n2")), encoding="utf-8"
    )
    rc = sa.main(["--combine", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert rc == 2
    assert "both claim arm" in capsys.readouterr().err


def test_combine_cli_all_three_arms_present_exits_zero(tmp_path) -> None:
    for arm in sa.ARMS:
        (tmp_path / f"{arm}.json").write_text(
            __import__("json").dumps(_fake_transcript(arm)), encoding="utf-8"
        )
    rc = sa.main(["--combine", *[str(tmp_path / f"{arm}.json") for arm in sa.ARMS]])
    assert rc == 0


# ---------------------------------------------------------------------------
# Qodo finding 9 — acceptance is engine-wide, and says so; contamination is
# detected where the engine makes it detectable.
# ---------------------------------------------------------------------------


def test_parse_metrics_text_collects_request_counter_summed_over_labels() -> None:
    text = (
        'vllm:request_success_total{finished_reason="stop",model_name="x"} 7.0\n'
        'vllm:request_success_total{finished_reason="length",model_name="x"} 3.0\n'
    )
    assert sa.parse_metrics_text(text)[_REQ_METRIC] == 10.0


def test_acceptance_delta_labels_scope_as_engine_wide_not_per_request() -> None:
    delta = sa.acceptance_delta(_metrics(0, 0, 0), _metrics(90, 100, 1))
    assert delta[_SCOPE_KEY] == sa.ENGINE_WIDE_SCOPE
    assert delta["request_scoped"] is False


def test_acceptance_delta_flags_contamination_from_foreign_traffic() -> None:
    # 4 requests completed in the window, but this tool issued only 1
    delta = sa.acceptance_delta(_metrics(0, 0, 10), _metrics(90, 100, 14))
    assert delta[_CONTAMINATED_KEY] is True
    assert delta["requests_in_window"] == 4.0
    assert delta[_RATE_KEY] == 0.9  # the number is still reported, just flagged
    assert "other clients" in delta["note"]


def test_acceptance_delta_clean_when_exactly_the_issued_request_completed() -> None:
    delta = sa.acceptance_delta(_metrics(0, 0, 10), _metrics(90, 100, 11))
    assert delta[_CONTAMINATED_KEY] is False
    assert delta["expected_requests"] == 1


def test_acceptance_delta_contamination_unknown_when_request_counter_absent() -> None:
    delta = sa.acceptance_delta(_metrics(0, 0), _metrics(90, 100))
    assert delta[_CONTAMINATED_KEY] is None
    assert delta["requests_in_window"] is None


def test_acceptance_delta_contamination_unknown_when_request_still_in_flight() -> None:
    delta = sa.acceptance_delta(_metrics(0, 0, 10), _metrics(90, 100, 10))
    assert delta[_CONTAMINATED_KEY] is None


def test_summarize_log_lines_can_never_be_shown_clean() -> None:
    summary = sa.summarize_log_lines([_LOG_LINE])
    assert summary[_SCOPE_KEY] == sa.ENGINE_WIDE_SCOPE
    assert summary[_CONTAMINATED_KEY] is None
    assert summary["request_scoped"] is False


def test_fmt_row_names_engine_wide_scope_and_contamination() -> None:
    entry = {
        "arm": "dspark",
        "ttft_ms": 100.0,
        _TOK_S_KEY: 20.0,
        "acceptance": {
            _RATE_KEY: 0.9,
            "surface": "vllm_metrics_http",
            _SCOPE_KEY: sa.ENGINE_WIDE_SCOPE,
            _CONTAMINATED_KEY: True,
        },
    }
    line = sa._fmt_row("code", entry)
    assert "engine-wide" in line
    assert "CONTAMINATED" in line


def test_no_acceptance_entries_never_fabricate_a_rate() -> None:
    entry = sa._no_acceptance("not_applicable")
    assert entry[_RATE_KEY] is None
    assert entry[_CONTAMINATED_KEY] is None
    assert entry[_SCOPE_KEY] is None


# ---------------------------------------------------------------------------
# Qodo finding 3 — chunks are not tokens; and finding 10 — the wall-clock
# deadline is real.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, *exc) -> bool:
        return False


def _sse(text: str) -> str:
    return "data: " + __import__("json").dumps({"choices": [{"delta": {"content": text}}]})


def _sse_usage(completion_tokens: int) -> str:
    return "data: " + __import__("json").dumps(
        {"choices": [{"delta": {}}], "usage": {"completion_tokens": completion_tokens}}
    )


def _fake_clock(step: float = 0.01):
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += step
        return state["t"]

    return clock


def _measure_with(monkeypatch, lines: list[str], *, step: float = 0.01, max_seconds: float = 60.0):
    monkeypatch.setattr(sa.time, "perf_counter", _fake_clock(step))
    monkeypatch.setattr(
        sa.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(lines)
    )
    return sa.measure_shape("http://x", "cortex", "code", sa.SHAPES["code"], None, max_seconds, 64)


def test_measure_shape_prefers_real_usage_completion_tokens(monkeypatch) -> None:
    lines = [_sse("a"), _sse("b"), _sse_usage(100), "data: [DONE]"]
    row = _measure_with(monkeypatch, lines)
    assert row[_COMPLETION_KEY] == 100
    assert row[_CHUNKS_KEY] == 2
    assert row[_ESTIMATED_KEY] is False
    assert row["throughput_basis"] == sa.BASIS_USAGE


def test_measure_shape_never_labels_chunk_count_as_completion_tokens(monkeypatch) -> None:
    lines = [_sse("a"), _sse("b"), _sse("c"), "data: [DONE]"]
    row = _measure_with(monkeypatch, lines)
    # the chunk count lives under its OWN key; completion_tokens stays unknown
    assert row[_COMPLETION_KEY] is None
    assert row[_CHUNKS_KEY] == 3
    assert row[_ESTIMATED_KEY] is True
    assert row["throughput_basis"] == sa.BASIS_CHUNKS
    assert "ESTIMATE" in row["throughput_note"]


def test_fmt_row_marks_estimated_throughput(monkeypatch) -> None:
    entry = {"arm": "none", "ttft_ms": 1.0, _TOK_S_KEY: 12.0, _ESTIMATED_KEY: True}
    assert "(est)" in sa._fmt_row("code", entry)
    entry_measured = {"arm": "none", "ttft_ms": 1.0, _TOK_S_KEY: 12.0, _ESTIMATED_KEY: False}
    assert "(est)" not in sa._fmt_row("code", entry_measured)


def test_consume_sse_stream_stops_at_wall_clock_deadline() -> None:
    # an endless stream that never stalls: every chunk arrives promptly, so
    # urllib's per-operation timeout would never fire
    def endless():
        while True:
            yield _sse("x")

    clock = _fake_clock(1.0)
    out = sa.consume_sse_stream(endless(), now=clock, t0=0.0, deadline=5.0)
    assert out["timed_out"] is True
    assert out["chunks"] > 0


def test_consume_sse_stream_completes_within_deadline() -> None:
    lines = [_sse("a"), _sse_usage(7), "data: [DONE]"]
    out = sa.consume_sse_stream(lines, now=_fake_clock(0.01), t0=0.0, deadline=1000.0)
    assert out["timed_out"] is False
    assert out["usage"]["completion_tokens"] == 7
    assert out["chunks"] == 1


def test_measure_shape_reports_timeout_as_error_not_a_measurement(monkeypatch) -> None:
    # 500 promptly-arriving chunks: no socket timeout ever fires, but the
    # 5-second wall clock is blown long before the stream ends
    row = _measure_with(monkeypatch, [_sse("x")] * 500, step=1.0, max_seconds=5.0)
    assert row["timed_out"] is True
    assert "deadline" in row["error"]
    # a timed-out shape is NOT a completed measurement
    assert _TOK_S_KEY not in row
    assert "TIMED OUT" in sa._fmt_row("code", {**row, "arm": "dspark"})


# ---------------------------------------------------------------------------
# Qodo finding 4 — a FAILED cell is not a comparison.
# ---------------------------------------------------------------------------


def _failed_transcript(arm: str) -> dict:
    data = _fake_transcript(arm)
    data["shapes"]["prose"] = {"shape": "prose", "error": "boom"}
    return data


def test_build_comparison_marks_errored_shape_as_failed() -> None:
    transcripts = {arm: _fake_transcript(arm) for arm in sa.ARMS}
    transcripts["dspark"] = _failed_transcript("dspark")
    rows, missing = sa.build_comparison(transcripts)
    assert missing == []
    prose = next(r for r in rows if r["arm"] == "dspark" and r["shape"] == "prose")
    assert prose["status"] == "FAILED"


def test_incomplete_cells_lists_both_missing_and_failed() -> None:
    transcripts = {"mtp-n2": _fake_transcript("mtp-n2"), "dspark": _failed_transcript("dspark")}
    rows, _ = sa.build_comparison(transcripts)
    cells = sa.incomplete_cells(rows)
    statuses = {c["status"] for c in cells}
    assert statuses == {"MISSING", "FAILED"}
    # 3 MISSING (the whole `none` arm) + 1 FAILED prose cell
    assert len(cells) == len(sa.SHAPES) + 1


def test_incomplete_cells_empty_when_every_cell_is_ok() -> None:
    rows, _ = sa.build_comparison({arm: _fake_transcript(arm) for arm in sa.ARMS})
    assert sa.incomplete_cells(rows) == []


def _write_transcripts(tmp_path, transcripts: dict) -> list[str]:
    paths = []
    for arm, data in transcripts.items():
        path = tmp_path / f"{arm}.json"
        path.write_text(__import__("json").dumps(data), encoding="utf-8")
        paths.append(str(path))
    return paths


def test_combine_cli_exits_nonzero_on_failed_shape(tmp_path, capsys) -> None:
    transcripts = {arm: _fake_transcript(arm) for arm in sa.ARMS}
    transcripts["dspark"] = _failed_transcript("dspark")
    rc = sa.main(["--combine", *_write_transcripts(tmp_path, transcripts)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "INCOMPLETE comparison" in err
    assert "prose/dspark=FAILED" in err


def test_combine_cli_allow_partial_accepts_failed_shape(tmp_path) -> None:
    transcripts = {arm: _fake_transcript(arm) for arm in sa.ARMS}
    transcripts["dspark"] = _failed_transcript("dspark")
    rc = sa.main(["--combine", *_write_transcripts(tmp_path, transcripts), "--allow-partial"])
    assert rc == 0


def test_combine_cli_json_reports_completeness(tmp_path, capsys) -> None:
    transcripts = {arm: _fake_transcript(arm) for arm in sa.ARMS}
    transcripts["dspark"] = _failed_transcript("dspark")
    rc = sa.main(["--combine", *_write_transcripts(tmp_path, transcripts), "--json"])
    payload = __import__("json").loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["complete"] is False
    assert payload["incomplete_cells"] == [{"shape": "prose", "arm": "dspark", "status": "FAILED"}]


def test_main_requires_url_and_arm_without_combine(capsys) -> None:
    try:
        sa.main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected argparse to exit on missing required args")
