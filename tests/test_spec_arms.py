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


def _load():
    spec = importlib.util.spec_from_file_location("spec_arms", _SCRIPT)
    assert spec and spec.loader
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
    assert delta["accepted_tokens"] == 90.0
    assert delta["drafted_tokens"] == 100.0
    assert delta["acceptance_rate"] == 0.9
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
    assert parsed["accepted_tokens"] == 55
    assert parsed["drafted_tokens"] == 62
    assert parsed["acceptance_rate"] == 0.887


def test_parse_acceptance_log_line_none_on_unrelated_line() -> None:
    assert sa.parse_acceptance_log_line("some unrelated INFO line") is None


def test_summarize_log_lines_aggregates_across_window() -> None:
    line2 = _LOG_LINE.replace(
        "Accepted: 55 tokens, Drafted: 62 tokens", "Accepted: 45 tokens, Drafted: 50 tokens"
    )
    summary = sa.summarize_log_lines([_LOG_LINE, line2, "unrelated"])
    assert summary["accepted_tokens"] == 100
    assert summary["drafted_tokens"] == 112
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


def test_main_requires_url_and_arm_without_combine(capsys) -> None:
    try:
        sa.main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected argparse to exit on missing required args")
