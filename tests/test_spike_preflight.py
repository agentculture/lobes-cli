"""Offline tests for `scripts/spike-preflight.sh`'s argv-token validator.

The spike harness exists to make ONE silent failure impossible: the fleet
compose template records as a MEASURED bug that a brace-containing substitution
corrupts compose's interpolation of every LATER brace pair, and
``PRIMARY_SPECULATIVE_CONFIG`` was the victim — it lost its closing brace. A
mangled token means vLLM boots healthy and serves with NO speculation, so every
throughput number measured against it is meaningless.

That validator therefore has to be right about mangled input, and it is the one
part of the harness that needs neither docker, nor ssh, nor a GPU. The script
exposes it as the read-only ``check-token`` mode precisely so it can be pinned
here: these tests shell out to the real script, so they test the shipped code
path rather than a Python re-implementation of it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "spike-preflight.sh"
CHECK_TOKEN = "check-token"


def _check(token: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), CHECK_TOKEN, token],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111, "spike-preflight.sh must be executable"


VALID = [
    # The deployed cortex default, verbatim from `docker inspect`.
    '--speculative-config={"method": "mtp", "num_speculative_tokens": 2}',
    # The DSpark shape the spike swaps in: longer, a model path, more braces.
    '--speculative-config={"method": "dspark", "model": "radixark/DSpark-Drafter-1.36B",'
    ' "num_speculative_tokens": 4, "draft_tensor_parallel_size": 1}',
    # The spark-lobe 1M YaRN override — the substitution whose placement the
    # template calls load-bearing.
    '--hf-overrides={"text_config": {"rope_parameters": {"rope_type": "yarn",'
    ' "factor": 4.0, "original_max_position_embeddings": 262144,'
    ' "mrope_interleaved": true, "mrope_section": [11, 11, 10],'
    ' "partial_rotary_factor": 0.25, "rope_theta": 10000000}}}',
    "--hf-overrides={}",
    '--default-chat-template-kwargs={"preserve_thinking": true}',
    # A brace inside a JSON string must not confuse the depth scan.
    '--speculative-config={"method": "mtp", "note": "a } inside a string"}',
]


@pytest.mark.parametrize("token", VALID, ids=range(len(VALID)))
def test_well_formed_tokens_pass(token):
    proc = _check(token)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


MANGLED = {
    # THE recorded bug: the closing brace is eaten by compose interpolation.
    "missing_closing_brace": '--speculative-config={"method": "mtp", "num_speculative_tokens": 2',
    # The same bug on the longer DSpark JSON the spike introduces.
    "dspark_missing_closing_brace": '--speculative-config={"method": "dspark",'
    ' "model": "radixark/DSpark-Drafter-1.36B", "num_speculative_tokens": 4',
    "extra_closing_brace": '--speculative-config={"method": "mtp"}}',
    "closing_before_opening": '--speculative-config=}"method": "mtp"{',
    # Dropping the template's single quotes lets the shell lexer split the JSON
    # on its spaces; the leading fragment is what lands in argv.
    "split_on_space": '--speculative-config={"method":',
    # ...and if the quotes are NOT consumed by the lexer they leak into argv.
    "quote_leaked": "--hf-overrides='{}'",
    "empty_payload": "--speculative-config=",
    "not_json": "--speculative-config=mtp",
    # A JSON array/scalar is parseable but is not a config object.
    "json_but_not_object": '--speculative-config=["mtp"]',
    "unterminated_string": '--speculative-config={"method": "mtp}',
}


@pytest.mark.parametrize("token", MANGLED.values(), ids=list(MANGLED))
def test_mangled_tokens_fail(token):
    proc = _check(token)
    assert proc.returncode != 0, (
        "validator ACCEPTED a mangled token — this is the exact silent failure "
        "the harness exists to prevent:\n" + proc.stdout
    )
    assert "MANGLED" in proc.stdout


def test_non_flag_token_is_rejected():
    proc = _check('{"method": "mtp"}')
    assert proc.returncode != 0
    assert "not a --flag=payload token" in proc.stdout


def test_unrelated_flag_is_rejected():
    # check-token validates JSON-bearing flags only; anything else is operator
    # error, not a silent pass.
    proc = _check("--max-model-len=1048576")
    assert proc.returncode != 0


def test_check_token_requires_an_argument():
    proc = subprocess.run(
        ["bash", str(SCRIPT), CHECK_TOKEN],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 2


def test_unknown_mode_is_a_usage_error():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "wat"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 2


def test_help_exits_zero_and_documents_every_mode():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    for mode in ("preflight", "stop", "restore", "check-token"):
        assert mode in proc.stdout
    # Mutation safety is part of the contract, not a footnote.
    assert "--apply" in proc.stdout
