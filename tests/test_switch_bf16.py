"""Tests for ``lobes switch`` bf16/unquantized ("none") quantization behaviour.

When the catalog marks a model with ``quantization="none"`` (or an operator passes
``--quantization none`` explicitly), ``lobes switch`` must NOT write
``VLLM_QUANTIZATION`` to ``.env`` — the single-model template's default
``--quantization=${VLLM_QUANTIZATION:-modelopt}`` would silently corrupt bf16
weights. Instead, a compose-edit notice fires telling the operator to REMOVE the
``--quantization`` line by hand (mirrors the moe_backend / MTP-flag surfacing
pattern in switch.py).
"""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.runtime import _compose


def _scaffold(path):
    _compose.write_scaffold(path, force=True)
    return path


# ---------------------------------------------------------------------------
# bf16/none via catalog (Qwen/Qwen3.5-4B)
# ---------------------------------------------------------------------------


def test_switch_bf16_minor_does_not_set_vllm_quantization(tmp_path, capsys) -> None:
    # Switching to Qwen/Qwen3.5-4B (catalog quantization="none") must NOT include
    # VLLM_QUANTIZATION in the dry-run plan — the template already defaults to
    # --quantization=modelopt; writing "none" or omitting it both leave the wrong
    # default. The operator must remove the line by hand per the compose-edit notice.
    # We check for the plan key=value form ("VLLM_QUANTIZATION="); the notice text
    # may mention "VLLM_QUANTIZATION" as prose but that's fine.
    _scaffold(tmp_path)
    rc = main(["switch", "Qwen/Qwen3.5-4B", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # Plan must not contain the key=value assignment line.
    assert "VLLM_QUANTIZATION=" not in out
    assert "bf16/unquantized" in out


def test_switch_bf16_minor_surfaces_remove_quantization_compose_edit(tmp_path, capsys) -> None:
    # The bf16/none gear must surface a compose-edit NOTE telling the operator to REMOVE
    # the --quantization flag — the template's default corrupts bf16 weights.
    _scaffold(tmp_path)
    rc = main(["switch", "Qwen/Qwen3.5-4B", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # A NOTE line must name REMOVE and --quantization together (the bf16 notice).
    note_lines = [line for line in out.splitlines() if "NOTE:" in line]
    assert any(
        "REMOVE" in line and "--quantization" in line for line in note_lines
    ), f"expected a REMOVE --quantization NOTE; got notes: {note_lines}"


def test_switch_bf16_minor_json_mode_has_no_quantization_in_env(tmp_path, capsys) -> None:
    # In JSON mode the plan's ``env`` dict must not contain VLLM_QUANTIZATION, and the
    # compose_edits list must include the REMOVE --quantization notice.
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "Qwen/Qwen3.5-4B",
            "--json",
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    env = payload["env"]
    assert "VLLM_QUANTIZATION" not in env, f"VLLM_QUANTIZATION must not be in plan: {env}"
    compose_edits = payload["compose_edits"]
    assert any(
        "REMOVE" in n and "--quantization" in n for n in compose_edits
    ), f"expected REMOVE --quantization in compose_edits; got: {compose_edits}"


# ---------------------------------------------------------------------------
# bf16/none via explicit --quantization none flag (any model)
# ---------------------------------------------------------------------------


def test_switch_explicit_quantization_none_does_not_set_env(tmp_path, capsys) -> None:
    # An explicit ``--quantization none`` on any model must suppress VLLM_QUANTIZATION
    # and emit the bf16/unquantized message (mirrors catalog-driven behaviour).
    # Check for the plan key=value form only ("VLLM_QUANTIZATION=").
    _scaffold(tmp_path)
    rc = main(["switch", "foo/bar", "--quantization", "none", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_QUANTIZATION=" not in out
    assert "bf16/unquantized" in out


def test_switch_explicit_quantization_none_json_no_env_key(tmp_path, capsys) -> None:
    # JSON mode: explicit --quantization none → VLLM_QUANTIZATION absent from plan env.
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "foo/bar",
            "--quantization",
            "none",
            "--json",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "VLLM_QUANTIZATION" not in payload["env"]


# ---------------------------------------------------------------------------
# Real quantization values are unaffected
# ---------------------------------------------------------------------------


def test_real_quantization_values_still_written(tmp_path, capsys) -> None:
    # The "none" handling must NOT affect real quantization values (modelopt /
    # modelopt_fp4 / compressed-tensors); those must still be written to the plan.
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "mmangkad/Qwen3.6-27B-NVFP4",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_QUANTIZATION=modelopt_fp4" in out

    rc2 = main(
        [
            "switch",
            "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "VLLM_QUANTIZATION=compressed-tensors" in out2


def test_switch_uncatalogued_quantization_none_surfaces_remove_notice(tmp_path, capsys) -> None:
    # Qodo finding 4: an explicit --quantization none on an UNCATALOGUED model must
    # still surface the REMOVE --quantization compose-edit NOTE (the notice keys off
    # the effective quantization choice, not just catalog metadata).
    _scaffold(tmp_path)
    rc = main(
        ["switch", "acme/some-bf16-model", "--quantization", "none", "--compose-dir", str(tmp_path)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_QUANTIZATION=" not in out
    note_lines = [line for line in out.splitlines() if "NOTE:" in line]
    assert any(
        "REMOVE" in line and "--quantization" in line for line in note_lines
    ), f"expected REMOVE --quantization NOTE (uncatalogued --quantization none); got: {note_lines}"
