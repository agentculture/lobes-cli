"""Tests for the model-ops runtime: .env r/w, dir resolution, switch/serve/stop/status."""

from __future__ import annotations

import json
import types

import pytest

from model_gear.cli import _runtime_ops, main
from model_gear.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from model_gear.runtime import _compose, _env, _health


def _ok() -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _scaffold(path):
    _compose.write_scaffold(path, force=True)
    return path


# --- _env -----------------------------------------------------------------


def test_env_read_write(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("VLLM_PORT=8000\nHF_TOKEN=\n", encoding="utf-8")
    assert _env.read_env(env, "VLLM_PORT") == "8000"
    # empty value (KEY=) reads as the caller's default
    assert _env.read_env(env, "HF_TOKEN", "fallback") == "fallback"
    # absent key reads as default
    assert _env.read_env(env, "NOPE", "x") == "x"
    # rewrite-if-present
    _env.set_env(env, "VLLM_PORT", "9001")
    assert _env.read_env(env, "VLLM_PORT") == "9001"
    # append-if-absent
    _env.set_env(env, "VLLM_MODEL", "foo/bar")
    assert _env.read_env(env, "VLLM_MODEL") == "foo/bar"


def test_read_env_missing_file_returns_default(tmp_path) -> None:
    assert _env.read_env(tmp_path / "nope.env", "K", "default") == "default"


def test_set_env_missing_file_raises(tmp_path) -> None:
    with pytest.raises(ModelGearError) as exc:
        _env.set_env(tmp_path / "nope.env", "K", "V")
    assert exc.value.code == EXIT_ENV_ERROR


# --- resolve_deployment_dir ----------------------------------------------


def test_resolve_explicit(tmp_path) -> None:
    _scaffold(tmp_path)
    assert _compose.resolve_deployment_dir(str(tmp_path)) == tmp_path


def test_resolve_explicit_missing_raises_user_error(tmp_path) -> None:
    with pytest.raises(ModelGearError) as exc:
        _compose.resolve_deployment_dir(str(tmp_path / "empty"))
    assert exc.value.code == EXIT_USER_ERROR


def test_resolve_env_var(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setenv("MODEL_GEAR_DIR", str(tmp_path))
    assert _compose.resolve_deployment_dir(None) == tmp_path


def test_resolve_default_missing_raises_env_error() -> None:
    # The autouse fixture points the default home at an empty tmp dir.
    with pytest.raises(ModelGearError) as exc:
        _compose.resolve_deployment_dir(None)
    assert exc.value.code == EXIT_ENV_ERROR


# --- port parsing ---------------------------------------------------------


def test_parse_port_invalid_raises_env_error() -> None:
    with pytest.raises(ModelGearError) as exc:
        _env.parse_port("not-a-number", "VLLM_PORT")
    assert exc.value.code == EXIT_ENV_ERROR


def test_invalid_env_port_gives_clean_error(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _env.set_env(tmp_path / ".env", "VLLM_PORT", "abc")
    rc = main(["status", "--compose-dir", str(tmp_path)])
    assert rc == EXIT_ENV_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err  # structured, not a generic "unexpected: ValueError"


# --- switch ---------------------------------------------------------------


def test_switch_dry_run_changes_nothing(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "VLLM_MODEL=foo/bar" in out
    # .env untouched
    assert (
        _env.read_env(tmp_path / ".env", "VLLM_MODEL") == "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
    )


def test_switch_apply_recreates_and_writes_env(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(_compose, "compose_down", lambda d: (calls.append("down"), _ok())[1])
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: (calls.append("up"), _ok())[1])
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)

    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path), "--apply", "--no-probe"])
    assert rc == 0
    assert calls == ["down", "up"]  # frees prior model before starting new one
    env = tmp_path / ".env"
    assert _env.read_env(env, "VLLM_MODEL") == "foo/bar"
    assert _env.read_env(env, "VLLM_SERVED_NAME") == "foo/bar"


def test_switch_writes_tool_call_parser_when_given(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_compose, "compose_down", lambda d: _ok())
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: _ok())
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)

    rc = main(
        [
            "switch",
            "nvidia/Qwen3-32B-NVFP4",  # would auto-infer hermes; the explicit flag must win
            "--tool-call-parser",
            "qwen3_coder",
            "--compose-dir",
            str(tmp_path),
            "--apply",
            "--no-probe",
        ]
    )
    assert rc == 0
    assert _env.read_env(tmp_path / ".env", "VLLM_TOOL_CALL_PARSER") == "qwen3_coder"


def test_switch_leaves_tool_call_parser_when_unknown_model(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # an unknown model (no inference rule) and no --tool-call-parser must neither
    # plan nor write VLLM_TOOL_CALL_PARSER, leaving the scaffolded default.
    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "VLLM_TOOL_CALL_PARSER" not in capsys.readouterr().out
    assert _env.read_env(tmp_path / ".env", "VLLM_TOOL_CALL_PARSER") == "qwen3_coder"


def test_switch_auto_selects_parser_for_known_model(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # Qwen3.6 needs qwen3_coder; switch must infer + plan it without the flag.
    rc = main(["switch", "mmangkad/Qwen3.6-27B-NVFP4", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_TOOL_CALL_PARSER=qwen3_coder" in out
    assert "auto-selected" in out


def test_switch_auto_selects_quantization_from_catalog(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # The RedHatAI Mistral fallback is compressed-tensors, not modelopt_fp4 —
    # switch must read that from the catalog and plan it without an explicit flag.
    rc = main(
        [
            "switch",
            "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_QUANTIZATION=compressed-tensors" in out
    assert "from catalog" in out


def test_switch_quantization_explicit_wins(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # An explicit --quantization overrides the catalog value (here for the 27B,
    # whose catalog value is modelopt_fp4).
    rc = main(
        [
            "switch",
            "mmangkad/Qwen3.6-27B-NVFP4",
            "--quantization",
            "compressed-tensors",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_QUANTIZATION=compressed-tensors" in out
    assert "explicit" in out


def test_switch_leaves_quantization_when_uncatalogued(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # An uncatalogued model with no --quantization must neither plan nor write
    # VLLM_QUANTIZATION, leaving the scaffolded default.
    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "VLLM_QUANTIZATION" not in capsys.readouterr().out


def test_switch_purpose_machine_resolve_into_plan(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # decode-heavy purpose + blackwell machine → the resolved VLLM_* knobs.
    rc = main(
        [
            "switch",
            "mmangkad/Qwen3.6-27B-NVFP4",
            "--purpose",
            "decode-heavy",
            "--machine",
            "blackwell",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_PURPOSE=decode-heavy" in out
    assert "VLLM_MACHINE=blackwell" in out
    assert "VLLM_MAX_NUM_SEQS=8" in out  # decode-heavy
    assert "VLLM_MAX_NUM_BATCHED_TOKENS=4096" in out  # decode-heavy
    assert "VLLM_GPU_MEM_UTIL=0.85" in out  # blackwell machine default
    assert "VLLM_MAX_MODEL_LEN=65536" in out  # blackwell machine default
    assert "VLLM_ATTENTION_BACKEND=flashinfer" in out


def test_switch_defaults_to_balanced(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "mmangkad/Qwen3.6-27B-NVFP4",
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_PURPOSE=balanced" in out  # default purpose
    # mmangkad is a non-MTP candidate now, so the balanced profile's seqs (4) stands
    # (only the MTP primary is force-capped to 2).
    assert "VLLM_MAX_NUM_SEQS=4" in out


def test_switch_explicit_overrides_machine_defaults(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "mmangkad/Qwen3.6-27B-NVFP4",
            "--machine",
            "blackwell",  # would default 0.85 / 65536
            "--gpu-mem-util",
            "0.5",
            "--max-model-len",
            "16384",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_GPU_MEM_UTIL=0.5" in out
    assert "VLLM_MAX_MODEL_LEN=16384" in out


def test_switch_moe_model_prints_compose_edit_notice(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "mmangkad/Qwen3.6-35B-A3B-NVFP4",
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # the MoE candidate needs --moe-backend ADDED (its own catalog extra)...
    assert "--moe-backend=marlin" in out
    # ...and, being non-MTP, it also gets the inverted notice to REMOVE the MTP
    # primary's baked-in flags (the template ships them by default now).
    assert "--speculative-config=" in out
    assert "REMOVE these" in out


def test_switch_to_mtp_primary_needs_no_compose_edit(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # The MTP build is the default primary now: its serve flags are baked into the
    # compose template, so switching TO it needs NO hand edit. switch should print
    # no compose-edit NOTE, force the MTP seq cap (2), and set the modelopt quant.
    rc = main(
        [
            "switch",
            "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "NOTE:" not in out  # MTP primary's flags are template defaults — nothing to edit
    assert "VLLM_MAX_NUM_SEQS=2" in out  # forced MTP cap (overrides the balanced 4)
    assert "MTP primary cap" in out
    assert "VLLM_MAX_MODEL_LEN=262144" in out  # spark serves the full 256K by default (load-tested)
    # quantization comes from the catalog (modelopt, not modelopt_fp4)
    assert any(line.strip() == "VLLM_QUANTIZATION=modelopt" for line in out.splitlines())


def test_switch_to_non_mtp_prints_remove_notice(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # The template ships the MTP primary's flags by default. Switching to a non-MTP
    # model must surface the inverse hand edit: REMOVE those 4 `command:` items.
    # Each is shown as an argv-safe YAML list item — --speculative-config uses the
    # `=` form (no shell space that would split into a broken token, Qodo #27).
    rc = main(
        [
            "switch",
            "nvidia/Qwen3-32B-NVFP4",  # a dense, non-MTP candidate
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "REMOVE these" in out
    assert "- '--speculative-config=" in out  # single quoted YAML list item
    assert "--speculative-config '" not in out  # no shell space form
    assert "qwen3_5_mtp" in out
    assert "--trust-remote-code" in out
    assert "--language-model-only" in out
    assert "--tokenizer=mmangkad/Qwen3.6-27B-NVFP4" in out
    # not an MoE checkpoint — no --moe-backend
    assert "--moe-backend" not in out


def test_switch_clamps_context_to_model_native_ceiling(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # spark's machine default is 262144 (256K, for the 256K-native MTP primary), but
    # nvidia/Qwen3-32B-NVFP4 is 32K-native — vLLM would refuse 262144 (no YaRN) and
    # fail to boot. switch must clamp the machine default DOWN to the model ceiling.
    rc = main(
        [
            "switch",
            "nvidia/Qwen3-32B-NVFP4",
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_MAX_MODEL_LEN=32768" in out  # clamped from spark's 262144 default
    assert "VLLM_MAX_MODEL_LEN=262144" not in out
    assert "clamped to model native ceiling" in out


def test_switch_no_clamp_when_model_fits_machine_default(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # The 256K-native MTP primary exactly meets spark's 256K default — no clamp, no notice.
    rc = main(
        [
            "switch",
            "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_MAX_MODEL_LEN=262144" in out  # full machine default stands
    assert "clamped to model native ceiling" not in out


def test_switch_explicit_max_model_len_overrides_clamp(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # An explicit --max-model-len wins even past the native ceiling: the operator is
    # opting into a YaRN/rope-scaling config (their responsibility), so don't clamp.
    rc = main(
        [
            "switch",
            "nvidia/Qwen3-32B-NVFP4",
            "--machine",
            "spark",
            "--max-model-len",
            "131072",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_MAX_MODEL_LEN=131072" in out  # explicit value respected, not clamped
    assert "clamped to model native ceiling" not in out


def test_switch_warns_on_uncatalogued_context_unclamped(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # An uncatalogued model (switch supports arbitrary IDs) has no native_max_model_len
    # to clamp against, so it inherits spark's 262144 default. vLLM would refuse to boot
    # if the checkpoint's native context is smaller — switch must NOT silently apply the
    # high default; it must warn and tell the user to pass --max-model-len. (Qodo #34.)
    rc = main(
        [
            "switch",
            "foo/bar-7b",
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "VLLM_MAX_MODEL_LEN=262144" in out  # machine default applied unclamped
    assert "uncatalogued model" in out  # the boot-safety warning fired
    assert "--max-model-len" in out  # and it points at the override
    assert "clamped to model native ceiling" not in out  # nothing to clamp


def test_switch_apply_writes_purpose_machine_env(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_compose, "compose_down", lambda d: _ok())
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: _ok())
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    rc = main(
        [
            "switch",
            "mmangkad/Qwen3.6-27B-NVFP4",
            "--purpose",
            "prompt-heavy",
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
            "--apply",
            "--no-probe",
        ]
    )
    assert rc == 0
    env = tmp_path / ".env"
    assert _env.read_env(env, "VLLM_PURPOSE") == "prompt-heavy"
    assert _env.read_env(env, "VLLM_MACHINE") == "spark"
    assert _env.read_env(env, "VLLM_MAX_NUM_BATCHED_TOKENS") == "16384"


def test_switch_apply_records_tool_probe(tmp_path, monkeypatch, capsys) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_compose, "compose_down", lambda d: _ok())
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: _ok())
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    seen: dict = {}

    def fake_probe(port, served):
        seen["port"], seen["served"] = port, served
        return {"ok": True, "tool_calls": ["finish"], "finish": "tool_calls", "error": None}

    monkeypatch.setattr(_runtime_ops, "probe_tool_calling", fake_probe)
    # nvidia/Qwen3-32B is a non-MTP catalog model, so --apply alone would block on
    # the compose edit; --force restarts (and runs the probe) anyway.
    rc = main(
        [
            "switch",
            "nvidia/Qwen3-32B-NVFP4",
            "--compose-dir",
            str(tmp_path),
            "--apply",
            "--force",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool_call_parser"] == "hermes"  # auto-selected for the dense model
    assert payload["tool_calling"]["ok"] is True
    assert payload["tool_calling"]["tool_calls"] == ["finish"]
    assert seen["served"] == "nvidia/Qwen3-32B-NVFP4"


def test_switch_apply_no_probe_skips(tmp_path, monkeypatch, capsys) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_compose, "compose_down", lambda d: _ok())
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: _ok())
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)

    def boom(*a, **k):  # the probe must not run with --no-probe
        raise AssertionError("probe ran despite --no-probe")

    monkeypatch.setattr(_runtime_ops, "probe_tool_calling", boom)
    rc = main(
        [
            "switch",
            "nvidia/Qwen3-32B-NVFP4",
            "--compose-dir",
            str(tmp_path),
            "--apply",
            "--force",  # non-MTP model: --force restarts past the compose-edit block
            "--no-probe",
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["tool_calling"] is None


def test_switch_apply_non_mtp_blocks_and_force_overrides(tmp_path, monkeypatch, capsys) -> None:
    # The template ships the MTP primary's flags. Switching --apply to a non-MTP
    # model must write .env but NOT recreate the container (the compose file is
    # incompatible until the MTP lines are removed) — otherwise it takes a healthy
    # deployment down and fails to come back. --force overrides.
    _scaffold(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(_compose, "compose_down", lambda d: (calls.append("down"), _ok())[1])
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: (calls.append("up"), _ok())[1])
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    monkeypatch.setattr(_runtime_ops, "probe_tool_calling", lambda *a, **k: None)

    # --apply alone on a non-MTP model: writes .env, blocks the restart.
    rc = main(
        ["switch", "nvidia/Qwen3-32B-NVFP4", "--compose-dir", str(tmp_path), "--apply", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["restarted"] is False
    assert payload["blocked_on_compose_edits"] is True
    assert payload["compose_edits"]  # the REMOVE-these-lines notice
    assert calls == []  # the container was NOT recreated
    assert (
        _env.read_env(tmp_path / ".env", "VLLM_MODEL") == "nvidia/Qwen3-32B-NVFP4"
    )  # .env written

    # --force overrides: now it recreates the container.
    rc = main(
        [
            "switch",
            "nvidia/Qwen3-32B-NVFP4",
            "--compose-dir",
            str(tmp_path),
            "--apply",
            "--force",
            "--no-probe",
            "--json",
        ]
    )
    assert rc == 0
    assert calls == ["down", "up"]


def test_probe_tool_calling_delegates(monkeypatch) -> None:
    # The CLI wrapper builds the local URL and forwards to assess.probe_tool_calls
    # (which owns the never-raises contract); it just passes the result through.
    captured: dict = {}

    def fake(url, model):
        captured["url"], captured["model"] = url, model
        return {"ok": True, "tool_calls": ["finish"], "finish": None, "error": None}

    monkeypatch.setattr(_runtime_ops.assess, "probe_tool_calls", fake)
    result = _runtime_ops.probe_tool_calling(8001, "foo/bar")
    assert result["ok"] is True
    assert captured == {"url": "http://localhost:8001", "model": "foo/bar"}


def test_switch_apply_surfaces_compose_failure(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(
        _compose,
        "compose_down",
        lambda d: types.SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == EXIT_ENV_ERROR


# --- switch --task (embed/score) ------------------------------------------


def test_switch_embed_model_auto_detects_task_json(tmp_path, capsys) -> None:
    # Qwen3-Embedding-0.6B is an embed catalog entry; switch must auto-detect
    # task=embed, set VLLM_TASK, apply the tiny-KV defaults, skip the tool-call
    # parser, skip quantization (empty catalog value), and add the compose notice.
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "Qwen/Qwen3-Embedding-0.6B",
            "--machine",
            "spark",
            "--json",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    env = payload["env"]
    # Task must be set in the plan.
    assert env.get("VLLM_TASK") == "embed"
    # Tiny-KV defaults for a pooling model.
    assert env.get("VLLM_MAX_MODEL_LEN") == "8192"
    assert float(env.get("VLLM_GPU_MEM_UTIL", 0)) == pytest.approx(0.06)
    # No tool-call parser (pooling model).
    assert "VLLM_TOOL_CALL_PARSER" not in env
    assert payload.get("tool_call_parser") is None
    # No quantization (catalog entry has quantization="").
    assert "VLLM_QUANTIZATION" not in env
    # Probe is suppressed for embed/score.
    assert payload["probe"] is False
    # Compose edit notice must give the pooling serve flags for this vLLM build
    # (--runner pooling --convert embed) plus the Matryoshka --hf-overrides.
    compose_edits = payload["compose_edits"]
    assert any("--runner=pooling" in n for n in compose_edits), compose_edits
    assert any("--convert=embed" in n for n in compose_edits), compose_edits
    assert any("--hf-overrides=" in n for n in compose_edits), compose_edits
    assert any("is_matryoshka" in n for n in compose_edits), compose_edits


def test_switch_generate_model_unchanged_by_task_feature(tmp_path, capsys) -> None:
    # A normal generate model (mmangkad/Qwen3.6-27B-NVFP4) must be byte-for-byte
    # unaffected: still gets a tool-call parser, no VLLM_TASK, no embed defaults.
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "mmangkad/Qwen3.6-27B-NVFP4",
            "--machine",
            "spark",
            "--json",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    env = payload["env"]
    # Tool-call parser must be resolved (qwen3_coder for this model).
    assert env.get("VLLM_TOOL_CALL_PARSER") == "qwen3_coder"
    assert payload.get("tool_call_parser") == "qwen3_coder"
    # No pooling task.
    assert "VLLM_TASK" not in env
    # Context must be the 256K machine default (the model is 256K-native; no clamp).
    assert env.get("VLLM_MAX_MODEL_LEN") == "262144"
    # GPU util must be the spark machine default (0.6), not the embed/score 0.06.
    assert float(env.get("VLLM_GPU_MEM_UTIL", 0)) == pytest.approx(0.6)
    # Probe enabled (no --no-probe, not an embed/score model).
    assert payload["probe"] is True


def test_switch_embed_explicit_task_flag_wins(tmp_path, capsys) -> None:
    # An explicit --task embed on an unknown model (no catalog entry) must still
    # activate the pooling path: VLLM_TASK set, tiny-KV defaults, no parser.
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "some/unknown-embed-model",
            "--task",
            "embed",
            "--machine",
            "spark",
            "--json",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    env = payload["env"]
    assert env.get("VLLM_TASK") == "embed"
    assert env.get("VLLM_MAX_MODEL_LEN") == "8192"
    assert float(env.get("VLLM_GPU_MEM_UTIL", 0)) == pytest.approx(0.06)
    assert "VLLM_TOOL_CALL_PARSER" not in env
    assert payload["probe"] is False


def test_switch_embed_explicit_overrides_win_over_defaults(tmp_path, capsys) -> None:
    # When --max-model-len and --gpu-mem-util are passed explicitly they must win
    # over the embed/score defaults (8192 / 0.06).
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "Qwen/Qwen3-Embedding-0.6B",
            "--machine",
            "spark",
            "--max-model-len",
            "4096",
            "--gpu-mem-util",
            "0.10",
            "--json",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    env = payload["env"]
    assert env.get("VLLM_MAX_MODEL_LEN") == "4096"
    assert float(env.get("VLLM_GPU_MEM_UTIL", 0)) == pytest.approx(0.10)


def test_switch_embed_skips_probe_even_without_no_probe(tmp_path, capsys) -> None:
    # Without --no-probe a generate model would run the probe; an embed/score
    # model must suppress it regardless.
    _scaffold(tmp_path)
    rc = main(
        [
            "switch",
            "Qwen/Qwen3-Embedding-0.6B",
            "--machine",
            "spark",
            "--compose-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "probe tool calling" not in out
    assert "post-switch probe: skipped" in out
    assert "/v1/embeddings" in out


# --- serve / stop ---------------------------------------------------------


def test_serve_dry_run(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["serve", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_serve_apply(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: (calls.append("up"), _ok())[1])
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    rc = main(["serve", "--compose-dir", str(tmp_path), "--apply", "--no-probe"])
    assert rc == 0
    assert calls == ["up"]


def test_serve_apply_records_tool_probe(tmp_path, monkeypatch, capsys) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: _ok())
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    monkeypatch.setattr(
        _runtime_ops,
        "probe_tool_calling",
        lambda port, served: {"ok": True, "tool_calls": ["finish"], "finish": None, "error": None},
    )
    rc = main(["serve", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["tool_calling"]["ok"] is True


def test_start_is_serve_alias(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["start", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_stop_dry_run(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["stop", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_stop_apply(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(_compose, "compose_down", lambda d: (calls.append("down"), _ok())[1])
    rc = main(["stop", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert calls == ["down"]


# --- status ---------------------------------------------------------------


def test_status_json(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["status", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["container"] == "model-gear-vllm"
    assert payload["state"] == "not created"  # offline _probe → None
    assert payload["health"] == "not responding"  # offline is_healthy → False
    assert payload["model"] == "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
    assert payload["tool_call_parser"] == "qwen3_coder"  # scaffolded default
