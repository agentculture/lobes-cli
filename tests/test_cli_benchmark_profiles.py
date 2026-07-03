"""Tests for ``lobes benchmark --profile`` — issue #81, task t9.

Covers the CLI wiring around :mod:`lobes.bench.compare`: argument parsing,
JSON/text output shape, read-only behaviour, and graceful offline
degradation. Network-shape correctness (the probes themselves) is covered by
``tests/test_bench_compare.py`` and ``tests/test_cli_measure.py`` — these
tests focus on the CLI glue.

No live model is ever required: "reachable" tests talk to a tiny canned local
HTTP server; "unreachable" tests talk to a closed OS port (instant
connection-refused, no real timeout wait).
"""

from __future__ import annotations

import argparse
import http.server
import json
import socket
import threading
import time

import pytest

import lobes.cli._commands.benchmark as benchmark_cmd
from lobes import roles_measure as RM
from lobes.bench.compare import PROFILE_NAMES
from lobes.cli import main
from lobes.runtime import _compose

# ---------------------------------------------------------------------------
# A canned local HTTP server for the reachable path (chat-completions shape).
# ---------------------------------------------------------------------------


class _LLMHandler(http.server.BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics":
            body = b""
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        time.sleep(0.005)
        if self.path == "/v1/chat/completions":
            self._write_json(
                200,
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
            )
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        pass


@pytest.fixture()
def llm_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LLMHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


def _closed_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _scaffold_fleet(path):
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


# ---------------------------------------------------------------------------
# register(): --profile / --timeout flags
# ---------------------------------------------------------------------------


def test_register_profile_flag_choices() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    benchmark_cmd.register(sub)

    for name in (*PROFILE_NAMES, "all"):
        args = parser.parse_args(["benchmark", "--profile", name])
        assert args.profile == name

    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "--profile", "not-a-real-profile"])


def test_register_profile_defaults_to_none() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    benchmark_cmd.register(sub)
    args = parser.parse_args(["benchmark"])
    assert args.profile is None


def test_register_timeout_flag_parseable() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    benchmark_cmd.register(sub)
    args = parser.parse_args(["benchmark", "--profile", "cortex-only", "--timeout", "3.5"])
    assert args.timeout == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Graceful offline degradation (acceptance 4) through the full CLI
# ---------------------------------------------------------------------------


def test_cli_profile_unreachable_backend_no_exception_marks_unavailable(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    port = _closed_port()
    rc = main(
        [
            "benchmark",
            "--profile",
            "cortex-only",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    result = payload["cortex-only"]
    assert result["available"] is False
    assert result["reason"]
    assert result["columns"]["cortex"]["ready"] is False


def test_cli_profile_qwen_variant_unavailable_by_default_catalog(tmp_path, capsys) -> None:
    """Today's real catalog has no bf16 27B Qwen -> reported unavailable, never fabricated."""
    _scaffold_fleet(tmp_path)
    port = _closed_port()
    rc = main(
        [
            "benchmark",
            "--profile",
            "qwen-nvfp4-vs-bf16",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    result = payload["qwen-nvfp4-vs-bf16"]
    assert result["available"] is False
    assert "catalog" in result["reason"].lower()
    assert result["columns"] == {}


# ---------------------------------------------------------------------------
# Reachable path: cortex+senses side by side, comparable metric keys
# ---------------------------------------------------------------------------


def test_cli_profile_cortex_plus_senses_reachable_json(tmp_path, llm_server, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(
        [
            "benchmark",
            "--profile",
            "cortex+senses",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(llm_server),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    result = payload["cortex+senses"]
    assert result["available"] is True
    assert set(result["columns"]) == {"cortex", "senses"}
    cortex_keys = set(result["columns"]["cortex"]["metrics"])
    senses_keys = set(result["columns"]["senses"]["metrics"])
    assert cortex_keys == senses_keys  # comparable across columns


def test_cli_profile_all_json_returns_all_four_profiles(tmp_path, llm_server, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(
        [
            "benchmark",
            "--profile",
            "all",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(llm_server),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(PROFILE_NAMES)
    # cortex-only / cortex+senses / senses-direct reachable via the canned server.
    for name in ("cortex-only", "cortex+senses", "senses-direct"):
        assert payload[name]["available"] is True
    # the catalog-gated profile stays unavailable regardless of reachability.
    assert payload["qwen-nvfp4-vs-bf16"]["available"] is False


def test_cli_profile_text_output_renders_markdown_blocks(tmp_path, llm_server, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(
        [
            "benchmark",
            "--profile",
            "all",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(llm_server),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    for name in PROFILE_NAMES:
        assert name in out
    assert "unavailable" in out.lower()  # the qwen profile's note


# ---------------------------------------------------------------------------
# RUNTIME-ONLY vocabulary (acceptance 3) through the CLI
# ---------------------------------------------------------------------------


def test_cli_profile_metric_keys_are_runtime_only(tmp_path, llm_server, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(
        [
            "benchmark",
            "--profile",
            "all",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(llm_server),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    for result in payload.values():
        for column in result["columns"].values():
            assert set(column["metrics"]) <= RM.ALLOWED_METRIC_KEYS


# ---------------------------------------------------------------------------
# Read-only: --profile never touches docker/compose
# ---------------------------------------------------------------------------


def test_cli_profile_never_touches_docker(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)
    port = _closed_port()

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("--profile must never invoke docker/compose")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    monkeypatch.setattr(_compose, "compose_down", boom)
    monkeypatch.setattr(_compose, "_run", boom)
    monkeypatch.setattr(_compose, "_probe", boom)
    rc = main(
        [
            "benchmark",
            "--profile",
            "all",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--json",
        ]
    )
    assert rc == 0


def test_cli_benchmark_has_no_apply_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["benchmark", "--apply"])
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Regression: plain (no --profile) namespaces used elsewhere still work
# ---------------------------------------------------------------------------


def test_cmd_benchmark_ignores_absent_profile_attribute(monkeypatch) -> None:
    """A Namespace with no 'profile' attribute at all must not raise (getattr default)."""
    import types

    monkeypatch.setattr(
        benchmark_cmd._assess,
        "run_benchmark",
        lambda *a, **kw: {
            "model": "m",
            "endpoint": "e",
            "max_model_len": 0,
            "purpose": "balanced",
            "input_len": 100,
            "output_len": 100,
            "decode_rates": [50.0],
            "prefill": {"prompt_tokens": 100, "seconds": 0.5},
        },
    )
    monkeypatch.setattr(benchmark_cmd._assess, "render_benchmark", lambda r: "## stub")
    args = types.SimpleNamespace(
        json=False,
        port=8000,
        compose_dir=None,
        model="test-model",
        purpose=None,
        input_len=None,
        output_len=None,
        runs=2,
        all_lobes=False,
        # no 'profile' attribute at all
    )
    rc = benchmark_cmd.cmd_benchmark(args)
    assert rc == 0
