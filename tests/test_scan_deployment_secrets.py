"""Tests for scripts/scan_deployment_secrets.py — the CI secret gate over
every committed deployment artifact (t3,
docs/plans/2026-08-29-deployment-lock-per-box.md).

Criterion 3 of t3 is the load-bearing one: prove the gate can actually
fail. These tests build a fixture tree twice — once clean, once with a
planted, obviously-fake token in a committed docker-compose.override.yml —
and assert the scanner fails on the planted tree and passes on the clean
one. No real credential appears anywhere in this file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "scan_deployment_secrets.py"
_SPEC = importlib.util.spec_from_file_location("scan_deployment_secrets", _SCRIPT_PATH)
scan_deployment_secrets = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scan_deployment_secrets
_SPEC.loader.exec_module(scan_deployment_secrets)  # type: ignore[union-attr]

scan_paths = scan_deployment_secrets.scan_paths
main = scan_deployment_secrets.main
DEFAULT_SCAN_GLOBS = scan_deployment_secrets.DEFAULT_SCAN_GLOBS

# An obviously-fake token that still matches the detection shape (a
# non-empty value assigned to a known secret key name). Never a real
# credential.
_FAKE_TOKEN = "sk-fake-not-a-real-token-0123456789abcdef"


def _write_clean_tree(root: Path) -> Path:
    box = root / "deployments" / "spark-box"
    box.mkdir(parents=True)

    (box / "deployment.lock.toml").write_text(
        "\n".join(
            [
                "[cortex]",
                'model = "unsloth/Qwen3.8-27B-NVFP4"',
                "max_model_len = 262144",
                "gpu_mem_util = 0.58",
                "",
            ]
        )
    )

    (box / "docker-compose.yml").write_text(
        "\n".join(
            [
                "services:",
                "  vllm-primary:",
                "    environment:",
                "      - GATEWAY_API_KEY=${GATEWAY_API_KEY}",
                "      - HF_TOKEN=${HF_TOKEN:-}",
                "      - PRIMARY_PEER_ORIGIN=${PRIMARY_PEER_ORIGIN}",
                "",
            ]
        )
    )

    (box / "docker-compose.override.yml").write_text(
        "\n".join(
            [
                "# operator-authored override — no inline secrets here",
                "services:",
                "  vllm-primary:",
                '    command: ["--speculative-config", \'{"method": "dspark"}\']',
                "",
            ]
        )
    )

    (box / "Dockerfile.vllm-primary").write_text(
        "\n".join(
            [
                "FROM vllm/vllm-openai:nightly",
                "ARG HF_TOKEN",
                "",
            ]
        )
    )

    return box


def test_clean_tree_passes(tmp_path: Path) -> None:
    _write_clean_tree(tmp_path)

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert findings == []
    assert main(["--root", str(tmp_path)]) == 0


def test_planted_token_in_override_fails(tmp_path: Path) -> None:
    box = _write_clean_tree(tmp_path)

    override = box / "docker-compose.override.yml"
    override.write_text(
        "\n".join(
            [
                "services:",
                "  vllm-primary:",
                "    environment:",
                f"      - GATEWAY_API_KEY={_FAKE_TOKEN}",
                "",
            ]
        )
    )

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.path == override
    assert finding.key == "GATEWAY_API_KEY"
    assert finding.value == _FAKE_TOKEN

    assert main(["--root", str(tmp_path)]) == 1


def test_planted_peer_api_key_fails(tmp_path: Path) -> None:
    box = _write_clean_tree(tmp_path)

    (box / "docker-compose.override.yml").write_text(
        "\n".join(
            [
                "services:",
                "  gateway:",
                "    environment:",
                f"      - WORKER_PEER_API_KEY={_FAKE_TOKEN}",
                "",
            ]
        )
    )

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [f.key for f in findings] == ["WORKER_PEER_API_KEY"]


def test_planted_hf_token_in_dockerfile_fails(tmp_path: Path) -> None:
    box = _write_clean_tree(tmp_path)

    (box / "Dockerfile.vllm-primary").write_text(
        "\n".join(
            [
                "FROM vllm/vllm-openai:nightly",
                f"ENV HF_TOKEN={_FAKE_TOKEN}",
                "",
            ]
        )
    )

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [f.key for f in findings] == ["HF_TOKEN"]


def test_planted_token_in_lock_fails(tmp_path: Path) -> None:
    box = _write_clean_tree(tmp_path)

    (box / "deployment.lock.toml").write_text(
        "\n".join(
            [
                "[cortex]",
                'model = "unsloth/Qwen3.8-27B-NVFP4"',
                f'GATEWAY_API_KEY = "{_FAKE_TOKEN}"',
                "",
            ]
        )
    )

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [f.key for f in findings] == ["GATEWAY_API_KEY"]


def test_peer_origin_is_treated_as_sensitive(tmp_path: Path) -> None:
    """Peer origins are internal information per operator decision (see
    CLAUDE.md's proxy-lobes section) and covered by the same suffix rule
    as the *_PEER_API_KEY family, so a hardcoded origin also fails."""
    box = _write_clean_tree(tmp_path)

    (box / "docker-compose.override.yml").write_text(
        "\n".join(
            [
                "services:",
                "  gateway:",
                "    environment:",
                "      - PRIMARY_PEER_ORIGIN=http://10.0.0.7:8000",
                "",
            ]
        )
    )

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [f.key for f in findings] == ["PRIMARY_PEER_ORIGIN"]


def test_non_deployment_files_are_not_scanned(tmp_path: Path) -> None:
    """The path list names the lock and the verbatim-committed
    compose/Dockerfiles under deployments/<box>/ specifically — a
    secret-shaped value sitting in an unrelated file elsewhere in the tree
    (e.g. a top-level README) is out of this gate's declared scope."""
    _write_clean_tree(tmp_path)

    (tmp_path / "README.md").write_text(f"GATEWAY_API_KEY={_FAKE_TOKEN}\n")

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert findings == []


def test_default_globs_name_the_lock_and_every_artifact_kind() -> None:
    """Criterion 2: the scanner's path list explicitly names the lock and
    every verbatim-committed compose/Dockerfile kind, not a generic
    repository-default scan."""
    joined = " ".join(DEFAULT_SCAN_GLOBS)

    assert "deployment.lock.toml" in joined
    assert "docker-compose" in joined
    assert "Dockerfile" in joined
    assert all(pattern.startswith("deployments/") for pattern in DEFAULT_SCAN_GLOBS)


def test_real_repo_tree_is_clean() -> None:
    """Run the actual scanner over this repo's current tree as an extra
    regression guard — should stay clean since deployments/ does not yet
    exist (t6/t9 land it later) and no other committed file matches the
    glob list."""
    repo_root = Path(__file__).resolve().parents[1]

    findings = scan_paths(repo_root, DEFAULT_SCAN_GLOBS)

    assert findings == [], [f.render(repo_root) for f in findings]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "${GATEWAY_API_KEY}",
        "${GATEWAY_API_KEY:-}",
        "${GATEWAY_API_KEY-}",
    ],
)
def test_template_placeholders_are_not_flagged(tmp_path: Path, value: str) -> None:
    box = tmp_path / "deployments" / "spark-box"
    box.mkdir(parents=True)
    (box / "docker-compose.yml").write_text(f"GATEWAY_API_KEY={value}\n")

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert findings == []


# --- Defect 1 (PR #223 review): interpolation defaults bypassed the scan ----
#
# ``${VAR:-literal}`` and ``${VAR-literal}`` are NOT template placeholders.
# Docker Compose substitutes the literal fallback whenever the variable is
# absent (``:-`` also when it is set-but-empty), so the committed file really
# does carry the credential at rest. Only a bare reference and an EXPLICITLY
# EMPTY default are safe.


@pytest.mark.parametrize(
    "value",
    [
        "${GATEWAY_API_KEY:-" + _FAKE_TOKEN + "}",
        "${GATEWAY_API_KEY-" + _FAKE_TOKEN + "}",
    ],
)
def test_interpolation_default_carrying_a_secret_is_flagged(tmp_path: Path, value: str) -> None:
    box = tmp_path / "deployments" / "spark-box"
    box.mkdir(parents=True)
    compose = box / "docker-compose.yml"
    compose.write_text(
        "\n".join(
            [
                "services:",
                "  gateway:",
                "    environment:",
                f"      - GATEWAY_API_KEY={value}",
                "",
            ]
        )
    )

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [f.key for f in findings] == ["GATEWAY_API_KEY"]
    assert findings[0].value == value
    assert main(["--root", str(tmp_path)]) == 1


def test_interpolation_default_in_a_dockerfile_is_flagged(tmp_path: Path) -> None:
    box = tmp_path / "deployments" / "spark-box"
    box.mkdir(parents=True)
    (box / "Dockerfile.vllm-primary").write_text(
        "\n".join(
            [
                "FROM vllm/vllm-openai:nightly",
                "ENV HF_TOKEN=${HF_TOKEN:-" + _FAKE_TOKEN + "}",
                "",
            ]
        )
    )

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [f.key for f in findings] == ["HF_TOKEN"]


@pytest.mark.parametrize("value", ["${HF_TOKEN}", "${HF_TOKEN:-}", "${HF_TOKEN-}"])
def test_dockerfile_safe_placeholders_are_not_flagged(tmp_path: Path, value: str) -> None:
    box = tmp_path / "deployments" / "spark-box"
    box.mkdir(parents=True)
    (box / "Dockerfile.vllm-primary").write_text(
        "\n".join(
            [
                "FROM vllm/vllm-openai:nightly",
                f"ENV HF_TOKEN {value}",
                f"ARG BUILD_HF_TOKEN={value}",
                "",
            ]
        )
    )

    assert scan_paths(tmp_path, DEFAULT_SCAN_GLOBS) == []


# --- Defect 2 (PR #223 review): Dockerfile whitespace assignment bypassed ---
#
# ``ENV KEY value`` and ``ARG KEY value`` are valid Dockerfile syntax with no
# ``=`` at all, so the YAML/env assignment regex skipped the line entirely.


def _dockerfile(root: Path, *lines: str) -> Path:
    box = root / "deployments" / "spark-box"
    box.mkdir(parents=True, exist_ok=True)
    path = box / "Dockerfile.vllm-primary"
    path.write_text("\n".join(("FROM vllm/vllm-openai:nightly", *lines, "")))
    return path


def test_dockerfile_env_whitespace_form_is_flagged(tmp_path: Path) -> None:
    path = _dockerfile(tmp_path, f"ENV HF_TOKEN {_FAKE_TOKEN}")

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert len(findings) == 1
    assert findings[0].path == path
    assert findings[0].key == "HF_TOKEN"
    assert findings[0].value == _FAKE_TOKEN
    assert findings[0].line_no == 2
    assert main(["--root", str(tmp_path)]) == 1


def test_dockerfile_arg_whitespace_form_is_flagged(tmp_path: Path) -> None:
    _dockerfile(tmp_path, f"ARG GATEWAY_API_KEY {_FAKE_TOKEN}")

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [(f.key, f.value) for f in findings] == [("GATEWAY_API_KEY", _FAKE_TOKEN)]


def test_dockerfile_lowercase_directive_is_flagged(tmp_path: Path) -> None:
    """Dockerfile instructions are case-insensitive in practice."""
    _dockerfile(tmp_path, f"env HF_TOKEN {_FAKE_TOKEN}", f"Arg WORKER_PEER_API_KEY={_FAKE_TOKEN}")

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [f.key for f in findings] == ["HF_TOKEN", "WORKER_PEER_API_KEY"]


def test_dockerfile_multi_variable_env_flags_the_secret_only(tmp_path: Path) -> None:
    _dockerfile(tmp_path, f"ENV VLLM_LOGGING_LEVEL=INFO HF_TOKEN={_FAKE_TOKEN}")

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [(f.key, f.value) for f in findings] == [("HF_TOKEN", _FAKE_TOKEN)]


def test_dockerfile_multi_variable_env_flags_every_secret(tmp_path: Path) -> None:
    _dockerfile(tmp_path, f"ENV HF_TOKEN={_FAKE_TOKEN} GATEWAY_API_KEY={_FAKE_TOKEN}")

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [f.key for f in findings] == ["HF_TOKEN", "GATEWAY_API_KEY"]


def test_dockerfile_quoted_whitespace_value_is_flagged(tmp_path: Path) -> None:
    _dockerfile(tmp_path, f'ENV HF_TOKEN "{_FAKE_TOKEN}"')

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [(f.key, f.value) for f in findings] == [("HF_TOKEN", _FAKE_TOKEN)]


def test_dockerfile_whitespace_value_keeps_embedded_spaces(tmp_path: Path) -> None:
    """The whitespace form takes the whole remainder as one value."""
    _dockerfile(tmp_path, "ENV HF_TOKEN fake token with spaces")

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [(f.key, f.value) for f in findings] == [("HF_TOKEN", "fake token with spaces")]


def test_dockerfile_line_continuation_is_flagged_at_the_first_line(tmp_path: Path) -> None:
    _dockerfile(
        tmp_path,
        "ENV VLLM_LOGGING_LEVEL=INFO \\",
        "    # a comment inside the continuation",
        f"    HF_TOKEN={_FAKE_TOKEN}",
    )

    findings = scan_paths(tmp_path, DEFAULT_SCAN_GLOBS)

    assert [(f.key, f.line_no) for f in findings] == [("HF_TOKEN", 2)]


def test_dockerfile_bare_arg_and_comments_stay_clean(tmp_path: Path) -> None:
    _dockerfile(
        tmp_path,
        "# ENV HF_TOKEN " + _FAKE_TOKEN,
        "ARG HF_TOKEN",
        "ARG GATEWAY_API_KEY=",
        "ENV VLLM_LOGGING_LEVEL=INFO",
        "RUN echo HF_TOKEN not-an-assignment",
        "LABEL org.opencontainers.image.title=lobes",
    )

    assert scan_paths(tmp_path, DEFAULT_SCAN_GLOBS) == []
