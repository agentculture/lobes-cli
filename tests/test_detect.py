"""Tests for host card detection (``lobes.runtime._detect``).

Detection is exercised entirely through injected fact-gathering functions so no
test touches real hardware, real ``/proc`` files, or spawns a real ``nvidia-smi``
subprocess. Real-world fact sets (the literal nvidia-smi CSV lines this box and
a GB10 report) are asserted to resolve to the right registered card; anything
unrecognized must resolve to :data:`lobes.runtime._detect.UNKNOWN`, never a
guessed fallback.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from lobes.runtime import _detect

# ---------------------------------------------------------------------------
# Real-world fact sets (from the task brief / this box)
# ---------------------------------------------------------------------------

# 'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader' on Thor (sm_110).
THOR_SMI_LINE = "NVIDIA Thor, 11.0"
# The GB10 (DGX Spark), sm_121.
SPARK_SMI_LINE = "NVIDIA GB10, 12.1"

THOR_DEVICE_TREE_MODEL_RAW = "NVIDIA Jetson AGX Thor Developer Kit\x00"
# device_tree_fn is injected post-parse (as _read_device_tree_model itself would
# return it, NUL already stripped) — the NUL-stripping test below exercises the
# raw reader directly.
THOR_DEVICE_TREE_MODEL = "NVIDIA Jetson AGX Thor Developer Kit"


def _card(
    smi_line: str | None = None,
    meminfo_gb: float | None = 128.0,
    device_tree: str | None = None,
    hostname: str | None = "host",
) -> _detect.DetectedCard:
    return _detect.detect_card(
        nvidia_smi_fn=lambda: smi_line,
        meminfo_fn=lambda: meminfo_gb,
        device_tree_fn=lambda: device_tree,
        hostname_fn=lambda: hostname,
    )


# ---------------------------------------------------------------------------
# Resolution: real-world fact sets
# ---------------------------------------------------------------------------


def test_thor_fact_set_resolves_to_thor() -> None:
    card = _card(smi_line=THOR_SMI_LINE, device_tree=THOR_DEVICE_TREE_MODEL)
    assert card.resolved == "thor"
    assert card.is_known
    assert card.device_name == "NVIDIA Thor"
    assert card.compute_capability == "sm_110"


def test_gb10_fact_set_resolves_to_spark_not_blackwell() -> None:
    # The GB10 is a Grace *Blackwell* part but must resolve to spark, matching
    # the registry's detection-precedence contract (spark before blackwell).
    card = _card(smi_line=SPARK_SMI_LINE)
    assert card.resolved == "spark"
    assert card.is_known
    assert card.device_name == "NVIDIA GB10"
    assert card.compute_capability == "sm_121"


def test_thor_and_gb10_are_distinguished_by_compute_capability_too() -> None:
    thor = _card(smi_line=THOR_SMI_LINE)
    spark = _card(smi_line=SPARK_SMI_LINE)
    assert thor.compute_capability == "sm_110"
    assert spark.compute_capability == "sm_121"
    assert thor.resolved != spark.resolved


def test_unrecognized_fact_set_yields_unknown_never_a_guess() -> None:
    card = _card(smi_line="NVIDIA H100, 9.0", device_tree=None, hostname="build-box")
    assert card.resolved == _detect.UNKNOWN
    assert not card.is_known
    # raw facts are still surfaced even though resolution is UNKNOWN
    assert card.device_name == "NVIDIA H100"
    assert card.compute_capability == "sm_90"


def test_all_probes_failing_yields_unknown_with_no_facts() -> None:
    card = _detect.detect_card(
        nvidia_smi_fn=lambda: None,
        meminfo_fn=lambda: None,
        device_tree_fn=lambda: None,
        hostname_fn=lambda: None,
    )
    assert card.resolved == _detect.UNKNOWN
    assert card.device_name is None
    assert card.compute_capability is None
    assert card.total_memory_gb is None
    assert card.hostname is None
    assert card.sources["device_name"] == "unavailable"
    assert card.sources["total_memory_gb"] == "unavailable"


def test_device_tree_model_is_a_fallback_when_nvidia_smi_is_unavailable() -> None:
    # Headless-ish path: no nvidia-smi line at all, but the device tree still
    # names the board — device_tree_fn is the only name-ish signal available.
    card = _card(smi_line=None, device_tree=THOR_DEVICE_TREE_MODEL)
    assert card.resolved == "thor"
    assert card.device_name is None  # nvidia-smi contributed nothing
    assert card.device_tree_model == "NVIDIA Jetson AGX Thor Developer Kit"


def test_hostname_alone_can_resolve_a_card() -> None:
    # DetectionSignature.matches() checks the hostname too; a host named after
    # its card should resolve even with no GPU probe data at all.
    card = _card(smi_line=None, device_tree=None, hostname="my-thor-01")
    assert card.resolved == "thor"


# ---------------------------------------------------------------------------
# Total memory: /proc/meminfo, never nvidia-smi
# ---------------------------------------------------------------------------


def test_total_memory_comes_from_meminfo_not_nvidia_smi() -> None:
    card = _card(smi_line=THOR_SMI_LINE, meminfo_gb=126.5)
    assert card.total_memory_gb == 126.5
    assert card.sources["total_memory_gb"] == "/proc/meminfo"


def test_meminfo_probe_failure_degrades_to_none_not_a_raise() -> None:
    card = _card(smi_line=THOR_SMI_LINE, meminfo_gb=None)
    assert card.total_memory_gb is None
    assert card.sources["total_memory_gb"] == "unavailable"


def test_read_total_memory_gb_parses_real_meminfo_format(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       132131328 kB\nMemFree:        32768000 kB\nSwapTotal: 0 kB\n"
    )
    gb = _detect._read_total_memory_gb(meminfo)
    assert gb == pytest.approx(126.0, abs=0.1)


def test_read_total_memory_gb_missing_file_returns_none(tmp_path: Path) -> None:
    assert _detect._read_total_memory_gb(tmp_path / "does-not-exist") is None


# ---------------------------------------------------------------------------
# HARD CONSTRAINT: nvidia-smi is never asked for memory fields
# ---------------------------------------------------------------------------


def test_nvidia_smi_query_never_requests_memory_fields() -> None:
    argv = _detect._nvidia_smi_argv()
    query = " ".join(argv)
    assert "memory" not in query.lower()
    assert "memory.used" not in query
    assert "memory.total" not in query
    # and it does ask for the two fields detection actually needs
    assert "name" in query
    assert "compute_cap" in query


def test_run_nvidia_smi_never_raises_on_missing_binary() -> None:
    # Simulate the binary not being on PATH.
    def _boom(*args, **kwargs):
        raise FileNotFoundError("no such file: nvidia-smi")

    import unittest.mock as mock

    with mock.patch("subprocess.run", side_effect=_boom):
        assert _detect._run_nvidia_smi() is None


def test_run_nvidia_smi_never_raises_on_timeout() -> None:
    import unittest.mock as mock

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

    with mock.patch("subprocess.run", side_effect=_timeout):
        assert _detect._run_nvidia_smi() is None


def test_run_nvidia_smi_nonzero_exit_returns_none() -> None:
    import unittest.mock as mock

    result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="oops")
    with mock.patch("subprocess.run", return_value=result):
        assert _detect._run_nvidia_smi() is None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cap_raw,expected",
    [
        ("11.0", "sm_110"),
        ("12.1", "sm_121"),
        ("9.0", "sm_90"),
        ("not-a-number", None),
        ("", None),
    ],
)
def test_compute_cap_to_sm(cap_raw: str, expected: str | None) -> None:
    assert _detect._compute_cap_to_sm(cap_raw) == expected


@pytest.mark.parametrize(
    "line,expected",
    [
        ("NVIDIA Thor, 11.0", ("NVIDIA Thor", "sm_110")),
        ("NVIDIA GB10, 12.1", ("NVIDIA GB10", "sm_121")),
        (None, (None, None)),
        ("", (None, None)),
        ("just-a-name", ("just-a-name", None)),
    ],
)
def test_parse_nvidia_smi_line(line, expected) -> None:
    assert _detect._parse_nvidia_smi_line(line) == expected


def test_device_tree_model_strips_trailing_nul(tmp_path: Path) -> None:
    path = tmp_path / "model"
    path.write_bytes(b"NVIDIA Jetson AGX Thor Developer Kit\x00")
    assert _detect._read_device_tree_model(path) == "NVIDIA Jetson AGX Thor Developer Kit"


def test_device_tree_model_missing_on_non_device_tree_hosts(tmp_path: Path) -> None:
    assert _detect._read_device_tree_model(tmp_path / "no-such-model") is None


# ---------------------------------------------------------------------------
# No torch import — stdlib only
# ---------------------------------------------------------------------------


def test_module_does_not_import_torch() -> None:
    source = Path(_detect.__file__).read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported


# ---------------------------------------------------------------------------
# DetectedCard shape
# ---------------------------------------------------------------------------


def test_detected_card_is_frozen_and_carries_sources() -> None:
    card = _card(smi_line=THOR_SMI_LINE)
    with pytest.raises(AttributeError):
        card.resolved = "spark"  # type: ignore[misc]
    assert set(card.sources) == {
        "device_name",
        "compute_capability",
        "total_memory_gb",
        "device_tree_model",
        "hostname",
    }
