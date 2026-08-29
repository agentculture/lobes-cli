"""Tests for :mod:`lobes.variation` — the machine-type variation id resolver.

A deployment variation is identified by machine type/setup, never hostname
(``docs/specs/2026-08-29-deployment-lock-per-box.md``). These tests exercise
the resolver entirely through :func:`lobes.runtime._detect.detect_card`'s
injected fact-gathering functions, so nothing here touches real hardware.
"""

from __future__ import annotations

from lobes import variation
from lobes.runtime import _detect

# Real-world fact sets, mirroring tests/test_detect.py.
THOR_SMI_LINE = "NVIDIA Thor, 11.0"
SPARK_SMI_LINE = "NVIDIA GB10, 12.1"


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


def test_same_machine_type_different_hostnames_resolve_to_same_variation() -> None:
    # Two physically different Sparks — only the hostname differs.
    spark_a = _card(smi_line=SPARK_SMI_LINE, hostname="spark-f8a9")
    spark_b = _card(smi_line=SPARK_SMI_LINE, hostname="dgx-lab-3")

    variation_a = variation.resolve_variation_id(spark_a)
    variation_b = variation.resolve_variation_id(spark_b)

    assert variation_a == variation_b == "spark"


def test_same_machine_type_different_serial_like_hostnames_resolve_to_same_variation() -> None:
    # Two Thors, hostnames that look like distinct serial-numbered assets.
    thor_a = _card(
        smi_line=THOR_SMI_LINE,
        device_tree="NVIDIA Jetson AGX Thor Developer Kit",
        hostname="thor-sn-0001",
    )
    thor_b = _card(
        smi_line=THOR_SMI_LINE,
        device_tree="NVIDIA Jetson AGX Thor Developer Kit",
        hostname="thor-sn-9427",
    )

    assert (
        variation.resolve_variation_id(thor_a) == variation.resolve_variation_id(thor_b) == "thor"
    )


def test_variation_id_never_contains_the_hostname() -> None:
    distinctive_hostname = "this-hostname-must-never-leak-xyz123"
    card = _card(smi_line=SPARK_SMI_LINE, hostname=distinctive_hostname)

    resolved = variation.resolve_variation_id(card)

    assert distinctive_hostname not in resolved
    # Sanity: the fact really was gathered (so the assertion above is
    # meaningful, not vacuous because the probe silently failed).
    assert card.hostname == distinctive_hostname


def test_unrecognised_card_resolves_to_explicit_unknown() -> None:
    card = _card(smi_line="Totally Unknown GPU, 9.9", hostname="anything")

    resolved = variation.resolve_variation_id(card)

    assert resolved == variation.UNKNOWN_VARIATION
    assert resolved == _detect.UNKNOWN
    assert not card.is_known


def test_no_fact_set_at_all_resolves_to_explicit_unknown_not_a_guess() -> None:
    card = _card(smi_line=None, meminfo_gb=None, device_tree=None, hostname=None)

    resolved = variation.resolve_variation_id(card)

    assert resolved == variation.UNKNOWN_VARIATION
