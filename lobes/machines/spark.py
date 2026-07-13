"""Spark — DGX Spark (GB10 Grace Blackwell, 128 GB unified, usually shared).

The load-tested primary target: the 256K-native MTP primary serves at the full
256K on the shared GB10 at a conservative util 0.6. Registered *before*
``blackwell`` so the GB10 (itself a Grace *Blackwell* part) is matched by its
specific ``gb10`` marker and never trips the discrete-Blackwell profile.
"""

from __future__ import annotations

from ._registry import register
from ._strategy import CardStrategy, DetectionSignature, Knob, MachineDefaults

STRATEGY = register(
    CardStrategy(
        name="spark",
        summary="DGX Spark (GB10 Grace Blackwell, 128 GB unified, usually shared)",
        signature=DetectionSignature(
            name_markers=("gb10", "dgx spark", "spark"),
            total_memory_gb=128,
        ),
        defaults=MachineDefaults(
            gpu_mem_util=Knob(0.6, "shared GB10: conservative — co-resides with other mesh agents"),
            max_model_len=Knob(
                262144, "load-tested 2026-06-03: 256K-native primary served at full 256K"
            ),
            attention_backend=Knob("flashinfer", "GB10 Blackwell: FlashInfer validated"),
        ),
        status="load-tested",
    )
)
