"""Blackwell — RTX PRO 6000 Blackwell (dedicated discrete VRAM).

Matched by the specific ``rtx pro 6000`` / ``6000 pro`` markers — never a bare
``rtx 6000`` (which would false-match the older, non-Blackwell RTX 6000 Ada /
Quadro RTX 6000). A dedicated discrete GPU, so it can reserve memory aggressively
and serve the report's longer context. Still a *configured* estimate — not
measured on this box.
"""

from __future__ import annotations

from ._registry import register
from ._strategy import CardStrategy, DetectionSignature, Knob, MachineDefaults

STRATEGY = register(
    CardStrategy(
        name="blackwell",
        summary="Blackwell 6000 Pro (RTX PRO 6000, dedicated VRAM)",
        signature=DetectionSignature(
            name_markers=("rtx pro 6000", "6000 pro"),
            total_memory_gb=96,
        ),
        defaults=MachineDefaults(
            gpu_mem_util=Knob(0.85, "dedicated discrete GPU — can reserve aggressively"),
            max_model_len=Knob(65536, "dedicated VRAM allows the report's longer context"),
            attention_backend=Knob("flashinfer", "discrete Blackwell: FlashInfer default"),
        ),
        status="configured",
    )
)
