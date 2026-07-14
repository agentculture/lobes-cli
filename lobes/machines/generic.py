"""Generic — the conservative fallback for an unknown Blackwell-class box.

Empty detection markers on purpose: it is never auto-matched (an empty signature
never matches), only ever chosen explicitly or as the legacy silent fallback of
:func:`lobes.profiles.detect_machine`. Registered last.
"""

from __future__ import annotations

from ._registry import register
from ._strategy import CardStrategy, DetectionSignature, Knob, MachineDefaults

STRATEGY = register(
    CardStrategy(
        name="generic",
        summary="unknown Blackwell-class box — conservative fallback",
        signature=DetectionSignature(name_markers=()),  # never auto-matched
        defaults=MachineDefaults(
            gpu_mem_util=Knob(0.6, "unknown box: conservative default"),
            max_model_len=Knob(32768, "unknown box: conservative context"),
            attention_backend=Knob("flashinfer", "unknown Blackwell-class: FlashInfer default"),
        ),
        status="configured",
    )
)
