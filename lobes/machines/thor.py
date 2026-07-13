"""Thor — Jetson AGX Thor (Blackwell-class, sm_110, 128 GB unified memory).

Load-tested here, and the values contradict the old guess. The pre-refactor
``thor`` row was an unvalidated estimate (``flashinfer`` / 32768 / util 0.6,
status ``configured``); live Thor testing of the fleet shows:

* ``cortex`` (generate lane) runs with ``kv_cache_dtype=auto``;
* ``embedder`` and ``reranker`` (pooling lanes) must use ``TRITON_ATTN`` because
  the FlashInfer/FLASH_ATTN pooling path hangs on sm_110, and the reranker also
  needs ``enforce_eager`` — both inherited from the shared :data:`SM_110` trait,
  not restated here.

The pooling knobs come from the sm_110 *trait* (composed, not copy-pasted); the
generate-lane knob is a Thor-specific ``role_overrides`` entry. For the legacy
single-model surface the context stays a sensible 32K and the attention backend
is the validated ``TRITON_ATTN`` (no longer the contradicted ``flashinfer``).
"""

from __future__ import annotations

from ._registry import register
from ._strategy import CardStrategy, DetectionSignature, Knob, MachineDefaults
from ._traits import SM_110

STRATEGY = register(
    CardStrategy(
        name="thor",
        summary="Jetson AGX Thor (Blackwell-class sm_110, 128 GB unified memory)",
        signature=DetectionSignature(
            name_markers=("thor",),
            compute_capability="sm_110",
            total_memory_gb=128,
        ),
        defaults=MachineDefaults(
            # Sensible single-model legacy values on a 128 GB unified board; the
            # attention backend is the sm_110-validated Triton path, not the
            # contradicted flashinfer guess.
            gpu_mem_util=Knob(0.6, "128 GB unified board: conservative single-model default"),
            max_model_len=Knob(32768, "sensible single-model cap on the unified board"),
            attention_backend=Knob(
                "TRITON_ATTN",
                "sm_110: flashinfer unvalidated/contradicted here — use the exercised Triton path",
            ),
        ),
        status="load-tested",
        # sm_110 pooling quirks (embedder/reranker) come from the shared trait.
        traits=(SM_110,),
        # Thor-specific measured generate-lane knob.
        role_overrides={
            "cortex": {
                "kv_cache_dtype": Knob(
                    "auto",
                    "thor sm_110: measured — cortex generate lane runs kv_cache_dtype=auto",
                ),
            },
        },
    )
)
