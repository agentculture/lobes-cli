"""Causal capability traits shared across boards — knowledge keyed by cause.

A trait captures a hardware fact and the knobs that follow from it, so boards
that share the fact reuse the knobs instead of copy-pasting them. The one live
trait today is :data:`SM_110`: the Blackwell-class sm_110 compute capability
(Jetson Thor, and any future sm_110 card) whose FLASH_ATTN / FlashInfer pooling
path hangs, forcing the two pooling lanes onto Triton — and the reranker
additionally onto eager execution because its CUDA-graph capture is unstable
there. Provenance strings name sm_110 as the cause so a second sm_110 board that
composes this trait inherits both the knobs and the *reason*.
"""

from __future__ import annotations

from ._strategy import Knob, Trait

# The measured sm_110 pooling quirk. Composed by the Thor strategy; reusable by
# any future sm_110 board with a single `traits=(SM_110,)` line — no knob copy.
SM_110 = Trait(
    name="sm_110",
    role_knobs={
        "embedder": {
            "attention_backend": Knob(
                "TRITON_ATTN",
                "sm_110: FLASH_ATTN/FlashInfer pooling path hangs — force TRITON_ATTN",
            ),
        },
        "reranker": {
            "attention_backend": Knob(
                "TRITON_ATTN",
                "sm_110: FLASH_ATTN/FlashInfer pooling path hangs — force TRITON_ATTN",
            ),
            "enforce_eager": Knob(
                True,
                "sm_110: reranker CUDA-graph capture unstable — enforce eager",
            ),
        },
    },
)
