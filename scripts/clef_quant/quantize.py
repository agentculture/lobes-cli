# flake8: noqa: E501
"""Quantize Cloudflare/clef with llm-compressor (plan orin-decision-models, t19 = Orin, t20 = Blackwell).

Runs INSIDE a container that has llmcompressor 0.14.0 (+ its pinned transformers /
compressed-tensors / accelerate). Not part of the lobes package.

    python quantize.py --scheme w4a16-g64 --src /src --calib /sets/calibration.jsonl --out /out/clef-W4A16-g64
    python quantize.py --scheme nvfp4     --src /src --calib /sets/calibration.jsonl --out /out/clef-NVFP4

Calibration inputs are the EXACT token sequences Clef sees: each calibration record
(t18, culture-dev/clef-calibration-set) is encoded with Cloudflare's own
``joint_schema_model.encode_record`` (state + typed questions + option spans), so the
backbone is calibrated on decision prompts, not generic chat text. Records longer than
``--max-tokens`` are dropped, never truncated (truncation would cut off the questions).

Keep-lists (stay bf16):
- w4a16-g64 (Orin, Marlin): lm_head, embed_tokens, the vision tower, and the
  GatedDeltaNet in_proj_a / in_proj_b (tiny; mirrors llm-compressor's own Qwen3.8-27B
  W4A16 example). Norms, conv1d, A_log, dt_bias are not Linear layers, so they are
  never quantized. INT4, symmetric, group 64 (GPTQ).
- nvfp4 (Blackwell, W4A4): lm_head, embed_tokens, the vision tower, and EVERY
  linear_attn projection — llm-compressor's own Qwen3.5 NVFP4 recipe documents that
  the fused GatedDeltaNet projections are incompatible with NVFP4.
The joint schema head is a separate file (joint_head.safetensors) and is copied
unchanged in bf16.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

KEEP_W4A16 = [
    "lm_head",
    "re:.*embed_tokens$",
    "re:.*visual.*",
    r"re:.*linear_attn\.in_proj_a$",
    r"re:.*linear_attn\.in_proj_b$",
]
KEEP_NVFP4 = ["lm_head", "re:.*embed_tokens$", "re:.*visual.*", "re:.*linear_attn.*"]
# Everything except the weight shards is copied from the source (head, its code, tokenizer, LICENSE, ...).
SKIP_COPY = ("model-", "model.safetensors.index.json", "config.json", ".cache", ".gitattributes")


def _load_clef_code(src: Path):
    spec = importlib.util.spec_from_file_location(
        "joint_schema_model", src / "joint_schema_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = mod  # its dataclasses resolve via sys.modules
    spec.loader.exec_module(mod)
    return mod


def calibration_dataset(src: Path, calib: Path, max_tokens: int):
    from datasets import Dataset
    from transformers import AutoProcessor

    clef = _load_clef_code(src)
    processor = AutoProcessor.from_pretrained(src)
    ids, dropped = [], 0
    for line in calib.read_text().splitlines():
        record = json.loads(line)["record"]
        enc = clef.encode_record(processor.tokenizer, record, max_length=16384, processor=processor)
        if len(enc.input_ids) > max_tokens:
            dropped += 1
            continue
        ids.append(list(enc.input_ids))
    ds = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})
    stats = {
        "records_used": len(ids),
        "records_dropped_over_max_tokens": dropped,
        "max_tokens": max_tokens,
        "total_tokens": sum(map(len, ids)),
    }
    return ds, processor, stats


def recipe_for(scheme: str):
    if scheme == "w4a16-g64":
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
        from llmcompressor.modifiers.gptq import GPTQModifier

        weights = QuantizationArgs(
            num_bits=4, type="int", symmetric=True, strategy="group", group_size=64
        )
        return GPTQModifier(
            config_groups={"group_0": QuantizationScheme(targets=["Linear"], weights=weights)},
            ignore=KEEP_W4A16,
        )
    if scheme == "nvfp4":
        from llmcompressor.modifiers.quantization import QuantizationModifier

        return QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=KEEP_NVFP4)
    raise SystemExit(f"unknown scheme {scheme}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=("w4a16-g64", "nvfp4"))
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=4096)
    # On unified-memory boxes (DGX Spark GB10) the default "cpu" offload of cached
    # activations HUNG in a pinned non_blocking H2D copy (2026-10-14, stage 49/65,
    # one CPU thread at 100% for ~1 h); keeping them on the GPU avoids the copy.
    ap.add_argument("--offload-device", default="cuda:0")
    a = ap.parse_args()

    import torch
    from llmcompressor import oneshot
    from llmcompressor.utils import load_context
    from transformers import Qwen3_5ForConditionalGeneration

    t0 = time.time()
    ds, processor, stats = calibration_dataset(a.src, a.calib, a.max_tokens)
    print(json.dumps({"event": "calibration", **stats}), flush=True)
    with load_context(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.src, dtype=torch.bfloat16)
    print(json.dumps({"event": "loaded", "s": round(time.time() - t0, 1)}), flush=True)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe_for(a.scheme),
        max_seq_length=a.max_tokens,
        num_calibration_samples=len(ds),
        sequential_offload_device=a.offload_device,
    )
    print(json.dumps({"event": "quantized", "s": round(time.time() - t0, 1)}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out, save_compressed=True)
    processor.save_pretrained(a.out)
    for f in a.src.iterdir():
        if f.is_file() and not f.name.startswith(SKIP_COPY) and not (a.out / f.name).exists():
            shutil.copy2(f, a.out / f.name)
    versions = {
        p: md.version(p)
        for p in ("llmcompressor", "compressed-tensors", "transformers", "torch", "accelerate")
    }
    (a.out / "quantization_provenance.json").write_text(
        json.dumps(
            {
                "source": "Cloudflare/clef",
                "source_revision": "2f3de3dd85f379784083b0814d997ab627200f0c",
                "scheme": a.scheme,
                "keep_bf16": KEEP_W4A16 if a.scheme == "w4a16-g64" else KEEP_NVFP4,
                "calibration": {"dataset": "culture-dev/clef-calibration-set", **stats},
                "versions": versions,
                "script": "lobes-cli scripts/clef_quant/quantize.py",
                "wall_s": round(time.time() - t0, 1),
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                "event": "saved",
                "out": str(a.out),
                "versions": versions,
                "s": round(time.time() - t0, 1),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
