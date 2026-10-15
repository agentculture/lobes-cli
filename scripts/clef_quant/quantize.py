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


PACK_FORMAT = {"w4a16-g64": "pack-quantized", "nvfp4": "nvfp4-pack-quantized"}
INDEX = "model.safetensors.index.json"


def write_weight_index(out: Path) -> None:
    """Give a single-file build an index that names only the backbone shard.

    Without one, vLLM loads EVERY *.safetensors in the dir, joint_head.safetensors
    included, and refuses to boot ('no module or parameter named evidence_layers';
    measured on the Orin, 2026-10-14). A multi-shard save already has its own index.
    """
    import struct

    single = out / "model.safetensors"
    if (out / INDEX).exists() or not single.exists():
        return
    with open(single, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__", None)
    total = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in header.values())
    (out / INDEX).write_text(
        json.dumps(
            {
                "metadata": {"total_size": total},
                "weight_map": {k: single.name for k in sorted(header)},
            },
            indent=2,
        )
    )


def compress_and_finish(model, scheme: str, src: Path, out: Path, stats: dict, t0: float) -> None:
    """Pack a FROZEN (GPTQ-done) model with compressed-tensors and write the release dir."""
    from compressed_tensors.compressors import ModelCompressor
    from transformers import AutoProcessor

    comp = ModelCompressor.from_pretrained_model(model, quantization_format=PACK_FORMAT[scheme])
    comp.compress_model(model)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    comp.update_config(out)
    write_weight_index(out)
    AutoProcessor.from_pretrained(src).save_pretrained(out)
    for f in src.iterdir():
        if (
            f.is_file()
            and not f.name.startswith(SKIP_COPY)
            # never copy source WEIGHT shards (any naming); the head's own file is the exception
            and not (".safetensors" in f.name and f.name != "joint_head.safetensors")
            and not (out / f.name).exists()
        ):
            shutil.copy2(f, out / f.name)
    versions = {
        p: md.version(p)
        for p in ("llmcompressor", "compressed-tensors", "transformers", "torch", "accelerate")
    }
    (out / "quantization_provenance.json").write_text(
        json.dumps(
            {
                "source": "Cloudflare/clef",
                "source_revision": "2f3de3dd85f379784083b0814d997ab627200f0c",
                "scheme": scheme,
                "pack_format": PACK_FORMAT[scheme],
                "keep_bf16": KEEP_W4A16 if scheme == "w4a16-g64" else KEEP_NVFP4,
                "calibration": {"dataset": "culture-dev/clef-calibration-set", **stats},
                "versions": versions,
                "script": "lobes-cli scripts/clef_quant/quantize.py",
                "wall_s": round(time.time() - t0, 1),
            },
            indent=1,
        )
    )
    print(json.dumps({"event": "saved", "out": str(out), "versions": versions}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=("w4a16-g64", "nvfp4"))
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--calib", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=4096)
    # On unified-memory boxes (DGX Spark GB10) the default "cpu" offload of cached
    # activations HUNG in a pinned non_blocking H2D copy (2026-10-14, stage 49/65,
    # one CPU thread at 100% for ~1 h); keeping them on the GPU avoids the copy.
    ap.add_argument("--offload-device", default="cuda:0")
    # Recovery: pack an existing FROZEN checkpoint in a fresh process (fresh CUDA context).
    ap.add_argument("--compress-only", type=Path, help="FROZEN checkpoint dir to pack")
    a = ap.parse_args()

    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    t0 = time.time()
    frozen = a.out.with_name(a.out.name + "-frozen")
    if a.compress_only:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            a.compress_only, dtype=torch.bfloat16
        )
        stats = json.loads((a.compress_only / "calibration_stats.json").read_text())
        compress_and_finish(model, a.scheme, a.src, a.out, stats, t0)
        return 0

    from llmcompressor import oneshot
    from llmcompressor.utils import load_context

    ds, _, stats = calibration_dataset(a.src, a.calib, a.max_tokens)
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
    # 2026-10-14: the in-process pack died with cudaErrorNotPermitted AFTER a 40-min GPTQ run
    # and lost everything. Save the FROZEN model first (GPTQ scales + fake-quant weights), so
    # a failed pack is recovered with --compress-only in a fresh process, never re-quantized.
    model.save_pretrained(frozen, save_compressed=False)
    (frozen / "calibration_stats.json").write_text(json.dumps(stats))
    print(
        json.dumps({"event": "frozen_saved", "dir": str(frozen), "s": round(time.time() - t0, 1)}),
        flush=True,
    )
    try:
        compress_and_finish(model, a.scheme, a.src, a.out, stats, t0)
    except Exception as exc:  # noqa: BLE001 — the frozen checkpoint is the recovery path
        print(
            json.dumps(
                {
                    "event": "pack_failed",
                    "error": repr(exc)[:300],
                    "recover": f"--compress-only {frozen}",
                }
            ),
            flush=True,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
