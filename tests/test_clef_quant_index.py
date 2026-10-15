"""quantize.py: a single-file build must ship model.safetensors.index.json.

Without an index, vLLM loads EVERY *.safetensors in the model dir — including Clef's
joint_head.safetensors — and refuses to boot ('no module or parameter named
evidence_layers in Qwen3_5ForEmbedding'; measured on the Orin, 2026-10-14).
"""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "clef_quantize", ROOT / "scripts" / "clef_quant" / "quantize.py"
)
quantize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quantize)


def _fake_safetensors(path: Path, tensors: dict[str, int]) -> None:
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, nbytes in tensors.items():
        header[name] = {"dtype": "U8", "shape": [nbytes], "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def test_single_file_build_gets_an_index_naming_only_the_backbone(tmp_path):
    _fake_safetensors(tmp_path / "model.safetensors", {"model.a": 4, "lm_head.weight": 6})
    _fake_safetensors(tmp_path / "joint_head.safetensors", {"evidence_layers.0.w": 8})
    quantize.write_weight_index(tmp_path)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {
        "lm_head.weight": "model.safetensors",
        "model.a": "model.safetensors",
    }
    assert index["metadata"]["total_size"] == 10


def test_an_existing_index_is_left_alone(tmp_path):
    _fake_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"model.a": 4})
    existing = {
        "metadata": {"total_size": 4},
        "weight_map": {"model.a": "model-00001-of-00002.safetensors"},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(existing))
    quantize.write_weight_index(tmp_path)
    assert json.loads((tmp_path / "model.safetensors.index.json").read_text()) == existing
