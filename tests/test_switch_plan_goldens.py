"""``tests/goldens/switch-plans.txt`` — the catalog's rendering surface, pinned.

The profile/shape goldens (``tests/test_profile_goldens.py``,
``tests/test_shape_goldens.py``) pin what a *card* renders to. Nothing pinned
what a *catalog entry* renders to — yet ``lobes switch`` resolves the served
``VLLM_*`` env, the auto-selected tool-call parser
(:func:`lobes.runtime._parser.infer_parser`), the catalog quantization and the
compose-edit notices straight out of :data:`lobes.catalog.SUPPORTED_MODELS`.

That gap mattered the moment the catalog grew an ENGINE axis
(qwen3-8-gguf-llamacpp plan t3): a llama.cpp-served GGUF gear had to become
declarable *without* moving a single byte of what any vLLM gear renders to.
This golden makes that a diff instead of an eyeball claim — adding a gear
appends exactly one block and leaves every other block byte-identical.

Regenerate deliberately, never reflexively::

    uv run python tests/goldens/regen.py

and read the diff: a block you did not mean to touch is precisely the signal
this file exists to catch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lobes.catalog import SUPPORTED_MODELS
from tests.goldens import regen

_GOLDEN = Path(__file__).resolve().parent / "goldens" / "switch-plans.txt"


def test_switch_plans_golden_is_byte_for_byte() -> None:
    assert _GOLDEN.is_file(), f"missing golden {_GOLDEN} — run tests/goldens/regen.py"
    assert regen.switch_plan_text() == _GOLDEN.read_text(encoding="utf-8"), (
        "switch-plan rendering drifted from tests/goldens/switch-plans.txt — "
        "regenerate with `uv run python tests/goldens/regen.py` and review the diff"
    )


def test_every_catalogued_gear_has_a_block() -> None:
    # The golden is only a guard if it covers the WHOLE catalog: a gear that
    # renders nothing here could change its rendering unnoticed.
    text = _GOLDEN.read_text(encoding="utf-8")
    for model in SUPPORTED_MODELS:
        assert f"### {model.id}\n" in text, f"{model.id}: no block in the switch-plan golden"
    assert text.count("### ") == len(SUPPORTED_MODELS)


def test_switch_plan_rendering_is_pure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same purity bar as the profile goldens: no host state, no GPU, no
    # subprocess. An explicit --machine is what keeps `_resolve_machine_name`
    # from shelling out to nvidia-smi, so a regression that ignored it would
    # make this golden host-dependent (and CI-flaky) rather than catalog-pure.
    def _boom(*args: object, **kwargs: object) -> None:  # pragma: no cover - guard
        raise AssertionError("switch-plan rendering must not shell out")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    assert regen.switch_plan_text() == regen.switch_plan_text()


def test_golden_records_the_inferred_parser_for_every_vllm_gear() -> None:
    # The "msg:" lines carry infer_parser's answer verbatim, so the golden is
    # ALSO the infer_parser pin the plan's acceptance criterion names. A vLLM
    # generate gear must show an auto-selected parser; nothing else may.
    text = _GOLDEN.read_text(encoding="utf-8")
    blocks = dict(_blocks(text))
    for model in SUPPORTED_MODELS:
        block = blocks[model.id]
        if model.task == "generate" and model.tool_parser:
            expected = f"  msg: tool-call parser (auto-selected): {model.tool_parser}"
            assert expected in block, f"{model.id}: golden does not pin {model.tool_parser!r}"


def _blocks(text: str) -> list[tuple[str, str]]:
    """Split the golden into ``(model_id, block_text)`` pairs."""
    pairs: list[tuple[str, str]] = []
    current_id: str | None = None
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("### "):
            if current_id is not None:
                pairs.append((current_id, "\n".join(current)))
            current_id = line[4:]
            current = []
        else:
            current.append(line)
    if current_id is not None:
        pairs.append((current_id, "\n".join(current)))
    return pairs
