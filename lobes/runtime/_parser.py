"""Infer the OpenAI ``--tool-call-parser`` to use for a given model.

vLLM needs a tool-call parser that matches how the served model emits tool
calls. Picking the wrong one silently breaks tool calling (the server returns
200 but no usable ``tool_calls``), so ``lobes switch`` auto-selects one per model
rather than leaving the caller to remember it. The rules below mirror the
guidance in ``templates/env.example`` (the single source of truth):

* Qwen3-Coder / Qwen3.6 checkpoints emit the XML function format → ``qwen3_coder``
* Qwen3 dense models emit Hermes-style JSON tool calls → ``hermes``
* Mistral checkpoints emit the ``[TOOL_CALLS]`` format → ``mistral``
* anything else → ``None`` (unknown; leave the configured parser untouched and
  let the caller pass ``--tool-call-parser`` explicitly)

The Qwen3 markers are deliberately **Qwen3-scoped** — a bare ``coder`` would also
match unrelated checkpoints (``deepseek-coder``, ``codellama``, ``Qwen2.5-Coder``)
and silently misconfigure their parser, so the coder rule requires the ``qwen3``
family. The ``mistral`` marker matches the Mistral family — note the common
``mistralai/`` org prefix contains the substring, so ``mistralai/Mixtral-…`` and
``mistralai/Ministral-…`` resolve to ``mistral`` too (correct: they share the
``[TOOL_CALLS]`` format). A bare ``mixtral``/``ministral`` basename without that
prefix does not match. Anything we haven't validated returns ``None`` (safe).

Pure string matching — no network, no model download. Extend ``_RULES`` to teach
a new family.
"""

from __future__ import annotations

# Ordered (substring-set, parser) rules, matched against the lowercased model id.
# A model matches a rule when it contains ANY of the rule's markers. The Coder /
# Qwen3.6 rule comes first because those ids also contain "qwen3"; its markers are
# Qwen3-scoped so a generic "*-coder" model doesn't get qwen3_coder.
_RULES: list[tuple[tuple[str, ...], str]] = [
    (("qwen3-coder", "qwen3_coder", "qwen3coder", "qwen3.6", "qwen3-6", "qwen3_6"), "qwen3_coder"),
    (("qwen3", "qwen-3"), "hermes"),
    (("mistral",), "mistral"),
]


def infer_parser(model: str) -> str | None:
    """Return the tool-call parser for ``model``, or ``None`` if unknown.

    ``None`` means "I can't tell" — the caller should leave the existing
    ``VLLM_TOOL_CALL_PARSER`` in place and rely on an explicit override.
    """
    name = (model or "").lower()
    for markers, parser in _RULES:
        if any(marker in name for marker in markers):
            return parser
    return None
