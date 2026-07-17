"""Infer the OpenAI ``--tool-call-parser`` to use for a given model.

vLLM needs a tool-call parser that matches how the served model emits tool
calls. Picking the wrong one silently breaks tool calling (the server returns
200 but no usable ``tool_calls``), so ``lobes switch`` auto-selects one per model
rather than leaving the caller to remember it. The rules below mirror the
guidance in ``templates/env.example`` (the single source of truth):

* Qwen3-Coder / Qwen3.5 / Qwen3.6 checkpoints emit the XML function format → ``qwen3_coder``
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
    (
        (
            "qwen3-coder",
            "qwen3_coder",
            "qwen3coder",
            "qwen3.5",
            "qwen3-5",
            "qwen3_5",
            "qwen3.6",
            "qwen3-6",
            "qwen3_6",
        ),
        "qwen3_coder",
    ),
    (("qwen3", "qwen-3"), "hermes"),
    (("mistral",), "mistral"),
    # Gemma 4 (Google DeepMind) does NOT emit Python-style calls: it emits its own
    # native `<|tool_call>call:name{args}<tool_call|>` syntax, whose delimiters are
    # SPECIAL TOKENS (ids 48/49). vLLM ships a purpose-built parser for it —
    # `gemma4` -> vllm.tool_parsers.gemma4_engine_tool_parser.Gemma4EngineToolParser,
    # which decodes with skip_special_tokens=False so it can see those delimiters.
    #
    # This rule USED to say "pythonic", carrying an explicit caveat that the value
    # was unconfirmed ("Risk r2 (pending #71): confirm against the served
    # checkpoint during live validation"). That live check finally ran against the
    # 31B muse lane on a physical Thor (2026-07-17) and the guess was WRONG:
    # `pythonic` is served with skip_special_tokens=True, so the delimiters are
    # stripped before it ever runs, it matches nothing, and the model's perfectly
    # well-formed call is silently relayed as assistant CONTENT with
    # tool_calls=null / finish_reason="stop" — a caller sees prose shaped like a
    # tool call and no callable one. Evidence:
    # docs/evidence/2026-07-17-accept-muse-tool-calling-thor.txt.
    #
    # Markers stay scoped to "gemma-4"/"gemma4" so they never match older Gemma
    # 1/2/3 checkpoints, whose tool-call story is different and unvalidated here.
    # VALIDATION SCOPE (#108): measured on the 31B (nvidia/Gemma-4-31B-IT-NVFP4).
    # The 12B senses/coder lanes inherit this family rule — the syntax is a Gemma 4
    # family trait and `gemma4_utils` is family-generic, not size-specific — but
    # have NOT been booted against it; they were on a parser proven wrong for the
    # family, so this is a strictly better default, not a validated claim.
    (("gemma-4", "gemma4"), "gemma4"),
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
