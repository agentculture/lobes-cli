"""Does the cortex strict+thinking structural tag let generation STOP after one tool call? (#310 t12)

CPU only. Same grammar (the nightly's JSON, which both xgrammar versions parse) and the Qwen3.8 tokenizer.
Feeds a complete thinking block plus ONE qwen_xml tool call, then asks the matcher whether EOS is allowed.
"""

import json

import xgrammar as xgr
from transformers import AutoTokenizer

GRAMMAR = {"type": "structural_tag", "format": {"type": "sequence", "elements": [
    {"type": "sequence", "elements": [
        {"type": "tag", "begin": "", "content": {"type": "any_text", "excludes": []}, "end": "</think>"},
        {"type": "const_string", "value": "\n\n"}]},
    {"type": "triggered_tags", "triggers": ["<tool_call>\n<function="], "excludes": ["<think>", "</think>"],
     "at_least_one": False, "stop_after_first": False,
     "tags": [{"type": "tag", "begin": "<tool_call>\n<function=finish>\n",
               "content": {"type": "json_schema", "style": "qwen_xml", "any_order": False,
                           "json_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                                           "required": ["summary"], "additionalProperties": False}},
               "end": "\n</function>\n</tool_call>"}]}]}}

tok = AutoTokenizer.from_pretrained("unsloth/Qwen3.8-27B-NVFP4")
info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=len(tok))
compiled = xgr.GrammarCompiler(info).compile_structural_tag(json.dumps(GRAMMAR))
text = ("\n\n<tool_call>\n<function=finish>\n<parameter=summary>\nhello\n"
        "</parameter>\n</function>\n</tool_call>")
ids = tok.encode(text, add_special_tokens=False)
m = xgr.GrammarMatcher(compiled)
accepted = sum(1 for t in ids if m.accept_token(t))
eos = [t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if isinstance(t, int)]
bitmask = xgr.allocate_token_bitmask(1, info.vocab_size)
m.fill_next_token_bitmask(bitmask)
import torch  # noqa: E402
allowed = lambda t: bool((bitmask[0, t // 32] >> (t % 32)) & 1)  # noqa: E731
print("xgrammar", xgr.__version__ if hasattr(xgr, "__version__") else "?",
      f"| accepted {accepted}/{len(ids)} tokens | terminated={m.is_terminated()} | EOS allowed after one call:",
      {tok.convert_ids_to_tokens(t): allowed(t) for t in eos},
      "| another '<tool_call>' allowed:", allowed(tok.encode("<tool_call>", add_special_tokens=False)[0]))
