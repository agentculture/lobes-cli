"""Dump the structural-tag grammar the cortex tool path builds for a strict tool call (#310 t12).

Runs inside a vllm/vllm-openai image, CPU only, offline. Loads the lobes plugin
(/opt/lobes/qwen3_thinking_tool_parser.py, mounted) over the image's own Qwen3EngineToolParser
and prints what get_structural_tag returns for the failing request shape (strict `finish`
tool, tool_choice auto) with thinking ON and OFF.
"""

import importlib.util
import inspect
import json
import sys

import vllm
from transformers import AutoTokenizer

spec = importlib.util.spec_from_file_location("plug", "/opt/lobes/qwen3_thinking_tool_parser.py")
plug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plug)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser as P  # noqa: E402

cls = [c for c in vars(plug).values() if inspect.isclass(c) and issubclass(c, P) and c is not P][0]
tok = AutoTokenizer.from_pretrained("unsloth/Qwen3.8-27B-NVFP4")
tools = [{"type": "function", "function": {
    "name": "finish", "description": "Finish the task with a summary.", "strict": True,
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                   "required": ["summary"], "additionalProperties": False}}}]
print("vllm", vllm.__version__)
print("parent get_structural_tag:", inspect.signature(P.get_structural_tag))
for thinking in (True, False):
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "Call the finish tool with summary hello."}],
                                tools=tools, tool_choice="auto",
                                chat_template_kwargs={"enable_thinking": thinking})
    parser = cls(tok)
    kw = {"reasoning": False}
    if "strict_level" in inspect.signature(P.get_structural_tag).parameters:
        import vllm.tool_parsers.tool_strict_level as tsl
        kw["strict_level"] = tsl.ToolStrictLevel.AUTO
    try:
        tag = parser.get_structural_tag(req, **kw)
    except Exception as e:  # report, don't hide
        print(f"thinking={thinking}: EXC {type(e).__name__}: {e}")
        continue
    if tag is None:
        print(f"thinking={thinking}: None (no grammar)")
        continue
    js = tag if isinstance(tag, (dict, list, str)) else getattr(tag, "model_dump", lambda: str(tag))()
    s = json.dumps(js, default=str, sort_keys=True) if not isinstance(js, str) else js
    print(f"thinking={thinking}: {len(s)} chars")
    print(s[:3000])
sys.stdout.flush()
