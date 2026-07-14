"""Tests for the think-aware vLLM tool-parser plugin (issue: strict tool
calling 500s on the Qwen3.6 thinking model — see
``lobes/vllm_plugins/qwen3_thinking_tool_parser.py`` for the full story).

Two modules under test:

- :mod:`lobes.vllm_plugins._thinking` — pure stdlib helper, no vllm
  dependency, imported and tested directly.
- :mod:`lobes.vllm_plugins.qwen3_thinking_tool_parser` — imports ``vllm`` at
  module scope. The real package is never installed in this offline CI
  environment, so it is exercised here by injecting fake ``vllm`` modules
  into ``sys.modules`` before import, then removing them afterwards (fixture).
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types

import pytest

from lobes.vllm_plugins._thinking import effective_reasoning

PLUGIN_MODULE_NAME = "lobes.vllm_plugins.qwen3_thinking_tool_parser"
_STUB_MODULE_NAMES = (
    "vllm",
    "vllm.tool_parsers",
    "vllm.tool_parsers.qwen3_engine_tool_parser",
)


# ---------------------------------------------------------------------------
# effective_reasoning — pure helper, no vllm involved
# ---------------------------------------------------------------------------


class _AttrRequest:
    """Attribute-style stand-in for vLLM's ChatCompletionRequest."""

    def __init__(self, chat_template_kwargs=None) -> None:
        self.chat_template_kwargs = chat_template_kwargs


def test_effective_reasoning_defaults_true_when_kwargs_absent_attr() -> None:
    request = object()  # no chat_template_kwargs attribute at all
    assert effective_reasoning(request) is True


def test_effective_reasoning_defaults_true_when_kwargs_none_attr() -> None:
    assert effective_reasoning(_AttrRequest(chat_template_kwargs=None)) is True


def test_effective_reasoning_defaults_true_when_key_absent_attr() -> None:
    # Non-empty dict (truthy) but no `enable_thinking` key -> still defaults True.
    assert effective_reasoning(_AttrRequest(chat_template_kwargs={"temperature": 0.7})) is True


def test_effective_reasoning_false_when_enable_thinking_false_attr() -> None:
    request = _AttrRequest(chat_template_kwargs={"enable_thinking": False})
    assert effective_reasoning(request) is False


def test_effective_reasoning_true_when_enable_thinking_true_attr() -> None:
    request = _AttrRequest(chat_template_kwargs={"enable_thinking": True})
    assert effective_reasoning(request) is True


def test_effective_reasoning_defaults_true_when_kwargs_absent_dict() -> None:
    assert effective_reasoning({}) is True


def test_effective_reasoning_defaults_true_when_kwargs_none_dict() -> None:
    assert effective_reasoning({"chat_template_kwargs": None}) is True


def test_effective_reasoning_false_when_enable_thinking_false_dict() -> None:
    request = {"chat_template_kwargs": {"enable_thinking": False}}
    assert effective_reasoning(request) is False


def test_effective_reasoning_true_when_enable_thinking_true_dict() -> None:
    request = {"chat_template_kwargs": {"enable_thinking": True}}
    assert effective_reasoning(request) is True


def test_thinking_module_imports_without_vllm_installed() -> None:
    # Re-import from scratch to prove no lingering vllm dependency crept in.
    mod = importlib.import_module("lobes.vllm_plugins._thinking")
    assert callable(mod.effective_reasoning)
    # No vllm module should have been pulled in as a side effect.
    assert "vllm" not in sys.modules


# ---------------------------------------------------------------------------
# qwen3_thinking_tool_parser — exercised via injected vllm stubs
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_vllm_stubs():
    """Ensure no stub/plugin modules leak between tests."""

    def _purge() -> None:
        for name in list(sys.modules):
            if name == PLUGIN_MODULE_NAME or name in _STUB_MODULE_NAMES:
                del sys.modules[name]

    _purge()
    yield
    _purge()


class _FakeToolParserManager:
    """Records registrations the way vLLM's real ToolParserManager would."""

    registered: dict = {}

    @classmethod
    def register_module(cls, name):
        def decorator(klass):
            cls.registered[name] = klass
            return klass

        return decorator


def _install_stub_vllm(get_structural_tag, *, define_class: bool = True) -> _FakeToolParserManager:
    """Inject fake ``vllm``/``vllm.tool_parsers``/``...qwen3_engine_tool_parser``
    modules into ``sys.modules`` and return the fake ToolParserManager class
    used, so callers can inspect ``.registered``.
    """
    vllm_mod = types.ModuleType("vllm")
    tool_parsers_mod = types.ModuleType("vllm.tool_parsers")
    qwen3_engine_mod = types.ModuleType("vllm.tool_parsers.qwen3_engine_tool_parser")

    manager = _FakeToolParserManager
    manager.registered = {}
    tool_parsers_mod.ToolParserManager = manager

    if define_class:
        fake_parent = type(
            "Qwen3EngineToolParser",
            (),
            {
                "structural_tag_model": "qwen_3_coder",
                "get_structural_tag": get_structural_tag,
            },
        )
        qwen3_engine_mod.Qwen3EngineToolParser = fake_parent

    # Standard parent/child package attribute linkage, mirroring how real
    # packages wire submodules onto their parent.
    vllm_mod.tool_parsers = tool_parsers_mod
    tool_parsers_mod.qwen3_engine_tool_parser = qwen3_engine_mod

    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.tool_parsers"] = tool_parsers_mod
    sys.modules["vllm.tool_parsers.qwen3_engine_tool_parser"] = qwen3_engine_mod

    return manager


def test_plugin_registers_under_pinned_name(clean_vllm_stubs) -> None:
    def fake_get_structural_tag(self, request, *, reasoning: bool = False):
        return {"reasoning": reasoning}

    manager = _install_stub_vllm(fake_get_structural_tag)

    mod = importlib.import_module(PLUGIN_MODULE_NAME)

    assert "qwen3_coder_thinking" in manager.registered
    assert manager.registered["qwen3_coder_thinking"] is mod.Qwen3ThinkingToolParser


def test_plugin_forwards_reasoning_false_when_thinking_disabled(clean_vllm_stubs) -> None:
    calls = []

    def fake_get_structural_tag(self, request, *, reasoning: bool = False):
        calls.append(reasoning)
        return {"reasoning": reasoning}

    _install_stub_vllm(fake_get_structural_tag)
    mod = importlib.import_module(PLUGIN_MODULE_NAME)

    parser = mod.Qwen3ThinkingToolParser()
    request = _AttrRequest(chat_template_kwargs={"enable_thinking": False})

    result = parser.get_structural_tag(request, reasoning=False)  # caller's hardcoded False

    assert calls == [False]
    assert result == {"reasoning": False}


def test_plugin_forwards_reasoning_true_when_kwargs_absent(clean_vllm_stubs) -> None:
    calls = []

    def fake_get_structural_tag(self, request, *, reasoning: bool = False):
        calls.append(reasoning)
        return {"reasoning": reasoning}

    _install_stub_vllm(fake_get_structural_tag)
    mod = importlib.import_module(PLUGIN_MODULE_NAME)

    parser = mod.Qwen3ThinkingToolParser()
    request = _AttrRequest(chat_template_kwargs=None)

    # vLLM's abstract_parser always calls with reasoning=False; the override
    # must ignore that and forward the request's effective (True) state.
    result = parser.get_structural_tag(request, reasoning=False)

    assert calls == [True]
    assert result == {"reasoning": True}


def test_import_surface_assert_raises_when_class_missing(clean_vllm_stubs) -> None:
    def fake_get_structural_tag(self, request, *, reasoning: bool = False):
        return reasoning

    _install_stub_vllm(fake_get_structural_tag, define_class=False)

    with pytest.raises(RuntimeError, match="Qwen3EngineToolParser"):
        importlib.import_module(PLUGIN_MODULE_NAME)


def test_import_surface_assert_raises_when_reasoning_not_keyword_only(clean_vllm_stubs) -> None:
    # `reasoning` present but positional-or-keyword, not keyword-only.
    def fake_get_structural_tag(self, request, reasoning=False):
        return reasoning

    _install_stub_vllm(fake_get_structural_tag)

    with pytest.raises(RuntimeError, match="get_structural_tag"):
        importlib.import_module(PLUGIN_MODULE_NAME)


def test_import_surface_assert_raises_when_reasoning_param_missing(clean_vllm_stubs) -> None:
    def fake_get_structural_tag(self, request):
        return None

    _install_stub_vllm(fake_get_structural_tag)

    with pytest.raises(RuntimeError, match="get_structural_tag"):
        importlib.import_module(PLUGIN_MODULE_NAME)


def test_sanity_fake_parent_signature_has_keyword_only_reasoning() -> None:
    # Guards the test fixture itself: the "good" stub really does present a
    # keyword-only `reasoning` param, so a passing suite isn't an accident.
    def fake_get_structural_tag(self, request, *, reasoning: bool = False):
        return reasoning

    sig = inspect.signature(fake_get_structural_tag)
    assert sig.parameters["reasoning"].kind == inspect.Parameter.KEYWORD_ONLY
