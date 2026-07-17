"""Tests for the qwen3_coder_thinking tool-parser plugin wiring on vllm-primary
(t2 of the devague plan "fleet template + init wiring for the think-aware
tool-parser plugin"; the plugin ITSELF is
``lobes/vllm_plugins/qwen3_thinking_tool_parser.py``, task t1's scope).

vllm-primary (the cortex/main generate lane) gains three things in
``lobes/templates/fleet/docker-compose.yml``:

1. ``PRIMARY_TOOL_CALL_PARSER`` defaults to ``qwen3_coder_thinking`` instead
   of the upstream ``qwen3_coder`` — a reasoning-aware variant registered by
   the plugin (upstream hardcodes ``reasoning=False`` on every emitted tool
   call, which breaks strict structural tags for a thinking model that must
   stay reasoning-aware across a tool turn).
2. ``--tool-parser-plugin=/opt/lobes/qwen3_thinking_tool_parser.py`` loads
   that plugin file.
3. A read-only volume mount lands the file inside the container at that path
   (``lobes init`` materialises it next to ``docker-compose.yml`` — see
   ``tests/test_init.py``'s "tool-parser plugin materialisation" section and
   ``lobes.runtime._compose.write_plugin_file``).

This is scoped to vllm-primary ONLY — every other fleet-compose service
(vllm-multimodal, vllm-multimodal-coder, vllm-embed, vllm-rerank, vllm-minor,
vllm-middle, gateway) must be byte-for-byte unchanged, proven below with a
sha256 hash of each service's sorted YAML subtree. These hashes were captured
from the SAME edit that added the vllm-primary changes above (t2's diff
touched only vllm-primary, so the non-primary services' rendering is
identical whether captured before or after) — if a FUTURE change to one of
those services is deliberate, recompute with:

    uv run python -c "
    import hashlib, yaml
    from pathlib import Path
    d = yaml.safe_load(Path('lobes/templates/fleet/docker-compose.yml').read_text())
    for name, svc in sorted(d['services'].items()):
        if name == 'vllm-primary':
            continue
        text = yaml.safe_dump(svc, sort_keys=True, default_flow_style=False)
        print(f'{name!r}: {hashlib.sha256(text.encode()).hexdigest()!r},')
    "

and paste the output into ``_EXPECTED_NON_PRIMARY_HASHES`` below.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"

_PLUGIN_DEST_PATH = "/opt/lobes/qwen3_thinking_tool_parser.py"
_PLUGIN_PARSER_NAME = "qwen3_coder_thinking"

_EXPECTED_NON_PRIMARY_HASHES = {
    # Recomputed when the muse role landed: the gateway service's environment:
    # block deliberately gained the MUSE_* passthroughs (BASE_URL/SERVED_NAME/
    # MAX_MODEL_LEN/FEASIBLE + the three MUSE_PEER_* channels), and the new
    # profile-gated vllm-muse service joined the fleet template. (Prior
    # recompute, t7 #127/#115: the inbound-auth pair + *_PEER_PROXY /
    # *_PEER_API_KEY knobs.) Every other service is byte-identical.
    "gateway": "aeb3c299060117abbdd5f2d4e26c0b5b5fd7c61ce5a8002b17fe687d40f76707",
    "vllm-embed": "63db52dc1121c1b861b5559c03d1b2c76699af86a575718908306f2440bd4b85",
    "vllm-middle": "efef630842164793e43313fff2b588b92d7f57aad35fffc941a3617cddc1a129",
    "vllm-minor": "ddca0c0c64eb06514ba23d5327f61ce410bf8de40d3d7f519c399c6b8c60bc01",
    "vllm-multimodal": "a809b5e4fce759646a63f1cfcb9221e3b3fbfafe8cfeed8dbe161ce22ff9f8fc",
    "vllm-multimodal-coder": "460f000fdd12eddbe4e6011b3a519acc66a22b44650a4b1bfe614aa92c6c6e93",
    "vllm-muse": "92b59e090a3db3a28eaf29b0d170dfb41393ae78b889fdc4cd639803c6b468cf",
    "vllm-rerank": "5929a5e6732c459ccd765ee629e04c8b32e1cc5fedf634e4cce2075d6ba49914",
}


def _load_fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


def _service_hash(service: dict) -> str:
    """A stable sha256 over a service's sorted YAML subtree."""
    text = yaml.safe_dump(service, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestVllmPrimaryToolParserPlugin:
    def test_tool_call_parser_default_is_thinking_aware(self) -> None:
        command = _load_fleet()["services"]["vllm-primary"]["command"]
        assert f"--tool-call-parser=${{PRIMARY_TOOL_CALL_PARSER:-{_PLUGIN_PARSER_NAME}}}" in command

    def test_tool_call_parser_still_env_overridable(self) -> None:
        # Same knob as before (PRIMARY_TOOL_CALL_PARSER), just a new default —
        # an operator can still override it in .env, e.g. back to plain
        # upstream qwen3_coder for a non-thinking primary.
        command = _load_fleet()["services"]["vllm-primary"]["command"]
        assert any(c.startswith("--tool-call-parser=${PRIMARY_TOOL_CALL_PARSER:-") for c in command)

    def test_tool_parser_plugin_flag_present(self) -> None:
        command = _load_fleet()["services"]["vllm-primary"]["command"]
        assert f"--tool-parser-plugin={_PLUGIN_DEST_PATH}" in command

    def test_auto_tool_choice_still_enabled(self) -> None:
        # The plugin wiring must not have dropped the existing flag it sits next to.
        command = _load_fleet()["services"]["vllm-primary"]["command"]
        assert "--enable-auto-tool-choice" in command

    def test_plugin_file_mounted_read_only(self) -> None:
        volumes = _load_fleet()["services"]["vllm-primary"]["volumes"]
        assert f"./qwen3_thinking_tool_parser.py:{_PLUGIN_DEST_PATH}:ro" in volumes


class TestOtherServicesUntouched:
    """Byte-precise proof that t2 touched ONLY vllm-primary's command/volumes."""

    def test_service_set_is_exactly_primary_plus_the_expected_others(self) -> None:
        services = _load_fleet()["services"]
        assert set(services) == {"vllm-primary"} | set(_EXPECTED_NON_PRIMARY_HASHES)

    def test_every_non_primary_service_hash_unchanged(self) -> None:
        services = _load_fleet()["services"]
        for name, expected in _EXPECTED_NON_PRIMARY_HASHES.items():
            actual = _service_hash(services[name])
            assert actual == expected, (
                f"{name!r} changed — t2 must touch ONLY vllm-primary's "
                "command/volumes; see this module's docstring to recompute "
                "hashes if the change is deliberate."
            )

    def test_no_other_service_mentions_the_tool_parser_plugin(self) -> None:
        compose = _load_fleet()
        for name, svc in compose["services"].items():
            if name == "vllm-primary":
                continue
            text = yaml.safe_dump(svc)
            assert _PLUGIN_PARSER_NAME not in text, name
            assert "tool-parser-plugin" not in text, name
            assert "qwen3_thinking_tool_parser" not in text, name

    def test_legacy_single_model_template_is_untouched(self) -> None:
        # The legacy (--single) template is a SEPARATE file this task must not
        # touch at all — no plugin flag, no plugin mount, no thinking-aware
        # parser default (it stays on plain upstream qwen3_coder).
        single_compose = _TEMPLATES / "docker-compose.yml"
        text = single_compose.read_text(encoding="utf-8")
        assert _PLUGIN_PARSER_NAME not in text
        assert "tool-parser-plugin" not in text
        assert "qwen3_thinking_tool_parser" not in text
