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
    # Recomputed for the opt-in `embed-deep` gear, on top of the 2026-07-17 Gemma 4
    # parser-pair correction. THREE deliberate stories are folded in here — this
    # recompute rebased the newest onto the other two, so all three matter:
    #
    #  * `embed-deep` (newest): the GATEWAY service's environment: block gained
    #    EMBED_DEEP_BASE_URL (empty default) + EMBED_DEEP_SERVED_NAME, and the
    #    profile-gated vllm-embed-deep service joined the set. vllm-embed itself is
    #    BYTE-IDENTICAL — the deep slot is a second gear beside the 0.6B, never an
    #    edit to it.
    #  * the three GEMMA lanes (vllm-multimodal, vllm-multimodal-coder,
    #    vllm-muse) moved from `--tool-call-parser=pythonic` to the `=gemma4`
    #    PAIR (tool + reasoning). Pythonic was a never-validated guess (its own
    #    comment said so) and the live 31B run disproved it — Gemma 4 emits
    #    `<|tool_call>call:name{...}` with special-token delimiters pythonic
    #    cannot see, so tool calls leaked out as plain content; and the tool
    #    parser alone then leaks `<|channel>` markers into content without its
    #    paired reasoning parser.
    #  * the GATEWAY service's environment: block gained the STT_/TTS_ FEASIBLE
    #    + PEER_ORIGIN/PEER_PROXY/PEER_API_KEY passthroughs (#129).
    #
    # (Prior recomputes: the muse role's MUSE_* passthroughs + the
    # profile-gated vllm-muse service; before that, t7 #127/#115's inbound-auth
    # pair + *_PEER_PROXY / *_PEER_API_KEY knobs.) Every other service is
    # byte-identical — this tripwire firing on exactly the intended services, and
    # NOTHING else, is itself the proof of each change's blast radius.
    "gateway": "3a029e22a7fbc7628215216a157caf434a133ecec0d249cc8801ff63a8c3157d",
    "vllm-embed": "63db52dc1121c1b861b5559c03d1b2c76699af86a575718908306f2440bd4b85",
    "vllm-embed-deep": "532b5b24c76c6cb90d06a4336ec42e6cc856a18ee112186aeff1141403f1143e",
    "vllm-middle": "efef630842164793e43313fff2b588b92d7f57aad35fffc941a3617cddc1a129",
    "vllm-minor": "ddca0c0c64eb06514ba23d5327f61ce410bf8de40d3d7f519c399c6b8c60bc01",
    "vllm-multimodal": "31cd10820f2411c6401a97ba84c54603ecde5434b5a7be6e309390047d847e11",
    "vllm-multimodal-coder": "f871a7d1aaac4a66eea8804c3ae4d9b4db1703bbaf1973b58a5ad2de5f7020e6",
    "vllm-muse": "6d61fb34b4ec56dfe7400021c23a41d61a0cc584d0e191df3d17f8de2bdaa2ae",
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


class TestGemma4ParserPair:
    """Gemma 4 lanes must wire the tool parser and the reasoning parser TOGETHER.

    vLLM ships Gemma 4 support as a matched pair, and half of it is worse than a
    clean miss (both halves measured live on the 31B muse lane, 2026-07-17 —
    docs/evidence/2026-07-17-accept-muse-tool-calling-thor.txt):

    * WITHOUT `--tool-call-parser=gemma4` (the old `pythonic` default): Gemma 4's
      `<|tool_call>call:name{...}<tool_call|>` delimiters are special tokens that
      pythonic — served with skip_special_tokens=True — never sees. It matches
      nothing, and the model's well-formed call is relayed as assistant CONTENT
      with tool_calls=null. Tool calling is silently, totally broken.
    * WITHOUT `--reasoning-parser=gemma4`: the tool parser forces
      skip_special_tokens=False (that is how it sees <|tool_call>), which also
      exposes Gemma's `<|channel>thought` markers. The tool parser does not strip
      those, so they leak into `content`.

    So neither flag is independently correct on a Gemma lane; this pins them as a
    unit, per-service, rather than as a global substring count.
    """

    GEMMA_SERVICES = ("vllm-multimodal", "vllm-multimodal-coder", "vllm-muse")

    def test_every_gemma_lane_wires_both_halves_of_the_pair(self) -> None:
        services = _load_fleet()["services"]
        for name in self.GEMMA_SERVICES:
            command = services[name]["command"]
            assert "--tool-call-parser=gemma4" in command, f"{name}: missing tool parser"
            assert "--reasoning-parser=gemma4" in command, f"{name}: missing reasoning parser"

    def test_no_gemma_lane_still_uses_the_disproven_pythonic_parser(self) -> None:
        """`pythonic` was a guess that a live 31B run disproved. It must not come
        back on any Gemma lane — the failure it causes is silent, so nothing else
        in CI would notice."""
        services = _load_fleet()["services"]
        for name in self.GEMMA_SERVICES:
            assert "--tool-call-parser=pythonic" not in services[name]["command"], (
                f"{name}: pythonic cannot parse Gemma 4's special-token tool-call "
                "delimiters — see docs/gemma-4-31b-nvfp4.md#tool-calling"
            )
