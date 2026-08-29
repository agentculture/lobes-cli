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
vllm-hand, vllm-middle, vllm-muse, vllm-worker, gateway) must be
byte-for-byte unchanged, proven below with a sha256 hash of each service's
sorted YAML subtree. These hashes were captured
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
import shlex
from pathlib import Path

import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"

_PLUGIN_DEST_PATH = "/opt/lobes/qwen3_thinking_tool_parser.py"
_PLUGIN_PARSER_NAME = "qwen3_coder_thinking"

_EXPECTED_NON_PRIMARY_HASHES = {
    # Recomputed 2026-08-10 for the `hand` lobe (hand-lobe plan t6). TWO
    # services moved and both are deliberate:
    #
    #  * `vllm-hand` is NEW — the ninth Colleague role's lane (LFM2.5-1.2B, the
    #    fine-tuning base). DEFAULT-ON, so it carries no `profiles:` gate,
    #    unlike vllm-minor/vllm-muse/vllm-worker: at ~2.4 GiB it co-resides on
    #    every card, which is the point of the role. Served bf16 (NO
    #    --quantization), text-only (NO --language-model-only — there is no ViT
    #    to drop), no --reasoning-parser (no thinking mode), and ARMED for LoRA
    #    with an EMPTY inventory.
    #  * `gateway` gained the HAND_* passthroughs (BASE_URL / SERVED_NAME /
    #    LORA_MODULES). Note the deliberate ABSENCE of HAND_PEER_ORIGIN /
    #    _PEER_PROXY / _PEER_API_KEY, which every other role's block carries:
    #    `hand` is never proxied (lobes.gateway._config.NEVER_PROXIED_BACKENDS)
    #    because it runs on every box, so there is no peer to refer it to.
    #
    # Every other service is byte-identical, which is this tripwire proving the
    # blast radius.
    #
    # Recomputed 2026-08-04 for the senses MTP off-switch (unsloth-QAT-senses plan,
    # t5): ONLY `vllm-multimodal` moved. Its `command:` changed shape from a YAML
    # list to a single shell-lexed STRING so the `--speculative-config` flag can be
    # dropped ENTIRELY from argv via `MULTIMODAL_SPECULATIVE_CONFIG=` in .env — a
    # list item cannot be conditionally omitted (an empty substitution renders as
    # an empty argv element, and `vllm serve` exits 2 on it). The RENDERED argv
    # with the knob unset is byte-identical to before; only the template's
    # representation of it changed, which is why this hash moves and no golden
    # does. See tests/test_senses_speculative_config.py and the lane's own comment
    # block. Every other service — the two other Gemma lanes included — is
    # byte-identical, which is this tripwire proving the blast radius.
    #
    # Recomputed 2026-07-31 for the multimodal-cortex promotion: the GATEWAY
    # service's PRIMARY_SERVED_NAME passthrough default moved from the outgoing
    # sakamakismile/…-Text-NVFP4-MTP to unsloth/Qwen3.6-27B-NVFP4, matching
    # vllm-primary's own --served-model-name default. Leaving them mismatched was
    # a real bug (caught in review): the gateway builds its routing table from
    # PRIMARY_SERVED_NAME, so a deployment whose .env omits the var would rewrite
    # model=cortex|main|hard to an id the engine does not serve — a 404 with a
    # healthy backend behind it, observed live during the 2026-07-31 boot.
    # ONLY `gateway` moved; every other service is byte-identical, which is this
    # tripwire proving the blast radius.
    #
    # Recomputed for the opt-in-core `worker` gear (thor-worker-lobe plan, t4),
    # on top of the `embed-deep` recompute and the 2026-07-17 Gemma 4 parser-pair
    # correction. The newest story rebases onto the earlier ones, so all matter:
    #
    #  * `worker` (newest): the profile-gated vllm-worker service (the eighth
    #    Colleague role — a Qwen3.5 multimodal MoE on the SAME Qwen nightly lane
    #    as the primary, self-draft MTP, served multimodal so NO
    #    --language-model-only) joined the set, and the GATEWAY service's
    #    environment: block gained the WORKER_* passthroughs (BASE_URL /
    #    SERVED_NAME / MAX_MODEL_LEN / FEASIBLE / PEER_ORIGIN / PEER_PROXY /
    #    PEER_API_KEY — all empty defaults, mirroring the muse passthroughs). Only
    #    `gateway` and the new `vllm-worker` move; every other service is
    #    byte-identical, which is the tripwire proving the blast radius.
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
    #
    # Recomputed 2026-08-19 for the qwen3.8-cortex-upgrade plan (t5): the shared
    # vLLM nightly digest bumped (sha256:7c5a10e9... -> sha256:8bd082c2...),
    # which moved every service whose image defaults off VLLM_NIGHTLY_IMAGE
    # (gateway's OPENAI_MODEL/env passthroughs plus vllm-embed, vllm-embed-deep,
    # vllm-hand, vllm-rerank, vllm-worker); the Gemma-lane services
    # (vllm-multimodal, vllm-multimodal-coder, vllm-muse, vllm-middle,
    # vllm-minor) stay on their own image and are unaffected, byte-identical to
    # the prior recompute.
    # 2026-08-20: +HAND feasibility/peer env passthrough (qodo PR #190 #2)
    #
    # Recomputed 2026-08-23 for the qwen3-8-gguf-llamacpp plan (t4): the gateway's
    # hardcoded `PRIMARY_URL=http://vllm-primary:8000` became the overridable
    # `${PRIMARY_URL:-http://vllm-primary:8000}` (same default, so every existing
    # deployment renders the identical value) so a card serving cortex on the
    # llama.cpp lane can point it at `llamacpp-primary` instead. `gateway` is the
    # ONLY pre-existing service that moved; the new `llamacpp-primary` lane is an
    # ADDITION and carries its own entry below.
    #
    # Recomputed 2026-08-25 for the cortex-replica-pool plan (issue #199, t3):
    # `gateway` gained the plural `*_PEER_ORIGINS`/`*_PEER_API_KEYS` replica
    # family, `GATEWAY_SELF_ORIGIN`, and the per-role declared lane
    # fingerprint (`*_QUANTIZATION`/`*_KV_CACHE_DTYPE`/`*_REASONING_PARSER`/
    # `*_TOOL_CALL_PARSER`/`*_SPECULATIVE_CONFIG`) passthroughs, for all nine
    # role prefixes — the config half of the replica pool (t4/t5 add the
    # data-plane use of these). `gateway` is again the ONLY service that
    # moved; every vLLM lane's own command/volumes are untouched.
    #
    # Recomputed 2026-08-25 for the lightning-on-orin plan (t6, the tenth
    # role): `gateway` gained the ASSOCIATE_ prefix on every channel the other
    # nine role prefixes already carry — BASE_URL/SERVED_NAME (wiring),
    # MAX_MODEL_LEN (served-context overlay), FEASIBLE, the four peer channels
    # (singular + plural) and the declared lane fingerprint. `gateway` is again
    # the ONLY pre-existing service that moved; `vllm-associate` (t7) is
    # untouched.
    #
    # Recomputed 2026-08-28 for the capacity-relative-pool-routing plan
    # (issue #199, t12): `gateway` gained the ten `<PREFIX>_MAX_ACTIVE`
    # declared-capacity knobs plus `GATEWAY_CAPACITY_KILL_SWITCH` — all
    # parsed by lobes/gateway/_config.py since t1 but absent from this
    # passthrough, so the capacity signal never reached the container in
    # any real deployment — and, found by the same guard,
    # `GATEWAY_FORCE_STRICT_TOOLS` and `FALLBACK_URL`/`FALLBACK_SERVED_NAME`,
    # which were inert for the same reason. `gateway` is again the ONLY
    # service that moved; no vLLM lane was touched. The new guard against
    # this whole class of gap is tests/test_gateway_env_passthrough_guard.py.
    "gateway": "e54d72e114cf4026f3ba1a516943864c53cacb90f4035af0df8b050797c173d3",
    # The opt-in llama.cpp cortex lane (t4), profile-gated behind `llamacpp` so
    # no existing deployment starts it. Hashed here from the day it landed, so a
    # later edit to it is as visible as an edit to any other lane.
    "llamacpp-primary": "56507a02c50560eb2ab2d620b33121c605fd2b0f3b6092f17451889cac7c2004",
    # NEW (lightning-on-orin plan, t7): the opt-in `vllm-associate` lane, gated
    # behind the `associate` profile so no existing deployment starts it. It
    # gives NVIDIA's published Jetson serve recipe's eight previously-
    # unexpressible flags a real home (five Mamba-cache flags,
    # --enable-prefix-caching, --max-num-batched-tokens, --trust-remote-code).
    # t6 then made `associate` the TENTH Colleague role and wired the gateway
    # to it; this LANE's own command/volumes were not touched by that, which is
    # exactly what this unchanged hash proves. See
    # tests/test_associate_compose.py.
    # Recomputed 2026-08-29 for the deployment-lock-per-box plan, t4: every
    # vllm-* lane that already read `.env` via `env_file:` gained a SECOND,
    # `required: false` entry for `.secrets.env` — a gitignored-by-suffix
    # sibling file so an operator (or scripts/gen-api-key.py) can drop
    # generated/file-supplied secrets there instead of `.env` itself.
    # `gateway` and `llamacpp-primary` are deliberately untouched (neither
    # used `env_file` before this task and neither does now — the gateway
    # reads only scoped, non-secret keys via `environment:` interpolation,
    # and llamacpp-primary serves a local GGUF with no HF_TOKEN need), which
    # is why their hashes above are unchanged. See
    # tests/test_fleet_secrets_env_file.py.
    "vllm-associate": "66850d08430be6496502d34f90fd7a56bb896fbfd2914f8fcb3caad60182a4d7",
    "vllm-embed": "52d6afc61fb6f23d23655251443ab0a50ab17ba350fb78ef3b1653a206117a1d",
    "vllm-embed-deep": "f73a2c1f7fe25664ea0ca12adb72ff445503cb5dd03ff9d027b6dc58cb1b0bcb",
    "vllm-hand": "7337db5b60adf2fcd47d1530eb149adf6c3652d029bb7102896f2da4432b7428",
    "vllm-middle": "cb09dd82886e67ac05d97815c8f567dbc97a422de5ba28dcf4055c5a0ea4f36b",
    "vllm-minor": "8e5b98ae680d8274eb563b897382b21f077a568af8a3aaf1f28f75b40f967b8f",
    "vllm-multimodal": "1bd3641b1df10ab26b5ca411fd5323e65ff8eb712a6c4c6ffbe85216b8e94f38",
    "vllm-multimodal-coder": "4e286dfd1fd9e250cb03190ea88151fa7030d3d5054f8901d8580ff910a5ecac",
    "vllm-muse": "00b16677ddedcd2605ef0e6a6d817fd23089751cb7946e8680d3b760e4307f24",
    "vllm-rerank": "2441f0e3d4c0d105e2803382f7128c68ee8fc03f6db7f2a2fa471cb2d0eafaf0",
    "vllm-worker": "8251c941e0667600d1d306db5a65d90ee9216391efdb6e246cdcc64f866c96e6",
}


def _load_fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


def _primary_command_tokens() -> list[str]:
    """vllm-primary's raw (unsubstituted) command, tokenized.

    ``command:`` is now a shell-lexed STRING (spec-knobs task, the same
    off-switch mechanism the senses lane established — see
    ``tests/test_senses_speculative_config.py``), not a YAML list. Tokenizing
    the raw text with ``shlex`` is safe: none of ``$``, ``{``, ``}`` are shell
    metacharacters, so every bare ``${VAR:-default}`` stays one token.
    """
    return shlex.split(_load_fleet()["services"]["vllm-primary"]["command"])


def _service_hash(service: dict) -> str:
    """A stable sha256 over a service's sorted YAML subtree."""
    text = yaml.safe_dump(service, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestVllmPrimaryToolParserPlugin:
    def test_tool_call_parser_default_is_thinking_aware(self) -> None:
        command = _primary_command_tokens()
        assert f"--tool-call-parser=${{PRIMARY_TOOL_CALL_PARSER:-{_PLUGIN_PARSER_NAME}}}" in command

    def test_tool_call_parser_still_env_overridable(self) -> None:
        # Same knob as before (PRIMARY_TOOL_CALL_PARSER), just a new default —
        # an operator can still override it in .env, e.g. back to plain
        # upstream qwen3_coder for a non-thinking primary.
        command = _primary_command_tokens()
        assert any(c.startswith("--tool-call-parser=${PRIMARY_TOOL_CALL_PARSER:-") for c in command)

    def test_tool_parser_plugin_flag_present(self) -> None:
        command = _primary_command_tokens()
        assert f"--tool-parser-plugin={_PLUGIN_DEST_PATH}" in command

    def test_auto_tool_choice_still_enabled(self) -> None:
        # The plugin wiring must not have dropped the existing flag it sits next to.
        command = _primary_command_tokens()
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
