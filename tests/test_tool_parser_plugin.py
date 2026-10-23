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
    # Recomputed for lane-safety d7 (FlashInfer JIT cache, vllm#62149): every vLLM
    # lane moved by GAINING FLASHINFER_WORKSPACE_BASE and a
    # ./compile-cache/<lane>-flashinfer bind at /cache/flashinfer; the non-vLLM
    # services (llamacpp-primary, comfyui, gateway) did not.
    # Recomputed for lane-safety t8 (per-lane compile cache): every vLLM lane moved
    # by GAINING VLLM_CACHE_ROOT and a ./compile-cache/<lane> bind mount; the
    # non-vLLM services (llamacpp-primary, comfyui, gateway) did not.
    # Recomputed for lane-safety t7 (boot hold): every non-primary service moved
    # because each healthcheck.start_period grew by the 180s maximum hold, and the
    # formerly unwrapped services (gateway, ...) gained the mg-logwrap entrypoint.
    # Recomputed for the /v1/render body caps (review finding 1 on the innereye
    # PR): ONLY `gateway` moved, and it moved by GAINING two environment
    # passthroughs — GATEWAY_RENDER_MAX_WORKFLOW_BYTES (1 MiB) and
    # GATEWAY_RENDER_MAX_UPLOAD_BYTES (32 MiB), the per-route request-body caps
    # the render facade enforces before buffering a submission or an input
    # image. Both are render-scoped: no other POST lane reads them, and no vLLM
    # lane reads them at all, which is why no other service's hash moves.
    # Required by tests/test_gateway_env_passthrough_guard.py — a knob
    # _config.py parses but compose never plumbs is inert in every real
    # deployment.
    #
    # Recomputed 2026-08-31 for issue #120: the three Gemma-family lanes
    # moved their attention backend off the dead VLLM_ATTENTION_BACKEND env
    # onto --attention-config (the #109 gate closed).
    # Recomputed 2026-08-29 for the gateway depends_on removal (issue #222):
    # ONLY `gateway` moved, and it moved by LOSING a key — its `depends_on` list
    # of the four core vLLM lanes is gone. `docker compose up -d <service>` walks
    # `depends_on`, so that "start order only" edge made EVERY gateway restart a
    # fleet-wide one: measured live on a Jetson AGX Thor 2026-08-28, a
    # gateway-only rebuild recreated the resident 27B cortex and started
    # vllm-multimodal on a box declaring MULTIMODAL_FEASIBLE=false. Every other
    # service is byte-identical, which is this tripwire proving the blast radius.
    #
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
    #
    # Recomputed 2026-10-18 for #310 R2: the shared image default moved to the
    # official vllm/vllm-openai v0.31.0-ubuntu2404 digest (moving every service
    # that defaults off VLLM_NIGHTLY_IMAGE), and vllm-worker also gained the
    # conditional WORKER_MM_ENCODER_ATTN_BACKEND slot (d1).
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
    #
    # Recomputed 2026-09-11 for the mesh-brain-join plan (t4): `gateway`
    # gained the six `LOBES_MESH_*` passthroughs (KEY/NAME/SEEDS/LEDGER_PATH
    # empty defaults; HEARTBEAT_S=60 / MISSED_MAX=3 carry their parser
    # defaults) and its FIRST `volumes:` block — the approval ledger's
    # read-write bind mount (spec c9; the host side defaults to
    # ./mesh-ledger.json in the deployment dir, the container side to
    # /home/gateway/mesh-ledger.json under the unprivileged gateway user).
    # `gateway` is again the ONLY service that moved — all twelve other
    # non-primary services stayed byte-identical, which is this tripwire
    # proving the blast radius. See tests/test_mesh_templates.py.
    #
    # Recomputed 2026-09-11 for t4 follow-up (volume defect fix): the gateway's
    # only volume changed from a file bind (P:P ledger) to a directory bind
    # (./mesh:/home/gateway/mesh), and the LOBES_MESH_LEDGER_PATH passthrough
    # gained a default (/home/gateway/mesh/ledger.json). LOBES_MESH_DIR is
    # also added to the passthrough. See tests/test_mesh_templates.py.
    #
    # Recomputed for t14 (env peer family retirement): `gateway` LOST the
    # <PREFIX>_PEER_ORIGIN/_PEER_PROXY/_PEER_API_KEY and the plural
    # <PREFIX>_PEER_ORIGINS/_PEER_API_KEYS passthrough lines for all ten role
    # prefixes — build_config no longer reads any of them. `gateway` is again
    # the ONLY service that moved.
    #
    # Recomputed 2026-09-16 for the `innereye` role registration (issue #82):
    # `gateway` GAINED seven INNEREYE_* passthrough lines — the feasibility
    # flag, the capacity knob, and the five lane-fingerprint keys the guard in
    # tests/test_gateway_env_passthrough_guard.py derives mechanically from
    # _config.FEASIBLE_ENV. No `comfyui` service exists yet (that is a later
    # task in the same plan), so `gateway` is again the ONLY service that
    # moved.
    #
    # Recomputed again 2026-09-16 for the /v1/render facade (issue #82, t9):
    # `gateway` GAINED the two INNEREYE_BASE_URL/INNEREYE_SERVED_NAME
    # passthrough lines, because build_config now READS them (an
    # `_optional_backend` like every sibling role) and
    # tests/test_gateway_env_passthrough_guard.py fails on any key the config
    # reads that compose does not plumb. Still no `comfyui` service (a later
    # task), so `gateway` remains the ONLY service that moved.
    #
    # Recomputed 2026-10-14 for the decider lane's POST /v1/systemone route
    # (odm t5): `gateway` GAINED the one DECIDER_DOMAINS passthrough line,
    # because build_config now reads it. `gateway` is the ONLY service moved.
    #
    # Recomputed 2026-10-14 for the decider domains advert (odm t7): `gateway`
    # GAINED the one DECIDER_DOMAIN_MODELS passthrough line (each domain's
    # checkpoint, read by /capabilities). `gateway` is the ONLY service moved.
    #
    # Recomputed 2026-10-15 for the decider's served window (PR #305 review,
    # finding 4): `gateway` GAINED the one DECIDER_DOMAIN_CONTEXTS passthrough
    # line (each domain's max_model_len, read by /capabilities in place of the
    # catalog's native ceiling). `gateway` is the ONLY service moved.
    # Recomputed for lane-safety #311 (deviation d1): the gateway gains the three
    # LOBES_THERMAL_* passthroughs.
    # Recomputed for stt-lanes: the gateway gains the four STT_HE_/STT_EN_
    # BASE_URL/FEASIBLE passthroughs (empty defaults). `gateway` is the ONLY
    # service moved.
    "gateway": "d19794e85500dfdae2dbc0c5c78439bd7868817be41a13f1ce1872feeca1ab9f",
    # The opt-in llama.cpp cortex lane (t4), profile-gated behind `llamacpp` so
    # no existing deployment starts it. Hashed here from the day it landed, so a
    # later edit to it is as visible as an edit to any other lane.
    "llamacpp-primary": "61e57bb2a1862a6adf6912fb6f0c3d0cc585ad008030c2ea1119038c66152de8",
    # NEW (innereye plan t3, issue #82; volumes/user added by t4): the opt-in
    # `comfyui` render-tenant service, gated behind the `innereye` profile so
    # no existing deployment starts it. Nothing else moved — vllm-primary's
    # tool-parser wiring above is untouched by this task.
    "comfyui": "9af5f5d01aeab2a08b4e4d1997ad0048e3016c5f114cd92de7378017554ca488",
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
    # Recomputed 2026-09-13 for the orin-associate-at-1m plan, t1 (approved
    # deviation d1, issue #260): the lane gained the conditional
    # ASSOCIATE_MAX_NUM_SEQS argv token (worker-lane idiom; unset renders
    # nothing) and a VLLM_ALLOW_LONG_MAX_MODEL_LEN environment passthrough
    # (default 0, mirroring vllm-primary) so a 1M-window lane can cap its
    # sequences. ONLY vllm-associate moved; every other hash here is
    # byte-identical. See tests/test_associate_compose.py.
    "vllm-associate": "cdaff038a73219f1aa0fce9e318cbd06a3c10120107002ee131277b940218d91",
    "vllm-embed": "6949a73055a5b9b702f14d72b86ea15c406b9aec5061c102fece22dad8dfbef7",
    "vllm-embed-deep": "5693868acac23c814781242c63355656b5a1ba316e71b9d726cca9cc6a1a6b74",
    "vllm-hand": "d7cd1bad8f2f09bafd6821889983bcd3f24504b6ee759bf0b48aca6391361bf6",
    "vllm-middle": "97ea5ec8da91f9ab84807a20d2fc3129c0c8f6e60a2c8441ba0b405d8a740717",
    "vllm-minor": "ae0017e69d4f4a5ce0e035e9f605692af9bea43ffe11d7f661dc5de44f75de7e",
    "vllm-multimodal": "77cbc0a1a5c223acbdabca5071557dd7448104c8d2732216f4970c2b610333de",
    "vllm-multimodal-coder": "487934e71c2e401de532a5398f0b937e9d2812e7d8e79ac25e357142aacfa03d",
    "vllm-muse": "5c5a1f4cd68dc904ddc45975b9c66b3863542962df39a5b029b05f3302f21b1e",
    # vllm-rerank recomputed for issue #227 t3 (--chat-template flag + the
    # qwen3_reranker.jinja read-only bind mount) — a deliberate change.
    "vllm-rerank": "5c968a7cf43750afefd33808d5878fd3d0aa5597a0413443a7ac0e9a5c207084",
    # Recomputed 2026-09-10 for the worker recipe knobs (worker-recipe-knobs
    # t3): ONLY `vllm-worker` moved, and only its `command` string +
    # environment comments — it grew the ${VAR:+…} / ${VAR-} slots for
    # kv-cache-dtype, attention-config, moe-backend, load-format,
    # max-num-seqs, max-num-batched-tokens, chunked-prefill,
    # async-scheduling, prefix-caching and an env-carried
    # --tool-call-parser (default unchanged: qwen3_coder). With none of
    # those set the RENDERED argv is byte-identical to before (proved
    # against real `docker compose config` in
    # tests/test_worker_recipe_knobs.py); this tripwire is hashing the
    # TEMPLATE text, which did change.
    "vllm-worker": "f653fab2281f6e33b497cf675fd82b60db004dac9de145bf74964cf9b1510fbc",
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
