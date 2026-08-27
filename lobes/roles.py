"""The role registry — the ten first-class, Colleague-facing lobes (issue #81).

lobes exposes the fleet not as a bag of model ids but as TEN discoverable
*roles*, each resolved to a live endpoint + metadata so a caller (Colleague)
can address a capability by role — ``cortex``, ``senses``, ``muse``,
``worker``, ``associate``, ``hand``, ``embedder``, ``reranker``, ``stt``,
``tts`` — without hardcoding any single model endpoint:

* ``cortex``   → the ``primary`` generate backend (Qwen 3.6 27B NVFP4 MTP).
  The authoritative reasoning/action/decision layer — the final authority.
* ``senses``   → the ``multimodal`` generate backend (Gemma 4 12B). The
  user-facing intake/perception/speak-back layer; it does NOT decide or act.
* ``muse``     → the ``muse`` generate backend (Gemma 4 31B NVFP4). The
  creative/ideation lobe — long-form writing, brainstorming, divergent
  second opinions; it proposes, never decides. OPT-IN: hosted only by a
  muse-hosting deployment shape (``lobes init --shape thor-muse``), never
  by the default ``machine-as-brain`` (a 31B cannot co-reside with the
  cortex+senses duo on a 128 GB box).
* ``worker``   → the ``worker`` generate backend (Qwen3.6-35B-A3B NVFP4 today;
  moving to ``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`` per issue
  #187 — an MoE ~3B-active-per-token lobe either way). The fast,
  **TEXT-ONLY, NON-CODING** doer — action selection, tool loops, RAG
  synthesis, summarization/log digestion, structured extraction, and repo
  **inspection**/navigation/running-already-authorized-commands, all UNDER
  cortex's direction. It is the first role besides ``cortex`` permitted
  ``repo_action`` (it may touch the repo — search, inspect diffs, run tests),
  but ``code_authoring`` is explicitly forbidden: "not coder" does not mean
  "cannot touch a repository" (issue #187) — new code or deep code reasoning
  escalates to ``cortex``. It never makes the final decision or a security
  call either. TEXT-ONLY: unlike the prior Qwen checkpoint's ViT, perception
  stays with ``senses``/the talker lane — worker is a doer, not a seer.
  OPT-IN like ``muse``: hosted only by a worker-hosting deployment shape,
  never by the default ``machine-as-brain``.
* ``associate`` → the ``associate`` generate backend (the SAME Nemotron 3.5
  Lightning checkpoint the ``worker`` seat holds). ``worker`` MINUS
  ``repo_action``: it executes, drafts, inspects and calls tools, then hands
  the result BACK rather than enacting it. OPT-IN, like muse/worker: hosted
  only by an explicit associate-hosting deployment shape, never by
  ``machine-as-brain``.
* ``hand``     → the ``hand`` generate backend (LiquidAI LFM2.5-1.2B-Instruct).
  The fleet's designated FINE-TUNING BASE and its trained specialist: one cheap
  base, many LoRA adapters, each mastering a domain ("muscle memory"). Where
  ``worker`` is an untrained generalist doer, ``hand`` knows a few things
  extremely well because someone taught it. At ~1.2B it is cheap enough to
  co-reside on EVERY card, so unlike ``muse``/``worker`` it is default-hosted
  and never proxied to a peer. It is also the ``minor``/``cheap`` capability
  tier (it replaced Qwen3.5-4B there) and the pressure-policy SERVABLE FLOOR —
  the one generate lane never shed under load. Addressed as ``model=hand`` for
  the base and ``model=hand:<domain>`` for an adapter.
* ``embedder`` → the ``embed`` pooling backend (Qwen3-Embedding-0.6B) →
  ``POST /v1/embeddings``.
* ``reranker`` → the ``score``/rerank backend (Qwen3-Reranker-0.6B) →
  ``POST /v1/rerank`` (+ ``/v1/score``).
* ``stt``      → the Parakeet sidecar behind the audio overlay →
  ``POST /v1/audio/transcriptions``. Opt-in (``lobes init --fleet --audio``).
  When the overlay is actually wired and not declared off, ``stt`` also
  advertises the ``/v1/realtime`` WebSocket server-VAD session capability
  (issue #149) — see :data:`STT_REALTIME_RESPONSIBILITY`.
* ``tts``      → the Chatterbox sidecar behind the audio overlay →
  ``POST /v1/audio/speech``. Opt-in.

This module is the SHARED core the CLI (``lobes capabilities``, t5) and the
gateway (``GET /capabilities``, t6) both consume, so the role→endpoint contract
has exactly one source of truth. It is pure/offline: it reads the same config
the gateway builds (a :class:`~lobes.gateway._routing.RoutingTable` +
:class:`~lobes.gateway._config.ServerConfig`) plus the static
:mod:`lobes.catalog`, and touches no sockets.

**Provisional wording (plan risk r2, issue #81):** the ``responsibilities`` /
``forbidden_responsibilities`` token lists below are issue #81's worked
examples. They describe the intended DIVISION OF LABOUR between the lobes; they
are *not* claims about answer correctness or task success — lobes emits a
runtime-only contract. The exact vocabulary is a build-time call and may be
refined without breaking the machine-readable shape.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass

from lobes.catalog import SUPPORTED_MODELS, SupportedModel
from lobes.gateway._config import ServerConfig, build_config
from lobes.gateway._replicas import UNCALIBRATED_WEIGHT
from lobes.gateway._replicas import UNKNOWN as _REPLICA_UNKNOWN
from lobes.gateway._replicas import ReplicaState
from lobes.gateway._routing import RoutingTable

# The ten first-class roles, in canonical order: generate lanes (cortex,
# senses, muse, worker, hand), pooling lanes, then the opt-in audio overlay.
# Downstream (CLI/gateway) iterate this for a stable ordering.
#
# ADDING A ROLE IS EFFECTIVELY IRREVERSIBLE. Every name here becomes a public
# address on `GET /capabilities`, `lobes capabilities`, the `model=` alias
# space and the `lobes up <role>` surface — and removing one later breaks every
# caller that learned to use it. Ten is the count today (the tenth,
# `associate`, landed with the lightning-on-orin plan's t6); read
# docs/colleague-stack.md before proposing a tenth.
ROLES: tuple[str, ...] = (
    "cortex",
    "senses",
    "muse",
    "worker",
    # The TENTH role (lightning-on-orin plan, t6): `associate` is `worker`
    # MINUS `repo_action` — it executes, drafts, inspects and calls tools, and
    # hands the result BACK rather than enacting it. It is a SEPARATE PUBLIC
    # ADDRESS on purpose, not a responsibilities token on `worker`: the
    # operator wants the `worker` seat left free for a possible future
    # worker/cortex switch, and only a distinct name can carry that.
    "associate",
    "hand",
    "embedder",
    "reranker",
    "stt",
    "tts",
)

# role → the internal gateway backend NAME that serves it — the key space the
# RoutingTable's feasibility/peer channels use. The seven gateway-fronted roles
# map to their vLLM backends; ``stt``/``tts`` (first-class since issue #129)
# map to themselves — they are path-routed audio lanes, not model-routed
# backends (still resolved from ``ServerConfig.audio_url`` below), but their
# names now ride the SAME ``FEASIBLE_ENV`` / peer origin/proxy/key channels,
# so :func:`annotate_peer_referrals` covers all ten roles uniformly.
# NOTE the name↔role_hint mismatch for the pooling lane: the *backend* is named
# ``embed``/``rerank`` while the *catalog* role_hint is ``embedding``/``reranker``.
# ``muse`` and ``worker`` each use their own name as their backend name.
ROLE_BACKEND: dict[str, str] = {
    "cortex": "primary",
    "senses": "multimodal",
    "muse": "muse",
    "worker": "worker",
    # `associate` is its own backend name too (the `vllm-associate` compose
    # lane, ASSOCIATE_BASE_URL) — like `muse`, `worker` and `hand`.
    "associate": "associate",
    "hand": "hand",
    "embedder": "embed",
    "reranker": "rerank",
    "stt": "stt",
    "tts": "tts",
}

# role → the catalog ``role_hint`` of its canonical model. Used to (a) look up
# context/quant/mtp for that role, and (b) name the model a role WOULD serve
# when its backend is not wired in this deployment (loaded=False but still named).
ROLE_ROLE_HINT: dict[str, str] = {
    "cortex": "primary",
    "senses": "multimodal",
    "muse": "muse",
    "worker": "worker",
    # `associate` serves the SAME catalog gear the `worker` role_hint names
    # (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) — one checkpoint,
    # two public addresses with different authority. The catalog holds ONE
    # entry per checkpoint id (tests/test_catalog.py::test_catalog_ids_are_unique),
    # so the honest mapping is a shared role_hint, not a duplicated entry —
    # the same name↔role_hint indirection the pooling lanes already use
    # (`embedder` → "embedding"). See lobes.catalog.BACKEND_ROLE_CATALOG_HINT,
    # which carries the identical alias for the tier layer.
    "associate": "worker",
    "hand": "hand",
    "embedder": "embedding",
    "reranker": "reranker",
}

# The chat path the three generate lobes share (SonarCloud: duplicated literal).
_CHAT_PATH = "/v1/chat/completions"

# role → the OpenAI path a caller hits. The reranker exposes both /v1/rerank and
# /v1/score; /v1/rerank is the canonical path advertised here.
ROLE_PATH: dict[str, str] = {
    "cortex": _CHAT_PATH,
    "senses": _CHAT_PATH,
    "muse": _CHAT_PATH,
    "worker": _CHAT_PATH,
    "associate": _CHAT_PATH,
    "hand": _CHAT_PATH,
    "embedder": "/v1/embeddings",
    "reranker": "/v1/rerank",
    "stt": "/v1/audio/transcriptions",
    "tts": "/v1/audio/speech",
}

# The two audio-overlay sidecars — hardcoded here (as in the gateway/realtime
# code) because they are NOT in the switchable catalog (lobes/catalog.py): they
# are fixed GPU sidecars behind the /v1/audio/* facade, activated together by
# ``lobes init --fleet --audio``.
_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"  # Parakeet TDT 0.6B, NeMo ASR
_STT_RUNTIME = "parakeet"
_TTS_MODEL = "ResembleAI/chatterbox"  # Chatterbox, Resemble AI 0.5B, Apache-2.0
_TTS_RUNTIME = "chatterbox"
_VLLM_RUNTIME = "vllm"  # the seven gateway-fronted roles all serve on vLLM

# Canonical responsibilities per role (issue #81 worked examples — PROVISIONAL,
# see the module docstring). A role's responsibilities are what it is EXPECTED to
# own in the division of labour, never a correctness/success claim.
ROLE_RESPONSIBILITIES: dict[str, tuple[str, ...]] = {
    "cortex": (
        "reasoning",
        "deciding",
        "planning",
        "tool_use",
        "code_repo_actions",
        "validation",
        "final_authority",
        # Added 2026-07-31 with the multimodal promotion (colleague#361). The
        # cortex checkpoint (then unsloth/Qwen3.6-27B-NVFP4, since replaced by
        # unsloth/Qwen3.8-27B-NVFP4) serves its own ViT and
        # image + video intake were VALIDATED live on the GB10 against negative
        # controls (docs/evidence/2026-07-31-accept-multimodal-cortex-spark.txt).
        #
        # Advertising them is REQUIRED, not cosmetic. Consumers are told to
        # resolve roles by name from this advert and NEVER to parse model names,
        # so omitting these made a seeing cortex look blind to every caller
        # obeying the contract — a lie by omission of exactly the kind the
        # "never advertise a capability you cannot serve" rule exists to
        # prevent, in the opposite direction. Reported by colleague#361.
        #
        # cortex is now the ONLY role that both SEES and DECIDES: `senses` is
        # forbidden final_decision/repo_action, and `worker` (issue #187) is
        # TEXT-ONLY — no image_understanding/video_understanding at all — as
        # well as forbidden final_decision/security_decision/code_authoring,
        # so cortex remains the sole role that both sees and has final say.
        "image_understanding",
        "video_understanding",
    ),
    "senses": (
        "intake",
        "normalize_input",
        "classify_intent",
        "prepare_context_packet",
        "speak_back",
    ),
    # `tool_use` alongside the creative tokens is NOT a widening of muse's
    # authority: the forbidden list below still bars final_decision /
    # repo_action / security_decision, so muse calls tools to RESEARCH a
    # proposal (read a file, search, fetch), never to enact one — cortex
    # remains the only lobe that acts. It is declared because muse's lane
    # genuinely serves tool calls (the fleet template's vllm-muse carries
    # --enable-auto-tool-choice --tool-call-parser=gemma4), and a
    # division-of-labour list silent on a capability the lane actually serves
    # tells a Colleague less than the truth.
    "muse": (
        "creative_generation",
        "long_form_writing",
        "ideation",
        "style_variation",
        "divergent_second_opinion",
        "tool_use",
    ),
    # `worker` is the fast, TEXT-ONLY, NON-CODING doer (issue #187,
    # superseding the earlier thor-worker-lobe plan's multimodal "seeing
    # doer" framing — the checkpoint moves to Nemotron 3.5 Lightning,
    # text-only). Unlike every other non-cortex role its list carries
    # `repo_action`: worker is the FIRST role besides cortex permitted to
    # ACT on the repo — but #187 draws an explicit line INSIDE that
    # permission, not just around it: "not coder does not mean cannot touch
    # a repository". So the vocabulary is split into two families:
    #
    #   * ALLOWED "touch the repo" tokens: `repo_action` (may act),
    #     `repo_inspection` (search code, inspect diffs, retrieve files,
    #     navigate), `run_authorized_commands` (run tests/already-authorized
    #     commands) — plus the general doer tokens (`execution`,
    #     `ground_work`, `bulk_transform`, `drafting`) and the explicit
    #     agent-work tokens #187 calls out by name: `action_selection`
    #     (choose the next tool/step), `retrieval_synthesis` (RAG answers
    #     from supplied evidence), `summarization`, `log_digestion`
    #     (summarize a long tool/log result), `structured_extraction`
    #     (extract/normalize structured data).
    #   * FORBIDDEN "author code" token: `code_authoring`, in
    #     ROLE_FORBIDDEN below, alongside final_decision/security_decision.
    #     New code or deep code reasoning ESCALATES to cortex — the
    #     forbidden list is what keeps `repo_action` from silently widening
    #     into coding authority now that the vocabulary can say so directly,
    #     rather than the policy hiding in prose or a model-name check.
    #
    # No image_understanding / video_understanding: perception stays with
    # `senses`/the talker lane; worker is text-only.
    "worker": (
        "execution",
        "ground_work",
        "bulk_transform",
        "drafting",
        "action_selection",
        "retrieval_synthesis",
        "summarization",
        "log_digestion",
        "structured_extraction",
        "repo_inspection",
        "run_authorized_commands",
        "tool_use",
        "repo_action",
    ),
    # The `associate` lobe (lightning-on-orin plan, t6): worker MINUS
    # `repo_action`. It DOES, but it does not ACT — every doer token worker
    # carries for producing a result is here (execution, ground_work,
    # bulk_transform, drafting, repo_inspection, run_authorized_commands,
    # tool_use), and the one token that lets a lobe CHANGE the repo is not.
    # associate inspects, drafts and hands the result back; cortex (or worker,
    # under cortex's direction) enacts it.
    #
    # The list is deliberately SHORTER than worker's, following the `hand`
    # precedent recorded in docs/colleague-stack.md: ADDING a responsibility
    # later is contract-compatible, REMOVING one is a break, so the
    # conservative list ships first. worker's agent-work tokens
    # (`action_selection`, `retrieval_synthesis`, `summarization`,
    # `log_digestion`, `structured_extraction`) are NOT claimed here — not
    # because associate could not serve them (it serves the same checkpoint),
    # but because nothing has been measured that says it should own them in
    # the division of labour, and a responsibilities list is a
    # division-of-labour claim, not a capability boast.
    #
    # No image_understanding / video_understanding: the checkpoint
    # (Nemotron 3.5 Lightning) carries no vision_config — text-only, exactly
    # like `worker`.
    "associate": (
        "execution",
        "ground_work",
        "bulk_transform",
        "drafting",
        "repo_inspection",
        "run_authorized_commands",
        "tool_use",
    ),
    # The `hand` lobe: a TRAINED SPECIALIST, not a generalist doer. Its value
    # is the LoRA adapter riding on it, so its responsibilities describe mastery
    # of a taught domain rather than raw capability. Deliberately NO
    # image_understanding / video_understanding — LFM2.5-1.2B-Instruct is
    # text-only (LiquidAI ships the vision variant as a separate architecture,
    # Lfm2VlForConditionalGeneration, which this checkpoint is not).
    "hand": (
        "domain_mastery",
        "learned_skill",
        "specialized_task",
        "tool_use",
    ),
    "embedder": ("vectorization", "memory_retrieval_input"),
    "reranker": ("retrieval_ordering", "relevance_refinement"),
    # NOTE: this base tuple deliberately does NOT list the realtime/VAD
    # session capability (issue #149) — see STT_REALTIME_RESPONSIBILITY
    # below. It stays static and unconditional so this dict remains a stable,
    # always-true description of what stt COULD serve; the honesty-gated
    # addition is applied at build time by _resolve_audio_role, never here.
    "stt": ("transcribe", "audio_input_to_text"),
    "tts": ("speech_output", "synthesize"),
}

# The /v1/realtime WebSocket server-VAD session capability (issue #149, task
# t4). Deliberately NOT a static member of ROLE_RESPONSIBILITIES["stt"]
# above — the honesty rule this repo already enforces for `loaded`/
# `feasible`/`ready` applies here too: a role must not claim a capability it
# cannot serve. A text-only fleet (no `lobes init --fleet --audio` overlay)
# or an operator-declared-off stt lane (`STT_FEASIBLE=false`) must not
# advertise it. _resolve_audio_role appends this token to stt's
# `responsibilities` tuple ONLY when the audio overlay is actually wired on
# THIS deployment (`AUDIO_URL` configured) AND the lane is feasible (not
# declared off) — never a new RoleInfo schema field, per the #149 t4 design
# (a new field would ripple into the CLI, gateway, tests, and
# docs/colleague-stack.md; an additive responsibilities token is
# contract-compatible).
STT_REALTIME_RESPONSIBILITY = "realtime_vad_session"

# What each role must NOT do. cortex is the final authority (nothing forbidden);
# senses is intake/perception only — it must not decide, act on the repo, or make
# security calls; muse proposes/creates but likewise never decides or acts.
# `worker` is the sole exception among the non-cortex generate lobes: it MAY act
# on the repo (repo_action is deliberately ABSENT from its forbidden list — see
# ROLE_RESPONSIBILITIES above), but it still must never make the final decision
# or a security call, so worker acts under cortex's direction, never on its own
# authority. Issue #187 adds a THIRD worker-forbidden token, `code_authoring`:
# "not coder does not mean cannot touch a repository" — worker may inspect,
# search, run tests/authorized commands (repo_action stays permitted), but
# authoring new code or deep code reasoning is explicitly barred and escalates
# to cortex. This is deliberately a vocabulary token, not a prose caveat or a
# model-name check, so a consumer reading only these two lists gets the whole
# policy. The service roles carry no forbidden list of their own.
ROLE_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "cortex": (),
    "senses": ("final_decision", "repo_action", "security_decision"),
    "muse": ("final_decision", "repo_action", "security_decision"),
    "worker": ("final_decision", "security_decision", "code_authoring"),
    # `associate` is worker's forbidden list PLUS `repo_action` — the single
    # token that separates the two roles. "They do, but not act": associate may
    # run authorized commands and inspect a repo, but it never changes one, and
    # like worker it never makes the final call, a security call, or authors
    # code. See ROLE_RESPONSIBILITIES above.
    "associate": (
        "final_decision",
        "security_decision",
        "code_authoring",
        "repo_action",
    ),
    # `hand` withholds repo_action deliberately for v1: ADDING a responsibility
    # later is contract-compatible, REMOVING one is a break, so the conservative
    # list ships first. Granting it once adapters exist is issue #180.
    "hand": ("final_decision", "repo_action", "security_decision"),
    "embedder": (),
    "reranker": (),
    "stt": (),
    "tts": (),
}

# role → the deployment env var that carries the SERVED ``--max-model-len`` for
# that role's backend (issue #81, t5). Mirrors the fleet compose template's
# `--max-model-len=${...}` flags (see docs/gateway-fleet.md / the fleet
# env.example). Only the six gateway-fronted roles carry one — stt/tts have no
# token context (see :func:`_audio_role`), so they are deliberately absent here.
ROLE_MAX_MODEL_LEN_ENV: dict[str, str] = {
    "cortex": "PRIMARY_MAX_MODEL_LEN",
    "senses": "MULTIMODAL_MAX_MODEL_LEN",
    "muse": "MUSE_MAX_MODEL_LEN",
    "worker": "WORKER_MAX_MODEL_LEN",
    "associate": "ASSOCIATE_MAX_MODEL_LEN",
    "hand": "HAND_MAX_MODEL_LEN",
    "embedder": "EMBED_MAX_MODEL_LEN",
    "reranker": "RERANK_MAX_MODEL_LEN",
}

# The roles the GATEWAY fronts as model-routed vLLM backends — every role
# except the two path-routed audio sidecars, which :func:`_resolve_audio_role`
# handles on a separate branch.
#
# DERIVED from :data:`ROLES`, deliberately: this was a hand-typed tuple inside
# ``build_role_registry`` until the ninth role landed, and a hand-typed copy of
# ROLES is precisely the thing that lets a new role half-land — it would be
# registered in every table above yet silently missing from the registry the
# CLI and ``GET /capabilities`` both read. Membership keys off
# :data:`ROLE_ROLE_HINT` (which the audio roles deliberately do not appear in,
# having no catalog entry), so adding a generate/pooling role to ROLES picks it
# up here automatically and adding an audio-style role does not.
GATEWAY_FRONTED_ROLES: tuple[str, ...] = tuple(r for r in ROLES if r in ROLE_ROLE_HINT)


@dataclass(frozen=True)
class RoleInfo:
    """Live metadata for one first-class role (a Colleague-facing lobe).

    Frozen so it is safe to share across gateway threads. JSON-serialisable with
    :func:`dataclasses.asdict` (tuples become arrays) — the CLI ``--json`` (t5)
    and the gateway ``GET /capabilities`` (t6) build their payloads from this.
    """

    role: str
    model: str  # the served model id this role resolves to (never hardcoded blank)
    runtime: str  # the serving stack: "vllm" | "parakeet" | "chatterbox"
    endpoint: str  # base URL of the service the caller hits ("" when not wired)
    path: str  # the OpenAI path, e.g. "/v1/chat/completions"
    # The SERVED context (tokens): the deployment's `--max-model-len` override
    # (ROLE_MAX_MODEL_LEN_ENV) when the env sets one, else the catalog native
    # (`SupportedModel.native_max_model_len`) — issue #81 t5. 0 for audio roles.
    context: int
    quant: str  # vLLM quantization for the model; "" when n/a (pooling/audio)
    mtp: bool  # speculative decoding (MTP draft head) active for this model
    # Does this role's endpoint accept OpenAI `tools` on a request? Derived from
    # the catalog entry's `tool_parser` being non-empty — the SAME field the
    # fleet template's `--enable-auto-tool-choice --tool-call-parser=<p>` pair is
    # built from (runtime._parser.infer_parser), so it cannot drift from what is
    # served without `tests/test_catalog.py`'s pairing guard failing first.
    # `False` for the pooling roles (embedder/reranker serve no chat lane) and
    # for stt/tts (no catalog entry at all).
    #
    # Deliberately a BOOL, not the parser name: the served parser can diverge
    # from the catalog's (the primary lane defaults to the `qwen3_coder_thinking`
    # PLUGIN over the catalog's base `qwen3_coder`, and `PRIMARY_TOOL_CALL_PARSER`
    # /`MIDDLE_TOOL_CALL_PARSER` can override it), so naming a parser here would
    # be a claim this module cannot honestly make. Whether tools are accepted
    # does not vary under that divergence; which parser produced them is an
    # implementation detail the OpenAI surface already abstracts away.
    #
    # NOT a claim about tool-call QUALITY or success — same runtime-only contract
    # as every other field here (see the module docstring's provisional wording).
    tools: bool
    responsibilities: tuple[str, ...]
    forbidden_responsibilities: tuple[str, ...]
    # Is this role even SERVABLE on this machine at all — the HARDWARE
    # dimension of issue #92's "advertised implies reachable" (plan
    # "per-machine profiles", task t6)? `True` unless this deployment's
    # RoutingTable named this role's backend in `table.infeasible` (from
    # `<PREFIX>_FEASIBLE=false`, see lobes.gateway._config.FEASIBLE_ENV) — a
    # fact about the MACHINE, independent of `loaded` (is a backend wired) and
    # `ready` (is it live right now). Since issue #129 this varies for stt/tts
    # too: an operator declares an audio lane off with STT_/TTS_FEASIBLE=false
    # (the audio roles stay outside the per-machine Profile TUNING schema, but
    # ride the same feasibility/peer channels); absent, the audio roles keep
    # their sleeping-lobe default — feasible:true, ready:false — so every
    # pre-#129 deployment renders byte-identically.
    feasible: bool = True
    # Runtime readiness — a caller-supplied LIVE signal, folded in by
    # build_role_registry: `backend_ready` (keyed by the ROLE_BACKEND name)
    # for the six gateway-fronted roles, `audio_ready` for stt/tts (issue
    # #89). Generalised from the stt/tts-only split (issue #89/#90) to all
    # ten roles (issue #81 t5) — `ready` is no longer a bare alias of `loaded`.
    #
    # `backend_ready` is TRI-STATE PER BACKEND but resolves to `ready` under a
    # SUPPLIED-vs-OMITTED rule the builder self-enforces (issue #92 / honesty
    # h14 — do not let this drift back to caller discipline):
    #   * mapping OMITTED entirely (`backend_ready is None`, the default) →
    #     back-compat: `ready == loaded`, the coarse "configured/wired" proxy.
    #     Still exercised by every non-HTTP caller (the CLI's non-live paths,
    #     most of this module's own test suite).
    #   * mapping SUPPLIED → AUTHORITATIVE: `ready = (backend_ready.get(name)
    #     is True)`. A present `None`, a present `False`, and a MISSING KEY all
    #     mean NOT ready — "no live signal" is never evidence of health.
    # THE TRAP this closes: `ReadinessCache.current()` reports a dead/missing/
    # unreachable backend as `None`. That cache-`None` means UNREACHABLE — the
    # OPPOSITE of "no signal, assume the wired/`loaded` default". A caller that
    # passes `current()` straight in (exactly what this contract invites) must
    # get `ready=False` for that backend, NOT a resurrected #92 `ready=True`.
    # Because the SUPPLIED branch is authoritative, it does.
    #
    # Structurally CLAMPED regardless: a role whose backend is not wired
    # (`loaded is False`), whose `endpoint` is empty, OR whose `feasible` is
    # `False` (task t6) can never report `ready=True`, no matter what signal a
    # caller passes in. This mirrors — and is enforced by the same code path
    # as — the stt/tts clamp on `audio_configured` (issue #89/#90 review
    # finding), now applied to all ten roles by build_role_registry itself,
    # not left to caller discipline. The `feasible` clamp is what makes an
    # infeasible-but-HEALTHY role (a live `backend_ready=True` signal) still
    # report `ready=False` — a healthy PROCESS is not evidence this MACHINE
    # can actually carry the role.
    ready: bool = False
    # Is this role's backend/service wired/present in THIS deployment? An
    # unconfigured/opt-in role is still returned, with loaded=False.
    loaded: bool = False


def _catalog_by_id(model_id: str) -> SupportedModel | None:
    """The catalog entry whose ``id`` == ``model_id`` (an operator's served name)."""
    return next((m for m in SUPPORTED_MODELS if m.id == model_id), None)


def _catalog_by_role_hint(role_hint: str) -> SupportedModel | None:
    """The canonical catalog entry for a role_hint (each is unique in the catalog)."""
    return next((m for m in SUPPORTED_MODELS if m.role_hint == role_hint), None)


def _gateway_base_url(server: ServerConfig) -> str:
    """The gateway's caller-facing base URL — NEVER fabricated from host:port.

    ``ServerConfig.host``/``.port`` (``GATEWAY_HOST``/``GATEWAY_PORT``) are the
    gateway process's own INTERNAL listen config — where it binds inside its
    container — not necessarily where a caller can reach it from outside. On
    the reference rig the gateway listens on internal container port 8000 but
    is PUBLISHED on host port 8001, and host port 8000 belongs to a wholly
    unrelated daemon (a stray uvicorn service). A URL built from
    ``host:port`` would therefore silently advertise that foreign daemon as
    if it were the gateway — a caller dialing it gets whatever happens to be
    listening there, not a 404 from the gateway, which is worse than an
    honest "unknown". This function must never do that.

    Returns ``server.public_url`` (the operator-declared, caller-reachable
    origin — ``GATEWAY_PUBLIC_URL``), rstripped of a trailing slash, when it
    is set; otherwise ``""``. An empty return here is not a degraded case to
    special-case downstream — :func:`build_role_registry` already treats an
    empty ``endpoint`` as a hard "never advertise ready=True" signal, so a
    caller either gets a real, dialable endpoint or an honest absence of one.

    Callers that know the real reachable address from elsewhere (a published
    host port, a tunnel URL, or — as the gateway's own HTTP route does,
    issue #87 — the request's own ``Host`` header) must pass it explicitly as
    ``gateway_url`` to :func:`build_role_registry`; that explicit value always
    wins over this fallback.
    """
    return (server.public_url or "").rstrip("/")


def _served_context(role: str, env: Mapping[str, str], native: int) -> int:
    """The SERVED context for ``role`` — issue #81 t5.

    Reads the deployment's ``--max-model-len`` override
    (:data:`ROLE_MAX_MODEL_LEN_ENV`) from ``env`` when present and numeric;
    falls back to the catalog ``native`` context otherwise (unset key, blank
    value, or a malformed override — never raises). ``role`` values with no
    entry in :data:`ROLE_MAX_MODEL_LEN_ENV` (the audio roles) always fall back
    to ``native`` (which :func:`_audio_role` always passes as ``0``).
    """
    key = ROLE_MAX_MODEL_LEN_ENV.get(role)
    if key is None:
        return native
    raw = env.get(key)
    if not raw:
        return native
    try:
        return int(raw)
    except (TypeError, ValueError):
        return native


def _resolve_ready(
    loaded: bool,
    feasible: bool,
    endpoint: str,
    ready_signal: bool | None,
    peer_signal: bool | None,
) -> bool:
    """Resolve ``RoleInfo.ready`` for a gateway-fronted role — the #92/#115 clamp.

    This is the structural enforcement :func:`_gateway_role`'s docstring
    promises: a caller passing a stale/wrong ``ready_signal`` (or ``peer_signal``)
    can never fabricate ``ready=True`` for a role with nothing to dial, nothing
    wired, or no hardware feasibility — the clamp is applied HERE, not left to
    caller discipline.

    ``ready`` is CLAMPED to ``False`` whenever the backend is not wired
    (``loaded is False``), the resolved ``endpoint`` is empty (see
    :func:`_gateway_base_url`), OR this machine's ``table.infeasible`` names
    this role's backend (``feasible is False`` — task t6, the HARDWARE
    dimension of the same invariant). This generalises, to all four
    gateway-fronted roles, the same clamp issue #89/#90 established for
    stt/tts (a caller-supplied signal can never override "nothing is wired"
    or "nothing to dial") — and now also "this machine can't run it at all",
    independent of wiring or a live health probe.

    When the role IS wired, feasible, and dialable: ``ready`` takes
    ``ready_signal`` directly when it is not ``None`` (an AUTHORITATIVE
    verdict — see :func:`_gateway_role`'s docstring for what produces one),
    else it falls back to ``loaded`` (the original t4 behaviour).

    ``peer_signal`` is the NEW live signal t5's clamp docstring demanded
    (proxy-lobes t6, issues #115/#127): the live PEER-probe verdict for a
    PROXIED role, threaded through by :func:`build_role_registry` from its
    ``peer_ready`` mapping — mirroring how ``backend_ready``/``audio_ready``
    thread their signals — and ``None`` for every other role and every caller
    without one. It is a SEPARATE channel from ``ready_signal``,
    deliberately: ``backend_ready`` (the LOCAL probe, folded into
    ``ready_signal`` upstream) still NEVER unclamps a proxied role — a
    healthy local process is not evidence the peer serves the model — while
    ``peer_signal`` reports a probe of the actual proxied path
    (:func:`lobes.gateway._readiness.probe_peer_ready`: the peer answered 200
    AND its own ``/v1/models`` lists the served id), so a proxied role's
    ``ready`` honestly reflects it (honesty h2 — a live proxied-path probe or
    ``False``, never hardcoded true). It is still clamped on an empty
    ``endpoint`` (nothing for a caller to dial — unchanged from every other
    role), and ``feasible`` stays ``False`` regardless: hosting is a hardware
    fact a forward does not change.
    """
    if loaded and feasible and endpoint:
        return ready_signal if ready_signal is not None else loaded
    if peer_signal is not None and endpoint:
        # PROXIED role with a live peer probe (t6): ready reflects the peer's
        # verified state — never `loaded` (it isn't, here) and never a local
        # backend_ready signal (see the two-channel rationale above).
        return peer_signal
    return False


def _gateway_role(
    role: str,
    table: RoutingTable,
    gateway: str,
    env: Mapping[str, str],
    ready_signal: bool | None,
    peer_signal: bool | None = None,
) -> RoleInfo:
    """Resolve a gateway-fronted role (cortex/senses/muse/worker/embedder/reranker).

    ``ready_signal`` carries only TWO meanings here, never the readiness cache's
    tri-state — :func:`build_role_registry` has already resolved that away:

    * ``True``/``False`` — an AUTHORITATIVE readiness verdict for this backend.
      The builder passes a concrete bool whenever a ``backend_ready`` mapping was
      supplied, having already collapsed a present ``None``, a present ``False``,
      and a missing key all to ``False`` (issue #92 / honesty h14). ``ready``
      takes this value directly (subject to the clamp in :func:`_resolve_ready`).
    * ``None`` — NO live signal at all (``backend_ready`` was omitted entirely),
      in which case ``ready`` falls back to the coarse ``loaded`` proxy — the
      original t4 behaviour.

    Crucially, ``None`` here is *only ever* "no mapping supplied", never "the
    cache said unreachable": those two ``None``s mean opposite things, and
    conflating them (reading the cache's unreachable-``None`` as "fall back to
    loaded=True") is the #92 defect. The builder resolves the cache's ``None`` to
    a concrete ``False`` on the supplied path so this function can never see it.

    ``ready`` itself — the clamp that makes a stale/wrong ``ready_signal`` (or
    ``peer_signal``) unable to fabricate ``ready=True`` for an unwired,
    undialable, or hardware-infeasible role, plus the separate proxied-role
    ``peer_signal`` channel — is computed by :func:`_resolve_ready`; see its
    docstring for the full rationale (issues #92, #115/#127).
    """
    backend = next((b for b in table.backends if b.name == ROLE_BACKEND[role]), None)
    loaded = backend is not None
    feasible = ROLE_BACKEND[role] not in table.infeasible
    if backend is not None:
        model_id = backend.served_name
    else:
        # Not wired: still name the model this role WOULD serve (catalog default).
        canonical = _catalog_by_role_hint(ROLE_ROLE_HINT[role])
        model_id = canonical.id if canonical else ""
    # Metadata: prefer the entry matching the served id; fall back to the role's
    # canonical entry when the operator serves a non-catalog name.
    entry = _catalog_by_id(model_id) or _catalog_by_role_hint(ROLE_ROLE_HINT[role])
    native_context = entry.native_max_model_len if entry else 0
    endpoint = gateway
    ready = _resolve_ready(loaded, feasible, endpoint, ready_signal, peer_signal)
    return RoleInfo(
        role=role,
        model=model_id,
        runtime=_VLLM_RUNTIME,
        endpoint=endpoint,
        path=ROLE_PATH[role],
        context=_served_context(role, env, native_context),
        quant=entry.quantization if entry else "",
        mtp=bool(entry.speculative_config) if entry else False,
        tools=bool(entry.tool_parser) if entry else False,
        responsibilities=ROLE_RESPONSIBILITIES[role],
        forbidden_responsibilities=ROLE_FORBIDDEN[role],
        feasible=feasible,
        ready=ready,
        loaded=loaded,
    )


def _audio_role(
    role: str,
    model: str,
    runtime: str,
    endpoint: str,
    loaded: bool,
    *,
    ready: bool | None = None,
    feasible: bool = True,
    responsibilities: tuple[str, ...] | None = None,
) -> RoleInfo:
    """Resolve an audio-overlay role (stt/tts). No catalog entry → 0/""/False.

    ``feasible`` is the #129 first-class channel: ``False`` when the operator
    declared the lane off (``STT_/TTS_FEASIBLE=false`` →
    ``table.infeasible``), which is what lets
    :func:`annotate_peer_referrals` attach ``hosted_by``/``proxied`` to an
    audio role exactly as it does to a dropped core role.

    ``responsibilities`` defaults to the static :data:`ROLE_RESPONSIBILITIES`
    entry for ``role`` when omitted; :func:`_resolve_audio_role` passes an
    explicit, honesty-gated tuple for ``stt`` (issue #149 t4 — see
    :data:`STT_REALTIME_RESPONSIBILITY`) so this function itself never has to
    know about the conditional.

    ``tools=False`` is a fact, not a fallback: the audio sidecars serve
    transcription/synthesis, not a chat lane that could accept ``tools``.
    """
    if ready is None:
        ready = loaded
    return RoleInfo(
        role=role,
        model=model,
        runtime=runtime,
        endpoint=endpoint,
        path=ROLE_PATH[role],
        context=0,
        quant="",
        mtp=False,
        tools=False,
        responsibilities=(
            responsibilities if responsibilities is not None else ROLE_RESPONSIBILITIES[role]
        ),
        forbidden_responsibilities=ROLE_FORBIDDEN[role],
        feasible=feasible,
        ready=ready,
        loaded=loaded,
    )


def build_role_registry(
    table: RoutingTable,
    server: ServerConfig,
    *,
    env: Mapping[str, str] | None = None,
    gateway_url: str | None = None,
    audio_ready: bool | None = None,
    backend_ready: Mapping[str, bool | None] | None = None,
    peer_ready: Mapping[str, bool | None] | None = None,
) -> dict[str, RoleInfo]:
    """Resolve the ten first-class roles to live metadata — the #81 contract.

    This is the ONE canonical builder both the CLI (t5) and gateway (t6) call.
    Its inputs are exactly what :func:`lobes.gateway._config.build_config`
    returns (``table``, ``server``), plus the raw ``env`` mapping for the
    served-context overlay below — no new config source is invented.

    :param table: the gateway routing table — its wired :class:`Backend` objects
        tell us which roles are ``loaded`` and each role's served model id.
    :param server: the gateway server config — supplies the audio overlay URL
        (``audio_url``) for stt/tts and, absent ``gateway_url``, the (very
        narrow) ``public_url`` fallback for the six gateway-fronted roles —
        see :func:`_gateway_base_url`.
    :param env: the deployment's environment mapping, consulted ONLY for the
        served ``--max-model-len`` overlay (:data:`ROLE_MAX_MODEL_LEN_ENV`) —
        so ``RoleInfo.context`` reports what the deployment actually SERVES
        (e.g. ``PRIMARY_MAX_MODEL_LEN``), not just the catalog native. ``None``
        (the default) or a mapping missing the relevant key falls back to the
        catalog native — the t4 behaviour is unchanged when ``env`` is omitted.
        Kept separate from ``table``/``server`` (typically built from the SAME
        env) so a caller assembling those by hand isn't forced to also pass it.
    :param gateway_url: the caller-facing gateway base URL for cortex / senses /
        embedder / reranker. When ``None``, it falls back to
        ``server.public_url`` (an operator-declared ``GATEWAY_PUBLIC_URL``) and,
        failing that, to ``""`` — it is NEVER fabricated from
        ``server.host``/``server.port`` (issue #81 t5; those are the gateway's
        INTERNAL listen config, not a client-reachable address — see
        :func:`_gateway_base_url`). Audio roles also use this origin as their
        endpoint when the overlay is wired (issue #87).
    :param audio_ready: optional live-readiness signal for stt/tts (issue #89).
        When not ``None`` it sets the audio roles' ``ready`` (the runtime signal)
        — ``loaded`` stays the config fact ``bool(audio_url)``. When ``None``,
        ``ready`` falls back to ``bool(audio_url)`` (the CLI/back-compat path).
    :param backend_ready: optional live-readiness signal for the six
        gateway-fronted roles (issue #81 t5), keyed by the internal
        :class:`~lobes.gateway._routing.Backend` name (:data:`ROLE_BACKEND`'s
        values — ``"primary"``/``"multimodal"``/``"embed"``/``"rerank"``), one
        tri-state value (``True``/``False``/``None``) per backend — exactly
        the shape :meth:`lobes.gateway._readiness.ReadinessCache.current`
        returns, so a caller passes ``current()`` STRAIGHT THROUGH with no
        translation and no per-call-site coercion. **When it is supplied it is
        AUTHORITATIVE**, and this builder self-enforces the invariant its shape
        implies (issue #92 / honesty h14): ``ready = (backend_ready.get(name)
        is True)`` — a present ``None``, a present ``False``, and a MISSING KEY
        all mean NOT ready. That matters because the readiness cache reports a
        dead/missing/unreachable backend as ``None`` — and the cache's ``None``
        means UNREACHABLE, the OPPOSITE of "no signal, assume the wired
        default". Reading that ``None`` as "fall back to ``loaded`` (=``True``
        for a wired backend)" is the exact #92 defect a dead backend advertised
        as ``ready=True``); because the supplied branch is authoritative, that
        cannot recur, and no caller-side ``_ready_iff_true``-style bridge is
        needed. Only when ``backend_ready`` is ``None`` (the default — the
        mapping OMITTED, not a per-backend ``None``) does ``ready`` fall back to
        ``loaded``, the original t4 behaviour, so every existing non-HTTP caller
        (the CLI, this module's own offline test suite) is unchanged. ``loaded``
        stays the config fact "is this backend wired" in all cases. ``roles.py``
        itself never probes anything to produce this signal — it is computed
        elsewhere (t3's :class:`~lobes.gateway._readiness.ReadinessCache`,
        socket-free to read) and handed in, exactly like ``audio_ready``.
    :param peer_ready: optional live-readiness signal for PROXIED roles
        (proxy-lobes t6, issues #115/#127) — the NEW, SEPARATE channel the
        t5 clamp docstring demanded, keyed by backend name like
        ``backend_ready`` but carrying the PEER-probe verdict
        (:func:`lobes.gateway._readiness.probe_peer_ready` via the readiness
        cache's peer thread: the declared peer answered 200 AND its own
        ``/v1/models`` lists the served id). Consulted ONLY for a role whose
        backend is in ``table.peer_proxied``; for exactly those roles
        ``ready`` reflects it (``is True`` — the h14 missing-key/None/False
        discipline applies), which is the live proxied-path probe honesty h2
        requires. ``backend_ready`` — the LOCAL probe channel — still never
        unclamps a proxied role (a healthy local process is not evidence the
        peer serves the model). ``None`` (the default — every pre-t6 caller,
        and every deployment with no proxied roles) leaves every role's
        ``ready`` exactly as before: a proxied role without a live peer
        signal is honestly not-ready, never hardcoded true.
    :returns: an ordered ``dict`` keyed by role name with EXACTLY the ten roles.
        Every role is always present — an unconfigured/opt-in role (stt/tts with
        ``audio_url`` unset, or an unwired embed/rerank/multimodal backend) is
        returned with ``loaded=False``, never omitted and never raising.

    Readiness (``RoleInfo.ready``) is no longer a bare alias of ``loaded``
    (issue #81 t5 — generalising the stt/tts split from issue #89/#90 to all
    ten roles). When a caller supplies ``backend_ready``/``audio_ready`` it is
    AUTHORITATIVE (a present ``None``/``False`` or a missing key ⇒ not ready);
    only an OMITTED signal falls back to the coarse "configured/wired"
    ``loaded`` proxy. Either way it is CLAMPED, here, to ``False`` whenever a
    role's backend is not wired OR its resolved ``endpoint`` is empty — a
    caller can never fabricate ``ready=True`` for a role with nothing to dial,
    regardless of what signal it passes in. ``roles.py`` stays pure/offline
    either way — it opens no socket to produce or consume this signal; true
    liveness is probed elsewhere (t3's ``ReadinessCache`` /
    ``probe_audio_ready``, issue #89) and handed in as a plain value.
    """
    resolved_env: Mapping[str, str] = env if env is not None else {}
    gateway = (gateway_url or _gateway_base_url(server)).rstrip("/")
    registry: dict[str, RoleInfo] = {}

    for role in GATEWAY_FRONTED_ROLES:
        if backend_ready is None:
            # NOT SUPPLIED → back-compat: no live signal at all, so fall back to
            # the coarse `loaded` proxy (the original t4 behaviour). `None` here
            # is `_gateway_role`'s "fall back to loaded" sentinel — never confused
            # with the AUTHORITATIVE branch below, which never passes it a `None`.
            signal = None
        else:
            # SUPPLIED → AUTHORITATIVE, and resolved to a concrete bool HERE so a
            # present `None`, a present `False`, and a MISSING KEY all collapse to
            # "not ready" (issue #92 / honesty h14). This is the invariant this
            # builder now SELF-ENFORCES rather than leaving to caller discipline:
            # a supplied mapping is the single source of truth, and "no live
            # signal" is never evidence of health. In particular
            # `ReadinessCache.current()` reports a dead/unreachable backend as
            # `None`; reading that `None` as "no signal → fall back to loaded"
            # (which for a wired backend is `True`) is the exact #92 defect — the
            # cache's `None` means UNREACHABLE, the opposite of "unknown, assume
            # configured". By passing `_gateway_role` a concrete `True`/`False`
            # (never `None`) on the supplied path, that trap cannot recur.
            signal = backend_ready.get(ROLE_BACKEND[role]) is True
        # The SEPARATE peer channel (t6): only a PROXIED role's backend, and
        # only when a live peer_ready mapping was supplied, gets a concrete
        # bool (missing key / present None / present False all collapse to
        # "not ready", the same h14 discipline as the local channel above).
        # Every other role — and every caller without a peer signal — passes
        # None, so _gateway_role's clamp behaves exactly as before.
        peer_signal = None
        if peer_ready is not None and ROLE_BACKEND[role] in table.peer_proxied:
            peer_signal = peer_ready.get(ROLE_BACKEND[role]) is True
        registry[role] = _gateway_role(role, table, gateway, resolved_env, signal, peer_signal)

    audio_url = (server.audio_url or "").rstrip("/")
    audio_configured = bool(audio_url)
    # Audio roles use the gateway origin when the overlay is wired (issue #87),
    # but fall back to empty endpoint when it is not wired.
    audio_endpoint = gateway if audio_configured else ""
    # `loaded` is a config fact — is the audio overlay wired in THIS deployment —
    # kept SEPARATE from `ready`, the runtime signal. `ready` is the gateway's
    # live probe (`audio_ready`) when it supplied one, else it falls back to the
    # configured signal. Keeping them apart means a warming backend reports
    # loaded=True/ready=False (deployed but not yet consumable) instead of
    # masquerading as not-deployed, and an unconfigured overlay never reports a
    # ready role with an empty endpoint.
    #
    # Clamp on `audio_configured` AND `audio_endpoint` so that last invariant
    # holds STRUCTURALLY, not merely by caller discipline: an unconfigured
    # overlay, or one whose endpoint came back empty because no gateway_url/
    # public_url was known (issue #81 t5, criterion 3), is never ready, no
    # matter what `audio_ready` a caller passes. When configured AND dialable,
    # use the live probe signal if one was supplied, else fall back to the
    # configured fact.
    audio_ready_signal = (
        audio_configured
        and bool(audio_endpoint)
        and (audio_ready if audio_ready is not None else True)
    )
    for role, model, runtime in (
        ("stt", _STT_MODEL, _STT_RUNTIME),
        ("tts", _TTS_MODEL, _TTS_RUNTIME),
    ):
        registry[role] = _resolve_audio_role(
            role,
            model,
            runtime,
            table,
            endpoint=audio_endpoint,
            configured=audio_configured,
            ready_signal=audio_ready_signal,
            peer_ready=peer_ready,
        )
    return registry


def _resolve_audio_role(
    role: str,
    model: str,
    runtime: str,
    table: RoutingTable,
    *,
    endpoint: str,
    configured: bool,
    ready_signal: bool,
    peer_ready: Mapping[str, bool | None] | None,
) -> RoleInfo:
    """One audio lane's :class:`RoleInfo`, with first-class feasibility (#129).

    An operator declares a lane off with ``STT_/TTS_FEASIBLE=false`` — the
    same channel as a dropped core role; absent (every pre-#129 deployment)
    the lane stays feasible, so the sleeping-lobe contract renders
    byte-identically. A declared-off lane is flagged (``feasible:false``),
    never hidden — ``loaded`` is honestly ``False`` and ``ready`` follows the
    PEER probe when the role is proxied and a live peer signal was supplied
    (the same h14 missing-key/None/False discipline the core roles use); a
    healthy LOCAL bridge is not evidence the peer serves the lane.

    The realtime/VAD session capability (issue #149 t4, see
    :data:`STT_REALTIME_RESPONSIBILITY`) is folded into ``stt``'s
    ``responsibilities`` HERE, and only on this — the feasible — branch,
    ``configured`` (an actually-wired audio overlay, i.e. ``AUDIO_URL`` is
    set) is ALSO required: a text-only fleet (no ``--audio`` overlay) leaves
    ``configured=False`` and gets the static, unconditional base tuple
    unchanged, exactly like an operator-declared-off lane does on the other
    branch below. Neither ``tts`` nor a declared-off ``stt`` ever sees the
    extra token.
    """
    if role not in table.infeasible:
        responsibilities = ROLE_RESPONSIBILITIES[role]
        if role == "stt" and configured:
            responsibilities = responsibilities + (STT_REALTIME_RESPONSIBILITY,)
        return _audio_role(
            role,
            model,
            runtime,
            endpoint,
            configured,
            ready=ready_signal,
            responsibilities=responsibilities,
        )
    peer_signal = False
    if peer_ready is not None and role in table.peer_proxied:
        peer_signal = peer_ready.get(role) is True
    return _audio_role(role, model, runtime, "", False, ready=peer_signal, feasible=False)


def annotate_peer_referrals(payload: dict[str, dict], table: RoutingTable) -> dict[str, dict]:
    """Add the honest referral — ``hosted_by: <peer origin>`` — to each unhosted role.

    The ONE shared annotator both honesty surfaces call (the gateway's
    ``GET /capabilities`` via :func:`lobes.gateway.server.capabilities_payload`,
    and the CLI's offline fallback in ``lobes capabilities``), so the referral
    contract has exactly one implementation. Mutates ``payload`` in place (and
    returns it for convenience): for each gateway-fronted role whose entry says
    ``feasible: false`` (this box does not host it — the #113 dropped-lobe
    channel) AND whose backend has an OPERATOR-DECLARED peer origin in
    ``table.peer_origins`` (:data:`lobes.gateway._config.PEER_ORIGIN_ENV`,
    mesh-brain t3), a ``hosted_by`` key naming that origin is added.

    Everything else is untouched — a hosted role is never annotated (even if
    an origin is declared for it: a referral says who hosts what THIS box does
    not), an unhosted role with no declared peer stays exactly as it was, and
    with ``table.peer_origins`` empty (the default) the payload is
    byte-identical to the pre-referral contract. The origin is metadata for
    the CALLER to dial directly; THIS FUNCTION never forwards a request to
    it — it only annotates. A name whose operator ALSO armed
    ``<PREFIX>_PEER_PROXY`` (see the THIRD state below) IS forwarded, but by
    the data-plane proxy branch in :mod:`lobes.gateway.server`
    (:func:`~lobes.gateway.server._proxy_to_peer`, proxy-lobes t6, issues
    #115/#127), never by this pure/offline annotator. Audio roles (stt/tts)
    joined the channel in issue #129 — first-class entries in
    ``ROLE_BACKEND`` and ``FEASIBLE_ENV``/``PEER_*_ENV`` — so a declared-off
    audio lane with a declared peer gets the same ``hosted_by``/``proxied``
    annotations as any dropped core role.

    **A THIRD honesty state — PROXIED (proxy-lobes t5/t6, issues #115/#127).**
    Referral above says "ask the peer yourself"; a role whose backend name is
    ALSO in ``table.peer_proxied`` (the operator's ``<PREFIX>_PEER_PROXY``
    opt-in — :data:`lobes.gateway._config.PEER_PROXY_ENV`, t1) is one this box
    has committed to answering ON THE PEER'S BEHALF — the gateway itself
    FORWARDS the request via the data-plane proxy branch
    (:func:`lobes.gateway.server._proxy_to_peer`, landed in t6; this module
    itself stays pure/offline and dials nothing — it only adds the marker
    below). That is a materially different claim from a bare referral, so it
    gets its own explicit marker,
    ``"proxied": true``, added ALONGSIDE (never instead of) ``hosted_by`` — the
    origin named there is unchanged: it is still "whoever ultimately serves
    this", now additionally reachable by asking THIS box too.
    ``table.peer_proxied`` is a subset of ``table.infeasible`` ∩
    ``table.peer_origins`` by construction (:func:`lobes.gateway._config.
    _peer_proxied`), so a proxied role always also gets ``hosted_by`` — the
    three states are told apart by KEY PRESENCE alone, never by a sentinel
    value:

    * **hosted** (this box serves it) — neither key present.
    * **referral-only** (dropped, no local proxy) — ``hosted_by`` present,
      ``proxied`` ABSENT (never ``false``) — mirrors ``hosted_by``'s own
      optional-key convention above: a key that doesn't apply is omitted, not
      set to a falsy sentinel, so ``"proxied" in entry`` is itself the signal.
    * **proxied** (dropped, this box forwards) — BOTH ``hosted_by`` and
      ``proxied: true`` present.

    ``feasible`` stays ``false`` for a proxied role in all cases — it remains
    a HARDWARE/deployment fact ("this box does not itself host the model"),
    independent of whether a request for it happens to be answerable via a
    forward. Likewise ``ready`` is never forced ``true`` here: it is left
    exactly as :func:`build_role_registry` already computed it — which, since
    proxy-lobes t6, means a proxied role's ``ready`` reflects the live
    PEER-probe verdict when the caller threaded one through the ``peer_ready``
    channel (see :func:`build_role_registry`), and stays the clamp's honest
    ``False`` otherwise. This annotator adds no readiness claim of its own.

    With ``table.peer_proxied`` empty (the default — every deployment that
    predates issues #115/#127, and every referral-only or no-peer deployment
    today) this branch never fires, so the payload is byte-identical to the
    pre-proxy contract, exactly as it already is byte-identical to the
    pre-referral one when ``table.peer_origins`` is empty.
    """
    for role, backend in ROLE_BACKEND.items():
        entry = payload.get(role)
        if not isinstance(entry, dict) or entry.get("feasible") is not False:
            continue
        origin = table.peer_origins.get(backend)
        if origin:
            entry["hosted_by"] = origin
            if backend in table.peer_proxied:
                entry["proxied"] = True
    return payload


# The five DECLARED lane-fingerprint suffixes a role's backend may carry in
# `table.lane_fingerprints` (see `lobes.gateway._config.LANE_FINGERPRINT_SUFFIXES`)
# do NOT include "RUNTIME" — no `<PREFIX>_RUNTIME` knob exists yet (t2 landed
# only the five engine knobs an operator would reasonably expect identical
# across a role's replica pool). So the offline fingerprint fallback below
# always reads "runtime" as unknown; that is honest, not a bug in this module
# — inventing a runtime from the catalog is exactly what c33/h25 forbid.
_FINGERPRINT_DECLARED_FIELDS: tuple[tuple[str, str], ...] = (
    ("runtime", "RUNTIME"),
    ("quantization", "QUANTIZATION"),
    ("kv_cache_dtype", "KV_CACHE_DTYPE"),
    ("reasoning_parser", "REASONING_PARSER"),
    ("tool_parser", "TOOL_CALL_PARSER"),  # the lane knob is *_TOOL_CALL_PARSER (Qodo, PR #213)
    ("speculative_config", "SPECULATIVE_CONFIG"),
)


def _offline_fingerprint(declared: Mapping[str, str], entry: Mapping[str, object]) -> dict:
    """Build a fingerprint dict with NO live probe: declared knobs + the
    payload's own served id/context, 'unknown' for everything else.

    Never touches the catalog (c33/h25) — `entry["model"]`/`entry["context"]`
    are already the SERVED values `lobes.roles.build_role_registry` resolved
    (deployment override, or the catalog default when unwired), which is the
    same honesty basis :class:`~lobes.gateway._replicas.Fingerprint` uses for
    its own LIVE `served_id`/`max_model_len` fields — just sourced from the
    payload instead of a live `/v1/models` probe.
    """
    served_id = entry.get("model") or _REPLICA_UNKNOWN
    max_len = entry.get("context") or None
    fingerprint: dict[str, object] = {
        "served_id": served_id,
        "max_model_len": max_len,
    }
    for field_name, suffix in _FINGERPRINT_DECLARED_FIELDS:
        fingerprint[field_name] = declared.get(suffix, _REPLICA_UNKNOWN)
    return fingerprint


def _replica_row_from_state(state: ReplicaState) -> dict[str, object]:
    return {
        "origin": state.origin,
        "local": state.local,
        "ready": state.ready,
        "busy": state.busy,
        "running": state.running,
        "waiting": state.waiting,
        "compatible": state.compatible,
        "reason": state.reason,
        "fingerprint": dataclasses.asdict(state.fingerprint) if state.fingerprint else None,
        # `weight` is the RESOLVED capacity `_selection.py` ranked by (post
        # clamp, post kill switch) — unchanged key/meaning from before t6.
        "weight": state.weight,
        # `capacity` (t6, additive) is the RAW capacity that replica claimed,
        # pre-clamp — `None` when none was published. Kept alongside
        # `weight` so a clamped peer (the diagnostic case: the two differ)
        # is explainable from /capabilities alone, per the spec's h1/c24.
        "capacity": state.capacity,
    }


def _offline_replica_rows(
    table: RoutingTable, backend: str, local_fingerprint: dict | None
) -> list[dict[str, object]]:
    """The declared-only replica view: no probe ran, so every live field is
    honestly `None` rather than guessed — see :func:`annotate_replicas`.

    ``weight``/``capacity`` follow the same honesty rule (t6): no probe ran,
    so no capacity was ingested — ``weight`` reports the
    :data:`~lobes.gateway._replicas.UNCALIBRATED_WEIGHT` sentinel (the
    resolved-capacity fallback :func:`~lobes.gateway._selection.select_replica`
    itself treats as "nothing published", never a measured one-slot capacity)
    and ``capacity`` reports ``None`` — a not-probed row never guesses a raw
    claimed number it never received.
    """
    local_origin = table.self_origin or "local"
    rows: list[dict[str, object]] = [
        {
            "origin": local_origin,
            "local": True,
            "ready": None,
            "busy": None,
            "running": None,
            "waiting": None,
            "compatible": None,
            "reason": "not probed (offline)",
            "fingerprint": local_fingerprint,
            "weight": UNCALIBRATED_WEIGHT,
            "capacity": None,
        }
    ]
    for origin in table.replica_origins.get(backend, ()):
        rows.append(
            {
                "origin": origin,
                "local": False,
                "ready": None,
                "busy": None,
                "running": None,
                "waiting": None,
                "compatible": None,
                "reason": "not probed (offline)",
                "fingerprint": None,
                "weight": UNCALIBRATED_WEIGHT,
                "capacity": None,
            }
        )
    return rows


def annotate_replicas(
    payload: dict[str, dict],
    table: RoutingTable,
    snapshot: Mapping[str, tuple[ReplicaState, ...]] | None = None,
) -> dict[str, dict]:
    """Add the additive per-role ``fingerprint``/``replicas`` keys (#199, t6).

    Sibling of :func:`annotate_peer_referrals` — same "one shared annotator,
    both the gateway and the CLI offline fallback call it" discipline, kept
    as a SEPARATE function rather than folded into that one so
    ``annotate_peer_referrals``'s own behaviour (and every test that pins it)
    stays byte-identical. Mutates ``payload`` in place and returns it.

    A role is annotated only when it has a declared replica pool: its
    backend name (:data:`ROLE_BACKEND`) appears in ``table.replica_origins``,
    OR ``snapshot`` carries a (non-empty) tuple for it. This is the gate that
    keeps a pre-pool deployment (no ``*_PEER_ORIGINS`` anywhere, no snapshot)
    byte-identical: with both empty, this function is a no-op for every role,
    so ``payload`` never gains a ``fingerprint``/``replicas`` key it did not
    already have — the spec's h1/c9 requirement.

    **fingerprint** — what THIS box's own replica of the role is actually
    serving, as a plain dict: ``{served_id, max_model_len, runtime,
    quantization, kv_cache_dtype, reasoning_parser, tool_parser,
    speculative_config}``. Two provenances, mirroring
    :class:`~lobes.gateway._replicas.Fingerprint`'s own rule and NEVER the
    catalog (c33/h25 — the exact defect that mislabels the Orin's llama.cpp
    replica as ``quant=modelopt``):

    * ``snapshot`` supplied and this role's local :class:`ReplicaState`
      carries a live-probed ``fingerprint`` → that fingerprint, verbatim
      (:func:`dataclasses.asdict`).
    * otherwise → :func:`_offline_fingerprint`: the payload's own SERVED
      ``model``/``context`` (already deployment-resolved by
      :func:`build_role_registry`) plus ``table.lane_fingerprints`` for the
      five declared engine knobs, ``"unknown"`` for anything undeclared.

    **replicas** — one entry per replica of the role, local first:
    ``{origin, local, ready, busy, running, waiting, compatible, reason,
    fingerprint, weight, capacity}``. With a ``snapshot`` this is a straight
    :class:`ReplicaState` → dict projection (live numbers, `None` fingerprint
    for an unprobed/incompatible peer). Without one (the CLI's offline path,
    which never has a live snapshot to hand) it is the DECLARED list only —
    :func:`_offline_replica_rows` — with every live field honestly ``None``
    and ``reason: "not probed (offline)"``, rather than guessing readiness
    from a config file (the same #96 lesson ``lobes capabilities``'s ``ready``
    clamp already applies one level up). ``weight`` (t6, additive) is the
    RESOLVED capacity ``_selection.py`` ranked by — the ingested value after
    the clamp and kill switch, or the
    :data:`~lobes.gateway._replicas.UNCALIBRATED_WEIGHT` sentinel when
    nothing was published or probed; ``capacity`` is the RAW capacity that
    replica claimed, pre-clamp, ``None`` when none was published/probed. The
    two are kept SEPARATE so a clamped peer — the case where they diverge —
    is explainable from this row alone, rather than collapsed into one
    number.

    Existing keys are never touched: ``feasible``/``proxied``/``hosted_by``/
    ``ready``/``loaded`` keep their documented type and single-owner meaning
    exactly as :func:`annotate_peer_referrals` left them.
    """
    resolved_snapshot: Mapping[str, tuple[ReplicaState, ...]] = snapshot or {}
    for role, backend in ROLE_BACKEND.items():
        entry = payload.get(role)
        if not isinstance(entry, dict):
            continue
        declared_peers = table.replica_origins.get(backend, ())
        role_snapshot = resolved_snapshot.get(role, ())
        if not declared_peers and not role_snapshot:
            continue  # no replica pool declared or probed for this role
        local_state = next((s for s in role_snapshot if s.local), None)
        if local_state is not None and local_state.fingerprint is not None:
            fingerprint = dataclasses.asdict(local_state.fingerprint)
        else:
            fingerprint = _offline_fingerprint(table.lane_fingerprints.get(backend, {}), entry)
        entry["fingerprint"] = fingerprint
        if role_snapshot:
            entry["replicas"] = [_replica_row_from_state(s) for s in role_snapshot]
        else:
            entry["replicas"] = _offline_replica_rows(table, backend, fingerprint)
    return payload


def role_registry_from_env(
    env: Mapping[str, str] | None = None,
    *,
    gateway_url: str | None = None,
    audio_ready: bool | None = None,
    backend_ready: Mapping[str, bool | None] | None = None,
) -> dict[str, RoleInfo]:
    """Build the role registry straight from an env mapping.

    A thin convenience over the one canonical builder: it runs the same
    :func:`lobes.gateway._config.build_config` the gateway uses, then delegates
    to :func:`build_role_registry` — threading the SAME resolved env through as
    ``env`` too, so the served-context overlay (``PRIMARY_MAX_MODEL_LEN`` and
    friends, t5) is applied automatically. Lets a host-side caller (the CLI, t5)
    build the registry from a deployment's ``.env`` without assembling a
    ``RoutingTable``/``ServerConfig`` pair by hand. ``env`` defaults to
    ``os.environ`` when omitted (matching :func:`build_config`'s default).
    ``audio_ready``/``backend_ready`` pass straight through to
    :func:`build_role_registry` (both default ``None`` — this offline
    convenience never probes anything itself; a caller with a live signal in
    hand supplies it here exactly as it would to the canonical builder).
    """
    resolved_env = os.environ if env is None else env
    table, server = build_config(resolved_env)
    return build_role_registry(
        table,
        server,
        env=resolved_env,
        gateway_url=gateway_url,
        audio_ready=audio_ready,
        backend_ready=backend_ready,
    )
