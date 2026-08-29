"""``lobes init [TARGET]`` — scaffold a deployment directory.

Copies the packaged compose + ``env.example``→``.env`` (+ gateway Dockerfile)
into ``TARGET`` (default ``~/.lobes``; ``lobes init .`` for the local folder).

The DEFAULT topology is the **fleet duo** (issue #69): the always-warm Qwen
generate primary + the multimodal Gemma gear, fronted by the stdlib gateway with
the co-resident embedding/reranker gears (the legacy 4B ``minor`` / 14B
``middle`` generate gears stay behind opt-in compose profiles). ``--single``
(alias ``--legacy``) restores the old single-model scaffold (one vLLM server, no
gateway). ``--fleet`` is now a default-implied no-op kept for back-compat.
``--audio`` layers the realtime audio overlay on the fleet (incompatible with
``--single``). Mutating: dry-run by default; ``--apply`` writes, ``--force``
overwrites.

``--shape <machine-as-brain|spark-lobe|thor-lobe|orin-lobe|orin-cortex|orin-associate|
thor-muse|thor-worker|orin-small>``
(brain-shapes t4, issue #113; ``orin-small`` added by the mesh-brain
end-state's t2, issue #112; ``thor-muse`` added with the seventh role, muse;
``orin-lobe`` added by the Orin variation; ``orin-cortex`` by the
qwen3-8-gguf-llamacpp plan; ``orin-associate`` by the lightning-on-orin plan's
t9)
selects the DEPLOYMENT-SHAPE axis — which roles THIS box hosts at all
— composed on top of whichever per-machine
:class:`~lobes.profiles.schema.Profile` detection/``--profile`` resolves (the
per-machine TUNING axis, issue #110). Fleet topology only (a fleet-scaffold
axis — incompatible with ``--single``). The default, ``machine-as-brain``,
hosts every one of the seven default-hosted Colleague roles this card can serve —
today's behaviour, unchanged — and t3's
:func:`~lobes.profiles.shape_render.render_shape` composes it as a strict
no-op over the profile, so a bare ``lobes init`` (no ``--shape`` at all) makes
zero new decisions and renders byte-identically to before this flag existed.
The mesh-brain alternatives drop one generate lobe to a peer box and reclaim
its GPU-memory budget: ``spark-lobe`` (drops ``senses``), ``thor-lobe`` (drops
``cortex``). ``orin-lobe`` is ``thor-lobe``'s sm_87 sibling — senses + the
pooling gears, no ``cortex`` (Ampere cannot run the NVFP4 primary) and no
``stt``/``tts`` (the Parakeet base image carries no sm_87 kernels, so audio is
forwarded to a peer via ``AUDIO_URL``). ``orin-cortex`` is the same board's
opposite answer — it keeps ``cortex`` LOCAL, on the **llama.cpp** lane (the
``orin`` card declares a GGUF gear, which is weight-only and decodes on
Ampere) and drops ``senses``, which cannot co-reside with it in 61.3 GiB.
``orin-associate`` is a THIRD answer on the same board — it hosts the opt-in
``associate`` lobe (Nemotron 3.5 Lightning, W4A16/FP8-mixed NVFP4, which
decodes on Ampere for the same weight-only reason the GGUF cortex does) and
drops BOTH ``cortex`` and ``senses`` to a peer, likewise with no ``stt``/
``tts``. ``orin-small`` drops BOTH heavy lobes and hosts the opt-in ``minor``
gear instead. All four Orin mesh-lobe shapes are **declared, UNVALIDATED**
data only (no physical Jetson AGX Orin has booted any of them; the #108
rule). An unknown ``--shape`` value is a user error naming the valid
(sorted) shapes.

A card may also REFUSE the default shape. A card profile can declare
mutually-exclusive roles (``[[exclusive_roles]]``, see
:class:`~lobes.profiles.schema.ExclusiveRoles`) — roles the board can each
serve, but not both at once — and ``machine-as-brain`` hosts every role a card
marks feasible, so on such a card the decision-free path is exactly the one
that would over-commit the box. :func:`_guard_coresidency` turns that into a
:class:`~lobes.cli._errors.ModelGearError` naming the shapes the card declares
as the way out, on the dry run as well as ``--apply``. An EXPLICIT ``--shape``
warns and proceeds instead — only the *defaulted* shape is refused. A card
declaring no group (every built-in profile but ``orin``) is unaffected.

A third thing ``--apply`` may generate is the **GPU-access override** pair
(``docker-compose.gpu.yml`` / ``docker-compose.gpu-audio.yml``): a card profile
that declares ``gpu_access = "runtime"`` (a board whose NVIDIA container
toolkit resolves to legacy csv mode) gets its GPU services' ``deploy:`` stanza
``!reset`` away in favour of ``runtime: nvidia``. Purely card-driven — no flag,
no shape involvement — and written on every ``--apply``, which is what makes it
survive a re-render. Every card that takes the default ``gpu_access =
"devices"`` writes nothing at all.

``--from-lock <path-or-dir>`` (deployment-lock plan, t7) is a **fourth thing
entirely: a SOURCE, not another input to the renderer.** The three axes above
(topology, ``--profile``, ``--shape``) all feed
:mod:`lobes.profiles.render`; ``--from-lock`` bypasses that path completely and
materialises a COMMITTED variation — the compose files, overrides and
Dockerfiles a box actually ran, digest-checked against its
``deployment.lock.toml`` — verbatim. That bypass is what makes a restore
byte-identical to what the box ran, hand edits included, instead of to what
the renderer would produce today.

Three consequences follow, and all three are enforced rather than documented:

* ``.env`` stays **merge-only** (:data:`lobes.runtime._compose.MERGE_ONLY_FILES`).
  A restore may replace compose files and Dockerfiles wholesale; it appends the
  lock's rendered knobs only where the ``.env`` lacks them and never rewrites an
  existing line. No secret is restorable from a lock — it carries none, by
  construction.
* Bypassing resolution also bypasses :func:`_sync_gpu_overrides`, so the
  card-driven csv-vs-devices GPU-access correction does not run. A lock whose
  declared variation differs from what :mod:`lobes.runtime._detect` +
  :mod:`lobes.variation` resolve on this box is therefore REFUSED
  (:func:`_guard_variation`); the override is its own explicit flag,
  ``--allow-variation-mismatch``, never ``--force``.
* The generated overlays the lock does NOT name are removed
  (:data:`RESTORE_SYNCED_FILES`) — the remove-on-mismatch behaviour of
  ``_sync_shape_override`` / ``_sync_gpu_overrides``, surviving a lock round
  trip. Nothing else is ever deleted.

Mutation safety is unchanged: dry-run by default, ``--apply`` to write.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lobes import __version__
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.cli._runtime_ops import resolve_init_profile
from lobes.profiles.schema import GPU_ACCESS_RUNTIME
from lobes.profiles.schema import ROLES as CORE_ROLES
from lobes.profiles.schema import ExclusiveRoles, Profile
from lobes.profiles.shape_render import (
    GATEWAY_SERVICE,
    ROLE_SERVICE,
    compose_profile,
    overcommitted_groups,
    render_shape,
    role_service,
)
from lobes.profiles.shapes import (
    OPT_IN_CORE_ROLES,
    Shape,
    builtin_shape_names,
    resolve_shape,
)
from lobes.runtime import _compose, _detect, _env
from lobes.runtime._buildability import check_lock_buildability
from lobes.runtime._lock import LOCK_FILENAME, DeploymentLock, file_digest, load_lock
from lobes.variation import resolve_variation_id

# The deployment shape a bare `lobes init` (no --shape) resolves: the
# whole-brain identity shape, hosting every role this card can serve. t3's
# render_shape composes it as a strict no-op over the resolved Profile (see
# tests/test_shape_goldens.py), which is what makes the bare path's rendering
# provably unchanged by the existence of this flag.
DEFAULT_SHAPE = "machine-as-brain"

# The compose profile a shape-dropped core service is parked in. NOTHING ever
# activates it (no COMPOSE_PROFILES entry, no CLI target), so `docker compose up`
# skips every service assigned to it — this is what makes a dropped lobe not RUN,
# rather than merely be flagged off in .env (brain-shapes t4b, issue #113).
SHAPE_DROPPED_PROFILE = "shape-dropped"


def _shape_dropped_services(shape: Shape, profile: Profile) -> list[str]:
    """The base-fleet core compose SERVICES a (shape, card) pair must NOT run, sorted.

    Primarily a function of the shape's ``hosts`` (the deployment-shape axis) via
    t3's render API ``ROLE_SERVICE`` map over the DEFAULT-HOSTED Profile-machinery
    core roles — never hardcoded per shape. Card feasibility is a SEPARATE axis
    (#107) and is deliberately not folded in: this override exists to enforce the
    SHAPE's own drop decision, so machine-as-brain (hosts every default role)
    always yields ``[]`` on every card whose roles are vLLM-served. Opt-in core
    roles (``muse``/``worker``) are excluded: their services are already parked
    behind their own Docker Compose profile in the base template (nothing
    activates it unless a hosting shape renders ``COMPOSE_PROFILES``), so a
    non-hosting shape has nothing to park.

    The card DOES enter through one narrow door: the ENGINE axis
    (qwen3-8-gguf-llamacpp t5). A role the shape HOSTS whose composed model is a
    non-vLLM catalog gear runs its alternative lane
    (:func:`~lobes.profiles.shape_render.role_service`), so the base template's
    vLLM service for that role must be parked too — otherwise ``docker compose
    up`` would start BOTH cortex lanes and the vLLM one would crash-loop trying
    to load a ``.gguf``. Every card whose roles are vLLM-served (all of them
    before this axis) yields exactly what it yielded before, machine-as-brain's
    empty list included.
    """
    composed = compose_profile(shape, profile)
    dropped: set[str] = set()
    for role in CORE_ROLES:
        if role in OPT_IN_CORE_ROLES:
            continue
        if not shape.hosts_role(role):
            dropped.add(ROLE_SERVICE[role])
            continue
        if role_service(role, composed.role(role)) != ROLE_SERVICE[role]:
            # Hosted, but by another engine's lane — park the vLLM one.
            dropped.add(ROLE_SERVICE[role])
    return sorted(dropped)


def render_shape_override(shape: Shape, profile: Profile) -> str | None:
    """The ``docker-compose.shape.yml`` override text for a (shape, card) pair.

    ``None`` when the shape drops no core role (machine-as-brain / bare init) — the
    caller writes no file, keeping the scaffold byte-identical to before this flag.
    Otherwise a docker-compose *override* (mirrors ``docker-compose.audio.yml``):
    each dropped core service is parked in the inert :data:`SHAPE_DROPPED_PROFILE`,
    and the gateway's ``depends_on`` is cleared with the compose ``!reset`` tag so
    it no longer references the now-profile-disabled service.
    """
    dropped = _shape_dropped_services(shape, profile)
    if not dropped:
        return None
    lines = [
        "# lobes deployment-SHAPE override — GENERATED by "
        "`lobes init --shape <mesh-shape> --apply`.",
        "#",
        "# brain-shapes t4b (issue #113). A docker-compose *override* layered LAST via",
        "#   docker compose -f docker-compose.yml [-f docker-compose.audio.yml] \\",
        "#                   -f docker-compose.shape.yml up -d",
        "# (`lobes fleet up --apply` auto-includes it when present). It parks each core",
        "# service this shape DROPS in the `shape-dropped` compose profile — a profile",
        "# NOTHING activates, so `docker compose up` skips the service. Without it the",
        "# base compose boots every core gear unconditionally, so a dropped lobe would",
        "# RUN and eat the GPU budget the shape reclaimed (proven live on the GB10).",
        "#",
        "# The gateway's base `depends_on` lists every core service, so once one is",
        "# profile-disabled that edge dangles (compose errors on / auto-enables a",
        "# depends_on to an inactive-profile service). The compose `!reset` tag CLEARS",
        "# the attribute (the value is ignored — list *replacement* is `!override`,",
        "# which is not what we want): the remaining core gears carry no profile and",
        "# start regardless of `depends_on`, and the gateway tolerates a backend still",
        "# loading (see its base comment), so dropping the start-order edge is safe.",
        "#",
        "# REQUIRES Docker Compose v2.24+ (compose-spec `!reset` merge tag).",
        "services:",
    ]
    for service in dropped:
        lines.append(f"  {service}:")
        lines.append(f'    profiles: ["{SHAPE_DROPPED_PROFILE}"]')
    lines.append(f"  {GATEWAY_SERVICE}:")
    lines.append("    depends_on: !reset null")
    return "\n".join(lines) + "\n"


def _sync_shape_override(target: Path, shape: Shape, profile: Profile) -> None:
    """Write the shape override for a role-dropping shape, or REMOVE a stale one.

    Re-initialising to machine-as-brain (or any shape that drops no core role) over
    a previous mesh-shape scaffold must scrub the stale ``docker-compose.shape.yml``
    — otherwise ``docker compose up`` would keep skipping the (now re-hosted) lobe
    (brain-shapes t4b, criterion 3).
    """
    override_path = target / _compose.SHAPE_OVERLAY
    text = render_shape_override(shape, profile)
    if text is None:
        if override_path.exists():
            override_path.unlink()
        return
    override_path.write_text(text, encoding="utf-8")


def _shape_override_plan(target: Path, shape: Shape, profile: Profile) -> dict:
    """What ``--apply`` would do to ``docker-compose.shape.yml`` — for the dry-run plan.

    ``action`` is ``write`` (shape drops a lobe), ``remove`` (a stale override is on
    disk but the selected shape drops nothing), or ``none``.
    """
    dropped = _shape_dropped_services(shape, profile)
    if dropped:
        return {"file": _compose.SHAPE_OVERLAY, "action": "write", "disables": dropped}
    if (target / _compose.SHAPE_OVERLAY).exists():
        return {"file": _compose.SHAPE_OVERLAY, "action": "remove", "disables": []}
    return {"file": _compose.SHAPE_OVERLAY, "action": "none", "disables": []}


def _shape_override_plan_line(plan: dict) -> str | None:
    """A human dry-run line for the shape-override plan, or ``None`` when nothing changes."""
    if plan["action"] == "write":
        return (
            f"  {plan['file']} (parks {', '.join(plan['disables'])} in the inert "
            f"'{SHAPE_DROPPED_PROFILE}' profile; !resets gateway depends_on)"
        )
    if plan["action"] == "remove":
        return f"  {plan['file']} (stale — would be REMOVED: this shape drops no lobe)"
    return None


def _shape_override_written(shape: Shape, profile: Profile) -> dict:
    """Post-``--apply`` shape-override state, for the JSON payload.

    ``_sync_shape_override`` has just run, so ``written`` (True iff the pair parks
    a core service) matches the file now on disk.
    """
    dropped = _shape_dropped_services(shape, profile)
    return {"file": _compose.SHAPE_OVERLAY, "written": bool(dropped), "disables": dropped}


# --- csv-mode GPU access (the card's gpu_access declaration) -----------------


def _gpu_override_text(services: tuple[str, ...], patches: str) -> str:
    """One GENERATED GPU-access override's text: ``!reset`` deploy, ask via ``runtime``.

    ``services`` are the GPU services declared by ``patches`` (the compose file
    this override layers onto) — a compose override may only name services some
    file in the same ``-f`` chain declares, which is why the two halves are
    separate files (see :data:`lobes.runtime._compose.GPU_OVERLAY`).
    """
    lines = [
        "# lobes GPU-ACCESS override — GENERATED by `lobes init --apply` when the resolved",
        f'# card profile declares gpu_access = "{GPU_ACCESS_RUNTIME}". Layers onto {patches}.',
        "#",
        "# WHY: a board whose NVIDIA container toolkit resolves to the legacy CSV mode",
        '# (nvidia-container-toolkit `mode = "auto"` on a Jetson AGX Orin, measured live',
        "# 2026-08-04 — docs/orin-profiles.md divergence 1) REFUSES the compose template's",
        "# `deploy.resources.reservations.devices` GPU request at container CREATE:",
        '#   "invoking the NVIDIA Container Runtime Hook directly … is not supported.',
        '#    Please use the NVIDIA Container Runtime"',
        "# csv mode wants the container told to use the `nvidia` runtime instead. Compose",
        "# has no conditional-block syntax, so no ${VAR} in the template can switch between",
        "# the two forms — only a second compose file can, which is what this is.",
        "#",
        "# The compose `!reset` tag CLEARS the base file's whole `deploy:` attribute (the",
        "# value is ignored; `!override` would REPLACE it, which is not what we want) —",
        "# every `deploy:` in the shipped templates holds nothing but the GPU reservation,",
        "# so nothing else is lost. REQUIRES Docker Compose v2.24+ (compose-spec `!reset`).",
        "#",
        "# Re-generated on every `lobes init --apply`, and REMOVED when the resolved card",
        "# no longer declares it — that is the whole point: the hand edit this replaces",
        "# did not survive a re-render.",
        "services:",
    ]
    for service in services:
        lines.append(f"  {service}:")
        lines.append("    deploy: !reset null")
        lines.append("    runtime: nvidia")
    return "\n".join(lines) + "\n"


def render_gpu_overrides(profile: Profile) -> dict[str, str] | None:
    """``{filename: text}`` for the card's GPU-access overrides, or ``None``.

    ``None`` for every card that takes the default ``gpu_access = "devices"``
    (base / spark / thor today) — the caller writes no file, so the deployment
    is byte-identical to before this knob existed.

    Both halves are rendered together and unconditionally: their content is a
    pure function of the shipped templates, never of what happens to be
    scaffolded, so the audio half is correct the moment ``--audio`` is added
    without a re-render being needed to make it so. The ``-f`` chain authority
    (:func:`lobes.runtime._compose.compose_file_args`) is what decides that the
    audio half is only ever passed alongside the audio overlay.
    """
    if profile.gpu_access != GPU_ACCESS_RUNTIME:
        return None
    return {
        _compose.GPU_OVERLAY: _gpu_override_text(_compose.GPU_SERVICES, _compose.COMPOSE_FILE),
        _compose.GPU_AUDIO_OVERLAY: _gpu_override_text(
            _compose.GPU_SERVICES_AUDIO, _compose.AUDIO_OVERLAY
        ),
    }


def _gpu_override_files() -> list[str]:
    """The two generated GPU-override filenames, in chain order."""
    return [_compose.GPU_OVERLAY, _compose.GPU_AUDIO_OVERLAY]


def _sync_gpu_overrides(target: Path, profile: Profile) -> None:
    """Write the card's GPU-access overrides, or REMOVE stale ones.

    Re-initialising a csv-mode deployment onto a card that takes the default
    GPU access must scrub the files — otherwise ``docker compose`` would keep
    asking for the GPU the legacy way on a board that wants the modern one
    (the mirror image of the bug this exists to fix).
    """
    texts = render_gpu_overrides(profile)
    for name in _gpu_override_files():
        path = target / name
        if texts is None:
            if path.exists():
                path.unlink()
            continue
        path.write_text(texts[name], encoding="utf-8")


def _gpu_override_plan(target: Path, profile: Profile) -> dict:
    """What ``--apply`` would do to the GPU overrides — for the dry-run plan.

    ``action`` is ``write`` (the card declares csv-mode GPU access), ``remove``
    (stale files on disk but this card takes the default), or ``none``.
    """
    files = _gpu_override_files()
    if profile.gpu_access == GPU_ACCESS_RUNTIME:
        return {"files": files, "action": "write", "gpu_access": profile.gpu_access}
    if any((target / name).exists() for name in files):
        return {"files": files, "action": "remove", "gpu_access": profile.gpu_access}
    return {"files": [], "action": "none", "gpu_access": profile.gpu_access}


def _gpu_override_plan_line(plan: dict) -> str | None:
    """A human dry-run line for the GPU-override plan, or ``None`` when nothing changes."""
    if plan["action"] == "write":
        return (
            f"  {', '.join(plan['files'])} (gpu_access={plan['gpu_access']}: "
            "!resets each GPU service's deploy stanza for `runtime: nvidia`)"
        )
    if plan["action"] == "remove":
        return (
            f"  {', '.join(plan['files'])} (stale — would be REMOVED: this card uses "
            f"gpu_access={plan['gpu_access']})"
        )
    return None


def _gpu_override_written(profile: Profile) -> dict:
    """Post-``--apply`` GPU-override state, for the JSON payload."""
    written = profile.gpu_access == GPU_ACCESS_RUNTIME
    return {
        "files": _gpu_override_files() if written else [],
        "written": written,
        "gpu_access": profile.gpu_access,
    }


def _templates(fleet: bool, audio: bool) -> dict[str, str]:
    if not fleet:
        return _compose.SINGLE_TEMPLATES
    templates = dict(_compose.FLEET_TEMPLATES)
    if audio:
        templates.update(_compose.AUDIO_TEMPLATES)
    return templates


def _resolve_fleet_profile(
    target: Path,
    profile_name: str | None,
    shape: Shape,
    *,
    shape_explicit: bool,
):
    """Resolve the per-machine profile for a fleet init; emits a stderr warning
    when ``--profile`` forces a name onto a card it wasn't validated for (or an
    undetected one), and likewise when detection itself comes back UNKNOWN with
    no ``--profile`` override — that case now resolves the conservative 'base'
    built-in (t14) rather than refusing; see ``resolve_init_profile``.

    The resolved card is also where the co-residency guard runs
    (:func:`_guard_coresidency`) — BOTH the dry-run and ``--apply`` paths reach
    the profile through here, so putting it here is what makes the check
    impossible to reach around."""
    profile, card, warning = resolve_init_profile(profile_name, target)
    if warning:
        emit_diagnostic(f"warning: {warning}")
    _guard_coresidency(shape, profile, shape_explicit=shape_explicit)
    return profile, card


def _coresidency_lines(
    shape: Shape, profile: Profile, groups: Sequence[ExclusiveRoles]
) -> list[str]:
    """One human line per over-hosted group: what clashes, why, and the way out."""
    lines = []
    for group in groups:
        hosted = ", ".join(role for role in group.roles if shape.hosts_role(role))
        line = (
            f"card profile {profile.name!r} declares {hosted} mutually exclusive, "
            f"but shape {shape.name!r} hosts them together"
        )
        if group.reason:
            line += f" — {group.reason}"
        lines.append(line)
    return lines


def _coresidency_shapes(groups: Sequence[ExclusiveRoles]) -> list[str]:
    """The resolving shapes the card itself names, de-duplicated, in declared order."""
    seen: list[str] = []
    for group in groups:
        for name in group.shapes:
            if name not in seen:
                seen.append(name)
    return seen


def _guard_coresidency(shape: Shape, profile: Profile, *, shape_explicit: bool) -> None:
    """Refuse a DEFAULTED shape that would over-host a card's exclusive roles.

    ``feasible`` answers "can this board serve this role at all?" — a per-role
    question both members of an exclusive group answer honestly with yes. The
    default ``machine-as-brain`` shape hosts every feasible role, so on a card
    that declares a co-residency limit the bare, decision-free path is exactly
    the one that renders a deployment expected to OOM at boot. This is where
    the card's declaration
    (:attr:`~lobes.profiles.schema.Profile.exclusive_roles`) reaches something
    real: it turns that silent over-commit into a user error that names the
    shapes resolving it.

    **An EXPLICIT ``--shape`` warns and proceeds**, it is not refused, and no
    override flag is added. That is this CLI's existing precedent, not a new
    one: ``--profile`` already documents "overrides detection, including
    forcing a profile onto a card it was not validated for (warns, but
    proceeds)" — an operator who types the shape has made the call knowingly,
    and lobes' job is then to be loud, not to be in the way. A second
    ``--force``-style flag would also be ambiguous next to the ``--force``
    ``init`` already has (overwrite existing files), and it would let the
    DEFAULT path be forced past the guard, which is the one thing this must
    never allow.

    Runs on the dry run as well as ``--apply``: a plan that quietly describes
    a deployment that cannot boot is the same bug one step earlier.
    """
    groups = overcommitted_groups(shape, profile)
    if not groups:
        return
    detail = "; ".join(_coresidency_lines(shape, profile, groups))
    if shape_explicit:
        emit_diagnostic(
            f"warning: {detail}. Proceeding — you named --shape {shape.name} explicitly."
        )
        return
    resolving = _coresidency_shapes(groups)
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"refusing to scaffold the default {shape.name!r} shape: {detail}",
        remediation=(
            f"choose a shape that hosts one of them: {', '.join(resolving)} "
            f"(e.g. 'lobes init --shape {resolving[0]}'). "
            f"To scaffold {shape.name} anyway, name it explicitly "
            f"('lobes init --shape {shape.name}') — it warns, then proceeds."
        ),
    )


def _profile_plan_lines(profile, card, profile_name: str | None, shape: Shape) -> list[str]:
    facts = (
        f"device_name={card.device_name!r}, compute_capability={card.compute_capability!r}, "
        f"total_memory_gb={card.total_memory_gb!r}"
    )
    if profile_name:
        why = f"forced via --profile; detected card={card.resolved!r}, {facts}"
    else:
        why = f"auto-detected: {facts}"
    lines = [f"Profile: {profile.name} ({why})"]
    lines.append(f"Shape: {shape.name} (hosts={list(shape.hosts)})")
    rendered = render_shape(shape, profile)
    lines.append(f"  would set {len(rendered.env)} env var(s) in {_compose.ENV_FILE}")
    return lines


def _values_equal(current: str, new: str) -> bool:
    """True when two env-var strings represent the same value.

    A straight string match covers everything but numbers; a profile-resolved
    float (``str(0.3) == "0.3"``) and the template's own literal (``"0.30"``,
    written for human readability — see ``env.example``) are the SAME value
    with different spellings, so a numeric-aware fallback avoids rewriting a
    key whose resolved value merely restates the shipped default in fewer
    digits. A non-numeric mismatch (e.g. an actually different model id or
    flag token) still compares unequal and gets written.
    """
    if current == new:
        return True
    try:
        return float(current) == float(new)
    except ValueError:
        return False


def _apply_profile_env(env_path: Path, env: dict[str, str]) -> None:
    """Write a profile's rendered env vars into ``.env``, skipping no-op writes.

    ``write_scaffold`` has already copied the template's own ``env.example``
    defaults into ``.env`` by the time this runs. When the resolved profile's
    value for a key is the SAME as what's already there (see
    :func:`_values_equal`), the original line is left untouched instead of
    being rewritten in a different (but equal) format — this keeps a
    zero-divergence profile (e.g. ``spark`` on a freshly scaffolded fleet
    ``.env``, which already ships spark's own defaults) byte-identical to
    today's plain scaffold.
    """
    current = _env.read_env_file(env_path)
    for key, value in env.items():
        existing = current.get(key)
        if existing is not None and _values_equal(existing, value):
            continue
        _env.set_env(env_path, key, value)


def _profile_plan_dict(profile, card, profile_name: str | None, shape: Shape) -> dict:
    rendered = render_shape(shape, profile)
    return {
        "profile": profile.name,
        "profile_forced": bool(profile_name),
        "detected_card": card.resolved,
        "detected_facts": {
            "device_name": card.device_name,
            "compute_capability": card.compute_capability,
            "total_memory_gb": card.total_memory_gb,
        },
        "shape": shape.name,
        "shape_hosts": list(shape.hosts),
        "profile_env": rendered.env,
    }


def _dry_run_payload(
    target: Path,
    fleet: bool,
    audio: bool,
    plan: list,
    profile,
    card,
    profile_name: str | None,
    shape: Shape | None,
    force: bool = False,
) -> dict:
    """The ``--json`` shape of a dry run: what ``--apply`` would write."""
    payload = {
        "dry_run": True,
        "fleet": fleet,
        "single": not fleet,
        "audio": audio,
        "target": str(target),
        "files": [
            {
                "name": name,
                "exists": exists,
                "action": _compose.scaffold_action(target, name, force=force),
            }
            for name, exists in plan
        ],
    }
    if fleet:
        payload.update(_profile_plan_dict(profile, card, profile_name, shape))
        payload["shape_override"] = _shape_override_plan(target, shape, profile)
        payload["gpu_override"] = _gpu_override_plan(target, profile)
    return payload


_SCAFFOLD_NOTES = {
    "create": "",
    "merge": " (exists; MERGED — missing keys appended, existing lines untouched)",
    "overwrite": " (exists; would be OVERWRITTEN by --force)",
    "skip": " (exists; left as-is — pass --force to overwrite)",
}


def _scaffold_note(target: Path, name: str, force: bool) -> str:
    """The dry-run suffix naming what ``--apply`` would do to one scaffolded file.

    A plan that says "needs --force" about ``.env`` would now be a lie twice
    over: ``.env`` is never overwritten (merge-only), and the other files no
    longer abort the command when they exist.
    """
    return _SCAFFOLD_NOTES[_compose.scaffold_action(target, name, force=force)]


def _dry_run_scope(fleet: bool, audio: bool) -> str:
    if fleet and audio:
        return "the fleet duo + audio overlay "
    if fleet:
        return "the fleet duo (main + multimodal) "
    return "the legacy single-model "


def _dry_run_lines(
    target: Path,
    fleet: bool,
    audio: bool,
    plan: list,
    profile,
    card,
    profile_name: str | None,
    shape: Shape | None,
    force: bool = False,
) -> list[str]:
    """The human-readable dry-run plan, one line per thing ``--apply`` would do.

    Every generated overlay gets a line: a plan that stays silent about a file
    ``--apply`` would write is how a re-render turns into a surprise (the
    GPU-access pair is exactly the omission that made a shadowed profile look
    like a working one).
    """
    lines = [f"DRY RUN — would scaffold {_dry_run_scope(fleet, audio)}into {target}:"]
    for name, _exists in plan:
        lines.append(f"  {name}{_scaffold_note(target, name, force)}")
    if audio:
        lines.append("  .env (+ audio keys appended)")
    if fleet:
        lines.extend(_profile_plan_lines(profile, card, profile_name, shape))
        for line in (
            _shape_override_plan_line(_shape_override_plan(target, shape, profile)),
            _gpu_override_plan_line(_gpu_override_plan(target, profile)),
        ):
            if line:
                lines.append(line)
    lines.append("Re-run with --apply to write.")
    return lines


def _emit_dry_run(
    target: Path,
    fleet: bool,
    audio: bool,
    json_mode: bool,
    profile_name: str | None,
    shape: Shape | None,
    shape_explicit: bool = False,
    force: bool = False,
) -> None:
    plan = _compose.scaffold_plan(target, _templates(fleet, audio))
    profile = card = None
    if fleet:
        # Detection/warning happens on a dry run too — the plan must be honest
        # about what --apply would do, including the fallback profile it would
        # serve on an UNKNOWN card.
        profile, card = _resolve_fleet_profile(
            target, profile_name, shape, shape_explicit=shape_explicit
        )
        # The tool-parser plugin file (t2) is fleet-only — mounted into
        # vllm-primary/cortex, never scaffolded for the legacy single-model dir.
        plan = plan + [_compose.plugin_plan(target)]
    if json_mode:
        emit_result(
            _dry_run_payload(target, fleet, audio, plan, profile, card, profile_name, shape, force),
            json_mode=True,
        )
        return
    lines = _dry_run_lines(target, fleet, audio, plan, profile, card, profile_name, shape, force)
    emit_result("\n".join(lines), json_mode=False)


def _write_fleet_render(
    target: Path,
    force: bool,
    shape: Shape | None,
    profile,
    written: list[Path],
) -> list[Path]:
    """Everything ``--apply`` writes on the FLEET path after the scaffold pass.

    Split out of ``_emit_apply`` so the report-what-changed bookkeeping (the two
    "never claim a file we did not write / never claim less than what changed"
    guards) reads as one concern instead of padding the caller's branch count.
    Returns the ``written`` list extended with anything this pass touched.
    """
    # The tool-parser plugin file (t2) is fleet-only — mounted into
    # vllm-primary/cortex, never scaffolded for the legacy single-model dir.
    # Single source of truth: written fresh from the packaged
    # lobes.vllm_plugins module, not a lobes/templates/ copy.
    plugin = _compose.write_plugin_file(target, force=force)
    if plugin is not None:
        written = written + [plugin]
    # Render the resolved (shape, profile) pair's knobs into .env, the same
    # way any other env value gets written here (lobes.runtime._env.set_env)
    # — skipping keys the composition merely restates from the template
    # default. machine-as-brain (the default shape) is a strict no-op over
    # the profile (t3), so this is byte-identical to the pre-shape
    # profile_env(profile) call it replaces.
    rendered = render_shape(shape, profile)
    _apply_profile_env(target / _compose.ENV_FILE, rendered.env)
    # Persist the profile choice itself for doctor/status to report
    _env.set_env(target / _compose.ENV_FILE, "LOBES_PROFILE", profile.name)
    # Pin the gateway image to the lobes-cli release that scaffolded this.
    _env.set_env(target / _compose.ENV_FILE, "MODEL_GEAR_VERSION", __version__)
    # A mesh-brain shape that DROPS a core role needs the generated compose
    # override so the dropped lobe does not RUN (t4b, #113); a shape that drops
    # nothing (machine-as-brain) writes none and scrubs any stale one.
    _sync_shape_override(target, shape, profile)
    # A csv-mode board (the card profile's gpu_access declaration) needs its
    # GPU request expressed as `runtime: nvidia` instead of the template's
    # deploy.resources stanza; every other card writes nothing and scrubs a
    # stale pair. This is what makes the fix survive a re-render.
    _sync_gpu_overrides(target, profile)
    # The fleet path ALWAYS edits .env after the scaffold pass — the shape/
    # profile knobs above, plus LOBES_PROFILE and MODEL_GEAR_VERSION. On a
    # re-render `write_scaffold` returns it only when the merge appended a
    # key, so without this the report would omit the one file that was
    # certainly rewritten. Never claim less than what changed.
    env_path = target / _compose.ENV_FILE
    if env_path not in written:
        written = written + [env_path]
    return written


def _apply_payload(
    target: Path,
    fleet: bool,
    audio: bool,
    written: list[Path],
    profile,
    card,
    profile_name: str | None,
    shape: Shape | None,
) -> dict:
    """The ``--apply --json`` result body."""
    payload = {
        "scaffolded": str(target),
        "fleet": fleet,
        "single": not fleet,
        "audio": audio,
        "files": [p.name for p in written],
    }
    if fleet:
        payload["profile"] = profile.name
        payload["profile_forced"] = bool(profile_name)
        payload["detected_card"] = card.resolved
        payload["shape"] = shape.name
        payload["shape_override"] = _shape_override_written(shape, profile)
        payload["gpu_override"] = _gpu_override_written(profile)
    return payload


def _override_note(shape: Shape | None, profile) -> str:
    """The human report's lines for the two GENERATED compose overlays.

    Fleet-only; the caller guards on that. Empty when the shape drops nothing
    and the card asks for its GPU the default (deploy.resources) way.
    """
    note = ""
    dropped = _shape_dropped_services(shape, profile)
    if dropped:
        note = (
            f"\n  {_compose.SHAPE_OVERLAY} (drops {', '.join(dropped)}: "
            f"parked in the inert '{SHAPE_DROPPED_PROFILE}' profile)"
        )
    if profile.gpu_access == GPU_ACCESS_RUNTIME:
        note += (
            f"\n  {', '.join(_gpu_override_files())} "
            f"(gpu_access={profile.gpu_access}: GPU asked for via `runtime: nvidia`)"
        )
    return note


def _emit_apply(
    target: Path,
    fleet: bool,
    audio: bool,
    force: bool,
    json_mode: bool,
    profile_name: str | None,
    shape: Shape | None,
    shape_explicit: bool = False,
) -> None:
    profile = card = None
    if fleet:
        # Resolve BEFORE writing anything — an explicit --profile mismatch, an
        # UNKNOWN card (falling back to the conservative 'base' profile, t14) or
        # a shape that would over-host the card's mutually-exclusive roles all
        # surface here, before any file is written.
        profile, card = _resolve_fleet_profile(
            target, profile_name, shape, shape_explicit=shape_explicit
        )
    written = _compose.write_scaffold(target, force=force, templates=_templates(fleet, audio))
    # Create the durable-log dir now (as the invoking user) so the compose bind-mount
    # source exists before `lobes serve` / `fleet up` — otherwise Docker makes it
    # root-owned. The mg-logwrap entrypoint writes per-boot logs here (issue #50).
    _compose.ensure_log_dir(target)
    if fleet:
        written = _write_fleet_render(target, force, shape, profile, written)
    if audio:
        # Extend the fleet .env with the audio keys (NGC_API_KEY, ports, AUDIO_URL …).
        # Independent of --shape: --audio is the sole switch that SCAFFOLDS the
        # overlay, and passing both is harmless/idempotent. What a shape decides
        # is whether the overlay's services RUN: every built-in shape hosts
        # stt/tts except `orin-lobe`, whose board cannot serve them (no sm_87
        # Parakeet image) — there the keys are still written, and AUDIO_URL
        # forwards /v1/audio/* to a peer instead.
        _compose.append_audio_env(target)
    if json_mode:
        emit_result(
            _apply_payload(target, fleet, audio, written, profile, card, profile_name, shape),
            json_mode=True,
        )
        return
    next_step = (
        "docker login nvcr.io && lobes fleet up --apply"
        if fleet
        else "docker login nvcr.io && lobes serve --apply"
    )
    profile_note = f"\n>> profile: {profile.name}\n>> shape: {shape.name}" if fleet else ""
    override_note = _override_note(shape, profile) if fleet else ""
    emit_result(
        f">> scaffolded {target}:\n"
        + "\n".join(f"  {p.name}" for p in written)
        + (f"\n  {_compose.ENV_FILE} (+ audio keys)" if audio else "")
        + override_note
        + profile_note
        + f"\n>> next: {next_step}",
        json_mode=False,
    )


# --- --from-lock: restore a committed deployment variation (lock plan, t7) ---
#
# `--from-lock` is a distinct SOURCE, not a fourth input to the renderer. The
# three axes above (topology, --profile, --shape) all feed
# `lobes/profiles/render.py`; this path never reaches it. That bypass is the
# whole point: a restore is byte-identical to what the box RAN, hand edits
# included, rather than to what the renderer would produce today.
#
# It is also why `_guard_variation` exists. Bypassing resolution also bypasses
# `_sync_gpu_overrides`, the card-driven correction that decides whether a
# deployment asks for its GPU the modern (`deploy.resources`) or the legacy
# (`runtime: nvidia`) way — so restoring a csv-mode variation onto a
# devices-mode board (or the reverse) would silently reproduce exactly the bug
# those overlays exist to fix. The lock's declared machine type is therefore
# checked against detection, and a mismatch refuses by default.

#: Overlays `--apply` GENERATES on the normal path (`_sync_shape_override` /
#: `_sync_gpu_overrides`). Each is written when the lock names it and REMOVED
#: when it does not — the remove-on-mismatch behaviour of those two syncs,
#: surviving a lock round-trip. Nothing outside this tuple is ever deleted: an
#: operator's own `docker-compose.override.yml` is not lobes' to remove.
RESTORE_SYNCED_FILES: tuple[str, ...] = (
    _compose.SHAPE_OVERLAY,
    _compose.GPU_OVERLAY,
    _compose.GPU_AUDIO_OVERLAY,
)

#: The `.env` block header a restore writes when the target has no `.env` yet.
_RESTORED_ENV_HEADER = (
    "# .env — RESTORED by `lobes init --from-lock` from a committed\n"
    f"# {LOCK_FILENAME}. These are the RENDERED KNOBS the lock carries.\n"
    "#\n"
    "# The lock is secret-free BY CONSTRUCTION (an allowlist of rendered keys,\n"
    "# never a copy of a deployed .env), so no credential can be restored from\n"
    "# it: GATEWAY_API_KEY, every *_PEER_*, COMPOSE_PROFILES and HF_TOKEN must\n"
    "# be generated (scripts/gen-api-key.py) or supplied from a gitignored\n"
    "# secret file before this deployment can serve.\n"
)

_MERGED_ENV_HEADER = (
    "",
    "# --- appended by `lobes init --from-lock`: lock keys this .env did not set ---",
    "# Existing lines above were left untouched (a restore never rewrites them).",
)


def _lock_source(raw: str) -> tuple[Path, Path]:
    """``(lock file, variation dir)`` for a ``--from-lock`` argument.

    Accepts either the variation FOLDER (the ``deployments/<id>/`` case) or the
    lock file itself, so a path a human copied out of either context works.
    """
    path = Path(raw).expanduser()
    lock_file = path / LOCK_FILENAME if path.is_dir() else path
    if not lock_file.is_file():
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"no {LOCK_FILENAME} at {path}",
            remediation=(
                f"pass a variation directory containing {LOCK_FILENAME}, or the " "lock file itself"
            ),
        )
    return lock_file, lock_file.parent


def _check_restorable_name(name: str) -> None:
    """Refuse a ``[files]`` entry that is not a plain, non-secret filename.

    Two hazards, one gate. A name carrying a separator or ``..`` would let a
    committed lock write OUTSIDE the deployment directory — a lock is adopted
    from a repo, so it is untrusted input, not lobes' own output. And a name in
    the ``.env`` SECRET family (the repo's positional gitignore rule: a ``.env``
    SUFFIX is ignored) is the one thing a committed variation may never carry:
    ``.env`` is merge-only and holds the operator-typed state
    :data:`lobes.runtime._compose.MERGE_ONLY_FILES` names.
    """
    if not name or name != Path(name).name or name in {".", ".."}:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"refusing to restore {name!r}: not a plain filename",
            remediation=(
                f"a {LOCK_FILENAME} [files] key must name one file inside the "
                "variation folder, never a path"
            ),
        )
    if name == _compose.ENV_FILE or name.endswith(_compose.ENV_FILE):
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"refusing to restore {name!r}: the .env family is never committed",
            remediation=(
                "secrets stay in the gitignored .env family; a lock records "
                "rendered knobs, which a restore MERGES into .env instead"
            ),
        )


def _lock_file_names(lock: DeploymentLock) -> list[str]:
    """The validated, sorted set of files a lock says a restore materialises."""
    names = sorted(lock.files)
    if not names:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"{LOCK_FILENAME} names no deployment files",
            remediation=(
                "a restorable variation records its compose files and Dockerfiles "
                "in the lock's [files] table — re-capture the lock"
            ),
        )
    for name in names:
        _check_restorable_name(name)
    return names


def _verify_source(source_dir: Path, lock: DeploymentLock, names: list[str]) -> None:
    """Every named file exists in the variation folder and matches its digest.

    Runs BEFORE anything is written (and on the dry run), so a variation folder
    that has drifted from its own lock is refused with nothing half-restored.
    """
    for name in names:
        path = source_dir / name
        if not path.is_file():
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"{name}: named by {LOCK_FILENAME} but missing from {source_dir}",
                remediation="the committed variation is incomplete — restore is not possible",
            )
        actual = file_digest(path)
        if actual != lock.files[name]:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=(
                    f"{name}: digest {actual} does not match the "
                    f"{LOCK_FILENAME} entry {lock.files[name]}"
                ),
                remediation=(
                    "the committed variation and its lock disagree — re-capture the "
                    "lock, or restore a variation whose files match it"
                ),
            )


def _detected_variation() -> str:
    """This box's variation id (machine type, never a hostname)."""
    return resolve_variation_id(_detect.detect_card())


def _guard_variation(lock: DeploymentLock, detected: str, *, allow_mismatch: bool) -> None:
    """Refuse a cross-machine-type restore unless the operator said so explicitly.

    An UNKNOWN card is a mismatch, not a pass: "we could not tell" is not
    evidence that the lock fits. The override is a flag of its own rather than
    the existing ``--force`` (which means "overwrite files") — a restore onto
    the wrong machine type is a different decision from clobbering a file, and
    conflating them would let one be taken while meaning the other.
    """
    if lock.variation == detected:
        return
    detail = (
        f"this box detects as {detected!r} but the lock declares variation " f"{lock.variation!r}"
    )
    if allow_mismatch:
        emit_diagnostic(
            f"warning: {detail}. Proceeding — you passed --allow-variation-mismatch. "
            "The restored GPU-access overlays are the LOCK's, not this card's."
        )
        return
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"refusing to restore a variation captured on another machine type: {detail}",
        remediation=(
            "--from-lock bypasses profile/shape resolution, so it also bypasses the "
            "card's gpu_access correction: a csv-mode variation restored onto a "
            "devices-mode board (or the reverse) asks for the GPU the wrong way. "
            f"Restore a {detected!r} variation, or pass --allow-variation-mismatch "
            "to take that risk deliberately."
        ),
    )


def _restore_removals(target: Path, names: list[str]) -> list[str]:
    """Generated overlays present in *target* that this lock does not name."""
    return [name for name in RESTORE_SYNCED_FILES if name not in names and (target / name).exists()]


def _merge_lock_env(env_path: Path, env: dict) -> list[str]:
    """Merge the lock's rendered knobs into ``.env``; returns the keys APPENDED.

    Append-only, by the same rule as
    :func:`lobes.runtime._compose.merge_env_template`: an existing line is never
    rewritten or reordered, so a live ``.env`` — the one file a restore may
    never clobber (:data:`lobes.runtime._compose.MERGE_ONLY_FILES`) — comes
    through a restore byte-identical when it already carries the lock's keys.
    Deliberately NOT :func:`lobes.runtime._env.set_env`, which rewrites the
    whole file even for a pure append.
    """
    existing = (
        _compose.env_keys(env_path.read_text(encoding="utf-8")) if env_path.exists() else set()
    )
    added = [key for key in sorted(env) if key not in existing]
    if not added:
        return []
    lines = [f"{key}={env[key]}" for key in added]
    if env_path.exists():
        with env_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join([*_MERGED_ENV_HEADER, *lines, ""]))
    else:
        env_path.write_text(_RESTORED_ENV_HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    return added


def _env_action(env_path: Path, env: dict) -> str:
    """What ``--apply`` would do to ``.env``: ``create``, ``merge`` or ``none``."""
    if not env_path.exists():
        return "create" if env else "none"
    existing = _compose.env_keys(env_path.read_text(encoding="utf-8"))
    return "merge" if any(key not in existing for key in env) else "none"


def _from_lock_payload(
    *,
    lock_file: Path,
    source_dir: Path,
    target: Path,
    lock: DeploymentLock,
    detected: str,
) -> dict:
    """The shared JSON skeleton both the dry run and ``--apply`` report."""
    return {
        "from_lock": True,
        "lock": str(lock_file),
        "source": str(source_dir),
        "target": str(target),
        "variation": lock.variation,
        "detected_variation": detected,
        "variation_mismatch": lock.variation != detected,
        "profile": lock.profile,
        "shape": lock.shape,
    }


def _from_lock_dry_run_lines(
    source_dir: Path, target: Path, names: list[str], removals: list[str], env_plan: dict
) -> list[str]:
    lines = [f"DRY RUN — would restore the variation in {source_dir} into {target}:"]
    for name in names:
        exists = (target / name).exists()
        lines.append(f"  {name}{' (exists; would be REPLACED verbatim)' if exists else ''}")
    for name in removals:
        lines.append(f"  {name} (stale — would be REMOVED: this lock does not name it)")
    if env_plan["action"] == "create":
        lines.append(
            f"  {_compose.ENV_FILE} (would be CREATED with the lock's "
            f"{len(env_plan['keys'])} rendered knob(s); no secret is restorable from a lock)"
        )
    elif env_plan["action"] == "merge":
        lines.append(
            f"  {_compose.ENV_FILE} (MERGE-ONLY — would append "
            f"{len(env_plan['keys'])} missing key(s); existing lines untouched)"
        )
    else:
        lines.append(f"  {_compose.ENV_FILE} (unchanged — it already sets every locked knob)")
    lines.append("Re-run with --apply to write.")
    return lines


def _guard_buildability(lock: DeploymentLock) -> None:
    """Refuse (or warn about) a variation whose gateway wheel will not install.

    The preflight t10's guard was built for, wired at the one moment a stale
    pin becomes consequential. ``Dockerfile.gateway`` installs the gateway as
    ``lobes-cli==${MODEL_GEAR_VERSION}``, so a variation can outlive the wheel
    it references — and without this the operator meets that as a ``pip``
    traceback nested in a ``docker build`` log, long after the restore claimed
    success.

    Deliberately OFFLINE and WARN-ONLY: no index is queried, because a restore
    must not depend on network reachability, and offline nothing here can PROVE
    a wheel uninstallable. ``lobes_version`` is optional in the lock schema, so
    its absence means "not recorded", never "broken" — refusing on it would
    reject every variation captured without one. The raising path
    (:func:`assert_buildable` on a definitive ``installable is False``) needs an
    index query, which stays opt-in; see issue tracking for wiring it to a
    network-permitted verb.
    """
    result = check_lock_buildability(lock)
    if result.risk == "ephemeral_dev":
        emit_diagnostic(
            f"warning: this variation pins a development wheel "
            f"({lock.lobes_version}) — those are published to TestPyPI by a PR "
            "and may no longer be installable; the gateway image build can fail"
        )
    elif result.risk == "unversioned":
        emit_diagnostic(
            "warning: this variation records no MODEL_GEAR_VERSION — "
            "`docker compose build gateway` will resolve an empty pin unless "
            "the deployment's own .env supplies one"
        )


def _emit_from_lock(
    raw_source: str,
    target: Path,
    *,
    apply: bool,
    json_mode: bool,
    allow_mismatch: bool,
) -> None:
    """Restore a committed variation — the whole ``--from-lock`` path.

    Everything that can refuse does so before the first byte is written: the
    lock parses, its ``[files]`` names are plain and non-secret, the variation
    folder matches its own digests, and the machine type agrees (or was
    explicitly overridden). The dry run runs the identical checks, so a plan
    never describes a restore that would then fail halfway.
    """
    lock_file, source_dir = _lock_source(raw_source)
    lock = load_lock(lock_file)
    names = _lock_file_names(lock)
    _verify_source(source_dir, lock, names)
    detected = _detected_variation()
    _guard_variation(lock, detected, allow_mismatch=allow_mismatch)
    _guard_buildability(lock)
    removals = _restore_removals(target, names)
    env_path = target / _compose.ENV_FILE
    payload = _from_lock_payload(
        lock_file=lock_file,
        source_dir=source_dir,
        target=target,
        lock=lock,
        detected=detected,
    )

    if not apply:
        env_plan = {
            "file": _compose.ENV_FILE,
            "action": _env_action(env_path, dict(lock.env)),
            "keys": sorted(
                key
                for key in lock.env
                if key
                not in (
                    _compose.env_keys(env_path.read_text(encoding="utf-8"))
                    if env_path.exists()
                    else set()
                )
            ),
        }
        payload.update(
            {
                "dry_run": True,
                "files": [
                    {
                        "name": name,
                        "action": "overwrite" if (target / name).exists() else "create",
                    }
                    for name in names
                ],
                "remove": removals,
                "env": env_plan,
            }
        )
        if json_mode:
            emit_result(payload, json_mode=True)
            return
        emit_result(
            "\n".join(_from_lock_dry_run_lines(source_dir, target, names, removals, env_plan)),
            json_mode=False,
        )
        return

    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        (target / name).write_bytes((source_dir / name).read_bytes())
    for name in removals:
        (target / name).unlink()
    added = _merge_lock_env(env_path, dict(lock.env))
    # The compose bind-mount source, exactly as the scaffold path creates it.
    _compose.ensure_log_dir(target)
    payload.update({"restored": str(target), "files": names, "removed": removals})
    payload["env_keys_added"] = added
    if json_mode:
        emit_result(payload, json_mode=True)
        return
    removed_note = "".join(f"\n  {name} (removed — not in this lock)" for name in removals)
    env_note = (
        f"\n  {_compose.ENV_FILE} (+{len(added)} locked knob(s); existing lines untouched)"
        if added
        else f"\n  {_compose.ENV_FILE} (unchanged)"
    )
    emit_result(
        f">> restored {target} from {lock_file}:\n"
        + "\n".join(f"  {name}" for name in names)
        + removed_note
        + env_note
        + f"\n>> variation: {lock.variation} (detected: {detected})"
        + "\n>> next: supply this box's secrets (GATEWAY_API_KEY, HF_TOKEN, any "
        "*_PEER_API_KEY), then 'lobes fleet up --apply'",
        json_mode=False,
    )


#: ``lobes init`` flags that feed the RENDERER — every one of them is a
#: conflict with ``--from-lock``, which is a different source entirely.
_RENDERER_AXES: tuple[tuple[str, str], ...] = (
    ("single", "--single"),
    ("audio", "--audio"),
    ("profile", "--profile"),
    ("shape", "--shape"),
)


def _guard_from_lock_axes(args: argparse.Namespace) -> None:
    """Refuse ``--from-lock`` alongside any renderer axis.

    Not a stylistic objection: a restore materialises the lock's files verbatim
    and never calls :func:`lobes.profiles.shape_render.render_shape`, so a
    ``--profile`` or ``--shape`` passed here would be silently inert — the one
    failure mode worse than an error.
    """
    for dest, flag in _RENDERER_AXES:
        if getattr(args, dest, None):
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"{flag} is incompatible with --from-lock",
                remediation=(
                    "--from-lock is a distinct SOURCE, not a renderer input: it "
                    "materialises a committed variation verbatim and never resolves "
                    f"a profile or shape. Drop {flag}, or drop --from-lock to render."
                ),
            )


def cmd_init(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    from_lock = getattr(args, "from_lock", None)
    if from_lock:
        # The restore SOURCE (lock plan, t7). Dispatched before any topology /
        # profile / shape decision is even computed — that is the bypass, and
        # putting it first is what makes it unreachable-around rather than a
        # branch some later code path could still fall through.
        _guard_from_lock_axes(args)
        target = (
            Path(args.target).expanduser() if args.target else _compose.default_deployment_dir()
        )
        _emit_from_lock(
            from_lock,
            target,
            apply=bool(args.apply),
            json_mode=json_mode,
            allow_mismatch=bool(getattr(args, "allow_variation_mismatch", False)),
        )
        return 0
    # The fleet duo is the DEFAULT (issue #69); --single (alias --legacy) opts out
    # to the legacy single-model scaffold. --fleet is a default-implied no-op alias.
    single = bool(getattr(args, "single", False))
    fleet = not single
    audio = bool(getattr(args, "audio", False))
    if audio and not fleet:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="--audio is incompatible with --single",
            remediation="the audio overlay layers on the fleet (the default): "
            "drop --single, e.g. 'lobes init --audio'",
        )
    shape_name = getattr(args, "shape", None)
    if shape_name is not None and single:
        # Shapes render the per-role fleet .env (a fleet-scaffold axis, brain-
        # shapes t4); --single has no per-role profile/shape resolution at all
        # (it never even calls detection — see
        # test_single_topology_never_calls_detection), so ANY explicit --shape
        # here — even spelling out the default machine-as-brain — is a
        # conflict, not a silent no-op.
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="--shape is incompatible with --single",
            remediation="shapes render the per-role fleet .env (a fleet-scaffold "
            "axis): drop --single, e.g. 'lobes init --shape spark-lobe'",
        )
    # Resolve the shape BEFORE writing anything, in both dry-run and --apply, so
    # an unknown --shape value always aborts before any file is touched. A bare
    # `lobes init` (shape_name is None) resolves DEFAULT_SHAPE — pure data, no
    # host probe, so this makes zero new decisions on the default path.
    shape = resolve_shape(shape_name or DEFAULT_SHAPE) if fleet else None
    target = Path(args.target).expanduser() if args.target else _compose.default_deployment_dir()
    profile_name = getattr(args, "profile", None)
    # An EXPLICIT --shape is the operator's own decision (see
    # _guard_coresidency): it warns past a co-residency clash, where the
    # defaulted shape refuses.
    shape_explicit = shape_name is not None
    if args.apply:
        _emit_apply(
            target, fleet, audio, args.force, json_mode, profile_name, shape, shape_explicit
        )
    else:
        _emit_dry_run(
            target,
            fleet,
            audio,
            json_mode,
            profile_name,
            shape,
            shape_explicit=shape_explicit,
            force=args.force,
        )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "init",
        help="Scaffold a deployment dir (default ~/.lobes; dry-run by default; --apply).",
    )
    p.add_argument(
        "target",
        nargs="?",
        help="Where to scaffold (default ~/.lobes; '.' for the current folder).",
    )
    # Topology selector. Default is the fleet duo (main primary + multimodal gear
    # + gateway + embed/rerank); --single (alias --legacy) restores the legacy
    # single-model scaffold. --fleet is the now-default-implied no-op kept for
    # back-compat. They are mutually exclusive.
    topology = p.add_mutually_exclusive_group()
    topology.add_argument(
        "--single",
        "--legacy",
        dest="single",
        action="store_true",
        help="Scaffold the legacy single-model deployment (one vLLM server, no "
        "gateway) instead of the default fleet duo.",
    )
    topology.add_argument(
        "--fleet",
        action="store_true",
        help="Default-implied no-op (kept for back-compat): the fleet duo — the "
        "Qwen primary + the multimodal gear behind 1 OpenAI gateway with the "
        "co-resident embedding/reranker gears — is now the default scaffold.",
    )
    p.add_argument(
        "--audio",
        action="store_true",
        help="Also scaffold the audio overlay (STT + TTS + realtime bridge). "
        "Layers on the fleet (the default); incompatible with --single.",
    )
    p.add_argument(
        "--profile",
        help="Per-machine profile to render into .env (default: auto-detect the "
        "host card — spark, thor, ... — via lobes.runtime._detect). Overrides "
        "detection, including forcing a profile onto a card it was not "
        "validated for (warns, but proceeds). Fleet topology only. On an "
        "UNKNOWN card with no --profile, init warns and serves the conservative "
        "'base' profile (small generate model + pooling gears, no 27B) instead "
        "of guessing or refusing.",
    )
    p.add_argument(
        "--shape",
        # Derived from the shipped shape TOMLs, never hand-listed — a new
        # built-in shape shows up in --help the moment its file lands (and the
        # resolver's own error message is derived the same way).
        metavar="{" + ",".join(builtin_shape_names()) + "}",
        help="Deployment shape to render (brain-shapes, issue #113): which "
        "roles this box hosts, composed on top of whichever --profile/"
        "detection resolves. Default 'machine-as-brain' (host every "
        "default first-class Colleague role this card can serve — today's "
        "behaviour; a bare 'lobes init' makes zero new decisions and renders "
        "byte-identically). Mesh-brain alternatives drop one generate lobe "
        "to a peer box and reclaim its GPU-memory budget: 'spark-lobe' "
        "(drops senses), 'thor-lobe' (drops cortex). 'orin-lobe' is "
        "thor-lobe's sm_87 sibling (senses + pooling gears; no cortex, and no "
        "stt/tts — the Parakeet image has no sm_87 kernels, so audio forwards "
        "to a peer); 'orin-cortex' is that board's opposite answer — cortex "
        "LOCAL on the llama.cpp GGUF lane, senses dropped (the two do not fit "
        "in 61.3 GiB); 'orin-associate' is a third answer on the same board — "
        "the opt-in Lightning 'associate' lobe LOCAL, both cortex and senses "
        "dropped, no stt/tts. 'thor-muse' drops BOTH "
        "heavy default lobes and hosts the opt-in 31B 'muse' creative lobe "
        "instead; 'thor-worker' drops BOTH and hosts the opt-in 35B-A3B "
        "multimodal 'worker' ground-work lobe instead. 'orin-small' (issue "
        "#112, DECLARED/UNVALIDATED — no "
        "physical Jetson AGX Orin has booted it) drops both heavy lobes and "
        "hosts the opt-in 'minor' generate gear instead. Fleet topology only "
        "— incompatible with --single. An unknown value is a user error "
        "naming the valid shapes. On a card that declares mutually-exclusive "
        "roles (today only 'orin', whose cortex and senses do not both fit in "
        "61.3 GiB), OMITTING --shape is refused with an error naming the shapes "
        "that resolve it; naming 'machine-as-brain' explicitly warns and "
        "proceeds.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing template files (compose, Dockerfiles, plugin). "
        "NEVER overwrites .env: that file is merge-only always — missing keys "
        "are appended, existing lines are left untouched. Without --force an "
        "existing file is skipped, not an error.",
    )
    p.add_argument(
        "--from-lock",
        metavar="PATH",
        help="Restore a COMMITTED deployment variation instead of rendering one "
        f"(lock plan, t7). PATH is a variation directory holding a {LOCK_FILENAME} "
        "(e.g. deployments/<variation>/) or that lock file itself. A distinct "
        "SOURCE, not a renderer input: the lock's compose files, overrides and "
        "Dockerfiles are materialised VERBATIM and no profile or shape is "
        "resolved, which is what makes a restore byte-identical to what the box "
        "ran — hand edits included. Incompatible with --single/--audio/--profile/"
        "--shape. '.env' is MERGE-ONLY as always: the lock's rendered knobs are "
        "appended when missing and every existing line is left untouched (no "
        "secret is restorable from a lock — it carries none by construction). "
        "The generated overlays this lock does not name are REMOVED. Because a "
        "restore bypasses the card's gpu_access correction, a lock whose declared "
        "variation differs from the detected machine type is REFUSED — see "
        "--allow-variation-mismatch. Dry-run by default; --apply writes.",
    )
    p.add_argument(
        "--allow-variation-mismatch",
        action="store_true",
        help="Restore --from-lock even though this box's detected machine type "
        "differs from the lock's declared variation (warns, then proceeds). The "
        "risk is concrete: the restored deployment asks for its GPU the way the "
        "CAPTURING card did, so a csv-mode variation on a devices-mode board (or "
        "the reverse) fails at container create. Separate from --force, which is "
        "only about overwriting files.",
    )
    p.add_argument("--apply", action="store_true", help="Actually write the files.")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_init)
