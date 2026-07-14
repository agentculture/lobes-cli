"""Render a (:class:`~lobes.profiles.shapes.Shape`, \
:class:`~lobes.profiles.schema.Profile`) pair into the concrete compose/.env a box runs.

This is the composition point of the two landed axes:

* the #108 **per-machine** :class:`~lobes.profiles.schema.Profile` — how each
  role is TUNED on a given card (``gpu_mem_util``, ``max_model_len``, …),
  rendered to ``.env`` by :func:`lobes.profiles.render.profile_env`;
* the brain-shapes t1 **deployment-shape** :class:`~lobes.profiles.shapes.Shape`
  — which roles a box HOSTS at all (``machine-as-brain`` hosts everything;
  ``spark-lobe`` / ``thor-lobe`` drop a lobe to a peer box in the mesh).

Rendering is a **pure function of (shape, profile, template)**: no GPU probe,
no host read, no subprocess — it runs identically on a GPU-less CI runner. It
never re-implements the role -> env mapping; it composes the pair into a
synthetic :class:`~lobes.profiles.schema.Profile` and hands that to the existing
:func:`lobes.profiles.render.profile_env`. That is what makes the whole-brain
``machine-as-brain`` shape a strict no-op: hosting every role with no override
yields exactly the bare card profile's rendering (the invariant the goldens
pin).

**Core roles vs audio roles vs the opt-in `minor` gear.** The four
Profile-machinery core roles (:data:`~lobes.profiles.schema.ROLES` —
``cortex`` / ``senses`` / ``embedder`` / ``reranker``) carry the ``.env``
knobs and map through ``profile_env``. The two audio-overlay roles (``stt`` /
``tts``) carry no per-machine vLLM knobs of their own; hosting either is what
turns on the **audio overlay compose file** (``docker-compose.audio.yml``) —
mirroring ``lobes init --fleet --audio`` / ``lobes fleet up`` auto-including
it when present. The opt-in ``minor`` gear (:data:`~lobes.profiles.shapes.OPT_IN_ROLES`,
added for the mesh-brain end-state's t2, issue #112) likewise carries no
Profile knobs — its service (``vllm-minor``) already lives, unconditionally,
in the base fleet compose file, gated only by the ``minor`` Docker Compose
profile; hosting it contributes no new ``.env`` key, only a service-set
entry. So a shape contributes to BOTH sides of "compose/.env": the ``.env``
via the core roles, and the compose file list + service set via what it
hosts.

**Dropped role -> flagged off, no service.** A core role a shape does NOT host
renders the #110-conventional ``<PREFIX>_FEASIBLE=false`` marker and nothing
else — no ``<PREFIX>_MODEL``, no knobs — exactly like a card that finds the
role infeasible (see :func:`lobes.profiles.render.profile_env`'s
``feasible=False`` convention). Its compose service is likewise absent from
:func:`shape_services`. A role the CARD marks infeasible is dropped the same
way even if the shape would host it — feasibility stays the card's call.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

from lobes.profiles.render import profile_env
from lobes.profiles.schema import ROLES, Profile, RoleProfile
from lobes.profiles.shapes import AUDIO_ROLES, OPT_IN_ROLES, Shape

# The base fleet compose file every deployment runs, plus the opt-in audio
# overlay. A Shape hosting an audio role (stt/tts) is what turns the overlay on
# -- mirrors `lobes init --fleet --audio` / `lobes fleet up` auto-including
# docker-compose.audio.yml when it is present (lobes/templates/fleet/).
FLEET_COMPOSE_FILE = "docker-compose.yml"
AUDIO_OVERLAY_FILE = "docker-compose.audio.yml"

# role -> the compose SERVICE (the `services:` key the compose template
# declares) that serves it. The four core roles and `minor` live in the base
# fleet; stt/tts live in the audio overlay. Kept as a constant mirroring the
# shipped template exactly -- same design as render.ROLE_ENV_PREFIX, and
# verified against the real compose files by tests/test_shape_goldens.py.
ROLE_SERVICE: dict[str, str] = {
    "cortex": "vllm-primary",
    "senses": "vllm-multimodal",
    "embedder": "vllm-embed",
    "reranker": "vllm-rerank",
    "stt": "stt",
    "tts": "chatterbox",
    "minor": "vllm-minor",
}

# The fleet front (always up) and the audio overlay's realtime bridge (up
# whenever the overlay is), neither of which is a per-role gear.
GATEWAY_SERVICE = "gateway"
REALTIME_SERVICE = "realtime"


def _overlay(base: RoleProfile, override: RoleProfile) -> RoleProfile:
    """Compose a shape's per-role budget override onto the card profile's role.

    The override WINS field-by-field wherever it takes a position (a non-``None``
    knob or ``model``); every field the override is silent on flows through from
    the card ``base`` unchanged. Feasibility is deliberately NOT the override's
    to set: whether a hosted role is feasible is the card
    :class:`~lobes.profiles.schema.Profile`'s call (a shape only re-derives
    budget), so ``feasible`` always comes from ``base``.
    """
    merged: dict = {}
    for f in fields(RoleProfile):
        if f.name == "feasible":
            merged["feasible"] = base.feasible
            continue
        override_value = getattr(override, f.name)
        merged[f.name] = override_value if override_value is not None else getattr(base, f.name)
    return RoleProfile(**merged)


def compose_profile(shape: Shape, profile: Profile) -> Profile:
    """The synthetic per-role :class:`Profile` a (shape, card) pair resolves to.

    Composes the deployment-shape axis (which roles this box HOSTS) with the
    per-machine axis (how each role is TUNED):

    * a core role the shape HOSTS -> the card's
      :class:`~lobes.profiles.schema.RoleProfile`, with the shape's budget
      override (if any) overlaid (:func:`_overlay`);
    * a core role the shape DROPS -> ``RoleProfile(feasible=False)`` -- the box
      does not serve it, so it renders the #110 flagged-off marker
      (``<PREFIX>_FEASIBLE=false``) and no model/knobs.

    Only the four Profile-machinery core roles (:data:`ROLES`) carry ``.env``
    knobs; ``stt``/``tts`` are audio-overlay sidecars handled by
    :func:`shape_compose_files` / :func:`shape_services`, not here. Pure.
    """
    roles: dict[str, RoleProfile] = {}
    for role in ROLES:
        if shape.hosts_role(role):
            roles[role] = _overlay(profile.role(role), shape.override(role))
        else:
            roles[role] = RoleProfile(feasible=False)
    return Profile(
        name=f"{shape.name}@{profile.name}",
        summary=f"shape={shape.name} card={profile.name}",
        roles=roles,
    )


def shape_env(shape: Shape, profile: Profile) -> dict[str, str]:
    """The ``.env`` projection for a (shape, card) pair.

    Pure passthrough of :func:`lobes.profiles.render.profile_env` over the
    composed :class:`Profile` (:func:`compose_profile`) -- never a
    reimplementation of the role -> env mapping. Byte-identical to
    ``profile_env(profile)`` for the whole-brain ``machine-as-brain`` shape
    (hosts every role, no overrides), which is the invariant the goldens pin.
    """
    return profile_env(compose_profile(shape, profile))


def shape_compose_files(shape: Shape) -> tuple[str, ...]:
    """The ordered docker-compose files this shape runs.

    Always the base fleet (:data:`FLEET_COMPOSE_FILE`); the audio overlay
    (:data:`AUDIO_OVERLAY_FILE`) is appended iff the shape hosts an audio role
    (``stt``/``tts``). Pure function of the shape alone.
    """
    files = [FLEET_COMPOSE_FILE]
    if any(shape.hosts_role(role) for role in AUDIO_ROLES):
        files.append(AUDIO_OVERLAY_FILE)
    return tuple(files)


def shape_services(shape: Shape, profile: Profile) -> tuple[str, ...]:
    """The compose services a (shape, card) pair actually brings up, sorted.

    A core role the shape drops -- or one the card marks infeasible -- has no
    service here (the "dropped role -> no running service" contract). The
    gateway always fronts the fleet; the realtime bridge comes up with the audio
    overlay. The opt-in `minor` gear (:data:`OPT_IN_ROLES`) has no card-level
    feasibility of its own -- unlike the four core roles, it is never
    conditioned on ``profile.role(...).feasible`` -- so it is added whenever
    the shape hosts it, exactly like the audio roles.
    """
    services = {GATEWAY_SERVICE}
    for role in ROLES:
        if shape.hosts_role(role) and profile.role(role).feasible:
            services.add(ROLE_SERVICE[role])
    audio_hosted = False
    for role in AUDIO_ROLES:
        if shape.hosts_role(role):
            services.add(ROLE_SERVICE[role])
            audio_hosted = True
    if audio_hosted:
        services.add(REALTIME_SERVICE)
    for role in OPT_IN_ROLES:
        if shape.hosts_role(role):
            services.add(ROLE_SERVICE[role])
    return tuple(sorted(services))


@dataclass(frozen=True)
class ShapeRender:
    """The concrete artifacts a (shape, card) pair renders to -- the "compose/.env".

    ``env`` is the ``.env`` projection (core roles); ``compose_files`` is the
    ordered ``docker compose -f ...`` file list; ``services`` is the sorted set
    of services that will run. This is the read-only result ``lobes init
    --shape`` (t4) consumes.
    """

    shape: str
    card: str
    env: Mapping[str, str]
    compose_files: tuple[str, ...]
    services: tuple[str, ...]

    def env_text(self) -> str:
        """The sorted ``KEY=VALUE\\n`` ``.env`` projection -- the golden file format.

        Identical formatting to ``tests/goldens/regen.py``'s
        ``profile_env_text`` so a shape golden and a profile golden are the same
        shape of file (and ``machine-as-brain`` is byte-identical to the bare
        profile golden).
        """
        lines = sorted(f"{key}={value}" for key, value in self.env.items())
        return "\n".join(lines) + "\n"


def render_shape(shape: Shape, profile: Profile) -> ShapeRender:
    """Render a (shape, card-profile) pair into its compose/.env artifacts.

    Pure function of ``(shape, profile, template)`` -- no GPU probe, no host
    read, no subprocess; runs identically on a GPU-less CI runner. This is the
    API ``lobes init --shape`` (t4) consumes.
    """
    return ShapeRender(
        shape=shape.name,
        card=profile.name,
        env=shape_env(shape, profile),
        compose_files=shape_compose_files(shape),
        services=shape_services(shape, profile),
    )
