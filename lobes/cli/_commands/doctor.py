"""``lobes doctor`` — diagnose (and, opt-in, heal) the local model deployment.

Real checks (no longer a stub): is docker available, is a deployment scaffolded,
is the ``.env`` coherent with ``culture.yaml``, is ``/health`` reachable, does
the deployed gateway's own ``lobes-cli`` release match this CLI's (issue #99),
are all expected scaffold files on disk, and does the deployed ``.env`` carry
the knobs the resolved machine profile requires (issue #119 — a stale scaffold
can serve for weeks with ``/health`` green while one lane silently hangs). A
down model is *not* an error (bringing it up is the tool's job) — only missing
docker, an un-scaffolded deployment, or a deployed-artifact version mismatch
fail the run.

**The heal lane (``--fix``, issue #119).** Plain ``doctor`` is read-only,
always. ``--fix`` prints the missing-only heal plan (still read-only);
``--fix --apply`` commits it — writes only ABSENT scaffold files and appends
only ABSENT ``.env`` keys, never rewriting an existing line (docker compose
``env_file`` semantics let the LAST duplicate key win, so appending over a set
key would silently clobber it). This is the safe path between ``lobes init``'s
two extremes — refuses (any file exists) and ``--force`` (clobbers the whole
template set, ``.env`` included: gateway key, peer config, reclaim values).
A profile-required key that IS present but still carries the template default
is reported (the render was never applied) yet deliberately NOT auto-fixed —
rewriting an existing line is init ``--force`` territory, not doctor's.

JSON contract: ``{healthy, checks:[{id, passed, severity, message, remediation}]}``
(+ ``fix_plan`` on fleet deployments, ``fix_applied`` after ``--fix --apply``).
"""

from __future__ import annotations

import argparse
from importlib.resources import files as _resource_files
from pathlib import Path

from lobes import __version__
from lobes.cli._commands.init import DEFAULT_SHAPE, _values_equal
from lobes.cli._commands.whoami import _find_culture_yaml
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.cli._runtime_ops import resolve_init_profile
from lobes.gateway._config import FEASIBLE_ENV
from lobes.profiles.render import ROLE_ENV_PREFIX
from lobes.profiles.shape_render import ROLE_SERVICE, render_shape
from lobes.profiles.shapes import resolve_shape
from lobes.runtime import _compose, _detect, _env, _health


def _culture_model_tail() -> str | None:
    """The model name after ``vllm-local/`` in ``culture.yaml`` (or ``None``)."""
    cfg = _find_culture_yaml()
    if cfg is None:
        return None
    try:
        text = cfg.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("model:"):
            _, _, value = stripped.partition("model:")
            value = value.strip().strip("'\"")
            prefix = "vllm-local/"
            return value[len(prefix) :] if value.startswith(prefix) else value
    return None


def _check(id_: str, passed: bool, severity: str, message: str, remediation: str = "") -> dict:
    return {
        "id": id_,
        "passed": passed,
        "severity": severity,
        "message": message,
        "remediation": remediation,
    }


def _docker_check() -> dict:
    ok = _compose.docker_available()
    return _check(
        "docker_available",
        ok,
        "error",
        "docker + docker compose are available" if ok else "docker / docker compose not found",
        "" if ok else "install Docker + the NVIDIA Container Toolkit",
    )


def _env_coherence_check(env_path) -> dict:
    served = _env.read_env(env_path, "VLLM_SERVED_NAME")
    expected = _culture_model_tail()
    if not served:
        return _check(
            "env_coherence",
            False,
            "warn",
            "VLLM_SERVED_NAME is not set in .env",
            "set it, or run 'lobes switch <model> --apply'",
        )
    if expected and served != expected:
        return _check(
            "env_coherence",
            False,
            "warn",
            f"VLLM_SERVED_NAME ({served}) != culture.yaml model tail ({expected})",
            "align them so the acp vllm-local provider resolves the model",
        )
    return _check("env_coherence", True, "info", f"VLLM_SERVED_NAME = {served}")


def _health_check(port: int) -> dict:
    healthy = _health.is_healthy(port)
    return _check(
        "health_reachable",
        healthy,
        "info",
        f"/health responding on :{port}" if healthy else f"/health not responding on :{port}",
        "" if healthy else "start the server with 'lobes serve --apply'",
    )


def _version_skew_remediation(deploy_dir: Path | None) -> str:
    """The exact fix for a version mismatch — names both the file and the pin
    to change, plus the follow-up rebuild, so this is copy-pasteable."""
    env_path = f"{deploy_dir}/.env" if deploy_dir is not None else "<deployment>/.env"
    return (
        f"set MODEL_GEAR_VERSION={__version__} in {env_path}, "
        "then docker compose up -d --build gateway"
    )


def _machine_profile_section(deploy_dir: Path | None) -> dict | None:
    """Report the detected card and active profile, warning if mismatched.

    Returns a dict with detected_card, device info, profile, and validated flag,
    or None if deployment not scaffolded (no profile to report).
    """
    if deploy_dir is None:
        return None

    # Detect the card on the host
    card = _detect.detect_card()

    # Read the persisted profile choice from .env (if any)
    env_path = deploy_dir / _compose.ENV_FILE
    profile_name = _env.read_env(env_path, "LOBES_PROFILE")

    # Determine if profile is validated for this card
    validated = card.is_known and profile_name == card.resolved

    # Build the warning message if there's a mismatch or unknown card
    warning = None
    if not card.is_known:
        warning = (
            f"unrecognized card (device_name={card.device_name!r}, "
            f"compute_capability={card.compute_capability!r}, "
            f"total_memory_gb={card.total_memory_gb!r}) — "
            "profile not validated for this machine; "
            "pass --profile <name> to init to force a known profile"
        )
    elif profile_name and profile_name != card.resolved:
        warning = (
            f"profile {profile_name!r} does not match detected card {card.resolved!r} — "
            "this profile was not validated for this machine"
        )

    return {
        "detected_card": card.resolved,
        "device_name": card.device_name,
        "compute_capability": card.compute_capability,
        "total_memory_gb": card.total_memory_gb,
        "profile": profile_name,
        "validated": validated,
        "warning": warning,
    }


def _version_skew_check(port: int, deploy_dir: Path | None) -> dict:
    """Detect deployed-artifact version skew between the gateway and this CLI.

    This is the structural fix for issue #99, the root cause behind issue #92:
    ``Dockerfile.gateway`` runs ``pip install "lobes-cli==${MODEL_GEAR_VERSION}"``
    with ``MODEL_GEAR_VERSION`` written ONCE, by ``lobes init``, at scaffold
    time — no verb ever re-pins it afterwards. A gateway container can
    therefore silently keep running a stale ``lobes-cli`` release for as long
    as the deployment stays up, even after the host's own ``lobes`` binary
    (and PyPI) have moved on. On the reference rig this went undetected for
    five days: the gateway ran ``0.36.0`` and the realtime bridge ``0.34.1``
    against a host CLI at ``0.39.0``, and issue #92 was filed and
    investigated as a fresh code regression when the fix behind it was
    already published and simply undeployed.

    This check is docker-free: the gateway now reports its own deployed
    ``lobes-cli`` version over ``GET /health`` (issue #99, additive —
    :mod:`lobes.gateway.server`), so this only needs
    :func:`lobes.runtime._health.fetch_health` (a bounded HTTP GET) to compare
    that against this process's own :data:`lobes.__version__`.

    The three outcomes are NOT symmetric, deliberately:

    * **match** — ``passed=True``. Nothing to report.
    * **mismatch** — ``passed=False``, ``severity="error"`` (this DOES fail
      the overall run): a real, actionable defect — the deployed gateway is
      running code the operator's own CLI no longer believes is current, and
      that gap is exactly what let issue #92 masquerade as a live bug.
    * **gateway unreachable** — ``passed=False``, ``severity="info"`` (this
      does NOT fail the run): a down gateway is ordinary here (per this
      module's own docstring, "bringing it up is the tool's job", same as
      ``health_reachable``), so it must not be conflated with a real skew
      defect. Critically this is ALSO not a silent pass: reporting
      ``passed=True`` ("versions match") when nothing was actually verified
      would be exactly the #96/#92 mistake this whole plan exists to close —
      an unverified claim standing in for a live observation. The message
      says plainly that verification did not happen.
    """
    payload = _health.fetch_health(port)
    if payload is None:
        return _check(
            "gateway_version_match",
            False,
            "info",
            f"gateway not reachable on :{port} — cannot verify deployed version",
            "start the server ('lobes serve --apply' or 'lobes fleet up --apply'), "
            "then re-run doctor",
        )
    gateway_version = payload.get("version")
    if not gateway_version:
        return _check(
            "gateway_version_match",
            False,
            "info",
            f"gateway on :{port} did not report a version — cannot verify (pre-#99 gateway build)",
            "rebuild the deployed gateway image to pick up /health's version field",
        )
    if gateway_version != __version__:
        return _check(
            "gateway_version_match",
            False,
            "error",
            f"deployed gateway reports lobes-cli {gateway_version}, this CLI is "
            f"{__version__} — deployed-artifact version skew (issue #99)",
            _version_skew_remediation(deploy_dir),
        )
    return _check(
        "gateway_version_match",
        True,
        "error",
        f"gateway and CLI both report lobes-cli {__version__}",
    )


# --- gateway passthrough (issue #199, t3) -----------------------------------
#
# The role-prefix set every per-role gateway env family (FEASIBLE, the
# singular/plural peer channels, the declared lane fingerprint) is keyed on.
# Derived from FEASIBLE_ENV rather than hand-typed so this list can never
# drift from the prefixes lobes.gateway._config already recognises.
_GATEWAY_ROLE_PREFIXES: tuple[str, ...] = tuple(
    key[: -len("_FEASIBLE")] for key in FEASIBLE_ENV.values()
)

# Per-role env suffixes the gateway service is expected to pass through:
# FEASIBLE (the existing shape/feasibility flag), the singular "refer to one
# peer" pair (PEER_ORIGIN/PEER_API_KEY, issue #112/#115/#127), and the
# plural "pool of replicas" pair (PEER_ORIGINS/PEER_API_KEYS, issue #199).
# PEER_PROXY is a boolean opt-in knob, not a peer identity, but it rides the
# exact same per-role channel and is just as silently inert if the gateway
# container never receives it — included for the same reason.
_GATEWAY_PEER_SUFFIXES: tuple[str, ...] = (
    "FEASIBLE",
    "PEER_ORIGIN",
    "PEER_ORIGINS",
    "PEER_PROXY",
    "PEER_API_KEY",
    "PEER_API_KEYS",
)

# The declared lane fingerprint (issue #199): the same five knobs a role's
# own vLLM service is started with, mirrored to the gateway so GET
# /capabilities and the replica pool's compatibility check can see what this
# box's lane actually declares. Not every prefix's lane consumes an env
# override for every one of these five suffixes (several vLLM services
# hardcode the flag instead — see the packaged compose template's own
# per-service comments) — that is a lane-authoring fact, irrelevant here:
# doctor only cares whether a KEY THE OPERATOR SET IN .env reaches the
# gateway container, and the packaged template passes all five through for
# every prefix uniformly.
_GATEWAY_FINGERPRINT_SUFFIXES: tuple[str, ...] = (
    "QUANTIZATION",
    "KV_CACHE_DTYPE",
    "REASONING_PARSER",
    "TOOL_CALL_PARSER",
    "SPECULATIVE_CONFIG",
)

# Gateway-scoped singletons with no per-role prefix: the inbound auth key and
# this box's own advertised origin (issue #199's X-Lobes-Served-By source).
_GATEWAY_SINGLETON_KEYS: tuple[str, ...] = ("GATEWAY_SELF_ORIGIN", "GATEWAY_API_KEY")


def _gateway_relevant_keys() -> tuple[str, ...]:
    """Every ``.env`` key the deployed gateway service is expected to read.

    Purely a naming enumeration (role prefix x suffix, plus the two
    singletons) — it does not read any file itself.
    """
    keys: list[str] = list(_GATEWAY_SINGLETON_KEYS)
    for prefix in _GATEWAY_ROLE_PREFIXES:
        for suffix in _GATEWAY_PEER_SUFFIXES + _GATEWAY_FINGERPRINT_SUFFIXES:
            keys.append(f"{prefix}_{suffix}")
    return tuple(keys)


def _gateway_environment_block(text: str) -> str:
    """Only the ``services.gateway.environment`` lines of one compose file.

    A ``KEY=${KEY...}`` substitution under ANOTHER service (or anywhere else
    in an overlay) does not reach the gateway container, so matching it would
    report a passthrough that is not there (Qodo, PR #213). Same stdlib
    indentation scan as :func:`lobes.runtime._compose._override_service_keys`
    (the runtime carries no YAML parser): a service is a two-space key, its
    ``environment:`` a four-space key, and the block is every deeper-indented
    line that follows until the indent comes back up.
    """
    out: list[str] = []
    in_gateway = False
    in_env = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 2 and stripped.endswith(":"):
            in_gateway = stripped == "gateway:"
            in_env = False
            continue
        if not in_gateway:
            continue
        if indent == 4:
            in_env = stripped == "environment:"
            continue
        if in_env and indent > 4:
            out.append(stripped)
    return "\n".join(out)


def _has_passthrough(compose_text: str, key: str) -> bool:
    """True iff the compose text contains a ``KEY=${KEY...`` substitution.

    Matches both ``${KEY:-default}`` and bare ``${KEY}`` forms — both start
    with the literal ``KEY=${KEY`` this checks for. A key name that is a
    strict prefix of another declared key (e.g. ``PRIMARY_PEER_ORIGIN`` vs
    ``PRIMARY_PEER_ORIGINS``) cannot false-match: the shared prefix is always
    followed by ``=`` in a real passthrough line, so a differently-suffixed
    neighbour (which continues with another letter, not ``=``) never matches.
    """
    return f"{key}=${{{key}" in compose_text


_PASSTHROUGH_COMPOSE_FILES: tuple[str, ...] = (
    _compose.COMPOSE_FILE,
    "docker-compose.audio.yml",
    "docker-compose.shape.yml",
    "docker-compose.override.yml",
)


def _gateway_passthrough_check(deploy_dir: Path) -> dict:
    """A ``.env`` key set (non-empty) must reach the gateway container.

    The 2026-07-17 incident this guards against: the ``MUSE_*`` keys were set
    in ``.env`` but the deployed ``docker-compose.yml`` had no passthrough
    line for them in the gateway service's ``environment:`` block, so the
    values silently never reached the gateway container while every other
    check (including ``/health``) stayed green. Read-only: this check only
    reports; it never patches ``docker-compose.yml`` (that is a compose
    re-scaffold via ``lobes init --apply``, never doctor's ``--fix``, per its
    own "never rewrites/patches compose" contract).

    Fleet-only (the legacy single-model scaffold has no gateway service at
    all) and tolerant of an absent compose file — that is the pre-existing
    ``scaffold_files``/``compose_present`` finding's job, not this one's.
    """
    if not _compose.is_fleet(deploy_dir):
        return _check(
            "gateway_passthrough",
            True,
            "info",
            "gateway passthrough check applies to fleet deployments only",
        )
    compose_path = deploy_dir / _compose.COMPOSE_FILE
    if not compose_path.is_file():
        return _check(
            "gateway_passthrough",
            True,
            "info",
            "docker-compose.yml absent — see scaffold_files",
        )
    deployed = _env.read_env_file(deploy_dir / _compose.ENV_FILE)
    # A passthrough may legitimately live in an overlay rather than the base
    # file: an operator-owned docker-compose.override.yml (the deployed Spark
    # already adds HAND_FEASIBLE there) or the generated shape/audio overlays.
    # Scan every overlay present so an override-placed passthrough is not
    # falsely reported missing — the check is about whether the value REACHES
    # the container, not which file carries it.
    compose_text = "\n".join(
        _gateway_environment_block((deploy_dir / name).read_text(encoding="utf-8"))
        for name in _PASSTHROUGH_COMPOSE_FILES
        if (deploy_dir / name).is_file()
    )
    missing = sorted(
        key
        for key in _gateway_relevant_keys()
        if (deployed.get(key) or "").strip() and not _has_passthrough(compose_text, key)
    )
    if missing:
        shown = ", ".join(missing[:8])
        more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        return _check(
            "gateway_passthrough",
            False,
            "warn",
            f"{len(missing)} .env key(s) set but missing a gateway passthrough "
            f"in docker-compose*.yml: {shown}{more}",
            "re-scaffold docker-compose.yml from the packaged template "
            "('lobes init --apply') — doctor never patches compose directly",
        )
    return _check(
        "gateway_passthrough",
        True,
        "info",
        "every set gateway-relevant .env key has a compose passthrough",
    )


# --- scaffold integrity + profile staleness (issue #119) --------------------

_FIX_REMEDIATION = (
    "run 'lobes doctor --fix' to see the missing-only heal plan, then "
    "'lobes doctor --fix --apply' to write it (absent files/keys only — an "
    "existing .env line is never rewritten)"
)

# --- the associate lane's authenticated front (lightning-on-orin plan, t10) --
# Grounded in a REAL incident, not a hypothetical. On 2026-08-25 the Lightning
# spike ran NVIDIA's published Jetson recipe verbatim, which binds an
# OpenAI-compatible 30B generate endpoint on the box's tailnet with no
# credential. Within seconds of the API server starting, two DISTINCT tailnet
# peers queried it, neither initiated by the operator — see
# docs/evidence/2026-08-25-spike-lightning-vllm-orin.txt (the two `GET
# /v1/models 200 OK` lines from 100.127.105.72 and 100.105.216.63) and
# docs/evidence/2026-08-26-associate-gateway-auth-front.txt.
#
# The shipped lane publishes NO host port (the gateway is the only front
# door), so the remaining way to leave it uncredentialed is to run the gateway
# itself with no inbound key. This check is what makes "the associate lane
# binds behind GATEWAY_API_KEY" a CHECKED property of a deployment rather than
# something the operator is trusted to have remembered.
#
# Deliberately scoped to a LOCALLY HOSTED associate (ASSOCIATE_BASE_URL set —
# the activation env an associate-hosting shape renders). A box that merely
# refers or proxies associate to a peer hosts nothing, and every pre-associate
# deployment is byte-identically unaffected: the finding is not emitted at all.
_ASSOCIATE_HOST_KEY = "ASSOCIATE_BASE_URL"
#: Same precedence the gateway itself resolves its inbound key by
#: (lobes.gateway._config.inbound_api_key): explicit knob first, legacy
#: Culture-wide channel second. Either one arms the bearer gate.
_INBOUND_KEY_VARS: tuple[str, ...] = ("GATEWAY_API_KEY", "CULTURE_VLLM_API_KEY")


def _associate_auth_gate_check(deploy_dir: Path) -> dict | None:
    """``associate`` is hosted here ⇒ the gateway must require a bearer key.

    Returns ``None`` (no finding at all) when this box does not host the lane.
    Never echoes key material — only whether a non-blank value is present.
    """
    env_path = deploy_dir / _compose.ENV_FILE
    hosted = (_env.read_env(env_path, _ASSOCIATE_HOST_KEY) or "").strip()
    if not hosted:
        return None
    armed = [var for var in _INBOUND_KEY_VARS if (_env.read_env(env_path, var) or "").strip()]
    if armed:
        return _check(
            "associate_auth_gate",
            True,
            "info",
            f"associate lane is behind the gateway bearer gate ({armed[0]} set)",
        )
    return _check(
        "associate_auth_gate",
        False,
        "error",
        "associate lane is hosted here but the gateway requires no credential "
        "(neither GATEWAY_API_KEY nor CULTURE_VLLM_API_KEY is set)",
        "set GATEWAY_API_KEY in .env (scripts/gen-api-key.py mints one), then "
        "'lobes fleet up --apply' — an uncredentialed generate lane was queried "
        "by two tailnet peers within seconds on 2026-08-25",
    )


# The packaged template tree every scaffold file is materialised from — the
# same resource root `lobes init` / `lobes.runtime._compose` read.
_TEMPLATES_PACKAGE = "lobes.templates"


def _parse_env_text(text: str) -> dict[str, str]:
    """``KEY=VALUE`` lines from template text — same contract as
    :func:`lobes.runtime._env.read_env_file`, which only reads paths."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip():
            out[key.strip()] = value.strip()
    return out


def _expected_templates(deploy_dir: Path) -> dict[str, str]:
    """Template name -> dest file this FLEET deployment is expected to carry.

    The audio set is expected iff the overlay compose file is scaffolded —
    audio is opt-in, so a no-audio deployment is never flagged for lacking it.
    """
    templates = dict(_compose.FLEET_TEMPLATES)
    if _compose.audio_overlay_present(deploy_dir):
        templates.update(_compose.AUDIO_TEMPLATES)
    return templates


def _template_env_defaults() -> dict[str, str]:
    """The fleet ``env.example`` defaults, keyed by env var."""
    root = _resource_files(_TEMPLATES_PACKAGE)
    return _parse_env_text(_compose._read_template(root, "fleet/env.example"))


def _audio_env_defaults() -> dict[str, str]:
    """The audio overlay's ``env.audio.example`` defaults, keyed by env var."""
    root = _resource_files(_TEMPLATES_PACKAGE)
    return _parse_env_text(_compose._read_template(root, _compose.AUDIO_ENV_TEMPLATE))


def _dropped_role_prefixes(deploy_dir: Path) -> tuple[str, ...]:
    """``PREFIX_`` env prefixes of the core roles the deployment SHAPE drops.

    Read back from the generated ``docker-compose.shape.yml`` (the single
    source of truth for the drop decision) so the staleness diff never demands
    knobs for a lobe this box deliberately does not host.
    """
    path = deploy_dir / _compose.SHAPE_OVERLAY
    if not path.is_file():
        return ()
    services = _compose._override_service_keys(path.read_text(encoding="utf-8")) - {"gateway"}
    service_to_role = {service: role for role, service in ROLE_SERVICE.items()}
    return tuple(
        ROLE_ENV_PREFIX[service_to_role[s]] + "_"
        for s in sorted(services)
        if s in service_to_role and service_to_role[s] in ROLE_ENV_PREFIX
    )


def _scaffold_files_check(deploy_dir: Path) -> tuple[dict, list[str]]:
    """Every expected scaffold file exists — the 2026-07-17 Spark incident was
    files silently absent (audio Dockerfiles) with ``/health`` green for hours."""
    expected = list(_expected_templates(deploy_dir).values())
    plugin_dest, plugin_exists = _compose.plugin_plan(deploy_dir)
    missing = sorted(dest for dest in expected if not (deploy_dir / dest).exists())
    if not plugin_exists:
        missing.append(plugin_dest)
    if missing:
        return (
            _check(
                "scaffold_files",
                False,
                "warn",
                f"missing scaffold file(s): {', '.join(missing)} — the lanes built "
                "from them cannot start",
                _FIX_REMEDIATION,
            ),
            missing,
        )
    return (
        _check(
            "scaffold_files",
            True,
            "info",
            f"all {len(expected) + 1} expected scaffold files present",
        ),
        [],
    )


def _resolve_deployment_profile(deploy_dir: Path, recorded: str | None):
    """``(profile, note)`` — the profile this deployment should be checked
    against, resolved via the SAME path ``lobes init`` uses (no forked render).

    A recorded ``LOBES_PROFILE`` wins (it is what the operator's deployment
    claims to be); an unresolvable recorded name degrades to detection with a
    note rather than failing the whole check.
    """
    try:
        profile, _card, warning = resolve_init_profile(recorded, deploy_dir)
        return profile, warning
    except ModelGearError as err:
        profile, _card, _warning = resolve_init_profile(None, deploy_dir)
        return profile, (
            f"recorded LOBES_PROFILE {recorded!r} did not resolve ({err.message}); "
            "checked against card detection instead"
        )


def _staleness_ok(profile_name: str, overridden: list[str], detail: str) -> dict:
    """The passing verdict — operator overrides only ever annotate it."""
    message = f"profile {profile_name}: every required .env key is present"
    if overridden:
        message += f" ({len(overridden)} operator-set value(s) differ — legitimate)"
    if detail:
        message += f" [{detail}]"
    return _check("profile_staleness", True, "info", message)


def _staleness_failure(
    profile_name: str, missing: dict[str, str], stale: list[str], detail: str
) -> dict:
    """The warn verdict — missing keys are fixable, stale ones named for the operator."""
    parts: list[str] = []
    remediations: list[str] = []
    if missing:
        shown = ", ".join(sorted(missing)[:6])
        more = "" if len(missing) <= 6 else f" (+{len(missing) - 6} more)"
        parts.append(f"{len(missing)} required key(s) missing from .env: {shown}{more}")
        remediations.append(_FIX_REMEDIATION)
    if stale:
        parts.append(
            f"{len(stale)} key(s) still carry the template default where profile "
            f"{profile_name} requires a divergence: {', '.join(stale)}"
        )
        remediations.append(
            "set the divergent key(s) in .env yourself — doctor --fix never "
            "rewrites an existing line"
        )
    if detail:
        parts.append(detail)
    return _check("profile_staleness", False, "warn", "; ".join(parts), "; ".join(remediations))


def _staleness_verdict(
    profile_name: str,
    missing: dict[str, str],
    stale: list[str],
    overridden: list[str],
    notes: list[str],
) -> dict:
    """Fold the staleness diff into one check dict (missing/stale warn; overrides info)."""
    detail = "; ".join(notes) if notes else ""
    if not missing and not stale:
        return _staleness_ok(profile_name, overridden, detail)
    return _staleness_failure(profile_name, missing, stale, detail)


def _profile_staleness_check(deploy_dir: Path) -> tuple[dict, dict[str, str]]:
    """The deployed ``.env`` carries what the resolved machine profile requires.

    The 2026-07-14 Thor incident: a pre-#110 ``.env`` missing the thor
    profile's SM_110 divergences served for weeks — ``/health`` green, rerank
    lane hanging. Three diff classes, honestly separated: a MISSING key is a
    stale/partial scaffold (warn, fixable); a key still carrying the TEMPLATE
    default where the profile requires a divergence is a never-applied render
    (warn, named but not auto-fixed — rewriting an existing line is not
    doctor's to do); any other difference is an operator override (info —
    reclaim values and hand-tuning are legitimate).
    """
    env_path = deploy_dir / _compose.ENV_FILE
    deployed = _env.read_env_file(env_path)
    recorded = deployed.get("LOBES_PROFILE") or None
    profile, note = _resolve_deployment_profile(deploy_dir, recorded)
    rendered = render_shape(resolve_shape(DEFAULT_SHAPE), profile).env
    dropped = _dropped_role_prefixes(deploy_dir)
    required = {k: str(v) for k, v in rendered.items() if not k.startswith(dropped)}
    missing = {k: v for k, v in required.items() if k not in deployed}
    if _compose.audio_overlay_present(deploy_dir):
        missing.update({k: v for k, v in _audio_env_defaults().items() if k not in deployed})
    template_defaults = _template_env_defaults()
    stale: list[str] = []
    overridden: list[str] = []
    for key, want in sorted(required.items()):
        have = deployed.get(key)
        if have is None or _values_equal(have, want):
            continue
        default = template_defaults.get(key)
        if default is not None and _values_equal(have, default):
            stale.append(key)
        else:
            overridden.append(key)
    notes = []
    if recorded is None:
        notes.append(
            "no LOBES_PROFILE recorded — the deployment predates per-machine "
            "profiles (#110) or was scaffolded by hand"
        )
    if note:
        notes.append(note)
    return _staleness_verdict(profile.name, missing, stale, overridden, notes), missing


def _apply_fix(deploy_dir: Path) -> list[str]:
    """Write the missing-only heal: absent files, then absent ``.env`` keys.

    Files first — a freshly written ``.env`` changes which keys are missing, so
    the key set is recomputed AFTER the file pass. Append-only on ``.env``:
    every pre-existing line survives byte-for-byte, and a key that is present
    (even empty) is never touched — compose ``env_file`` last-wins semantics
    would let an appended duplicate silently clobber the operator's value.
    """
    _files_verdict, missing_files = _scaffold_files_check(deploy_dir)
    actions = _write_missing_files(deploy_dir, missing_files)
    _env_verdict, missing_env = _profile_staleness_check(deploy_dir)
    return actions + _append_missing_env(deploy_dir, missing_env)


def _write_missing_file(deploy_dir: Path, dest: str, by_dest: dict[str, str]) -> None:
    """Materialise ONE absent scaffold file from its packaged template."""
    if dest == _compose.PLUGIN_DEST_NAME:
        _compose.write_plugin_file(deploy_dir, force=False)
        return
    target = deploy_dir / dest
    root = _resource_files(_TEMPLATES_PACKAGE)
    target.write_text(_compose._read_template(root, by_dest[dest]), encoding="utf-8")
    if dest == _compose.ENV_FILE:
        try:
            target.chmod(0o600)  # secrets file — owner-only, like the scaffold
        except OSError:
            pass


def _write_missing_files(deploy_dir: Path, missing_files: list[str]) -> list[str]:
    actions: list[str] = []
    by_dest = {dest: name for name, dest in _expected_templates(deploy_dir).items()}
    for dest in missing_files:
        if (deploy_dir / dest).exists():  # missing-only, by construction; never overwrite
            continue
        _write_missing_file(deploy_dir, dest, by_dest)
        actions.append(f"wrote {dest}")
    return actions


def _append_missing_env(deploy_dir: Path, missing_env: dict[str, str]) -> list[str]:
    if not missing_env:
        return []
    env_path = deploy_dir / _compose.ENV_FILE
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write("\n# --- appended by 'lobes doctor --fix --apply' (missing-only heal, #119) ---\n")
        for key in sorted(missing_env):
            fh.write(f"{key}={missing_env[key]}\n")
    return [f"appended {key}" for key in sorted(missing_env)]


def _diagnose(compose_dir: str | None = None) -> dict[str, object]:
    checks: list[dict] = [_docker_check()]

    deploy_dir: Path | None = None
    try:
        deploy_dir = _compose.resolve_deployment_dir(compose_dir)
        checks.append(
            _check("compose_present", True, "error", f"deployment scaffolded at {deploy_dir}")
        )
    except ModelGearError as err:
        checks.append(_check("compose_present", False, "error", err.message, err.remediation))

    port = 8000
    fix_plan: dict[str, object] | None = None
    if deploy_dir is not None:
        env_path = deploy_dir / _compose.ENV_FILE
        checks.append(_env_coherence_check(env_path))
        port = _env.parse_port(_env.read_env(env_path, "VLLM_PORT", "8000"))
        # Scaffold integrity + profile staleness (issue #119) — fleet-only:
        # the legacy single-model scaffold has no per-role profile render.
        if _compose.is_fleet(deploy_dir):
            files_check, missing_files = _scaffold_files_check(deploy_dir)
            stale_check, missing_env = _profile_staleness_check(deploy_dir)
            passthrough_check = _gateway_passthrough_check(deploy_dir)
            checks.extend([files_check, stale_check, passthrough_check])
            # Only emitted when this box actually HOSTS the associate lane.
            auth_gate = _associate_auth_gate_check(deploy_dir)
            if auth_gate is not None:
                checks.append(auth_gate)
            fix_plan = {"files": missing_files, "env": missing_env}

    checks.append(_health_check(port))
    checks.append(_version_skew_check(port, deploy_dir))

    # Gather machine profile info (if scaffolded)
    machine_profile = _machine_profile_section(deploy_dir)

    # Only error-severity failures make the run unhealthy.
    healthy_overall = all(c["passed"] for c in checks if c["severity"] == "error")
    result: dict[str, object] = {"healthy": healthy_overall, "checks": checks}
    if machine_profile is not None:
        result["machine_profile"] = machine_profile
    if fix_plan is not None:
        result["fix_plan"] = fix_plan
    return result


def _mark(check: dict) -> str:
    if check["passed"]:
        return "ok"
    return "FAIL" if check["severity"] == "error" else check["severity"]


def _render_check_lines(checks: list[dict]) -> list[str]:
    """Render one ``[mark] id: message`` line per check, plus a remediation
    hint line for any that failed with one."""
    lines: list[str] = []
    for check in checks:
        lines.append(f"[{_mark(check)}] {check['id']}: {check['message']}")
        if not check["passed"] and check["remediation"]:
            lines.append(f"  hint: {check['remediation']}")
    return lines


def _render_machine_profile_lines(mp: dict) -> list[str]:
    """Render the ``machine profile:`` section (detected card, active
    profile, and any mismatch/unknown-card warning)."""
    lines = ["", "machine profile:", f"  detected card: {mp['detected_card']}"]
    if mp["device_name"]:
        lines.append(f"  device:       {mp['device_name']}")
    if mp["compute_capability"]:
        lines.append(f"  compute:      {mp['compute_capability']}")
    if mp["total_memory_gb"]:
        lines.append(f"  memory:       {mp['total_memory_gb']} GB")
    if mp["profile"]:
        lines.append(f"  profile:      {mp['profile']}")
        if not mp["validated"]:
            lines.append("  status:       NOT VALIDATED FOR THIS CARD (forced/unvalidated)")
    if mp.get("warning"):
        lines.append(f"  warning:      {mp['warning']}")
    return lines


def _render_fix_plan_lines(plan: dict, *, applied: list[str] | None) -> list[str]:
    """The ``--fix`` section: the missing-only heal plan, or what was applied."""
    if applied is not None:
        lines = ["", "fix applied (missing-only heal, #119):"]
        lines.extend(f"  {action}" for action in applied)
        if not applied:
            lines.append("  nothing to heal — no absent files or keys")
        return lines
    files = plan.get("files") or []
    env = plan.get("env") or {}
    lines = ["", "fix plan (DRY RUN — re-run with --fix --apply to write):"]
    lines.extend(f"  would write {dest}" for dest in files)
    lines.extend(f"  would append {key}={env[key]}" for key in sorted(env))
    if not files and not env:
        lines.append("  nothing to heal — no absent files or keys")
    return lines


def _render_text(report: dict) -> str:
    status = "healthy" if report["healthy"] else "unhealthy"
    lines = [f"lobes doctor: {status}", ""]
    lines.extend(_render_check_lines(report["checks"]))
    if "machine_profile" in report:
        lines.extend(_render_machine_profile_lines(report["machine_profile"]))
    if "fix_plan" in report and report.get("fix_requested"):
        lines.extend(_render_fix_plan_lines(report["fix_plan"], applied=report.get("fix_applied")))
    return "\n".join(lines)


def cmd_doctor(args: argparse.Namespace) -> int:
    fix = bool(getattr(args, "fix", False))
    apply = bool(getattr(args, "apply", False))
    if apply and not fix:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="--apply requires --fix",
            remediation="run 'lobes doctor --fix' for the heal plan, then add --apply",
        )
    compose_dir = getattr(args, "compose_dir", None)
    json_mode = bool(getattr(args, "json", False))
    report = _diagnose(compose_dir)
    if fix and "fix_plan" not in report:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="--fix needs a scaffolded FLEET deployment to heal",
            remediation="scaffold one first ('lobes init --apply'); the legacy "
            "single-model dir has no profile render to heal against",
        )
    if fix and apply:
        deploy_dir = _compose.resolve_deployment_dir(compose_dir)
        emit_diagnostic(f">> healing {deploy_dir} (missing-only)")
        applied = _apply_fix(deploy_dir)
        # Re-diagnose so the report describes the AFTER state — the proof the
        # heal worked is the checks passing, not the writes having happened.
        report = _diagnose(compose_dir)
        report["fix_applied"] = applied
    if fix:
        report["fix_requested"] = True
    emit_result(report if json_mode else _render_text(report), json_mode=json_mode)
    return 0 if report["healthy"] else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "doctor",
        help="Diagnose docker, the scaffold, .env coherence, profile staleness, "
        "and /health; --fix heals missing files/keys (dry-run; --apply commits).",
    )
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument(
        "--fix",
        action="store_true",
        help="Show the missing-only heal plan (writes nothing without --apply): "
        "absent scaffold files + absent .env keys. Never rewrites an existing line.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="With --fix: actually write the heal plan (absent files/keys only).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_doctor)
