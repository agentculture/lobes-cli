"""``lobes doctor`` — diagnose the local model deployment.

Real checks (no longer a stub): is docker available, is a deployment scaffolded,
is the ``.env`` coherent with ``culture.yaml``, is ``/health`` reachable, and
does the deployed gateway's own ``lobes-cli`` release match this CLI's (issue
#99). A down model is *not* an error (bringing it up is the tool's job) — only
missing docker, an un-scaffolded deployment, or a deployed-artifact version
mismatch fail the run.

JSON contract: ``{healthy, checks:[{id, passed, severity, message, remediation}]}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lobes import __version__
from lobes.cli._commands.whoami import _find_culture_yaml
from lobes.cli._errors import ModelGearError
from lobes.cli._output import emit_result
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
    if deploy_dir is not None:
        env_path = deploy_dir / _compose.ENV_FILE
        checks.append(_env_coherence_check(env_path))
        port = _env.parse_port(_env.read_env(env_path, "VLLM_PORT", "8000"))

    checks.append(_health_check(port))
    checks.append(_version_skew_check(port, deploy_dir))

    # Gather machine profile info (if scaffolded)
    machine_profile = _machine_profile_section(deploy_dir)

    # Only error-severity failures make the run unhealthy.
    healthy_overall = all(c["passed"] for c in checks if c["severity"] == "error")
    result: dict[str, object] = {"healthy": healthy_overall, "checks": checks}
    if machine_profile is not None:
        result["machine_profile"] = machine_profile
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


def _render_text(report: dict) -> str:
    status = "healthy" if report["healthy"] else "unhealthy"
    lines = [f"lobes doctor: {status}", ""]
    lines.extend(_render_check_lines(report["checks"]))
    if "machine_profile" in report:
        lines.extend(_render_machine_profile_lines(report["machine_profile"]))
    return "\n".join(lines)


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _diagnose(getattr(args, "compose_dir", None))
    json_mode = bool(getattr(args, "json", False))
    emit_result(report if json_mode else _render_text(report), json_mode=json_mode)
    return 0 if report["healthy"] else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "doctor",
        help="Diagnose docker, the deployment scaffold, .env coherence, and /health.",
    )
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_doctor)
