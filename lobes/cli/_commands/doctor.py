"""``lobes doctor`` — diagnose the local model deployment.

Real checks (no longer a stub): is docker available, is a deployment scaffolded,
is the ``.env`` coherent with ``culture.yaml``, and is ``/health`` reachable. A
down model is *not* an error (bringing it up is the tool's job) — only missing
docker or an un-scaffolded deployment fail the run.

JSON contract: ``{healthy, checks:[{id, passed, severity, message, remediation}]}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lobes.cli._commands.whoami import _find_culture_yaml
from lobes.cli._errors import ModelGearError
from lobes.cli._output import emit_result
from lobes.runtime import _compose, _env, _health


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

    # Only error-severity failures make the run unhealthy.
    healthy_overall = all(c["passed"] for c in checks if c["severity"] == "error")
    return {"healthy": healthy_overall, "checks": checks}


def _mark(check: dict) -> str:
    if check["passed"]:
        return "ok"
    return "FAIL" if check["severity"] == "error" else check["severity"]


def _render_text(report: dict) -> str:
    status = "healthy" if report["healthy"] else "unhealthy"
    lines = [f"lobes doctor: {status}", ""]
    for check in report["checks"]:
        lines.append(f"[{_mark(check)}] {check['id']}: {check['message']}")
        if not check["passed"] and check["remediation"]:
            lines.append(f"  hint: {check['remediation']}")
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
