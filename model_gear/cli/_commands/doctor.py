"""``model doctor`` — diagnose the local model deployment.

Real checks (no longer a stub): is docker available, is a deployment scaffolded,
is the ``.env`` coherent with ``culture.yaml``, and is ``/health`` reachable. A
down model is *not* an error (bringing it up is the tool's job) — only missing
docker or an un-scaffolded deployment fail the run.

JSON contract: ``{healthy, checks:[{id, passed, severity, message, remediation}]}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model_gear.cli._commands.whoami import _find_culture_yaml
from model_gear.cli._errors import ModelGearError
from model_gear.cli._output import emit_result
from model_gear.runtime import _compose, _env, _health


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


def _diagnose(compose_dir: str | None = None) -> dict[str, object]:
    checks: list[dict] = []

    docker_ok = _compose.docker_available()
    checks.append(
        _check(
            "docker_available",
            docker_ok,
            "error",
            (
                "docker + docker compose are available"
                if docker_ok
                else "docker / docker compose not found"
            ),
            "" if docker_ok else "install Docker + the NVIDIA Container Toolkit",
        )
    )

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
        served = _env.read_env(env_path, "VLLM_SERVED_NAME")
        expected = _culture_model_tail()
        if not served:
            checks.append(
                _check(
                    "env_coherence",
                    False,
                    "warn",
                    "VLLM_SERVED_NAME is not set in .env",
                    "set it, or run 'model switch <model> --apply'",
                )
            )
        elif expected and served != expected:
            checks.append(
                _check(
                    "env_coherence",
                    False,
                    "warn",
                    f"VLLM_SERVED_NAME ({served}) != culture.yaml model tail ({expected})",
                    "align them so the acp vllm-local provider resolves the model",
                )
            )
        else:
            checks.append(_check("env_coherence", True, "info", f"VLLM_SERVED_NAME = {served}"))
        port = _env.parse_port(_env.read_env(env_path, "VLLM_PORT", "8000"))

    healthy = _health.is_healthy(port)
    checks.append(
        _check(
            "health_reachable",
            healthy,
            "info",
            f"/health responding on :{port}" if healthy else f"/health not responding on :{port}",
            "" if healthy else "start the server with 'model serve --apply'",
        )
    )

    # Only error-severity failures make the run unhealthy.
    healthy_overall = all(c["passed"] for c in checks if c["severity"] == "error")
    return {"healthy": healthy_overall, "checks": checks}


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _diagnose(getattr(args, "compose_dir", None))
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(report, json_mode=True)
    else:
        status = "healthy" if report["healthy"] else "unhealthy"
        lines = [f"model doctor: {status}", ""]
        for check in report["checks"]:
            if check["passed"]:
                mark = "ok"
            else:
                mark = "FAIL" if check["severity"] == "error" else check["severity"]
            lines.append(f"[{mark}] {check['id']}: {check['message']}")
            if not check["passed"] and check["remediation"]:
                lines.append(f"  hint: {check['remediation']}")
        emit_result("\n".join(lines), json_mode=False)
    return 0 if report["healthy"] else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "doctor",
        help="Diagnose docker, the deployment scaffold, .env coherence, and /health.",
    )
    p.add_argument(
        "--compose-dir", help="Deployment dir (default: $MODEL_GEAR_DIR or ~/.model-gear)."
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_doctor)
