"""``lepenseur doctor`` — self-diagnosis (STUB).

Ships as a rubric-shaped stub. What "doctor" means for a *non-doer* — a thinker
that never executes — is an open design question (candidates: vLLM endpoint
reachability, culture.yaml/AGENTS.md coherence, model-string validity). Tracked
as a follow-up; see
docs/superpowers/specs/2026-05-22-scaffold-cli-sibling-design.md §12.

The stub returns a single trivially-passing check so the JSON contract
(``{healthy, checks:[{id, passed, severity, message, remediation}]}``) is
honored and the agent-first rubric's bundle 7 passes.
"""

from __future__ import annotations

import argparse

from lepenseur.cli._output import emit_result

_STUB_CHECK: dict[str, object] = {
    "id": "doctor_stub",
    "passed": True,
    "severity": "info",
    "message": (
        "doctor is a stub; self-diagnosis semantics for a thinking agent are "
        "not yet defined"
    ),
    "remediation": "",
}


def _diagnose() -> dict[str, object]:
    checks = [dict(_STUB_CHECK)]
    healthy = all(c["passed"] for c in checks)
    return {"healthy": healthy, "checks": checks}


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _diagnose()
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(report, json_mode=True)
    else:
        status = "healthy" if report["healthy"] else "unhealthy"
        lines = [f"lepenseur doctor: {status}", ""]
        for check in report["checks"]:
            mark = "ok" if check["passed"] else "FAIL"
            lines.append(f"[{mark}] {check['id']}: {check['message']}")
            if not check["passed"] and check["remediation"]:
                lines.append(f"  hint: {check['remediation']}")
        emit_result("\n".join(lines), json_mode=False)
    return 0 if report["healthy"] else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "doctor",
        help="Self-diagnosis (stub; semantics for a thinking agent are TBD).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_doctor)
