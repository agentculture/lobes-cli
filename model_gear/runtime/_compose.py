"""Resolve the deployment directory and drive ``docker compose`` (stdlib only).

The deployment directory holds ``docker-compose.yml`` + ``.env``. ``model init``
scaffolds it (default ``~/.model-gear``); every model-ops verb resolves it via
:func:`resolve_deployment_dir`. All subprocess calls use fixed argv lists (no
shell) — the bandit ``B404``/``B603`` skips in ``pyproject.toml`` cover them.
"""

from __future__ import annotations

import os
import subprocess  # noqa: S404 - fixed argv lists, never shell=True
from importlib.resources import files
from pathlib import Path

from model_gear.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError

CONTAINER = "model-gear-vllm"
COMPOSE_FILE = "docker-compose.yml"
ENV_FILE = ".env"

# Template filename -> destination filename written by the scaffold.
_TEMPLATES = {"docker-compose.yml": COMPOSE_FILE, "env.example": ENV_FILE}


def default_deployment_dir() -> Path:
    """The global deployment home: ``~/.model-gear``."""
    return Path.home() / ".model-gear"


def resolve_deployment_dir(explicit: os.PathLike | str | None) -> Path:
    """Resolve the directory holding ``docker-compose.yml``.

    Order: ``--compose-dir`` (explicit) → ``$MODEL_GEAR_DIR`` → ``~/.model-gear``.
    Raises :class:`ModelGearError` when the resolved directory has no
    ``docker-compose.yml`` (USER_ERROR for an explicit bad path, ENV_ERROR when
    the default home simply hasn't been scaffolded yet).
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        source, code = "--compose-dir", EXIT_USER_ERROR
    elif os.environ.get("MODEL_GEAR_DIR"):
        candidate = Path(os.environ["MODEL_GEAR_DIR"]).expanduser()
        source, code = "$MODEL_GEAR_DIR", EXIT_USER_ERROR
    else:
        candidate = default_deployment_dir()
        source, code = "default ~/.model-gear", EXIT_ENV_ERROR
    if not (candidate / COMPOSE_FILE).is_file():
        raise ModelGearError(
            code=code,
            message=f"no {COMPOSE_FILE} in {candidate} ({source})",
            remediation="run 'model init --apply' to scaffold ~/.model-gear, "
            "or pass --compose-dir / set MODEL_GEAR_DIR",
        )
    return candidate


# --- scaffolding -----------------------------------------------------------


def scaffold_plan(target: Path) -> list[tuple[str, bool]]:
    """Return ``(dest_name, already_exists)`` for each file ``init`` would write."""
    return [(dest, (target / dest).exists()) for dest in _TEMPLATES.values()]


def write_scaffold(target: os.PathLike | str, *, force: bool) -> list[Path]:
    """Copy the packaged templates into ``target``. Returns written paths.

    Refuses to overwrite an existing file unless ``force`` is set.
    """
    dest_dir = Path(target).expanduser()
    template_root = files("model_gear.templates")
    written: list[Path] = []
    for tname, dest_name in _TEMPLATES.items():
        dest = dest_dir / dest_name
        if dest.exists() and not force:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"{dest} already exists",
                remediation="re-run with --force to overwrite",
            )
    dest_dir.mkdir(parents=True, exist_ok=True)
    for tname, dest_name in _TEMPLATES.items():
        content = (template_root / tname).read_text(encoding="utf-8")
        dest = dest_dir / dest_name
        dest.write_text(content, encoding="utf-8")
        # .env is meant to hold secrets (HF_TOKEN); keep it owner-only on shared
        # hosts. Best-effort — chmod can fail on some filesystems (e.g. Windows).
        if dest_name == ENV_FILE:
            try:
                dest.chmod(0o600)
            except OSError:
                pass
        written.append(dest)
    return written


# --- docker compose --------------------------------------------------------


def _run(argv: list[str], *, cwd: str | None = None, timeout: int | None = None):
    """Run a command that MUST exist (docker); raise ENV_ERROR if it doesn't."""
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv, cwd=cwd, capture_output=True, text=True, check=False, timeout=timeout
        )
    except OSError as exc:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"could not run {argv[0]}: {exc}",
            remediation="ensure docker + the NVIDIA Container Toolkit are installed and on PATH",
        ) from exc


def _probe(argv: list[str], *, timeout: int = 10):
    """Run a best-effort command; return the result or ``None`` if it can't run."""
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv, capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def compose_down(deploy_dir: os.PathLike | str):
    return _run(["docker", "compose", "down"], cwd=str(deploy_dir))


def compose_up_detached(deploy_dir: os.PathLike | str):
    return _run(["docker", "compose", "up", "-d"], cwd=str(deploy_dir))


def docker_available() -> bool:
    """True if both ``docker`` and ``docker compose`` resolve."""
    d = _probe(["docker", "version"])
    if d is None or d.returncode != 0:
        return False
    c = _probe(["docker", "compose", "version"])
    return c is not None and c.returncode == 0


def container_status(container: str = CONTAINER) -> str:
    """Bare container lifecycle status (``running`` / ``exited`` / ``missing``)."""
    r = _probe(["docker", "inspect", "-f", "{{.State.Status}}", container])
    if r is None or r.returncode != 0:
        return "missing"
    return r.stdout.strip() or "missing"


def inspect_state(container: str = CONTAINER) -> str:
    """Status with health, e.g. ``running (healthy)`` — or ``not created``."""
    r = _probe(
        ["docker", "inspect", "-f", "{{.State.Status}} ({{.State.Health.Status}})", container]
    )
    if r is None or r.returncode != 0:
        return "not created"
    return r.stdout.strip() or "not created"


def container_image(container: str = CONTAINER) -> str:
    r = _probe(["docker", "inspect", container, "--format", "{{.Config.Image}}"])
    if r is None or r.returncode != 0:
        return "?"
    return r.stdout.strip() or "?"


def container_logs(container: str = CONTAINER, tail: int = 30) -> str:
    r = _probe(["docker", "logs", "--tail", str(tail), container])
    if r is None:
        return ""
    return (r.stdout or "") + (r.stderr or "")


def gpu_engine_mem() -> str:
    """Best-effort GPU memory used by the vLLM EngineCore process (via nvidia-smi)."""
    r = _probe(
        ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader"]
    )
    if r is None or r.returncode != 0:
        return "?"
    for line in r.stdout.splitlines():
        low = line.lower()
        if "enginecore" in low or "vllm" in low:
            parts = line.split(",")
            if len(parts) >= 2:
                return parts[1].strip()
    return "?"
