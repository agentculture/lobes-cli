"""Resolve the deployment directory and drive ``docker compose`` (stdlib only).

The deployment directory holds ``docker-compose.yml`` + ``.env``. ``model init``
scaffolds it (default ``~/.model-gear``); every model-ops verb resolves it via
:func:`resolve_deployment_dir`. All subprocess calls use fixed argv lists (no
shell) — the bandit ``B404``/``B603`` skips in ``pyproject.toml`` cover them.
"""

from __future__ import annotations

import os
import subprocess  # fixed argv lists only, never shell=True
from importlib.resources import files
from pathlib import Path

from model_gear.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError

CONTAINER = "model-gear-vllm"
COMPOSE_FILE = "docker-compose.yml"
ENV_FILE = ".env"
DOCKERFILE_GATEWAY = "Dockerfile.gateway"

# Fleet container names (model init --fleet / model fleet ...): two always-warm
# vLLM backends plus the stdlib gateway that fronts them on one OpenAI port.
FLEET_PRIMARY = "model-gear-vllm-primary"
FLEET_FALLBACK = "model-gear-vllm-fallback"
FLEET_GATEWAY = "model-gear-gateway"
FLEET_CONTAINERS = (FLEET_PRIMARY, FLEET_FALLBACK, FLEET_GATEWAY)

# Audio overlay (model init --fleet --audio): STT + TTS + the realtime bridge,
# layered on the base fleet via a compose override and fronted by the gateway.
AUDIO_OVERLAY = "docker-compose.audio.yml"
FLEET_STT = "model-gear-stt"
FLEET_TTS = "model-gear-tts"
FLEET_REALTIME = "model-gear-realtime"
FLEET_AUDIO_CONTAINERS = (FLEET_STT, FLEET_TTS, FLEET_REALTIME)

# Template filename -> destination filename written by the scaffold. The single
# template set is the default (every existing caller stays unchanged); the fleet
# set scaffolds the 3-container gateway deployment (model init --fleet).
# Cloudflare Tunnel example scaffolded alongside both deployments (the tunnel
# fronts the single-model :8000 or the fleet gateway). Copied verbatim as an
# `.example`; the owner copies it to the gitignored `.cf-tunnel.env` and edits.
CF_TUNNEL_EXAMPLE = "cf-tunnel.env.example"
SINGLE_TEMPLATES = {
    "docker-compose.yml": COMPOSE_FILE,
    "env.example": ENV_FILE,
    CF_TUNNEL_EXAMPLE: CF_TUNNEL_EXAMPLE,
}
FLEET_TEMPLATES = {
    "fleet/docker-compose.yml": COMPOSE_FILE,
    "fleet/env.example": ENV_FILE,
    "fleet/Dockerfile.gateway": DOCKERFILE_GATEWAY,
    CF_TUNNEL_EXAMPLE: CF_TUNNEL_EXAMPLE,
}
# The --audio extras layered on FLEET_TEMPLATES: the compose override, the two
# image build files, and the vendored Parakeet server. The audio .env keys are
# appended to .env separately (env.audio.example → AUDIO_ENV_TEMPLATE) so they
# extend the fleet .env instead of clobbering it.
AUDIO_TEMPLATES = {
    "fleet/docker-compose.audio.yml": AUDIO_OVERLAY,
    "fleet/Dockerfile.realtime": "Dockerfile.realtime",
    "fleet/Dockerfile.parakeet": "Dockerfile.parakeet",
    "fleet/listen_server.py": "listen_server.py",
    # _readiness.py is COPY'd into the Parakeet image (Dockerfile.parakeet), so
    # it MUST land at the deployment-dir root or `docker compose build stt`
    # fails on the COPY. Vendored twin of model_gear/realtime/_readiness.py.
    "fleet/_readiness.py": "_readiness.py",
}
AUDIO_ENV_TEMPLATE = "fleet/env.audio.example"
# Back-compat alias: the single set was the only one before the fleet existed.
_TEMPLATES = SINGLE_TEMPLATES


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


def _read_template(template_root, name: str) -> str:
    """Read a packaged template by its (possibly ``fleet/``-prefixed) name.

    ``importlib.resources`` traversables join one path segment per ``/`` call,
    so split the name and chain rather than passing ``"fleet/foo"`` in one go.
    """
    node = template_root
    for part in name.split("/"):
        node = node / part
    return node.read_text(encoding="utf-8")


def scaffold_plan(
    target: Path, templates: dict[str, str] = SINGLE_TEMPLATES
) -> list[tuple[str, bool]]:
    """Return ``(dest_name, already_exists)`` for each file ``init`` would write."""
    return [(dest, (target / dest).exists()) for dest in templates.values()]


def write_scaffold(
    target: os.PathLike | str, *, force: bool, templates: dict[str, str] = SINGLE_TEMPLATES
) -> list[Path]:
    """Copy the packaged templates into ``target``. Returns written paths.

    Refuses to overwrite an existing file unless ``force`` is set. ``templates``
    selects the template set (single-model by default, ``FLEET_TEMPLATES`` for
    the gateway deployment).
    """
    dest_dir = Path(target).expanduser()
    template_root = files("model_gear.templates")
    written: list[Path] = []
    for tname, dest_name in templates.items():
        dest = dest_dir / dest_name
        if dest.exists() and not force:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"{dest} already exists",
                remediation="re-run with --force to overwrite",
            )
    dest_dir.mkdir(parents=True, exist_ok=True)
    for tname, dest_name in templates.items():
        content = _read_template(template_root, tname)
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


def append_audio_env(target: os.PathLike | str) -> Path:
    """Append the audio overlay's env keys (``env.audio.example``) to ``.env``.

    The fleet ``.env`` is written first (``write_scaffold``); ``--audio`` then
    appends its keys (NGC_API_KEY, ports, voices, AUDIO_URL …) so they extend the
    fleet config rather than overwrite it. Returns the ``.env`` path.
    """
    env_path = Path(target).expanduser() / ENV_FILE
    content = _read_template(files("model_gear.templates"), AUDIO_ENV_TEMPLATE)
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write(content if content.startswith("\n") else "\n" + content)
    return env_path


# --- docker compose --------------------------------------------------------


def audio_overlay_present(deploy_dir: os.PathLike | str) -> bool:
    """True when the ``--audio`` overlay (``docker-compose.audio.yml``) is scaffolded."""
    return (Path(deploy_dir) / AUDIO_OVERLAY).is_file()


def fleet_containers(deploy_dir: os.PathLike | str) -> tuple[str, ...]:
    """Fleet container names, including the audio trio when the overlay is present."""
    if audio_overlay_present(deploy_dir):
        return FLEET_CONTAINERS + FLEET_AUDIO_CONTAINERS
    return FLEET_CONTAINERS


def _compose_files(deploy_dir: os.PathLike | str) -> list[str]:
    """``-f`` args: just the base file, or base + audio overlay when present.

    Returns ``[]`` when no overlay (``docker compose`` finds docker-compose.yml on
    its own), so a plain fleet keeps its current argv unchanged.
    """
    if audio_overlay_present(deploy_dir):
        return ["-f", COMPOSE_FILE, "-f", AUDIO_OVERLAY]
    return []


def _run(argv: list[str], *, cwd: str | None = None, timeout: int | None = None):
    """Run a command that MUST exist (docker); raise ENV_ERROR if it doesn't."""
    try:
        return subprocess.run(  # fixed argv, no shell
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
        return subprocess.run(  # fixed argv, no shell
            argv, capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def compose_down(deploy_dir: os.PathLike | str):
    return _run(["docker", "compose"] + _compose_files(deploy_dir) + ["down"], cwd=str(deploy_dir))


def compose_up_detached(deploy_dir: os.PathLike | str):
    return _run(["docker", "compose", "up", "-d"], cwd=str(deploy_dir))


def compose_up_build(deploy_dir: os.PathLike | str):
    """``docker compose up -d --build`` — used by the fleet, whose gateway service
    is built from a local ``Dockerfile.gateway`` (``--build`` picks up a new image
    on a re-run; first run builds either way). When the ``--audio`` overlay is
    scaffolded, the audio services (built from ``Dockerfile.realtime`` /
    ``Dockerfile.parakeet``) are layered in via ``-f docker-compose.audio.yml``."""
    return _run(
        ["docker", "compose"] + _compose_files(deploy_dir) + ["up", "-d", "--build"],
        cwd=str(deploy_dir),
    )


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
