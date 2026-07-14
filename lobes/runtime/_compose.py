"""Resolve the deployment directory and drive ``docker compose`` (stdlib only).

The deployment directory holds ``docker-compose.yml`` + ``.env``. ``lobes init``
scaffolds it (default ``~/.lobes``); every model-ops verb resolves it via
:func:`resolve_deployment_dir`. All subprocess calls use fixed argv lists (no
shell) — the bandit ``B404``/``B603`` skips in ``pyproject.toml`` cover them.
"""

from __future__ import annotations

import os
import subprocess  # fixed argv lists only, never shell=True
from importlib.resources import files
from pathlib import Path

from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError

CONTAINER = "model-gear-vllm"
COMPOSE_FILE = "docker-compose.yml"
ENV_FILE = ".env"
DOCKERFILE_GATEWAY = "Dockerfile.gateway"

# Durable logs (issue #50). `mg-logwrap.sh` is scaffolded next to docker-compose.yml
# and bind-mounted into each vLLM service as the entrypoint; it tees stdout+stderr to
# a per-boot file under the host log dir so a crash trace survives restart/recreate.
# Host dir: $MODEL_GEAR_LOG_DIR (in .env) or <deploy>/logs (matches the compose default
# `${MODEL_GEAR_LOG_DIR:-./logs}`); in-container it mounts at /logs/model-gear.
LOG_WRAPPER = "mg-logwrap.sh"
LOG_DIRNAME = "logs"
LOG_DIR_ENV = "MODEL_GEAR_LOG_DIR"

# Fleet container names (lobes init --fleet / lobes fleet ...): the always-warm
# Qwen generate primary, the always-on multimodal (Gemma 4 12B) generate gear, the
# co-resident embedding + reranker gears, and the stdlib gateway that fronts them on
# one OpenAI port. All five are in the default fleet compose, so all five are in the
# default container set. A *generate* fallback (FLEET_FALLBACK) is opt-in (not in the
# default compose), so it is not.
FLEET_PRIMARY = "model-gear-vllm-primary"
FLEET_MULTIMODAL = "model-gear-vllm-multimodal"
FLEET_EMBED = "model-gear-vllm-embed"
FLEET_RERANK = "model-gear-vllm-rerank"
FLEET_FALLBACK = "model-gear-vllm-fallback"  # opt-in; not started by default
FLEET_GATEWAY = "model-gear-gateway"
FLEET_CONTAINERS = (FLEET_PRIMARY, FLEET_MULTIMODAL, FLEET_EMBED, FLEET_RERANK, FLEET_GATEWAY)

# Audio overlay (lobes init --fleet --audio): STT + TTS + the realtime bridge,
# layered on the base fleet via a compose override and fronted by the gateway.
AUDIO_OVERLAY = "docker-compose.audio.yml"
FLEET_STT = "model-gear-stt"
# The TTS sidecar's container is `model-gear-chatterbox` (docker-compose.audio.yml,
# the Chatterbox sidecar that replaced Magpie in 0.25) — must match that
# container_name or `lobes fleet status` reports the TTS gear as "not created".
FLEET_TTS = "model-gear-chatterbox"
FLEET_REALTIME = "model-gear-realtime"
FLEET_AUDIO_CONTAINERS = (FLEET_STT, FLEET_TTS, FLEET_REALTIME)

# Deployment-SHAPE override (lobes init --shape <mesh-shape>): brain-shapes t4b,
# issue #113. GENERATED (not a packaged template) by `lobes init` when the chosen
# shape DROPS a core role — it parks the dropped core service in an inert compose
# profile so `docker compose up` skips it, mirroring how AUDIO_OVERLAY layers on the
# base fleet. Without it, the base compose boots every core service unconditionally,
# so a dropped lobe RUNS and eats the GPU budget the shape reclaimed (proven live on
# the GB10: spark-lobe still booted model-gear-vllm-multimodal). machine-as-brain /
# bare init drop nothing, so no such file is written.
SHAPE_OVERLAY = "docker-compose.shape.yml"
# Core compose SERVICE name -> its container. A shape-dropped core service is read
# back out of the override file (see `shape_dropped_containers`) to exclude its
# container from the expected fleet set. Mirrors the base fleet's four core gears;
# the gateway is never a dropped role, so it is not here.
_CORE_SERVICE_CONTAINER: dict[str, str] = {
    "vllm-primary": FLEET_PRIMARY,
    "vllm-multimodal": FLEET_MULTIMODAL,
    "vllm-embed": FLEET_EMBED,
    "vllm-rerank": FLEET_RERANK,
}

# Template filename -> destination filename written by the scaffold. The library
# helpers below keep SINGLE_TEMPLATES as their function-level default (every
# existing caller stays unchanged); the FLEET set scaffolds the gateway
# deployment (the duo: primary + multimodal gear + embed/rerank gears). NOTE: the
# `lobes init` CLI default is now the FLEET set (issue #69) — `--single` opts back
# to SINGLE; the function default below is only the back-compat library helper.
# Cloudflare Tunnel example scaffolded alongside both deployments (the tunnel
# fronts the single-model :8000 or the fleet gateway). Copied verbatim as an
# `.example`; the owner copies it to the gitignored `.cf-tunnel.env` and edits.
CF_TUNNEL_EXAMPLE = "cf-tunnel.env.example"
SINGLE_TEMPLATES = {
    "docker-compose.yml": COMPOSE_FILE,
    "env.example": ENV_FILE,
    LOG_WRAPPER: LOG_WRAPPER,
    CF_TUNNEL_EXAMPLE: CF_TUNNEL_EXAMPLE,
}
FLEET_TEMPLATES = {
    "fleet/docker-compose.yml": COMPOSE_FILE,
    "fleet/env.example": ENV_FILE,
    "fleet/Dockerfile.gateway": DOCKERFILE_GATEWAY,
    # Custom vLLM image for the Gemma 4 12B multimodal gear (issue #71).
    # Layers a Transformers build (gemma4_unified) on the NGC 26.05 base.
    # Authored in t1; wired to vllm-multimodal's build: block in t2.
    "fleet/Dockerfile.vllm-gemma4": "Dockerfile.vllm-gemma4",
    LOG_WRAPPER: LOG_WRAPPER,
    CF_TUNNEL_EXAMPLE: CF_TUNNEL_EXAMPLE,
}
# The --audio extras layered on FLEET_TEMPLATES: the compose override, the three
# image build files, and the vendored Parakeet server. The audio .env keys are
# appended to .env separately (env.audio.example → AUDIO_ENV_TEMPLATE) so they
# extend the fleet .env instead of clobbering it.
AUDIO_TEMPLATES = {
    "fleet/docker-compose.audio.yml": AUDIO_OVERLAY,
    "fleet/Dockerfile.realtime": "Dockerfile.realtime",
    "fleet/Dockerfile.parakeet": "Dockerfile.parakeet",
    # The chatterbox TTS service in docker-compose.audio.yml builds from this
    # Dockerfile, so it MUST land at the deployment-dir root or `docker compose
    # build chatterbox` fails with "Dockerfile.chatterbox: no such file".
    "fleet/Dockerfile.chatterbox": "Dockerfile.chatterbox",
    "fleet/listen_server.py": "listen_server.py",
    # _readiness.py is COPY'd into the Parakeet image (Dockerfile.parakeet), so
    # it MUST land at the deployment-dir root or `docker compose build stt`
    # fails on the COPY. Vendored twin of lobes/realtime/_readiness.py.
    "fleet/_readiness.py": "_readiness.py",
}
AUDIO_ENV_TEMPLATE = "fleet/env.audio.example"
_INIT_REMEDIATION = (
    "run 'lobes init --apply' to scaffold ~/.lobes, or pass --compose-dir / set LOBES_DIR"
)
# Back-compat alias: the single set was the only one before the fleet existed.
_TEMPLATES = SINGLE_TEMPLATES

# Think-aware tool-parser plugin (t2, devague plan: fleet template + init wiring
# for the qwen3_coder_thinking tool-parser plugin). Materialised from the
# PACKAGED PYTHON MODULE lobes.vllm_plugins.qwen3_thinking_tool_parser — a
# DIFFERENT mechanism than SINGLE_TEMPLATES/FLEET_TEMPLATES/AUDIO_TEMPLATES
# above, deliberately: those all read from the `lobes.templates` resource tree
# (data files with no other job), while the plugin file is real Python package
# code (`lobes/vllm_plugins/`) — single source of truth, read fresh via
# importlib.resources rather than duplicated as a second copy under
# `lobes/templates/fleet/` (contrast the audio overlay's `_readiness.py`
# "vendored twin", which IS a manually-synced duplicate because that file must
# import cleanly with zero `lobes` package on the Parakeet container's
# PYTHONPATH — the tool-parser plugin has no such constraint: vLLM loads it by
# file PATH via --tool-parser-plugin, never by Python import name, so one copy
# read fresh at scaffold time is strictly simpler and can't drift).
# Fleet-topology only (mounted into vllm-primary/cortex only, see the fleet
# compose) — the legacy single-model scaffold never materialises this file.
PLUGIN_PACKAGE = "lobes.vllm_plugins"
PLUGIN_MODULE_FILE = "qwen3_thinking_tool_parser.py"
PLUGIN_DEST_NAME = "qwen3_thinking_tool_parser.py"


def plugin_source() -> str:
    """The packaged tool-parser plugin's current source text.

    Read fresh via ``importlib.resources`` on every call — never cached, never
    duplicated under ``lobes/templates/`` — so the deployment dir always gets
    whatever ships in ``lobes.vllm_plugins.qwen3_thinking_tool_parser`` for the
    installed ``lobes-cli`` version.
    """
    return files(PLUGIN_PACKAGE).joinpath(PLUGIN_MODULE_FILE).read_text(encoding="utf-8")


def plugin_plan(target: os.PathLike | str) -> tuple[str, bool]:
    """``(dest_name, already_exists)`` for the plugin file — same shape as one
    entry of :func:`scaffold_plan`, so ``lobes init``'s dry-run listing can
    just append it to that function's return value."""
    return (PLUGIN_DEST_NAME, (Path(target) / PLUGIN_DEST_NAME).exists())


def write_plugin_file(target: os.PathLike | str, *, force: bool) -> Path:
    """Write the tool-parser plugin file into the deployment dir. Returns the path.

    Mirrors :func:`write_scaffold`'s per-file exists/force contract: refuses to
    overwrite an existing file unless ``force`` is set. Fleet-only — the caller
    (``lobes init``) never invokes this for the legacy single-model scaffold.
    """
    dest_dir = Path(target).expanduser()
    dest = dest_dir / PLUGIN_DEST_NAME
    if dest.exists() and not force:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"{dest} already exists",
            remediation="re-run with --force to overwrite",
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(plugin_source(), encoding="utf-8")
    return dest


def default_deployment_dir() -> Path:
    """The global deployment home: ``~/.lobes``."""
    return Path.home() / ".lobes"


def durable_log_dir(deploy_dir: os.PathLike | str, configured: str | None = None) -> Path:
    """Host directory holding the per-boot vLLM logs that ``mg-logwrap`` writes.

    Mirrors the compose default ``${MODEL_GEAR_LOG_DIR:-./logs}``: an absolute
    ``configured`` path (``MODEL_GEAR_LOG_DIR`` from ``.env``) wins; a relative one
    resolves against the deployment dir; unset falls back to ``<deploy>/logs``. So
    the CLI reads logs from exactly where compose mounts them.
    """
    base = Path(deploy_dir).expanduser()
    if configured:
        p = Path(configured).expanduser()
        return p if p.is_absolute() else (base / p)
    return base / LOG_DIRNAME


def ensure_log_dir(deploy_dir: os.PathLike | str, configured: str | None = None) -> Path:
    """Create the durable-log dir before ``docker compose up``.

    The compose bind-mounts this host dir into each vLLM container; if it doesn't
    exist when compose runs, Docker creates it **root-owned**. Creating it here (as
    the invoking user) keeps the logs user-readable. Best-effort — returns the dir
    even if mkdir fails (the wrapper falls back to plain exec, never blocking serve).
    """
    d = durable_log_dir(deploy_dir, configured)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def resolve_deployment_dir(explicit: os.PathLike | str | None) -> Path:
    """Resolve the directory holding ``docker-compose.yml``.

    Resolution order:
    1. ``explicit`` (``--compose-dir``) — USER_ERROR if bad.
    2. ``$LOBES_DIR`` — USER_ERROR if bad.
    3. ``$MODEL_GEAR_DIR`` (back-compat fallback) — USER_ERROR if bad.
    4. ``~/.lobes`` if it has a ``docker-compose.yml``.
    5. ``~/.model-gear`` back-compat: if ``~/.lobes`` lacks compose but
       ``~/.model-gear`` has it, use that instead (live fleet upgrade path).
    6. Default ``~/.lobes`` — ENV_ERROR (not scaffolded yet).

    Raises :class:`ModelGearError` when the resolved directory has no
    ``docker-compose.yml`` (USER_ERROR for an explicit bad path, ENV_ERROR when
    the default home simply hasn't been scaffolded yet).
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        source, code = "--compose-dir", EXIT_USER_ERROR
        if not (candidate / COMPOSE_FILE).is_file():
            raise ModelGearError(
                code=code,
                message=f"no {COMPOSE_FILE} in {candidate} ({source})",
                remediation=_INIT_REMEDIATION,
            )
        return candidate
    if os.environ.get("LOBES_DIR"):
        candidate = Path(os.environ["LOBES_DIR"]).expanduser()
        source, code = "$LOBES_DIR", EXIT_USER_ERROR
        if not (candidate / COMPOSE_FILE).is_file():
            raise ModelGearError(
                code=code,
                message=f"no {COMPOSE_FILE} in {candidate} ({source})",
                remediation=_INIT_REMEDIATION,
            )
        return candidate
    if os.environ.get("MODEL_GEAR_DIR"):
        candidate = Path(os.environ["MODEL_GEAR_DIR"]).expanduser()
        source, code = "$MODEL_GEAR_DIR (legacy)", EXIT_USER_ERROR
        if not (candidate / COMPOSE_FILE).is_file():
            raise ModelGearError(
                code=code,
                message=f"no {COMPOSE_FILE} in {candidate} ({source})",
                remediation=_INIT_REMEDIATION,
            )
        return candidate
    # Prefer ~/.lobes; fall back to ~/.model-gear for live fleets that haven't
    # been migrated yet, then fall through to ENV_ERROR if neither is scaffolded.
    lobes_default = default_deployment_dir()  # normally ~/.lobes; monkeypatched in tests
    legacy_default = lobes_default.parent / ".model-gear"
    if (lobes_default / COMPOSE_FILE).is_file():
        return lobes_default
    if (legacy_default / COMPOSE_FILE).is_file():
        return legacy_default
    raise ModelGearError(
        code=EXIT_ENV_ERROR,
        message=f"no {COMPOSE_FILE} in {lobes_default} (default ~/.lobes)",
        remediation=_INIT_REMEDIATION,
    )


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
    template_root = files("lobes.templates")
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
    content = _read_template(files("lobes.templates"), AUDIO_ENV_TEMPLATE)
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write(content if content.startswith("\n") else "\n" + content)
    return env_path


# --- docker compose --------------------------------------------------------


def audio_overlay_present(deploy_dir: os.PathLike | str) -> bool:
    """True when the ``--audio`` overlay (``docker-compose.audio.yml``) is scaffolded."""
    return (Path(deploy_dir) / AUDIO_OVERLAY).is_file()


def shape_overlay_present(deploy_dir: os.PathLike | str) -> bool:
    """True when the deployment-shape override (``docker-compose.shape.yml``) is scaffolded.

    Present only for a mesh-brain shape that drops a core role (``spark-lobe`` /
    ``thor-lobe``); machine-as-brain / bare init never write it. Mirrors
    :func:`audio_overlay_present`.
    """
    return (Path(deploy_dir) / SHAPE_OVERLAY).is_file()


def _override_service_keys(text: str) -> set[str]:
    """The top-level ``services:`` keys declared in an override file's YAML text.

    A stdlib line scan (the runtime carries no YAML parser — see the module
    docstring): a service key is exactly two-space-indented, non-blank, ends in
    ``:``. The 4-space ``profiles:`` / ``depends_on:`` lines inside a block are
    deeper-indented (or lack the trailing colon) and so are skipped.
    """
    keys: set[str] = set()
    for line in text.splitlines():
        if line.startswith("  ") and line[2:3] not in ("", " "):
            stripped = line.strip()
            if stripped.endswith(":"):
                keys.add(stripped[:-1])
    return keys


def shape_dropped_containers(deploy_dir: os.PathLike | str) -> tuple[str, ...]:
    """Container names the shape override disables, read from ``docker-compose.shape.yml``.

    The override file itself is the single source of truth: ``lobes init`` GENERATES
    it listing exactly the core services it parks in the inert ``shape-dropped``
    compose profile (plus the ``gateway`` whose ``depends_on`` it resets — not a
    core gear, so never returned here). Empty tuple when no override is scaffolded.
    """
    path = Path(deploy_dir) / SHAPE_OVERLAY
    if not path.is_file():
        return ()
    keys = _override_service_keys(path.read_text(encoding="utf-8"))
    return tuple(
        container for service, container in _CORE_SERVICE_CONTAINER.items() if service in keys
    )


def is_fleet(deploy_dir: os.PathLike | str) -> bool:
    """True when the deploy dir is a fleet deployment (``Dockerfile.gateway`` present)."""
    return (Path(deploy_dir) / DOCKERFILE_GATEWAY).is_file()


def fleet_containers(deploy_dir: os.PathLike | str) -> tuple[str, ...]:
    """Fleet container names.

    Excludes any core gear a deployment-shape override drops (its service is parked
    in an inert profile, so ``docker compose up`` never starts it — see
    :func:`shape_dropped_containers`), and includes the audio trio when the audio
    overlay is present.
    """
    dropped = shape_dropped_containers(deploy_dir)
    containers = tuple(c for c in FLEET_CONTAINERS if c not in dropped)
    if audio_overlay_present(deploy_dir):
        return containers + FLEET_AUDIO_CONTAINERS
    return containers


def _compose_files(deploy_dir: os.PathLike | str) -> list[str]:
    """``-f`` args: the base file plus whichever overlays are present.

    Returns ``[]`` when NO overlay (``docker compose`` finds docker-compose.yml on
    its own), so a plain fleet keeps its current argv unchanged. The shape override
    is placed LAST — its compose ``!reset`` on the gateway ``depends_on`` clears the
    dangling edge to a profile-disabled dropped service, and applying it after the
    audio overlay (which never touches ``depends_on``, only ``environment``)
    guarantees no later file re-introduces that edge.
    """
    audio = audio_overlay_present(deploy_dir)
    shape = shape_overlay_present(deploy_dir)
    if not audio and not shape:
        return []
    files = ["-f", COMPOSE_FILE]
    if audio:
        files += ["-f", AUDIO_OVERLAY]
    if shape:
        files += ["-f", SHAPE_OVERLAY]
    return files


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


def compose_service_argv(action: str, compose_files: list[str], services: list[str]) -> list[str]:
    """Build the ``docker compose ...`` argv for a role-targeted up/stop (t7, #81).

    ``action`` is ``"up"`` (→ ``up -d``) or ``"stop"`` (→ ``stop``). ``compose_files``
    is the ``-f`` prefix — ``[]`` for the base file only, or ``["-f", COMPOSE_FILE,
    "-f", AUDIO_OVERLAY]`` when the target reaches into the audio overlay (stt/tts).
    ``services`` are the compose SERVICE names to target; only these are
    (re)started/stopped, so one role toggles without disturbing the rest of the
    fleet. ``lobes up`` renders its dry-run PLAN from this same argv it later runs
    under ``--apply``, so the two are byte-identical. Note ``stop`` (not ``down``):
    a project-wide ``docker compose down`` would remove EVERY container.
    """
    verb = ["up", "-d"] if action == "up" else ["stop"]
    return ["docker", "compose"] + list(compose_files) + verb + list(services)


def run_compose(deploy_dir: os.PathLike | str, argv: list[str]):
    """Run a prebuilt ``docker compose ...`` argv in the deployment dir (t7)."""
    return _run(argv, cwd=str(deploy_dir))


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
