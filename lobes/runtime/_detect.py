"""Host card detection — gather host facts, resolve them via the chip registry.

This module is the **fact-gatherer**, not a second card table. The single source
of truth for "which card has which signature" is :mod:`lobes.machines` (a
:class:`~lobes.machines.CardStrategy` per card, each carrying a
:class:`~lobes.machines.DetectionSignature`); this module's job is narrower —
collect the host's raw facts (GPU device name, compute capability, total system
memory, hostname, and — on Jetson boards — the device-tree model string) from the
live host, then hand the name-ish facts to :func:`lobes.machines.detect` and
report whatever it resolves, honestly, including ``UNKNOWN`` when nothing
matches. It never guesses a "closest" card and never falls back to a default
card — ``UNKNOWN`` is a first-class result a caller renders as a warning, not an
error swallowed here.

Two hardware quirks shape the fact-gathering:

* **Never read nvidia-smi's memory fields.** On Thor's integrated GPU,
  ``nvidia-smi --query-gpu=memory.used,memory.total`` reports ``[N/A]`` for
  both — parsing that breaks the probe for no reason, since these are
  unified-memory boards where "GPU memory" is not a distinct pool anyway. The
  nvidia-smi probe here only ever queries ``name,compute_cap``, and
  :func:`_nvidia_smi_argv` is a small, directly testable builder so a test can
  assert the query string never grows a memory field.
* **Total memory comes from ``/proc/meminfo`` instead.** ``MemTotal`` is the
  honest, always-present system total on these unified-memory boards.

Every probe (nvidia-smi, ``/proc/meminfo``, ``/proc/device-tree/model``,
``socket.gethostname``) is exposed as an injectable function on
:func:`detect_card` and degrades independently and gracefully: a missing
binary, a timeout, a missing file, or unparsable output never raises out of the
public function — the corresponding fact is simply ``None`` and resolution
falls through toward ``UNKNOWN``. Stdlib only — no ``torch`` import, matching
the rest of :mod:`lobes.runtime`.
"""

from __future__ import annotations

import re
import socket
import subprocess  # fixed argv lists only, never shell=True (see pyproject bandit skips)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from lobes import machines

# The honest "no match" result. Never silently substituted for a real card name.
UNKNOWN = "unknown"

_NVIDIA_SMI_TIMEOUT_S = 5.0
# NEVER add memory.used / memory.total here — see module docstring. name and
# compute_cap are the only fields this probe ever asks nvidia-smi for.
_NVIDIA_SMI_FIELDS = "name,compute_cap"

_MEMINFO_PATH = Path("/proc/meminfo")
_DEVICE_TREE_MODEL_PATH = Path("/proc/device-tree/model")

_COMPUTE_CAP_RE = re.compile(r"^(\d+)\.(\d+)$")


@dataclass(frozen=True)
class DetectedCard:
    """The raw host facts gathered, plus the card name (or :data:`UNKNOWN`) they resolve to.

    The raw facts are always populated (or ``None`` on a failed probe) even when
    ``resolved`` is :data:`UNKNOWN` — a later task renders them in a warning so an
    unrecognized host is diagnosable, not just silently unsupported.
    """

    resolved: str
    device_name: str | None
    compute_capability: str | None
    total_memory_gb: float | None
    hostname: str | None
    device_tree_model: str | None
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def is_known(self) -> bool:
        return self.resolved != UNKNOWN


def _nvidia_smi_argv() -> list[str]:
    """The fixed nvidia-smi argv — name + compute_cap only, no memory fields.

    Split out so a test can assert on the built query string without spawning a
    subprocess (and without hardware in CI).
    """
    return ["nvidia-smi", f"--query-gpu={_NVIDIA_SMI_FIELDS}", "--format=csv,noheader"]


def _run_nvidia_smi(timeout: float = _NVIDIA_SMI_TIMEOUT_S) -> str | None:
    """Run the fixed nvidia-smi probe; the raw first CSV line, or ``None`` on any failure.

    Every failure mode — missing binary, timeout, non-zero exit, empty output —
    collapses to ``None`` uniformly; a caller only ever needs "got a line" vs
    "didn't", never why.
    """
    try:
        result = subprocess.run(  # fixed argv, no shell
            _nvidia_smi_argv(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _compute_cap_to_sm(cap_raw: str) -> str | None:
    """``"11.0"`` -> ``"sm_110"``, ``"12.1"`` -> ``"sm_121"``; ``None`` if unparsable."""
    match = _COMPUTE_CAP_RE.match(cap_raw.strip())
    if not match:
        return None
    major, minor = match.groups()
    return f"sm_{major}{minor}"


def _parse_nvidia_smi_line(line: str | None) -> tuple[str | None, str | None]:
    """``"NVIDIA Thor, 11.0"`` -> ``("NVIDIA Thor", "sm_110")``; degrades to ``(None, None)``."""
    if not line:
        return None, None
    parts = [p.strip() for p in line.split(",")]
    name = parts[0] or None
    compute_capability = _compute_cap_to_sm(parts[1]) if len(parts) > 1 else None
    return name, compute_capability


def _read_total_memory_gb(path: Path = _MEMINFO_PATH) -> float | None:
    """Total system memory in GB from ``/proc/meminfo`` ``MemTotal`` — never nvidia-smi.

    Unified-memory boards report ``[N/A]`` for GPU memory.used/memory.total, so
    ``/proc/meminfo`` is the only honest total on this hardware class.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "MemTotal:" and len(parts) >= 2:
            try:
                total_kb = float(parts[1])
            except ValueError:
                return None
            return round(total_kb / (1024 * 1024), 1)
    return None


def _read_device_tree_model(path: Path = _DEVICE_TREE_MODEL_PATH) -> str | None:
    """``/proc/device-tree/model`` (Jetson boards only; absent elsewhere, NUL-terminated)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    model = text.strip("\x00").strip()
    return model or None


def _read_hostname() -> str | None:
    try:
        return socket.gethostname()
    except OSError:
        return None


def detect_card(
    *,
    nvidia_smi_fn: Callable[[], str | None] = _run_nvidia_smi,
    meminfo_fn: Callable[[], float | None] = _read_total_memory_gb,
    device_tree_fn: Callable[[], str | None] = _read_device_tree_model,
    hostname_fn: Callable[[], str | None] = _read_hostname,
) -> DetectedCard:
    """Gather host facts and resolve them to a registered card name, or :data:`UNKNOWN`.

    Every fact-gathering function is injectable (defaults probe the real host) so
    tests never touch real hardware. Resolution rides
    :func:`lobes.machines.detect` — the single source of truth for card
    signatures — first against the nvidia-smi device name, falling back to the
    device-tree model string (the only name-ish signal on a headless Jetson board
    where nvidia-smi is unavailable). Never raises: every probe failure degrades
    to a ``None`` fact and the resolution degrades toward :data:`UNKNOWN`.
    """
    sources: dict[str, str] = {}

    try:
        smi_line = nvidia_smi_fn()
    except Exception:  # a probe must never raise out of detect_card
        smi_line = None
    device_name, compute_capability = _parse_nvidia_smi_line(smi_line)
    sources["device_name"] = "nvidia-smi" if device_name else "unavailable"
    sources["compute_capability"] = "nvidia-smi" if compute_capability else "unavailable"

    try:
        total_memory_gb = meminfo_fn()
    except Exception:
        total_memory_gb = None
    sources["total_memory_gb"] = "/proc/meminfo" if total_memory_gb is not None else "unavailable"

    try:
        device_tree_model = device_tree_fn()
    except Exception:
        device_tree_model = None
    sources["device_tree_model"] = "/proc/device-tree/model" if device_tree_model else "unavailable"

    try:
        hostname = hostname_fn()
    except Exception:
        hostname = None
    sources["hostname"] = "socket.gethostname" if hostname else "unavailable"

    strategy = machines.detect(device_name, hostname)
    if strategy is None and device_tree_model:
        strategy = machines.detect(device_tree_model, hostname)
    resolved = strategy.name if strategy is not None else UNKNOWN

    return DetectedCard(
        resolved=resolved,
        device_name=device_name,
        compute_capability=compute_capability,
        total_memory_gb=total_memory_gb,
        hostname=hostname,
        device_tree_model=device_tree_model,
        sources=sources,
    )
