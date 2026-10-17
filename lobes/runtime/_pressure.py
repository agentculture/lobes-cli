"""Read-only host memory-pressure sampler (``/proc``-based, stdlib only).

On a DGX Spark the CPU and GPU share **unified memory**, so classic GPU-VRAM
metrics do not capture full-system pressure.  Two ``/proc`` signals together
form a useful thrash indicator:

* **swap_used_percent** — how much of the swap device is consumed.  Computed
  from ``/proc/meminfo`` as ``(SwapTotal - SwapFree) / SwapTotal * 100``.
  Returns ``0.0`` when the system has no swap device (``SwapTotal == 0``).

* **iowait_percent** — how much of the CPU's time was spent waiting for I/O
  over the last ~150 ms interval.  Cumulative counters from ``/proc/stat``
  become meaningful only as a **delta**: take two snapshots 150 ms apart and
  compute ``iowait_delta / total_cpu_time_delta * 100``.  Returns ``0.0``
  when the two snapshots are identical (no CPU activity in the window).

Design rule: the module is **side-effect-free** — it only reads ``/proc``.  No
writes, no subprocesses that mutate state, no container lifecycle calls.

Pure parser functions are split out from the top-level sampler so that tests
can exercise the arithmetic with fixture strings (no live ``/proc`` or timing).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure parsers (accept file text as input — testable without /proc)
# ---------------------------------------------------------------------------


def parse_swap_percent(meminfo_text: str) -> float:
    """Return swap-used percentage parsed from ``/proc/meminfo`` text.

    Formula: ``(SwapTotal - SwapFree) / SwapTotal * 100``.

    Returns ``0.0`` when ``SwapTotal`` is zero (no swap device) to avoid a
    ``ZeroDivisionError``.

    Args:
        meminfo_text: Raw contents of ``/proc/meminfo``.

    Returns:
        Swap-used percentage in the range ``[0.0, 100.0]``.
    """
    total_kb: float = 0.0
    free_kb: float = 0.0

    for line in meminfo_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "SwapTotal:":
            total_kb = float(parts[1])
        elif key == "SwapFree:":
            free_kb = float(parts[1])

    if total_kb <= 0.0:
        return 0.0
    return (total_kb - free_kb) / total_kb * 100.0


def parse_iowait_percent(stat_text_before: str, stat_text_after: str) -> float:
    """Return iowait percentage from two ``/proc/stat`` snapshots.

    The ``cpu`` aggregate line in ``/proc/stat`` holds cumulative counters
    since boot; a single reading is not a meaningful percentage.  This
    function computes a **delta** across two snapshots and returns:

        ``iowait_delta / total_cpu_time_delta * 100``

    Returns ``0.0`` when ``total_delta`` is zero (snapshots taken too close
    together or no CPU activity) to avoid a ``ZeroDivisionError``.

    ``/proc/stat`` cpu line field order (all cumulative jiffies):
        user nice system idle iowait irq softirq steal guest guest_nice

    Args:
        stat_text_before: Raw contents of ``/proc/stat`` at time T₀.
        stat_text_after:  Raw contents of ``/proc/stat`` at time T₁.

    Returns:
        iowait percentage over the inter-snapshot interval.
    """

    def _cpu_fields(text: str) -> list[int]:
        for line in text.splitlines():
            if line.startswith("cpu "):
                # "cpu  100 20 30 800 50 0 0 0 0 0"
                parts = line.split()
                return [int(v) for v in parts[1:]]
        return []

    before = _cpu_fields(stat_text_before)
    after = _cpu_fields(stat_text_after)

    if not before or not after:
        return 0.0

    # Pad shorter list to the length of the longer one (kernel may emit
    # fewer optional fields on older configs).
    length = max(len(before), len(after))
    before += [0] * (length - len(before))
    after += [0] * (length - len(after))

    total_before = sum(before)
    total_after = sum(after)
    total_delta = total_after - total_before

    if total_delta == 0:
        return 0.0

    # iowait is the 5th field (index 4, 0-based after the "cpu" label is stripped)
    iowait_before = before[4] if len(before) > 4 else 0
    iowait_after = after[4] if len(after) > 4 else 0
    iowait_delta = iowait_after - iowait_before

    return iowait_delta / total_delta * 100.0


# ---------------------------------------------------------------------------
# Top-level sampler (reads /proc, thin wrapper around the pure parsers)
# ---------------------------------------------------------------------------

_PROC_MEMINFO = "/proc/meminfo"
_PROC_STAT = "/proc/stat"
_SAMPLE_INTERVAL_SECONDS = 0.150  # 150 ms between /proc/stat snapshots


_THERMAL_MIN_C = -40.0
_THERMAL_MAX_C = 150.0


def read_thermal_c(env: "os._Environ[str] | dict[str, str] | None" = None) -> float | None:
    """Return the hottest plausible temperature (deg C) of the selected zones.

    Zones live under ``${LOBES_SYS_ROOT:-/sys}/class/thermal/thermal_zone*``; only
    those whose ``type`` is listed in ``LOBES_THERMAL_ZONES`` (comma-separated,
    whitespace-trimmed) count, and the hottest wins. A zone whose ``temp`` read
    fails (e.g. ENODATA on the Orin cv0-2 zones) or lands outside -40..150 deg C is
    skipped. Nothing matching (or the selector unset/empty) returns ``None`` --
    unknown, never a reason to shed.

    Container access: probed 2026-10-17, model-gear-gateway (uid 10001) on spark
    and orin reads these zones with no extra mount.
    """
    environ = os.environ if env is None else env
    wanted = {z.strip() for z in environ.get("LOBES_THERMAL_ZONES", "").split(",") if z.strip()}
    if not wanted:
        return None
    root = Path(environ.get("LOBES_SYS_ROOT") or "/sys") / "class" / "thermal"
    hottest: float | None = None
    try:
        zones = sorted(root.glob("thermal_zone*"))
    except OSError:
        return None
    for zone in zones:
        try:
            if (zone / "type").read_text().strip() not in wanted:
                continue
            temp_c = int((zone / "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        if not _THERMAL_MIN_C <= temp_c <= _THERMAL_MAX_C:
            continue
        if hottest is None or temp_c > hottest:
            hottest = temp_c
    return hottest


def sample_pressure(
    env: "os._Environ[str] | dict[str, str] | None" = None,
) -> dict[str, float | None]:
    """Sample host memory pressure and return a snapshot dict.

    Reads ``/proc/meminfo`` once for swap metrics and ``/proc/stat`` twice
    (150 ms apart) to compute a meaningful iowait percentage.

    Returns:
        A dict with two float keys plus ``thermal_c`` (float or ``None``):
        - ``"swap_used_percent"``: fraction of swap device in use (0–100).
        - ``"iowait_percent"``: CPU iowait % over the last ~150 ms interval (0–100).
        - ``"thermal_c"``: hottest selected zone in deg C, ``None`` if unknown.

    ``env`` is passed to :func:`read_thermal_c` (default: this process's
    environment); ``lobes status --pressure`` passes the deployment ``.env``
    merged under the process env.

    Raises:
        OSError: if ``/proc/meminfo`` or ``/proc/stat`` cannot be read.
    """
    with open(_PROC_MEMINFO) as fh:
        meminfo_text = fh.read()

    with open(_PROC_STAT) as fh:
        stat_before = fh.read()

    time.sleep(_SAMPLE_INTERVAL_SECONDS)

    with open(_PROC_STAT) as fh:
        stat_after = fh.read()

    return {
        "swap_used_percent": parse_swap_percent(meminfo_text),
        "iowait_percent": parse_iowait_percent(stat_before, stat_after),
        "thermal_c": read_thermal_c(env=env),
    }
