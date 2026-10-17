"""Graded thermal governor for a heavy host-side container (lane-safety adjunct).

Packaged from ``~/odm/telemetry/governor2.sh``: the CPU quota of one container
follows the hottest thermal zone, and the container is paused only as the last
resort. Default bands (hottest zone, deg C -> ``docker update --cpus``):

    <84 -> 16 | <88 -> 12 | <91 -> 8 | <93.5 -> 4 | >=93.5 -> pause

A paused container resumes below 86 C at 4 cpus (the hysteresis). No root: it
only drives ``docker update --cpus`` / ``pause`` / ``unpause`` on a container the
invoking user can already reach.

History (2026-10-14): the bash governor2 was retired after ``docker update
--cpus`` was blamed for two ``cudaErrorNotPermitted`` exits in a quantization
job; the blame was later marked unproven (lobes#304). t3 of lobes#310 measured
it on a RUNNING vLLM 0.31.0 container (432/432 requests OK, 0 restarts, 0 CUDA
errors: ``docs/evidence/2026-10-17-verify-cpucap-running-vllm-spark.txt``), which
is the only evidence this mechanism rests on.

Own-pause tracking: the old pause-only governor resumed ANY paused container
below its low threshold, including a pause something else made. This one
resumes (and unpauses on exit) ONLY a pause it issued itself; a container found
paused by someone else is left alone and noted once.

The decision is the pure function :func:`decide`; :class:`Governor` is the thin
stateful shell with injectable docker / temperature / log, so nothing here
needs a real container or real thermal zones to test.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from lobes.runtime import _pressure

DEFAULT_POLL_S = 3.0
LOG_NAME = "governor.log"
LOG_ENV = "LOBES_GOVERNOR_LOG"


@dataclass(frozen=True)
class Bands:
    """``steps`` is ascending ``(limit_c, cpus)``: below ``limit_c`` the cap is the
    PREVIOUS entry's cpus (``top_cpus`` for the first). So the default is
    ``<84 -> 16``, ``<88 -> 12``, ``<91 -> 8``, ``<93.5 -> 4``."""

    steps: tuple[tuple[float, int], ...] = ((84.0, 16), (88.0, 12), (91.0, 8), (93.5, 4))
    pause_c: float = 93.5
    resume_c: float = 86.0
    resume_cpus: int = 4

    @property
    def top_cpus(self) -> int:
        return self.steps[0][1]


@dataclass(frozen=True)
class Decision:
    kind: str  # "cpus" | "pause" | "resume" | "hold"
    cpus: int | None = None


def decide(temp_c: float, *, paused: bool, bands: Bands) -> Decision:
    """Pure band selection. ``paused`` means paused BY THE GOVERNOR."""
    if paused:
        if temp_c < bands.resume_c:
            return Decision("resume", bands.resume_cpus)
        return Decision("hold")
    if temp_c >= bands.pause_c:
        return Decision("pause")
    for limit, cpus in bands.steps:
        if temp_c < limit:
            return Decision("cpus", cpus)
    return Decision("cpus", bands.steps[-1][1])


def parse_bands(
    spec: str, *, pause_c: float, resume_c: float, resume_cpus: int, top_cpus: int
) -> Bands:
    """``"84:12,88:8,91:4"`` -> a cap of ``top_cpus`` below 84, 12 below 88, ...
    where the last entry's cpus applies up to ``pause_c``. Limits must ascend."""
    pairs: list[tuple[float, int]] = []
    try:
        for part in spec.split(","):
            lim, cpus = part.strip().split(":")
            pairs.append((float(lim), int(cpus)))
    except ValueError as exc:
        raise ValueError(f"bad band spec {spec!r}: expected 'limit_c:cpus,...'") from exc
    limits = [p[0] for p in pairs]
    if not pairs or limits != sorted(set(limits)) or limits[-1] >= pause_c:
        raise ValueError(f"band limits must ascend and stay below the pause threshold: {spec!r}")
    # the cap below limit[i] is the cpus of the band that ENDS at limit[i]
    caps = [top_cpus] + [c for _, c in pairs]
    steps = tuple((lim, caps[i]) for i, (lim, _) in enumerate(pairs)) + ((pause_c, caps[-1]),)
    return Bands(steps=steps, pause_c=pause_c, resume_c=resume_c, resume_cpus=resume_cpus)


DockerRun = Callable[[list[str]], "tuple[int, str]"]


def run_docker(argv: list[str]) -> tuple[int, str]:  # pragma: no cover - thin shell
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout


def read_hottest_c(env: "dict[str, str] | None" = None) -> float | None:
    """Hottest plausible zone. Like governor2 it considers EVERY zone unless
    ``LOBES_THERMAL_ZONES`` narrows it; reuses ``read_thermal_c`` (so
    ``LOBES_SYS_ROOT`` and the plausibility window apply)."""
    environ = dict(os.environ if env is None else env)
    if not environ.get("LOBES_THERMAL_ZONES", "").strip():
        root = Path(environ.get("LOBES_SYS_ROOT") or "/sys") / "class" / "thermal"
        types: set[str] = set()
        try:
            for zone in root.glob("thermal_zone*"):
                try:
                    types.add((zone / "type").read_text().strip())
                except OSError:
                    continue
        except OSError:
            return None
        environ["LOBES_THERMAL_ZONES"] = ",".join(sorted(t for t in types if t))
    return _pressure.read_thermal_c(environ)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Governor:
    def __init__(
        self,
        container: str,
        bands: Bands,
        *,
        docker: DockerRun,
        read_temp: Callable[[], "float | None"],
        log: Callable[[str], None],
        apply: bool,
        now: Callable[[], str] = utc_now,
    ) -> None:
        self.container = container
        self.bands = bands
        self._docker = docker
        self._read_temp = read_temp
        self._log_sink = log
        self.apply = apply
        self._now = now
        self.cur: int | None = None  # last cap we set
        self.we_paused = False  # a pause WE issued (the only one we may undo)
        self._foreign_noted = False
        self.last_temp: float | None = None
        self.last_action: str | None = None

    # -- helpers
    def _log(self, msg: str, temp: "float | None" = None) -> None:
        tz = f" tz={temp:.1f}C" if temp is not None else ""
        prefix = "" if self.apply else "dry-run would "
        self._log_sink(f"{self._now()} {prefix}{msg}{tz}")

    def _do(self, *args: str) -> bool:
        if not self.apply:
            return True
        rc, _ = self._docker(["docker", *args])
        return rc == 0

    def _status(self) -> str:
        rc, out = self._docker(["docker", "inspect", "-f", "{{.State.Status}}", self.container])
        return out.strip() if rc == 0 else "missing"

    def _set_cpus(self, cpus: int, temp: "float | None", label: str | None = None) -> None:
        if self._do("update", "--cpus", str(cpus), self.container):
            self.cur = cpus
            self.last_action = f"cpus={cpus}"
            self._log(label or f"cpus={cpus}", temp)
        else:
            self._log(f"FAILED docker update --cpus {cpus}", temp)

    # -- one poll; False means the container is gone and the loop should end
    def tick(self) -> bool:
        status = self._status()
        if status in ("exited", "dead", "missing"):
            return False
        # dry-run never really pauses, so simulate our own pause
        paused = status == "paused" or (not self.apply and self.we_paused)
        if self.we_paused and not paused:
            self.we_paused = False  # someone else unpaused it; re-assert the cap
            self.cur = None
        if paused and not self.we_paused:
            if not self._foreign_noted:
                self._foreign_noted = True
                self._log("foreign pause (not ours): leaving container alone")
            return True
        self._foreign_noted = False
        temp = self._read_temp()
        self.last_temp = temp
        if temp is None:
            return True
        d = decide(temp, paused=self.we_paused, bands=self.bands)
        if d.kind == "pause":
            if self._do("pause", self.container):
                self.we_paused = True
                self.last_action = "pause"
                self._log("pause", temp)
            else:
                self._log("FAILED docker pause", temp)
        elif d.kind == "resume":
            # cap first so the thawed container does not resume at full speed
            if self._do("update", "--cpus", str(d.cpus), self.container) and self._do(
                "unpause", self.container
            ):
                self.we_paused = False
                self.cur = d.cpus
                self.last_action = f"resume cpus={d.cpus}"
                self._log(f"resume cpus={d.cpus}", temp)
            else:
                self._log("FAILED docker resume", temp)
        elif d.kind == "cpus" and d.cpus != self.cur:
            self._set_cpus(d.cpus, temp)
        return True

    def cleanup(self) -> None:
        """On exit: top-band cap back, and unpause only a pause we made."""
        top = self.bands.top_cpus
        if self.cur is not None and self.cur != top:
            if self._do("update", "--cpus", str(top), self.container):
                self.cur = top
                self._log(f"exit: restored cpus={top}")
            else:
                self._log(f"exit: FAILED to restore cpus={top}")
        if self.we_paused:
            if self._do("unpause", self.container):
                self.we_paused = False
                self._log("exit: unpaused (our own pause)")
            else:
                self._log("exit: FAILED to unpause our own pause")


def file_log(path: Path) -> Callable[[str], None]:
    def write(line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    return write


def run_forever(gov: Governor, *, poll_s: float = DEFAULT_POLL_S) -> None:  # pragma: no cover
    """Poll until the container exits or SIGINT/SIGTERM; always run cleanup."""
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        while not stop.is_set() and gov.tick():
            stop.wait(poll_s)
    finally:
        gov.cleanup()
