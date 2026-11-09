"""Member step: the idempotent upgrade of ONE member to a target lobes version.

Order (each write only when it is still needed):

1. take the member's rollout lock (``_member_lock``);
2. read the member's state (installed CLI, ``.env`` pin, running gateway);
   a member already at the target on all three is a no-op that succeeds;
3. refuse while a benchmark window is active (machine CLI ``window show``,
   ``docs/contracts/machine-cli-window.md``) unless ``allow_window``;
4. back up ``.env`` (only while it still carries the pre-apply pin, so a
   resumed run never shadows the backup rollback restores);
5. ``uv tool install lobes-cli==<target>`` by absolute path, then confirm
   ``lobes --version`` really reports the target (index/CDN lag);
6. ``lobes doctor --repin-version --apply`` — judged by the resulting ``.env``
   value, NOT its exit code: doctor exits non-zero on the very version skew it
   is fixing (the 2026-11-09 trap);
7. ``docker tag`` the running gateway image as ``lobes-gateway:<old-version>``
   (the target ``_rollback`` consumes);
8. ``lobes up gateway --build --apply`` — compose builds before it recreates,
   so a failed build leaves the old gateway running; nothing else is touched;
9. wait for ``/health`` to report the target.

Everything remote goes through ``_transport``. Records and errors carry no
secret values (transport output is redacted; errors are redacted again).
"""

from __future__ import annotations

import json
import posixpath
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from lobes.rollout import _transport
from lobes.rollout._member_lock import LockHeld, member_lock
from lobes.rollout._redact import redact
from lobes.rollout._rollback import GATEWAY_CONTAINER, GATEWAY_IMAGE_REPO, rollback_command
from lobes.rollout._transport import Runner, as_target
from lobes.rollout._versions import parse_version

OK = "ok"
FAILED = "failed"
SKIPPED = "skipped"

NO_MACHINE_CLI_EXIT = 3
_NO_MACHINE_CLI = "NO_MACHINE_CLI"
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_VERSION_TOKEN = re.compile(r"(\d+(?:\.\d+)+[0-9A-Za-z.+-]*)")

READ_TIMEOUT = 30.0
INSTALL_TIMEOUT = 600.0
BUILD_TIMEOUT = 1800.0

# $1 = deployment dir ("" -> $HOME/.lobes); scripts take args, never interpolation.
_DIR = 'd="${1:-$HOME/.lobes}"; '
_READ_PIN = _DIR + "sed -n 's/^MODEL_GEAR_VERSION=//p' \"$d/.env\" | tail -n 1"
_HEALTH = (
    _DIR + 'p=$(sed -n \'s/^VLLM_PORT=//p\' "$d/.env" 2>/dev/null | tail -n 1 | tr -d "\\"\' "); '
    'u="http://127.0.0.1:${p:-8000}/health"; '
    'if command -v curl >/dev/null 2>&1; then curl -fsS --max-time 5 "$u"; '
    "else python3 -c 'import sys,urllib.request as r;"
    'print(r.urlopen(sys.argv[1],timeout=5).read().decode())\' "$u"; fi'
)
_BACKUP = _DIR + 'b="$d/.env.bak-$2-rollout"; ' 'cp -p "$d/.env" "$b" && printf \'%s\\n\' "$b"'
_WINDOW_PROBE = (
    "for c in spark thor orin; do "
    'for p in "$HOME/.local/bin/$c" "$(command -v "$c" 2>/dev/null)"; do '
    'if [ -n "$p" ] && [ -x "$p" ]; then exec "$p" window show --json; fi; '
    "done; done; echo " + _NO_MACHINE_CLI + "; exit " + str(NO_MACHINE_CLI_EXIT)
)

Journal = Callable[..., Any]


@dataclass(frozen=True)
class StepRecord:
    step: str
    result: str
    detail: dict = field(default_factory=dict)


@dataclass
class MemberState:
    installed: Optional[str] = None
    pinned: Optional[str] = None
    running: Optional[str] = None


@dataclass
class MemberStepResult:
    member: str
    target_version: str
    ok: bool = False
    changed: bool = False
    skipped: bool = False
    reason: str = ""
    error: str = ""
    failed_step: str = ""
    old_version: Optional[str] = None
    old_gateway_version: Optional[str] = None
    backup_path: Optional[str] = None
    before: MemberState = field(default_factory=MemberState)
    steps: list = field(default_factory=list)


class _StepFailed(Exception):
    def __init__(self, step: str, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.step = step
        self.detail = detail or {}


def same_version(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    try:
        return parse_version(a) == parse_version(b)
    except ValueError:
        return a.strip() == b.strip()


def _sh(script: str, *args: str) -> list:
    return ["sh", "-c", script, "sh", *args]


def _parse_version_output(text: str) -> Optional[str]:
    found = _VERSION_TOKEN.findall(text or "")
    return found[-1] if found else None


class _Step:
    """One run of the member step; holds the injected seams."""

    def __init__(
        self,
        member: str,
        resident: Any,
        target_version: str,
        *,
        deploy_dir: str,
        uv_path: str,
        machine_cli: Optional[str],
        allow_window: bool,
        journal: Optional[Journal],
        runner: Runner,
        sleep: Callable[[float], Any],
        health_attempts: int,
        health_interval: float,
        stamp: str,
        secrets: Sequence[str],
    ):
        self.member = member
        self.resident = resident
        self.target = target_version
        self.deploy_dir = deploy_dir
        self.uv_path = uv_path
        self.machine_cli = machine_cli
        self.allow_window = allow_window
        self.journal = journal
        self.runner = runner
        self.sleep = sleep
        self.health_attempts = max(1, health_attempts)
        self.health_interval = health_interval
        self.stamp = stamp
        self.secrets = tuple(secrets)
        self.result = MemberStepResult(member, target_version)

    # plumbing -----------------------------------------------------------
    def _clean(self, text: str) -> str:
        return redact(text or "", self.secrets)

    def record(self, step: str, result: str, current: Optional[str] = None, **detail: Any):
        detail = {k: (self._clean(v) if isinstance(v, str) else v) for k, v in detail.items()}
        self.result.steps.append(StepRecord(step, result, detail))
        if self.journal is not None:
            self.journal(
                self.member, step, result, target=self.target, current=current, detail=detail
            )

    def run(self, argv: Sequence[str], timeout: float = READ_TIMEOUT):
        return _transport.run(
            self.resident, argv, timeout=timeout, runner=self.runner, secrets=self.secrets
        )

    def run_lobes(self, args: Sequence[str], timeout: float = READ_TIMEOUT):
        if self.deploy_dir:
            args = [*args, "--compose-dir", self.deploy_dir]
        return _transport.run_lobes(
            self.resident, args, timeout=timeout, runner=self.runner, secrets=self.secrets
        )

    # reads --------------------------------------------------------------
    def installed_version(self) -> Optional[str]:
        # --version takes no --compose-dir, so bypass self.run_lobes.
        r = _transport.run_lobes(
            self.resident, ["--version"], runner=self.runner, secrets=self.secrets
        )
        return _parse_version_output(r.stdout) if r.ok else None

    def pinned_version(self) -> Optional[str]:
        r = self.run(_sh(_READ_PIN, self.deploy_dir))
        if not r.ok:
            return None
        value = r.stdout.strip().strip("\"'").strip()
        return value or None

    def health_version(self) -> Optional[str]:
        r = self.run(_sh(_HEALTH, self.deploy_dir))
        if not r.ok:
            return None
        try:
            data = json.loads(r.stdout)
        except ValueError:
            return None
        v = data.get("version") if isinstance(data, dict) else None
        return v if isinstance(v, str) and v else None

    def read_state(self) -> MemberState:
        return MemberState(self.installed_version(), self.pinned_version(), self.health_version())

    # steps --------------------------------------------------------------
    def window_check(self) -> Optional[str]:
        """None when clear to proceed, else the refusal reason."""
        if self.machine_cli:
            r = self.run([self.machine_cli, "window", "show", "--json"])
        else:
            r = self.run(_sh(_WINDOW_PROBE))
        if r.returncode == NO_MACHINE_CLI_EXIT and _NO_MACHINE_CLI in r.stdout:
            self.record("window_check", OK, active=False, note="no machine CLI on the member")
            return None
        try:
            data = json.loads(r.stdout)
            if not isinstance(data, dict):
                raise ValueError("not an object")
        except ValueError:
            data = None
        if data is None:
            reason = "benchmark window state unreadable (machine CLI window show)"
        elif data.get("active"):
            reason = "benchmark window active on the member (machine CLI window show)"
        else:
            self.record("window_check", OK, active=False)
            return None
        if self.allow_window:
            self.record("window_check", OK, active=data is not None, overridden=True, note=reason)
            return None
        self.record("window_check", SKIPPED, reason=reason)
        return reason

    def backup_env(self) -> None:
        r = self.run(_sh(_BACKUP, self.deploy_dir, self.stamp))
        path = r.stdout.strip().splitlines()[-1] if r.ok and r.stdout.strip() else ""
        if not r.ok or not path.startswith("/"):
            raise _StepFailed("backup_env", f".env backup failed (exit {r.returncode})")
        self.result.backup_path = path
        self.record("backup_env", OK, path=path)

    def install_cli(self) -> None:
        spec = f"lobes-cli=={self.target}"
        r = self.run([self.uv_path, "tool", "install", spec], timeout=INSTALL_TIMEOUT)
        installed = self.installed_version()
        if not same_version(installed, self.target):
            raise _StepFailed(
                "install_cli",
                f"{spec} did not land (install exit {r.returncode}, "
                f"lobes --version reports {installed or 'nothing'}): {r.stderr.strip()[-300:]}",
                {"install_exit": r.returncode, "installed": installed},
            )
        self.record("install_cli", OK, current=installed, install_exit=r.returncode)

    def repin(self) -> None:
        r = self.run_lobes(["doctor", "--repin-version", "--apply"])
        pinned = self.pinned_version()
        # Judge by the .env value: doctor exits non-zero on the skew it fixes.
        if not same_version(pinned, self.target):
            raise _StepFailed(
                "repin",
                f"MODEL_GEAR_VERSION is {pinned or 'unset'} after repin "
                f"(doctor exit {r.returncode}), expected {self.target}",
                {"doctor_exit": r.returncode, "pinned": pinned},
            )
        self.record("repin", OK, current=pinned, doctor_exit=r.returncode)

    def tag_image(self, old: Optional[str]) -> None:
        if not old or same_version(old, self.target):
            self.record("tag_image", SKIPPED, reason="old gateway version unknown or current")
            return
        if not _TAG_RE.match(old):
            raise _StepFailed("tag_image", f"old version {old!r} is not a valid image tag")
        r = self.run(["docker", "inspect", "--format", "{{.Image}}", GATEWAY_CONTAINER])
        image_id = r.stdout.strip()
        if not r.ok or not image_id:
            raise _StepFailed(
                "tag_image", f"could not read the running gateway image (exit {r.returncode})"
            )
        tag = f"{GATEWAY_IMAGE_REPO}:{old}"
        t = self.run(["docker", "tag", image_id, tag])
        if not t.ok:
            raise _StepFailed("tag_image", f"docker tag {tag} failed (exit {t.returncode})")
        self.record("tag_image", OK, tag=tag, image=image_id)

    def rebuild_gateway(self) -> None:
        r = self.run_lobes(["up", "gateway", "--build", "--apply"], timeout=BUILD_TIMEOUT)
        if r.ok:
            self.record("rebuild_gateway", OK)
            return
        # compose builds before it recreates: the old gateway should still serve.
        self.result.old_gateway_version = self.health_version()
        raise _StepFailed(
            "rebuild_gateway",
            f"gateway build/recreate failed (exit {r.returncode}); old gateway "
            f"reports {self.result.old_gateway_version or 'no health'}; "
            f"to roll back: {rollback_command(self.member)} :: {r.stderr.strip()[-300:]}",
            {"exit": r.returncode, "old_gateway_version": self.result.old_gateway_version},
        )

    def wait_health(self) -> None:
        seen = None
        for attempt in range(self.health_attempts):
            seen = self.health_version()
            if same_version(seen, self.target):
                self.record("wait_health", OK, current=seen, attempts=attempt + 1)
                return
            if attempt + 1 < self.health_attempts:
                self.sleep(self.health_interval)
        raise _StepFailed(
            "wait_health",
            f"/health did not report {self.target} after {self.health_attempts} tries "
            f"(last: {seen or 'unreachable'}); to roll back: {rollback_command(self.member)}",
            {"last": seen},
        )

    # driver -------------------------------------------------------------
    def execute(self) -> MemberStepResult:
        res = self.result
        try:
            with member_lock(self.resident, runner=self.runner):
                self.record("lock", OK)
                self._locked()
        except LockHeld as exc:
            self._fail("lock", str(exc))
        except _StepFailed as exc:
            self._fail(exc.step, str(exc), exc.detail)
        except RuntimeError as exc:  # lock could not be taken for another reason
            self._fail("lock", str(exc))
        return res

    def _fail(self, step: str, message: str, detail: Optional[dict] = None) -> None:
        res = self.result
        res.ok, res.failed_step, res.error = False, step, self._clean(message)
        self.record(step, FAILED, error=res.error, **(detail or {}))

    def _locked(self) -> None:
        res = self.result
        st = self.read_state()
        res.before = st
        res.old_version = st.running or st.pinned
        need_install = not same_version(st.installed, self.target)
        need_repin = not same_version(st.pinned, self.target)
        need_rebuild = not same_version(st.running, self.target)
        if not (need_install or need_repin or need_rebuild):
            res.ok, res.changed, res.reason = True, False, "already at target"
            return
        refusal = self.window_check()
        if refusal:
            res.skipped, res.reason = True, refusal
            return
        res.changed = True
        if need_repin:
            self.backup_env()
        else:
            self.record("backup_env", OK, note=".env already pinned; pre-apply backup kept")
        if need_install:
            self.install_cli()
        else:
            self.record("install_cli", OK, current=st.installed, note="already installed")
        if need_repin:
            self.repin()
        else:
            self.record("repin", OK, current=st.pinned, note="already pinned")
        if need_rebuild:
            self.tag_image(st.running)
            self.rebuild_gateway()
        else:
            self.record("tag_image", SKIPPED, reason="gateway already at target")
            self.record("rebuild_gateway", OK, note="gateway already at target")
        self.wait_health()
        res.ok = True


def run_member_step(
    member: str,
    resident: Any,
    target_version: str,
    *,
    deploy_dir: str = "",
    uv_path: Optional[str] = None,
    machine_cli: Optional[str] = None,
    allow_window: bool = False,
    journal: Optional[Journal] = None,
    runner: Runner = _transport.default_runner,
    sleep: Callable[[float], Any] = time.sleep,
    health_attempts: int = 60,
    health_interval: float = 5.0,
    stamp: Optional[str] = None,
    secrets: Sequence[str] = (),
) -> MemberStepResult:
    """Upgrade one member to ``target_version``; safe to re-run.

    ``journal`` is called as ``journal(member, step, result, target=,
    current=, detail=)`` — :meth:`lobes.rollout._journal.Journal.append`.
    """
    parse_version(target_version)  # raises ValueError on anything odd
    if not _TAG_RE.match(target_version):
        raise ValueError(f"unsupported target version: {target_version!r}")
    t = as_target(resident)
    if not t.lobes_path.startswith("/"):
        raise ValueError("lobes_path must be absolute")
    uv = uv_path or posixpath.join(posixpath.dirname(t.lobes_path), "uv")
    if not uv.startswith("/"):
        raise ValueError("uv_path must be absolute")
    if deploy_dir and not deploy_dir.startswith("/"):
        raise ValueError("deploy_dir must be absolute")
    return _Step(
        member,
        resident,
        target_version,
        deploy_dir=deploy_dir,
        uv_path=uv,
        machine_cli=machine_cli,
        allow_window=allow_window,
        journal=journal,
        runner=runner,
        sleep=sleep,
        health_attempts=health_attempts,
        health_interval=health_interval,
        stamp=stamp or time.strftime("%Y%m%d-%H%M%S"),
        secrets=secrets,
    ).execute()
