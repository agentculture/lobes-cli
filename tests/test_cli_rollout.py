"""``lobes rollout plan|apply|status|rollback`` (gradual-fleet-rollout t13).

Everything remote is faked: one ``FakeMesh`` stands in for every member box,
answering both the SSH runner (``lobes.rollout._transport`` argv) and the
HTTP opener (``/health``, ``/capabilities``, ``/mesh/roster``, probe POSTs),
from one shared per-box state, so a gateway rebuild over "ssh" is what the
next "HTTP" ``/health`` reports. No real host is ever contacted.
"""

from __future__ import annotations

import io
import json
import shlex
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from lobes.cli import main
from lobes.cli._commands import rollout as rollout_cmd

OLD, NEW = "0.94.0", "0.95.0"
MESH_KEY = "mesh-secret-value-123"
OP_KEY = "operator-gateway-key-456"
KEYS = {
    "cultureos": "sk-cultureos-key-aaa",
    "spark": "sk-spark-key-bbb",
    "thor": "sk-thor-key-ccc",
}
SECRETS = [MESH_KEY, OP_KEY, *KEYS.values()]

TEMPLATE = """\
services:
  gateway:
    environment:
      - GATEWAY_PORT=8000
      - STT_EN_BASE_URL=${STT_EN_BASE_URL:-}
      - STT_EN_FEASIBLE=${STT_EN_FEASIBLE:-}
"""
DRIFTED = """\
services:
  gateway:
    environment:
      - GATEWAY_PORT=8000
"""
OVERRIDE_WITH_SECRET = f"""\
services:
  gateway:
    environment:
      - GATEWAY_API_KEY={KEYS['spark']}
"""


def _caps(*roles, proxied=False, ready=True):
    return {r: {"feasible": True, "ready": ready, "proxied": proxied} for r in roles}


class Box:
    def __init__(
        self,
        name,
        host,
        user,
        port,
        roles=(),
        proxied_roles=(),
        version=OLD,
        reachable=True,
        key=None,
        compose=TEMPLATE,
        override=None,
        lock_file=False,
        local=False,
    ):
        self.name, self.host, self.user, self.port = name, host, user, port
        self.installed = self.pinned = self.running = version
        self.caps = {**_caps(*roles), **_caps(*proxied_roles, proxied=True)}
        self.reachable = reachable
        self.key = key
        self.files = {"docker-compose.yml": compose}
        if override is not None:
            self.files["docker-compose.override.yml"] = override
        self.lock_file = lock_file
        self.local = local
        self.lock_held = False
        self.fail_build = False
        self.regress_after = None  # role that goes not-ready once running NEW
        self.window = None
        self.interrupt_on_probe = False
        self.calls = []
        self.tags = {}

    @property
    def origin(self):
        return f"http://{self.host}.mesh:{self.port}"


class FakeMesh:
    WRITE_KINDS = {
        "lock_acquire",
        "backup",
        "install",
        "repin",
        "tag",
        "build",
        "override_write",
        "restore",
        "recreate",
    }

    def __init__(self, boxes):
        self.boxes = {b.name: b for b in boxes}
        self.http = []

    # -- SSH runner ------------------------------------------------------
    def _box_for_argv(self, argv):
        if argv[0] == "ssh":
            dest = argv[5]
            host = dest.split("@", 1)[-1]
            for b in self.boxes.values():
                if b.host == host:
                    return b, argv[6]
            raise AssertionError(f"unknown ssh dest {dest}")
        for b in self.boxes.values():
            if b.local:
                return b, " ".join(shlex.quote(a) for a in argv)
        raise AssertionError(f"local command with no local box: {argv}")

    @staticmethod
    def kind(cmd):
        if "rollout.lock" in cmd:
            return "lock_release" if cmd.startswith("sh -c 'rm") else "lock_acquire"
        if "@@LOBES-FILE" in cmd:
            return "fetch_compose"
        if "docker-compose.override.yml" in cmd:
            return "override_write"
        if "window show" in cmd:
            return "window"
        if "tool install" in cmd:
            return "install"
        if "doctor --repin-version" in cmd:
            return "repin"
        if "up gateway --build" in cmd:
            return "build"
        if "up gateway --apply" in cmd:
            return "recreate"
        if "docker tag" in cmd:
            return "tag"
        if "docker inspect" in cmd:
            return "inspect"
        if "/health" in cmd:
            return "health"
        if cmd.startswith("cp -p "):
            return "restore"
        if ".env.bak" in cmd:
            return "backup"
        if "GATEWAY_API_KEY" in cmd:
            return "read_key"
        if "MODEL_GEAR_VERSION" in cmd:
            return "read_pin"
        if "--version" in cmd:
            return "version"
        return "other"

    def runner(self, argv, timeout):
        box, cmd = self._box_for_argv(list(argv))
        k = self.kind(cmd)
        box.calls.append((k, cmd))
        if not box.reachable:
            return 255, "", "ssh: connect to host: No route to host"
        if k == "lock_acquire":
            if box.lock_held:
                return 17, "HELD\nhost=other pid=9 started=x\n", ""
            box.lock_held = True
            return 0, "", ""
        if k == "lock_release":
            box.lock_held = False
            return 0, "", ""
        if k == "fetch_compose":
            out = ""
            for n, text in box.files.items():
                out += f"\n@@LOBES-FILE {n}\n{text}"
            if box.lock_file:
                out += "\n@@LOBES-FILE deployment.lock.toml\n"
            return 0, out, ""
        if k == "override_write":
            parts = shlex.split(cmd)
            # sh -c SCRIPT sh DIR BACKUP CONTENT
            backup, content = parts[-2], parts[-1]
            if backup:
                box.files[backup] = box.files.get("docker-compose.override.yml", "")
            box.files["docker-compose.override.yml"] = content
            return 0, "", ""
        if k == "read_key":
            return 0, (box.key or ""), ""
        if k == "version":
            return 0, f"lobes {box.installed}\n", ""
        if k == "read_pin":
            return 0, box.pinned + "\n", ""
        if k == "health":
            return 0, json.dumps({"version": box.running}), ""
        if k == "window":
            if box.window is None:
                return 3, "NO_MACHINE_CLI\n", ""
            return 0, json.dumps(box.window), ""
        if k == "backup":
            return 0, f"/home/{box.user}/.lobes/.env.bak-STAMP-rollout\n", ""
        if k == "install":
            box.installed = NEW
            return 0, "Installed 1 executable: lobes\n", ""
        if k == "repin":
            box.pinned = box.installed
            return 1, "", f"FAIL gateway_version_match GATEWAY_API_KEY={box.key}"
        if k == "inspect":
            return 0, "sha256:deadbeef\n", ""
        if k == "tag":
            parts = shlex.split(cmd)
            box.tags[parts[-1]] = parts[-2]
            return 0, "", ""
        if k == "build":
            if box.fail_build:
                return 1, "", "build failed"
            box.running = box.pinned
            return 0, "", ""
        if k == "restore":
            box.pinned = OLD
            return 0, "", ""
        if k == "recreate":
            box.running = box.pinned
            return 0, "", ""
        return 0, "", ""

    def writes(self, name):
        return [k for k, _ in self.boxes[name].calls if k in self.WRITE_KINDS]

    def all_writes(self):
        return {n: self.writes(n) for n in self.boxes if self.writes(n)}

    # -- HTTP opener -----------------------------------------------------
    def _box_for_url(self, url):
        p = urllib.parse.urlsplit(url)
        for b in self.boxes.values():
            if p.hostname == f"{b.host}.mesh" and p.port == b.port:
                return b
            if b.local and p.hostname in ("127.0.0.1", "localhost") and p.port == b.port:
                return b
        return None

    def opener(self, req, timeout=None):
        url = req.full_url
        method = req.get_method()
        auth = req.get_header("Authorization") or ""
        self.http.append((method, url))
        box = self._box_for_url(url)
        if box is None or not box.reachable:
            raise urllib.error.URLError("unreachable")
        path = urllib.parse.urlsplit(url).path
        if path == "/health":
            return _Resp({"status": "ok", "version": box.running})
        if path == "/capabilities":
            if box.key and auth != f"Bearer {box.key}":
                raise urllib.error.HTTPError(url, 401, "unauthorized", {}, io.BytesIO(b""))
            return _Resp({"roles": self._caps(box)})
        if path == "/mesh/roster":
            if auth != f"Bearer {MESH_KEY}":
                raise urllib.error.HTTPError(url, 401, "unauthorized", {}, io.BytesIO(b""))
            return _Resp(
                {
                    "members": [
                        {"name": b.name, "origin": b.origin, "verified": True}
                        for b in self.boxes.values()
                    ]
                }
            )
        if method == "POST":
            if box.interrupt_on_probe:
                box.interrupt_on_probe = False
                raise KeyboardInterrupt
            if box.key and auth != f"Bearer {box.key}":
                raise urllib.error.HTTPError(url, 401, "unauthorized", {}, io.BytesIO(b""))
            if path == "/v1/chat/completions":
                return _Resp({"choices": [{"message": {"content": "ok"}}]})
            if path == "/v1/embeddings":
                return _Resp({"data": [{"embedding": [0.1]}]})
            if path == "/v1/rerank":
                return _Resp({"results": [{"index": 0}]})
            return _Resp({"text": ""})
        raise urllib.error.HTTPError(url, 404, "nf", {}, io.BytesIO(b""))

    def _caps(self, box):
        caps = json.loads(json.dumps(box.caps))
        if box.regress_after and box.running == NEW:
            caps[box.regress_after]["ready"] = False
        return caps


class _Resp:
    status = 200

    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakePreflight:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        return (0, "Saved lobes_cli.whl") if self.ok else (1, "No matching distribution")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

RESIDENTS = """\
[resident.cultureos]
ssh = "cultureos"
user = "ori"
port = 8001
lobes_path = "/home/ori/.local/bin/lobes"
gateway_key = true

[resident.spark]
ssh = "spark-f8a9"
user = "nvidia"
port = 8001
lobes_path = "/home/nvidia/.local/bin/lobes"
gateway_key = true

[resident.thor]
ssh = "thor"
user = "thor"
port = 8000
lobes_path = "/home/thor/.local/bin/lobes"
gateway_key = true

[resident.orin]
ssh = "orin"
user = "orin"
port = 8000
lobes_path = "/home/orin/.local/bin/lobes"
gateway_key = false

[guest.orin-nano]
ssh = "orin-nano"
"""


def default_boxes():
    return [
        # operator box: gateway-only, hosts nothing, forwards everything
        Box(
            "cultureos",
            "cultureos",
            "ori",
            8001,
            proxied_roles=("cortex", "worker"),
            key=KEYS["cultureos"],
            local=True,
        ),
        Box("spark", "spark-f8a9", "nvidia", 8001, roles=("cortex", "hand"), key=KEYS["spark"]),
        Box("thor", "thor", "thor", 8000, roles=("worker",), key=KEYS["thor"]),
        Box("orin", "orin", "orin", 8000, roles=("embedder",), reachable=False),
    ]


@pytest.fixture
def deploy(tmp_path):
    d = tmp_path / "lobes"
    d.mkdir()
    (d / "docker-compose.yml").write_text(TEMPLATE)
    (d / ".env").write_text(
        "VLLM_PORT=8001\nLOBES_MESH_NAME=cultureos\n"
        f"LOBES_MESH_KEY={MESH_KEY}\nGATEWAY_API_KEY={OP_KEY}\n"
    )
    (d / "residents.toml").write_text(RESIDENTS)
    (tmp_path / "template.yml").write_text(TEMPLATE)
    return d


@pytest.fixture
def mesh(monkeypatch):
    m = FakeMesh(default_boxes())
    pre = FakePreflight()
    monkeypatch.setattr(
        rollout_cmd,
        "SEAMS",
        rollout_cmd.Seams(
            opener=m.opener, runner=m.runner, preflight_runner=pre, sleep=lambda s: None
        ),
    )
    m.preflight = pre
    return m


def run(deploy, *args):
    template = str(deploy.parent / "template.yml")
    argv = ["rollout", *args, "--compose-dir", str(deploy)]
    if args and args[0] in ("plan", "apply"):
        argv += ["--template", template]
    return main(argv)


def snapshot(d: Path):
    return {p: p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def journal_text(deploy):
    p = deploy / "rollout-journal.jsonl"
    return p.read_text() if p.exists() else ""


def journal(deploy):
    return [json.loads(line) for line in journal_text(deploy).splitlines() if line.strip()]


def assert_no_secrets(*texts):
    for t in texts:
        for s in SECRETS:
            assert s not in t, f"secret {s!r} leaked"


# ---------------------------------------------------------------------------
# registration / read-only surfaces
# ---------------------------------------------------------------------------


def test_registered_and_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rollout", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for verb in ("plan", "apply", "status", "rollback"):
        assert verb in out


def test_plan_is_read_only_and_ordered(deploy, mesh, capsys):
    before = snapshot(deploy)
    assert run(deploy, "plan", NEW) == 0
    out = capsys.readouterr().out
    assert snapshot(deploy) == before
    assert mesh.all_writes() == {}
    assert mesh.preflight.calls == []
    assert all(m == "GET" for m, _ in mesh.http)
    # canary first (thor hosts fewest unique roles), operator box last
    assert out.index("thor") < out.index("spark") < out.rindex("cultureos")
    assert "canary" in out
    assert "orin" in out and "skipped-offline" in out
    assert "orin-nano" in out and "skipped-guest" in out
    # addressing is shown, never guessed
    assert "nvidia@spark-f8a9" in out and "/home/nvidia/.local/bin/lobes" in out


def test_plan_json(deploy, mesh, capsys):
    assert run(deploy, "plan", NEW, "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["target"] == NEW and data["local"] == "cultureos"
    assert data["canary"] == "thor"
    assert [w["members"] for w in data["waves"]] == [["thor"], ["spark"], ["cultureos"]]
    by = {m["name"]: m for m in data["members"]}
    assert by["orin"]["status"] == "skipped-offline" and by["orin"]["pending"]
    assert by["orin-nano"]["status"] == "skipped-guest"
    assert by["spark"]["ssh"] == "spark-f8a9" and by["spark"]["port"] == 8001
    assert data["preflight"]["ran"] is False


def test_plan_reports_drift_and_lock(deploy, mesh, capsys):
    mesh.boxes["spark"].files["docker-compose.yml"] = DRIFTED
    mesh.boxes["thor"].lock_file = True
    assert run(deploy, "plan", NEW, "--json") == 0
    data = json.loads(capsys.readouterr().out)
    by = {m["name"]: m for m in data["members"]}
    assert by["spark"]["drift"]["missing"] == ["STT_EN_BASE_URL", "STT_EN_FEASIBLE"]
    assert by["thor"]["drift"]["lock_affected"] is True
    assert data["lock_affected"] == ["thor"]
    assert run(deploy, "plan", NEW) == 0
    out = capsys.readouterr().out
    assert "STT_EN_BASE_URL" in out and "lock" in out


def test_plan_refuses_downgrade_member(deploy, mesh, capsys):
    mesh.boxes["thor"].running = "0.96.0.dev1"
    assert run(deploy, "plan", NEW, "--json") == 0
    by = {m["name"]: m for m in json.loads(capsys.readouterr().out)["members"]}
    assert by["thor"]["status"] == "refused-downgrade"
    assert run(deploy, "plan", NEW, "--json", "--allow-downgrade") == 0
    by = {m["name"]: m for m in json.loads(capsys.readouterr().out)["members"]}
    assert by["thor"]["status"] == "planned"


def test_status_without_journal(deploy, mesh, capsys):
    assert run(deploy, "status") == 0
    assert "no rollout journal" in capsys.readouterr().out
    assert run(deploy, "status", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["members"] == {} and data["target"] is None
    assert not (deploy / "rollout-journal.jsonl").exists()


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def test_apply_without_apply_is_dry_run(deploy, mesh, capsys):
    before = snapshot(deploy)
    assert run(deploy, "apply", NEW) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "--apply" in out
    assert snapshot(deploy) == before
    assert mesh.all_writes() == {}
    assert mesh.preflight.calls == []
    assert run(deploy, "apply", NEW, "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["dry_run"] is True and data["state"] == "plan"


def test_apply_happy_path(deploy, mesh, capsys):
    assert run(deploy, "apply", NEW, "--apply") == 0
    out = capsys.readouterr().out
    assert len(mesh.preflight.calls) == 1
    for n in ("thor", "spark", "cultureos"):
        assert mesh.boxes[n].running == NEW
        assert "build" in mesh.writes(n)
    assert mesh.writes("orin") == []  # offline: never written to
    # wave order: canary, then spark, operator box last
    builds = [
        r["member"]
        for r in journal(deploy)
        if r["step"] == "rebuild_gateway" and r["result"] == "ok"
    ]
    assert builds == ["thor", "spark", "cultureos"]
    verified = [
        r["member"] for r in journal(deploy) if r["step"] == "verify" and r["result"] == "ok"
    ]
    assert verified == ["thor", "spark", "cultureos"]
    st = {r["member"]: r["result"] for r in journal(deploy) if r["member"] == "orin"}
    assert st == {"orin": "skipped_offline"}
    assert "complete" in out
    # the old gateway image was kept for rollback
    assert "lobes-gateway:0.94.0" in mesh.boxes["spark"].tags


def test_apply_json_complete(deploy, mesh, capsys):
    assert run(deploy, "apply", NEW, "--apply", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["state"] == "complete" and data["dry_run"] is False
    assert data["preflight"]["ok"] is True
    assert [m["member"] for m in data["results"]] == ["thor", "spark", "cultureos"]
    assert data["pending"] == ["orin"]


def test_apply_preflight_failure_touches_no_member(deploy, mesh, capsys):
    mesh.preflight.ok = False
    rc = run(deploy, "apply", NEW, "--apply")
    err = capsys.readouterr().err
    assert rc == 2
    assert mesh.all_writes() == {}
    assert "did not resolve" in err and "No member was contacted" in err
    assert not (deploy / "rollout-journal.jsonl").exists()


def test_apply_halts_at_first_failed_verification(deploy, mesh, capsys):
    mesh.boxes["spark"].regress_after = "cortex"
    rc = run(deploy, "apply", NEW, "--apply")
    cap = capsys.readouterr()
    assert rc == 2
    text = cap.out + cap.err
    assert "spark" in text and "capability_regression" in text and "cortex" in text
    assert "lobes rollout rollback --member spark --apply" in text
    # canary done, spark upgraded but not verified, operator box untouched
    assert mesh.boxes["thor"].running == NEW
    assert mesh.writes("cultureos") == []
    # never rolls back on its own
    assert "restore" not in mesh.writes("spark") and "recreate" not in mesh.writes("spark")
    assert mesh.boxes["spark"].running == NEW
    v = [r for r in journal(deploy) if r["step"] == "verify" and r["member"] == "spark"]
    assert v and v[-1]["result"] == "failed"


def test_apply_halt_json(deploy, mesh, capsys):
    mesh.boxes["spark"].regress_after = "cortex"
    rc = run(deploy, "apply", NEW, "--apply", "--json")
    cap = capsys.readouterr()
    assert rc == 2
    data = json.loads(cap.out)
    assert data["state"] == "halted"
    assert data["halt"]["member"] == "spark"
    assert data["halt"]["check"] == "capability_regression"
    assert data["halt"]["role"] == "cortex"
    assert data["halt"]["rollback"] == "lobes rollout rollback --member spark --apply"
    err = json.loads(cap.err)
    assert "rollback --member spark" in err["remediation"]


def test_apply_halts_on_failed_member_step(deploy, mesh, capsys):
    mesh.boxes["thor"].fail_build = True
    rc = run(deploy, "apply", NEW, "--apply", "--json")
    data = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert data["halt"]["member"] == "thor" and data["halt"]["check"] == "rebuild_gateway"
    assert mesh.writes("spark") == [] and mesh.writes("cultureos") == []


def test_apply_resumes_from_journal(deploy, mesh, capsys):
    # interrupted while verifying spark (after its gateway was rebuilt)
    mesh.boxes["spark"].interrupt_on_probe = True
    with pytest.raises(KeyboardInterrupt):
        run(deploy, "apply", NEW, "--apply")
    capsys.readouterr()
    assert mesh.boxes["spark"].running == NEW
    assert not mesh.boxes["spark"].lock_held  # released on interrupt
    thor_calls = len(mesh.boxes["thor"].calls)
    assert run(deploy, "status", "--json") == 0
    st = json.loads(capsys.readouterr().out)
    assert st["members"]["thor"]["state"] == "done"
    assert st["members"]["spark"]["state"] == "in_progress"
    assert st["resume"]["member"] == "spark"

    assert run(deploy, "apply", NEW, "--apply") == 0
    capsys.readouterr()
    # the verified canary is not redone; spark is verified; operator box done
    new_thor = [k for k, _ in mesh.boxes["thor"].calls[thor_calls:]]
    assert not set(new_thor) & FakeMesh.WRITE_KINDS
    verified = [
        r["member"] for r in journal(deploy) if r["step"] == "verify" and r["result"] == "ok"
    ]
    assert verified == ["thor", "spark", "cultureos"]
    # spark's "before" capabilities came from the first run's journal record
    assert run(deploy, "status", "--json") == 0
    st = json.loads(capsys.readouterr().out)
    assert {n: s["state"] for n, s in st["members"].items()} == {
        "thor": "done",
        "spark": "done",
        "cultureos": "done",
        "orin": "pending",
    }


def test_offline_member_upgraded_on_later_apply(deploy, mesh, capsys):
    assert run(deploy, "apply", NEW, "--apply") == 0
    mesh.boxes["orin"].reachable = True
    # orin has no gateway key (gateway_key = false): still upgraded and verified
    assert run(deploy, "apply", NEW, "--apply") == 0
    capsys.readouterr()
    assert mesh.boxes["orin"].running == NEW
    assert run(deploy, "status", "--json") == 0
    st = json.loads(capsys.readouterr().out)
    assert st["members"]["orin"]["state"] == "done"


def test_window_active_member_skipped_with_reason(deploy, mesh, capsys):
    mesh.boxes["thor"].window = {"active": True}
    assert run(deploy, "apply", NEW, "--apply", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["state"] == "incomplete"
    thor = next(r for r in data["results"] if r["member"] == "thor")
    assert thor["result"] == "skipped" and "benchmark window" in thor["reason"]
    assert mesh.boxes["thor"].running == OLD
    assert mesh.boxes["spark"].running == NEW


def test_lock_held_member_halts(deploy, mesh, capsys):
    mesh.boxes["thor"].lock_held = True
    rc = run(deploy, "apply", NEW, "--apply", "--json")
    data = json.loads(capsys.readouterr().out)
    assert rc == 2 and data["halt"]["member"] == "thor" and data["halt"]["check"] == "lock"
    assert "build" not in mesh.writes("thor")


def test_drift_refused_without_patch_flag(deploy, mesh, capsys):
    mesh.boxes["spark"].files["docker-compose.yml"] = DRIFTED
    rc = run(deploy, "apply", NEW, "--apply")
    err = capsys.readouterr().err
    assert rc == 1
    assert "spark" in err and "STT_EN_BASE_URL" in err and "--patch-drift" in err
    assert mesh.all_writes() == {}
    assert mesh.preflight.calls == []


def test_drift_dry_run_warns(deploy, mesh, capsys):
    mesh.boxes["spark"].files["docker-compose.yml"] = DRIFTED
    assert run(deploy, "apply", NEW, "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["would_refuse"] and "spark" in data["would_refuse"]
    assert mesh.all_writes() == {}


def test_drift_patched_with_flag(deploy, mesh, capsys):
    spark = mesh.boxes["spark"]
    spark.files["docker-compose.yml"] = DRIFTED
    spark.files["docker-compose.override.yml"] = OVERRIDE_WITH_SECRET
    assert run(deploy, "apply", NEW, "--apply", "--patch-drift") == 0
    cap = capsys.readouterr()
    new_override = spark.files["docker-compose.override.yml"]
    assert "STT_EN_BASE_URL=${STT_EN_BASE_URL:-}" in new_override
    assert f"GATEWAY_API_KEY={KEYS['spark']}" in new_override  # raw value preserved
    backups = [n for n in spark.files if n.startswith("docker-compose.override.yml.bak.")]
    assert len(backups) == 1 and spark.files[backups[0]] == OVERRIDE_WITH_SECRET
    assert "docker-compose.yml" in spark.files and spark.files["docker-compose.yml"] == DRIFTED
    assert "+      - STT_EN_BASE_URL" in cap.out  # the diff is printed
    rec = [r for r in journal(deploy) if r["step"] == "patch_override"]
    assert rec and rec[0]["member"] == "spark" and rec[0]["result"] == "ok"
    assert_no_secrets(cap.out, cap.err, journal_text(deploy))


def test_no_secret_values_anywhere(deploy, mesh, capsys):
    mesh.boxes["spark"].regress_after = "cortex"
    run(deploy, "plan", NEW)
    run(deploy, "plan", NEW, "--json")
    run(deploy, "apply", NEW)
    run(deploy, "apply", NEW, "--apply")
    run(deploy, "apply", NEW, "--apply", "--json")
    run(deploy, "status")
    run(deploy, "status", "--json")
    run(deploy, "rollback", "--member", "spark")
    run(deploy, "rollback", "--member", "spark", "--json")
    cap = capsys.readouterr()
    assert_no_secrets(cap.out, cap.err, journal_text(deploy))


def test_verification_uses_each_members_own_key(deploy, mesh, capsys):
    assert run(deploy, "apply", NEW, "--apply") == 0
    # spark's /capabilities only accepts spark's own key; a pass proves it was used
    v = [r for r in journal(deploy) if r["step"] == "verify" and r["member"] == "spark"]
    assert v[-1]["result"] == "ok"
    assert "cortex" in v[-1]["detail"]["probed"]


def test_gateway_only_operator_box_is_last_and_verified(deploy, mesh, capsys):
    assert run(deploy, "apply", NEW, "--apply", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    last = data["results"][-1]
    assert last["member"] == "cultureos" and last["result"] == "verified"
    # forwarded roles were exercised through the operator box's own gateway
    probes = [u for m, u in mesh.http if m == "POST" and "127.0.0.1:8001" in u]
    assert probes


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def _halted_run(deploy, mesh, capsys):
    mesh.boxes["spark"].regress_after = "cortex"
    assert run(deploy, "apply", NEW, "--apply") == 2
    capsys.readouterr()


def test_rollback_dry_run_writes_nothing(deploy, mesh, capsys):
    _halted_run(deploy, mesh, capsys)
    calls_before = len(mesh.boxes["spark"].calls)
    before = snapshot(deploy)
    assert run(deploy, "rollback", "--member", "spark") == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "/home/nvidia/.lobes/.env.bak-STAMP-rollout" in out or ".env.bak-" in out
    assert "lobes-gateway:0.94.0" in out
    assert "lobes rollout rollback --member spark --apply" in out
    assert snapshot(deploy) == before
    assert len(mesh.boxes["spark"].calls) == calls_before
    assert run(deploy, "rollback", "--member", "spark", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["dry_run"] is True and data["old_version"] == OLD
    assert len(data["actions"]) == 3


def test_rollback_apply_restores_previous_version(deploy, mesh, capsys):
    _halted_run(deploy, mesh, capsys)
    assert run(deploy, "rollback", "--member", "spark", "--apply", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["applied"] is True and data["ok"] is True
    kinds = mesh.writes("spark")
    assert "restore" in kinds and "recreate" in kinds
    assert mesh.boxes["spark"].running == OLD
    assert data["health_version"] == OLD
    rec = [r for r in journal(deploy) if r["step"] == "rollback"]
    assert rec and rec[-1]["member"] == "spark" and rec[-1]["result"] == "ok"


def test_rollback_without_journal_refuses(deploy, mesh, capsys):
    rc = run(deploy, "rollback", "--member", "spark")
    err = capsys.readouterr().err
    assert rc == 1 and "backup" in err
    assert mesh.all_writes() == {}


def test_rollback_unknown_member(deploy, mesh, capsys):
    rc = run(deploy, "rollback", "--member", "nope")
    assert rc == 1
    assert "nope" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        ("plan", NEW),
        ("apply", NEW),
        ("status",),
    ],
)
def test_every_read_or_dry_subcommand_emits_json(deploy, mesh, capsys, args):
    assert run(deploy, *args, "--json") == 0
    json.loads(capsys.readouterr().out)


def test_halt_after_patch_still_shows_the_override_diff(deploy, mesh, capsys):
    spark = mesh.boxes["spark"]
    spark.files["docker-compose.yml"] = DRIFTED
    spark.regress_after = "cortex"
    assert run(deploy, "apply", NEW, "--apply", "--patch-drift", "--json") == 2
    data = json.loads(capsys.readouterr().out)
    assert data["halt"]["member"] == "spark"
    assert "+      - STT_EN_BASE_URL" in data["halt"]["diff"]
