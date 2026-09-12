"""mesh-brain-join t4 — template and scan plumbing for the six LOBES_MESH_* keys.

The gateway does not read .env via env_file (the scoped-environment design,
see the note in templates/fleet/docker-compose.yml's gateway block): a mesh
key the operator sets in .env but that is ABSENT from the gateway's explicit
``environment:`` list never reaches the container, and the mesh is silently
OFF — the same silent-inert trap the capacity block documents. So the whole
contract this task owns is:

  1. every key the gateway's MeshConfig parses (t1's ``_mesh_config.py``)
     is listed on the gateway service's environment passthrough, and a
     rendered compose with ``LOBES_MESH_LEDGER_PATH`` set mounts that file
     read-write into the gateway — its first bind mount (spec c9);
  2. ``env.example`` documents the keys, with the secret flagged;
  3. ``.gitignore`` covers the runtime ledger's default file name (the
     ledger is a gitignored runtime file under the deployment dir, never a
     lock or a golden);
  4. the goldens regenerate unchanged except
     ``template-defaults.env`` gaining the new keys' defaults.

No docker runs here: ``_interpolate`` emulates compose's ``${...}``
interpolation for the exact operator subset this template uses
(``${VAR}``, ``${VAR:-default}``, ``${VAR-default}``, ``${VAR:+alternate}``,
brace-depth aware so the nested ``HF_CACHE`` default and the ``{}`` JSON
defaults survive), which is the same surface ``tests/goldens/regen.py``
walks.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

from lobes.profiles.loader import builtin_names
from tests.goldens.regen import (
    profile_env_text,
    shape_env_text,
    shape_golden_pairs,
    shape_golden_path,
    switch_plan_text,
    template_defaults_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_FLEET_COMPOSE = REPO_ROOT / "lobes" / "templates" / "fleet" / "docker-compose.yml"
_ENV_EXAMPLE = REPO_ROOT / "lobes" / "templates" / "fleet" / "env.example"
_GITIGNORE = REPO_ROOT / ".gitignore"
_GOLDENS = REPO_ROOT / "tests" / "goldens"

# The six keys t1's MeshConfig parses — every one of them must reach the
# gateway container through the explicit environment list.
MESH_KEYS = (
    "LOBES_MESH_KEY",
    "LOBES_MESH_NAME",
    "LOBES_MESH_SEEDS",
    "LOBES_MESH_HEARTBEAT_S",
    "LOBES_MESH_MISSED_MAX",
    "LOBES_MESH_LEDGER_PATH",
)

# The host-side default for the mesh runtime directory (compose interpolation);
# the in-container mount is /home/gateway/mesh (the unprivileged gateway user's
# home, per Dockerfile.gateway — uid 10001).
_DEFAULT_LEDGER_HOST_NAME = "./mesh"
_DEFAULT_LEDGER_CONTAINER_PATH = "/home/gateway/mesh"


# --- compose interpolation emulation (no docker in the test suite) ---------


_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _split_operator(inner: str) -> tuple[str, str, str]:
    """Split an interpolation body into ``(name, operator, rest)``.

    The operator is ``""`` (bare ``${VAR}``), ``":"-"`` / ``":"+"`` (the
    colon forms), or ``"-"`` (the colonless ``${VAR-default}`` form the
    SPECULATIVE_CONFIG lines use — whose default is a JSON string that
    itself contains colons, so the two-character ``:-`` / ``:+`` sequences
    are located by SCAN, not by 'first colon'). The FIRST such sequence in
    the body is the outermost operator: nested substitutions always sit
    INSIDE a default/alternate, after it.
    """
    positions = [(inner.find(op), op) for op in (":-", ":+") if inner.find(op) != -1]
    if positions:
        i, op = min(positions)
        return inner[:i], op, inner[i + 2 :]
    if "-" in inner:
        name, _, rest = inner.partition("-")
        if _NAME_RE.fullmatch(name):
            return name, "-", rest
    if _NAME_RE.fullmatch(inner):
        return inner, "", ""
    # Not a valid variable body — compose leaves such sequences as LITERALS
    # (e.g. the ${*_MAX_MODEL_LEN} prose in a comment), so do the same.
    return None, None, None


def _interpolate(text: str, env: dict[str, str]) -> str:
    """Render ``text`` the way ``docker compose`` interpolates it for ``env``.

    Unset variables: bare → empty, ``:-d``/``-d`` → ``d`` (``:-`` also when
    set-but-empty), ``:+a`` → empty. A chosen default/alternate is itself
    rendered recursively (the nested ``${HOME:-/root}`` inside HF_CACHE's
    default), while an operator-set value is taken literally.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        start = text.find("${", i)
        if start == -1:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:start])
        depth, j = 1, start + 2
        while j < n and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        name, op, rest = _split_operator(text[start + 2 : j - 1])
        if name is None:
            out.append(text[start:j])
            i = j
            continue
        value = env.get(name)
        if op in (":-", "-"):
            if op == ":-" and value in (None, ""):
                value = _interpolate(rest, env)
            elif value is None:
                value = _interpolate(rest, env)
        elif op == ":+":
            value = _interpolate(rest, env) if value not in (None, "") else ""
        elif value is None:
            value = ""
        out.append(value)
        i = j
    return "".join(out)


def _render_gateway(compose_env: dict[str, str]) -> dict:
    """The template rendered for ``compose_env``, as parsed YAML — what
    ``docker compose config`` would show the gateway service."""
    text = _FLEET_COMPOSE.read_text(encoding="utf-8")
    rendered = _interpolate(text, compose_env)
    return yaml.safe_load(rendered)["services"]["gateway"]


def _gateway_env_lines() -> dict[str, str]:
    """The UNRENDERED template's gateway ``environment:`` list as {key: raw}."""
    compose = yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in compose["services"]["gateway"]["environment"]:
        assert "=" in entry, f"non key=value gateway environment entry: {entry!r}"
        key, _, value = entry.partition("=")
        out[key] = value
    return out


# --- criterion 1a: every _mesh_config key is on the passthrough list -------


class TestMeshKeysOnGatewayPassthrough:
    def test_every_mesh_key_is_listed(self) -> None:
        env = _gateway_env_lines()
        for key in MESH_KEYS:
            assert key in env, (
                f"{key} is missing from the gateway environment list — a key set "
                "in .env but absent here never reaches the container (the "
                "gateway does not use env_file), so the mesh would be silently "
                "OFF. See the silent-inert note in the capacity block above."
            )

    def test_every_mesh_key_is_a_plain_passthrough_of_its_own_name(self) -> None:
        env = _gateway_env_lines()
        for key in MESH_KEYS:
            assert env[key].startswith(f"${{{key}"), (
                f"{key} must pass through the operator's own {key} value — " f"got {env[key]!r}"
            )

    def test_key_name_seeds_and_ledger_path_default_to_empty(self) -> None:
        """Mesh is OPT-IN: with nothing in .env the rendered gateway gets
        empty values for the key/identity/seeds keys, so t1's
        build_mesh_config sees them unset (enabled=False, c13). The ledger
        path defaults to a path inside the mounted mesh directory — the
        process never opens it when LOBES_MESH_KEY is empty."""
        gateway = _render_gateway({})
        env = {e.partition("=")[0]: e.partition("=")[2] for e in gateway["environment"]}
        for key in (
            "LOBES_MESH_KEY",
            "LOBES_MESH_NAME",
            "LOBES_MESH_SEEDS",
        ):
            assert env[key] == "", f"{key} must render empty on the default fleet, got {env[key]!r}"

    def test_heartbeat_and_missed_max_render_their_parser_defaults(self) -> None:
        """The container never reads .env, so these two must carry their
        parser defaults (60 / 3) in the compose line — an empty passthrough
        would hand t1's parser a value it rejects with a named error. An
        operator-set value still wins."""
        gateway = _render_gateway({})
        env = {e.partition("=")[0]: e.partition("=")[2] for e in gateway["environment"]}
        assert env["LOBES_MESH_HEARTBEAT_S"] == "60"
        assert env["LOBES_MESH_MISSED_MAX"] == "3"
        gateway = _render_gateway({"LOBES_MESH_HEARTBEAT_S": "90", "LOBES_MESH_MISSED_MAX": "7"})
        env = {e.partition("=")[0]: e.partition("=")[2] for e in gateway["environment"]}
        assert env["LOBES_MESH_HEARTBEAT_S"] == "90"
        assert env["LOBES_MESH_MISSED_MAX"] == "7"


# --- criterion 1b: the ledger's first bind mount ----------------------------


class TestGatewayLedgerMount:
    def test_rendered_compose_mounts_ledger_read_write_when_set(self) -> None:
        """Acceptance criterion: a rendered compose with
        LOBES_MESH_LEDGER_PATH set still mounts the mesh directory read-write
        into the gateway. The value controls the in-container ledger path
        (defaults to /home/gateway/mesh/ledger.json); the volume itself is
        LOBES_MESH_DIR's interpolation. The mount carries no :ro."""
        path = "/home/spark/.lobes/mesh/ledger.json"
        gateway = _render_gateway({"LOBES_MESH_LEDGER_PATH": path})
        volumes = gateway["volumes"]
        # The volume is the mesh directory bind (LOBES_MESH_DIR interpolation),
        # not P:P — the file path is controlled by LOBES_MESH_LEDGER_PATH env.
        assert "./mesh:/home/gateway/mesh" in volumes, (
            f"expected the mesh dir bind mount on the rendered gateway "
            f"service, got volumes={volumes!r}"
        )
        entry = volumes[volumes.index("./mesh:/home/gateway/mesh")]
        assert ":ro" not in entry, "the mesh dir is read-WRITE — the gateway appends approvals"
        # The custom path lands in the env passthrough.
        env = {e.partition("=")[0]: e.partition("=")[2] for e in gateway["environment"]}
        assert env["LOBES_MESH_LEDGER_PATH"] == path

    def test_ledger_mount_is_the_gateway_first_volume(self) -> None:
        """c9: the gateway service has NO volumes today — the mesh dir bind is
        its first (and, on the base template, only) mount."""
        gateway = _render_gateway({})
        assert gateway["volumes"] == [f"{_DEFAULT_LEDGER_HOST_NAME}:/home/gateway/mesh"], (
            "the gateway's volume list must be exactly the mesh dir bind — the "
            "first volume the gateway service has ever had"
        )

    def test_default_volume_is_mesh_dir_not_ledger_file(self) -> None:
        """With no .env value the default volume is the mesh runtime
        directory (./mesh:/home/gateway/mesh), not a file bind. The container
        side defaults to /home/gateway/mesh/ledger.json inside that mount."""
        gateway = _render_gateway({})
        env = {e.partition("=")[0]: e.partition("=")[2] for e in gateway["environment"]}
        assert env["LOBES_MESH_LEDGER_PATH"] == "/home/gateway/mesh/ledger.json"
        volume = gateway["volumes"][0]
        host, _, target = volume.partition(":")
        assert host == _DEFAULT_LEDGER_HOST_NAME
        assert target == "/home/gateway/mesh"


# --- env.example documents the keys -----------------------------------------


class TestEnvExampleDocumentsMesh:
    def _mesh_section(self) -> str:
        text = _ENV_EXAMPLE.read_text(encoding="utf-8")
        start = text.index("Mesh join")
        # Retired (t14): "# --- Honest referral to peer boxes" used to mark
        # the end of the mesh section; that block is deleted along with the
        # rest of the env peer family. The retirement note that replaced it
        # is the new end-of-section marker.
        return text[start : text.index("# --- Retired: honest referral")]

    def test_every_mesh_key_is_documented(self) -> None:
        section = self._mesh_section()
        for key in MESH_KEYS:
            assert f"{key}=" in section, f"{key} is not documented in env.example's mesh section"

    def test_join_key_is_flagged_secret(self) -> None:
        section = self._mesh_section().lower()
        assert "secret" in section
        assert "never commit" in section or "do not commit" in section

    def test_mesh_is_opt_in_and_off_by_default(self) -> None:
        section = self._mesh_section().lower()
        assert "opt-in" in section
        assert "byte-identical" in section

    def test_no_real_fleet_hostname_in_the_mesh_section(self) -> None:
        section = self._mesh_section().lower()
        for token in ("spark", "thor", "orin", "tail0be7e0"):
            assert token not in section, f"{token!r} must not appear in env.example's mesh section"


# --- .gitignore covers the runtime ledger ------------------------------------


class TestGitignoreCoversRuntimeLedger:
    def test_default_ledger_name_is_git_ignored(self, tmp_path: Path) -> None:
        """A real ``git check-ignore`` in a scratch repo carrying this
        .gitignore: the ledger's default dir is ignored wherever it lands
        (including a deployment dir that ``lobes init .`` places inside a
        working tree)."""
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        (tmp_path / ".gitignore").write_text(
            _GITIGNORE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (tmp_path / "mesh").mkdir()
        (tmp_path / "mesh" / "ledger.json").write_text("{}", encoding="utf-8")
        result = subprocess.run(
            ["git", "-C", str(tmp_path), "check-ignore", "-q", "mesh/ledger.json"],
            check=False,
        )
        assert result.returncode == 0, (
            "mesh/ledger.json is NOT gitignored — the runtime approval ledger "
            "would be committable (c9/h11: a gitignored runtime file the lock "
            "never captures and the goldens never render)"
        )

    def test_gitignore_rule_is_positional_not_scoped(self) -> None:
        """The rule must name the directory (anywhere under the tree), the same
        positional style as the ``*.env`` secret-dotfile rule — not a
        deployment-dir-scoped path this repo's .gitignore has never needed."""
        text = _GITIGNORE.read_text(encoding="utf-8")
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert any(
            line == "mesh/" for line in lines
        ), "expected a bare positional `mesh/` rule in .gitignore"
        assert not any(line.startswith("!") and "mesh/" in line for line in lines)


# --- criterion 3: goldens ----------------------------------------------------


class TestGoldens:
    def test_committed_goldens_are_byte_identical_to_a_fresh_regeneration(self) -> None:
        """Every golden ``regen.py`` owns reproduces byte-for-byte from the
        current tree. (no-pool-gateway.json is the one golden that captures
        a live gateway — it exercises gateway CODE, not the compose template
        this task edits, so it is out of this comparison.)"""
        for name in builtin_names():
            committed = (_GOLDENS / f"{name}.env").read_text(encoding="utf-8")
            assert profile_env_text(name) == committed, f"tests/goldens/{name}.env drifted"
        committed = (_GOLDENS / "template-defaults.env").read_text(encoding="utf-8")
        assert template_defaults_text() == committed, "tests/goldens/template-defaults.env drifted"
        for shape_name, card_name in shape_golden_pairs():
            committed = shape_golden_path(shape_name, card_name).read_text(encoding="utf-8")
            assert (
                shape_env_text(shape_name, card_name) == committed
            ), f"tests/goldens/shapes/{shape_name}__{card_name}.env drifted"
        committed = (_GOLDENS / "switch-plans.txt").read_text(encoding="utf-8")
        assert switch_plan_text() == committed, "tests/goldens/switch-plans.txt drifted"

    def test_template_defaults_carry_exactly_the_new_keys_defaults(self) -> None:
        """template-defaults.env's mesh surface is the six new keys'
        defaults — and nothing else of this task's edit may have moved the
        golden (verified at commit time by the regen diff)."""
        lines = set(template_defaults_text().splitlines())
        expected = {
            "LOBES_MESH_DIR=./mesh",
            "LOBES_MESH_KEY=",
            "LOBES_MESH_NAME=",
            "LOBES_MESH_SEEDS=",
            "LOBES_MESH_HEARTBEAT_S=60",
            "LOBES_MESH_MISSED_MAX=3",
            "LOBES_MESH_LEDGER_PATH=/home/gateway/mesh/ledger.json",
        }
        assert expected <= lines, f"missing from the golden: {expected - lines}"
        mesh_only = {line for line in lines if line.startswith("LOBES_MESH_")}
        assert (
            mesh_only == expected
        ), f"the golden carries unexpected LOBES_MESH_* lines: {mesh_only - expected}"

    def test_other_goldens_carry_no_mesh_keys(self) -> None:
        """Membership state is runtime state, never a rendered .env key (c9):
        no profile/shape golden may gain a LOBES_MESH_* line from this task."""
        for name in builtin_names():
            text = (_GOLDENS / f"{name}.env").read_text(encoding="utf-8")
            assert "LOBES_MESH_" not in text, f"tests/goldens/{name}.env leaked a mesh key"
        for shape_name, card_name in shape_golden_pairs():
            text = shape_golden_path(shape_name, card_name).read_text(encoding="utf-8")
            assert (
                "LOBES_MESH_" not in text
            ), f"tests/goldens/shapes/{shape_name}__{card_name}.env leaked a mesh key"
        assert "LOBES_MESH_" not in switch_plan_text()
