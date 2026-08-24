"""Offline tests for `scripts/spike-preflight.sh`'s argv-token validator.

The spike harness exists to make ONE silent failure impossible: the fleet
compose template records as a MEASURED bug that a brace-containing substitution
corrupts compose's interpolation of every LATER brace pair, and
``PRIMARY_SPECULATIVE_CONFIG`` was the victim — it lost its closing brace. A
mangled token means vLLM boots healthy and serves with NO speculation, so every
throughput number measured against it is meaningless.

That validator therefore has to be right about mangled input, and it is the one
part of the harness that needs neither docker, nor ssh, nor a GPU. The script
exposes it as the read-only ``check-token`` mode precisely so it can be pinned
here: these tests shell out to the real script, so they test the shipped code
path rather than a Python re-implementation of it.

The rest of the harness — the peer read, the ``docker compose -f`` chain
resolution and the refusal paths that guard the production mutation — is
exercised the same way, against a PATH of stub executables (``ssh``, ``docker``,
``lobes``, ``curl``). No network, no docker daemon, no fleet: the stubs are what
makes "an unreachable peer" and "a failing compose resolver" reproducible, and
the assertions are about the REAL script's decisions, not a re-implementation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "spike-preflight.sh"
CHECK_TOKEN = "check-token"
PREFLIGHT = "preflight"
STOP = "stop"
APPLY = "--apply"
PEER_FLAG = "--peer"
DEPLOY_DIR_FLAG = "--deploy-dir"
ALLOW_UNREACHABLE = "--allow-unreachable-peers"
UNREACHABLE = "UNREACHABLE"

# A well-formed cortex argv, so the argv proof passes and the ONLY thing under
# test in a given case is the peer read or the compose-chain resolution.
_GOOD_ARGS_JSON = r'["--speculative-config={\"method\": \"mtp\", \"num_speculative_tokens\": 2}"]'

_DOCKER_STUB = f"""#!/usr/bin/env bash
echo "docker $*" >> "$STUB_LOG"
case "$1" in
  inspect)
    case "$*" in
      *json\\ .Args*) printf '%s\\n' '{_GOOD_ARGS_JSON}' ;;
      *) echo "sha256:stub" ;;
    esac ;;
  ps) echo "  model-gear-vllm-primary  Up 3 hours  image=vllm:stub" ;;
  compose) echo "COMPOSE-RAN" >> "$STUB_LOG" ;;
esac
exit 0
"""

_SSH_STUB = """#!/usr/bin/env bash
echo "ssh $*" >> "$STUB_LOG"
[ "${STUB_SSH_RC:-0}" = "0" ] || exit "${STUB_SSH_RC}"
# The peer .env read: a peer that hosts its own cortex and proxies nothing.
echo "__DIR__=/home/peer/.lobes"
echo "PRIMARY_FEASIBLE=true"
exit 0
"""

_LOBES_STUB = """#!/usr/bin/env bash
echo "lobes $*" >> "$STUB_LOG"
[ "${STUB_LOBES_RC:-0}" = "0" ] || exit "${STUB_LOBES_RC}"
printf '%s' "${STUB_LOBES_OUT:-}"
exit 0
"""

_CURL_STUB = """#!/usr/bin/env bash
echo "curl $*" >> "$STUB_LOG"
exit 1
"""


class _Harness:
    """A stub PATH + deployment dir; `run()` shells out to the real script."""

    def __init__(self, root: Path):
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir(parents=True)
        for name, body in (
            ("docker", _DOCKER_STUB),
            ("ssh", _SSH_STUB),
            ("lobes", _LOBES_STUB),
            ("curl", _CURL_STUB),
        ):
            path = self.bin / name
            path.write_text(body)
            path.chmod(0o755)
        self.deploy = root / "deploy"
        self.deploy.mkdir()
        (self.deploy / ".env").write_text(
            "VLLM_PORT=8000\nPRIMARY_FEASIBLE=true\nPRIMARY_MODEL=stub/model\n"
        )
        self.log = root / "stub.log"
        self.log.write_text("")

    def run(self, *args: str, **stub_env: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.pop("LOBES_SPIKE_PEERS", None)
        # A deliberately ISOLATED PATH: the stubs, then only the system dirs
        # that carry bash/python3/coreutils. A real `lobes`, `docker` or `ssh`
        # installed on the box must never leak in — the whole point is that
        # deleting a stub simulates the tool being absent.
        env["PATH"] = os.pathsep.join([str(self.bin), "/usr/bin", "/bin"])
        env["STUB_LOG"] = str(self.log)
        env.update(stub_env)
        return subprocess.run(
            ["bash", str(SCRIPT), *args, DEPLOY_DIR_FLAG, str(self.deploy)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=env,
        )

    @property
    def compose_ran(self) -> bool:
        return "COMPOSE-RAN" in self.log.read_text()


@pytest.fixture
def harness(tmp_path) -> _Harness:
    return _Harness(tmp_path)


def _check(token: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), CHECK_TOKEN, token],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111, "spike-preflight.sh must be executable"


VALID = [
    # The deployed cortex default, verbatim from `docker inspect`.
    '--speculative-config={"method": "mtp", "num_speculative_tokens": 2}',
    # The DSpark shape the spike swaps in: longer, a model path, more braces.
    '--speculative-config={"method": "dspark", "model": "radixark/DSpark-Drafter-1.36B",'
    ' "num_speculative_tokens": 4, "draft_tensor_parallel_size": 1}',
    # The spark-lobe 1M YaRN override — the substitution whose placement the
    # template calls load-bearing.
    '--hf-overrides={"text_config": {"rope_parameters": {"rope_type": "yarn",'
    ' "factor": 4.0, "original_max_position_embeddings": 262144,'
    ' "mrope_interleaved": true, "mrope_section": [11, 11, 10],'
    ' "partial_rotary_factor": 0.25, "rope_theta": 10000000}}}',
    "--hf-overrides={}",
    '--default-chat-template-kwargs={"preserve_thinking": true}',
    # A brace inside a JSON string must not confuse the depth scan.
    '--speculative-config={"method": "mtp", "note": "a } inside a string"}',
]


@pytest.mark.parametrize("token", VALID, ids=range(len(VALID)))
def test_well_formed_tokens_pass(token):
    proc = _check(token)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


MANGLED = {
    # THE recorded bug: the closing brace is eaten by compose interpolation.
    "missing_closing_brace": '--speculative-config={"method": "mtp", "num_speculative_tokens": 2',
    # The same bug on the longer DSpark JSON the spike introduces.
    "dspark_missing_closing_brace": '--speculative-config={"method": "dspark",'
    ' "model": "radixark/DSpark-Drafter-1.36B", "num_speculative_tokens": 4',
    "extra_closing_brace": '--speculative-config={"method": "mtp"}}',
    "closing_before_opening": '--speculative-config=}"method": "mtp"{',
    # Dropping the template's single quotes lets the shell lexer split the JSON
    # on its spaces; the leading fragment is what lands in argv.
    "split_on_space": '--speculative-config={"method":',
    # ...and if the quotes are NOT consumed by the lexer they leak into argv.
    "quote_leaked": "--hf-overrides='{}'",
    "empty_payload": "--speculative-config=",
    "not_json": "--speculative-config=mtp",
    # A JSON array/scalar is parseable but is not a config object.
    "json_but_not_object": '--speculative-config=["mtp"]',
    "unterminated_string": '--speculative-config={"method": "mtp}',
}


@pytest.mark.parametrize("token", MANGLED.values(), ids=list(MANGLED))
def test_mangled_tokens_fail(token):
    proc = _check(token)
    assert proc.returncode != 0, (
        "validator ACCEPTED a mangled token — this is the exact silent failure "
        "the harness exists to prevent:\n" + proc.stdout
    )
    assert "MANGLED" in proc.stdout


def test_non_flag_token_is_rejected():
    proc = _check('{"method": "mtp"}')
    assert proc.returncode != 0
    assert "not a --flag=payload token" in proc.stdout


def test_unrelated_flag_is_rejected():
    # check-token validates JSON-bearing flags only; anything else is operator
    # error, not a silent pass.
    proc = _check("--max-model-len=1048576")
    assert proc.returncode != 0


def test_check_token_requires_an_argument():
    proc = subprocess.run(
        ["bash", str(SCRIPT), CHECK_TOKEN],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 2


def test_unknown_mode_is_a_usage_error():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "wat"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 2


def test_help_exits_zero_and_documents_every_mode():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    for mode in ("preflight", "stop", "restore", "check-token"):
        assert mode in proc.stdout
    # Mutation safety is part of the contract, not a footnote.
    assert "--apply" in proc.stdout


# ---------------------------------------------------------------------------
# Qodo finding 7 — an UNREACHABLE peer is UNKNOWN, and UNKNOWN must not read as
# safe. It fails the preflight, and therefore blocks the production mutation.
# ---------------------------------------------------------------------------


def test_unreachable_peer_fails_preflight(harness):
    proc = harness.run(PREFLIGHT, PEER_FLAG, "thor@thor", STUB_SSH_RC="255")
    assert proc.returncode != 0, (
        "an unreachable peer was treated as a successful probe — the harness "
        "cannot know whether that peer proxies cortex here:\n" + proc.stdout
    )
    assert UNREACHABLE in proc.stdout
    assert ALLOW_UNREACHABLE in proc.stdout, "the failure must name its opt-out"


def test_reachable_peer_passes_preflight(harness):
    proc = harness.run(PREFLIGHT, PEER_FLAG, "thor@thor")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert UNREACHABLE not in proc.stdout


def test_unreachable_peer_blocks_stop_apply(harness):
    proc = harness.run(STOP, APPLY, PEER_FLAG, "thor@thor", STUB_SSH_RC="255")
    assert proc.returncode != 0
    assert not harness.compose_ran, (
        "stop --apply mutated the production cortex lane while a peer's "
        "dependency on it was UNKNOWN"
    )


def test_allow_unreachable_peers_overrides_and_is_loud(harness):
    proc = harness.run(PREFLIGHT, PEER_FLAG, "thor@thor", ALLOW_UNREACHABLE, STUB_SSH_RC="255")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # The override has to land in an evidence transcript as a decision, not as
    # a clean check.
    assert "OPERATOR OVERRIDE IN EFFECT" in proc.stdout
    assert "OVERRIDDEN" in proc.stdout
    assert UNREACHABLE in proc.stdout


def test_allow_unreachable_peers_lets_stop_apply_through(harness):
    proc = harness.run(STOP, APPLY, PEER_FLAG, "thor@thor", ALLOW_UNREACHABLE, STUB_SSH_RC="255")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert harness.compose_ran
    assert "OPERATOR OVERRIDE IN EFFECT" in proc.stdout


# ---------------------------------------------------------------------------
# Qodo finding 8 — a failed compose resolver is not an empty -f chain.
# ---------------------------------------------------------------------------

CHAIN = (
    "-f\ndocker-compose.yml\n-f\ndocker-compose.audio.yml\n"
    "-f\ndocker-compose.shape.yml\n-f\ndocker-compose.override.yml\n"
)
NO_PEERS = (PEER_FLAG, "none")


def test_resolved_chain_is_printed_and_passed_to_compose(harness):
    proc = harness.run(STOP, APPLY, *NO_PEERS, STUB_LOBES_OUT=CHAIN)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "docker-compose.override.yml" in proc.stdout
    log = harness.log.read_text()
    for name in (
        "docker-compose.yml",
        "docker-compose.audio.yml",
        "docker-compose.shape.yml",
        "docker-compose.override.yml",
    ):
        assert name in log, f"{name} never reached the docker compose argv"


def test_resolver_failure_refuses_the_operation(harness):
    proc = harness.run(STOP, APPLY, *NO_PEERS, STUB_LOBES_RC="3")
    assert proc.returncode != 0
    assert "FAILED" in proc.stdout
    assert "UNKNOWN" in proc.stdout
    assert not harness.compose_ran, (
        "docker compose ran without the intended -f chain — it would have "
        "implicitly discovered files in the deployment dir"
    )


def test_resolver_failure_is_distinguishable_from_an_empty_chain(harness):
    failed = harness.run(PREFLIGHT, *NO_PEERS, STUB_LOBES_RC="3")
    empty = harness.run(PREFLIGHT, *NO_PEERS, STUB_LOBES_OUT="")
    assert failed.returncode != 0
    # An empty chain is a documented, legitimate result (`lobes fleet files`
    # prints nothing for a plain deployment) — it must NOT be a failure...
    assert empty.returncode == 0, empty.stdout + empty.stderr
    # ...and the two must not read the same in a transcript.
    assert "EMPTY" in empty.stdout
    assert "not a resolver failure" in empty.stdout
    assert "EMPTY" not in failed.stdout


def test_missing_resolver_refuses_the_operation(harness):
    (harness.bin / "lobes").unlink()
    proc = harness.run(STOP, APPLY, *NO_PEERS)
    assert proc.returncode != 0
    assert "not on PATH" in proc.stdout
    assert not harness.compose_ran


def test_preflight_never_mutates_even_with_a_good_chain(harness):
    proc = harness.run(PREFLIGHT, *NO_PEERS, STUB_LOBES_OUT=CHAIN)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not harness.compose_ran
    assert "nothing was mutated" in proc.stdout


def test_stop_dry_run_never_mutates(harness):
    proc = harness.run(STOP, *NO_PEERS, STUB_LOBES_OUT=CHAIN)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not harness.compose_ran
    assert "DRY RUN" in proc.stdout
    assert "docker-compose.shape.yml" in proc.stdout


# ---------------------------------------------------------------------------
# Qodo finding 1 — the peer set is configuration, not a code default.
# ---------------------------------------------------------------------------

PEER_SOURCE = "peer source"


def test_peer_set_comes_from_the_cli_flag(harness):
    proc = harness.run(PREFLIGHT, PEER_FLAG, "a@a", PEER_FLAG, "b@b")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"{PEER_SOURCE}    : --peer flag(s)" in proc.stdout
    assert "peers          : a@a b@b" in proc.stdout


def test_peer_set_comes_from_the_environment(harness):
    proc = harness.run(PREFLIGHT, LOBES_SPIKE_PEERS="a@a, b@b")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "$LOBES_SPIKE_PEERS" in proc.stdout
    assert "peers          : a@a b@b" in proc.stdout


def test_cli_peer_flag_beats_the_environment(harness):
    proc = harness.run(PREFLIGHT, PEER_FLAG, "z@z", LOBES_SPIKE_PEERS="a@a")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "peers          : z@z" in proc.stdout
    assert "--peer flag(s)" in proc.stdout


def test_fallback_default_is_labelled_as_a_fallback(harness):
    proc = harness.run(PREFLIGHT)
    assert "built-in fallback default" in proc.stdout, (
        "the transcript must say the peer addresses came from a code default "
        "rather than from configuration"
    )
    assert "thor@thor orin@orin" in proc.stdout


def test_peer_none_reads_no_peers(harness):
    proc = harness.run(PREFLIGHT, *NO_PEERS)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "(none)" in proc.stdout
    assert "ssh" not in harness.log.read_text()


def test_help_documents_the_peer_configuration_and_the_override():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    assert "LOBES_SPIKE_PEERS" in proc.stdout
    assert ALLOW_UNREACHABLE in proc.stdout
