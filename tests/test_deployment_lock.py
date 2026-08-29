"""The committed deployment lock writer (deployment-lock-per-box plan, t6).

Every test here is PURE: no GPU, no docker, no network, no host detection.
Profile/shape resolution is a pure function of ``(profile, shape)`` (the
pattern ``tests/test_profile_goldens.py`` established), and the lock writer
itself only ever sees a plain ``Mapping`` plus a ``tmp_path``.

The security-critical property under test is that the lock's contents are an
**allowlist** of rendered knob keys, never a denylist-filtered copy of a
deployed ``.env``. A denylist silently ships the next secret key someone adds;
an allowlist structurally cannot, and
:func:`test_an_invented_secret_key_is_excluded_with_no_code_naming_it` /
:func:`test_injecting_any_new_key_never_changes_the_lock` are the two tests
that would fail against a denylist implementation.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import profile_env
from lobes.profiles.shape_render import render_shape
from lobes.profiles.shapes import resolve_shape
from lobes.runtime._env import read_env_file
from lobes.runtime._lock import (
    LOCK_FILENAME,
    SCHEMA_VERSION,
    DeploymentLock,
    allowlist_env,
    build_lock,
    file_digest,
    is_excluded,
    load_lock,
    lock_keys,
    lock_path,
    lock_toml,
    write_lock,
)
from tests.goldens.regen import shape_golden_pairs, shape_golden_path

_GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"

# A deployed .env carries far more than the rendered knobs: the operator-typed
# state lobes/runtime/_compose.py's MERGE_ONLY_FILES docstring names (the
# inbound bearer key, every *_PEER_*, COMPOSE_PROFILES, HF_TOKEN) lives in the
# very same file. None of it may reach a committed lock.
_OPERATOR_TYPED = {
    "GATEWAY_API_KEY": "sk-live-do-not-commit-me",
    "HF_TOKEN": "hf_do_not_commit_me_either",
    "PRIMARY_PEER_ORIGIN": "http://spark-f8a9.tailnet.example:8080",
    "PRIMARY_PEER_API_KEY": "sk-peer-copy",
    "PRIMARY_PEER_ORIGINS": "http://a:8080,http://b:8080",
    "PRIMARY_PEER_API_KEYS": "sk-a,sk-b",
    "WORKER_PEER_ORIGIN": "http://thor.tailnet.example:8080",
    "GATEWAY_SELF_ORIGIN": "http://spark-f8a9.tailnet.example:8080",
    "COMPOSE_PROFILES": "llamacpp,muse",
}


def _golden_env(name: str) -> dict[str, str]:
    """A committed profile golden, parsed with the deployment ``.env`` reader."""
    return read_env_file(_GOLDENS_DIR / f"{name}.env")


def _deployed_env(name: str) -> dict[str, str]:
    """A plausible deployed ``.env``: the rendered knobs plus operator secrets."""
    return {**_golden_env(name), **_OPERATOR_TYPED}


# --- criterion 1: no secret ever enters the lock -----------------------------


@pytest.mark.parametrize("card", builtin_names())
def test_operator_typed_state_never_enters_the_lock(card: str) -> None:
    """A .env carrying GATEWAY_API_KEY / HF_TOKEN / *_PEER_ORIGIN yields none of them."""
    lock = build_lock(variation="test-variation", env=_deployed_env(card))
    for key in _OPERATOR_TYPED:
        assert key not in lock.env, f"{key} leaked into the lock's env table"


@pytest.mark.parametrize("card", builtin_names())
def test_no_secret_value_appears_anywhere_in_the_rendered_toml(card: str) -> None:
    """Not just absent as a KEY — the VALUE must not appear in the file at all."""
    text = lock_toml(build_lock(variation="test-variation", env=_deployed_env(card)))
    for key, value in _OPERATOR_TYPED.items():
        assert key not in text, f"{key} leaked into the rendered lock"
        assert value not in text, f"the value of {key} leaked into the rendered lock"


def test_the_allowlist_itself_names_no_secret_shaped_key() -> None:
    """Belt and braces: the derived key set carries no credential-shaped name."""
    for key in lock_keys():
        assert "API_KEY" not in key
        assert "TOKEN" not in key
        assert "PEER" not in key
        assert "SECRET" not in key
        assert "PASSWORD" not in key


# --- criterion 2: an ALLOWLIST, provably — not a denylist --------------------


def test_an_invented_secret_key_is_excluded_with_no_code_naming_it() -> None:
    """A brand-new secret key stays out with NO change to the writer.

    ``SOMETHING_NEW_API_KEY`` is invented here and appears nowhere in
    ``lobes/runtime/_lock.py`` — asserted below, so this test cannot be
    satisfied by adding it to a denylist.
    """
    env = {**_golden_env("spark"), "SOMETHING_NEW_API_KEY": "sk-tomorrows-secret"}
    lock = build_lock(variation="test-variation", env=env)
    assert "SOMETHING_NEW_API_KEY" not in lock.env
    assert "sk-tomorrows-secret" not in lock_toml(lock)

    source = Path(__file__).resolve().parents[1] / "lobes" / "runtime" / "_lock.py"
    assert "SOMETHING_NEW" not in source.read_text(encoding="utf-8")


def test_injecting_any_new_key_never_changes_the_lock() -> None:
    """The definitive allowlist property, and the one a denylist fails.

    Filtering is closed over its input: adding ANY key the renderer does not
    itself produce leaves the lock byte-identical. A denylist can only ever
    enumerate the keys it already knows about, so an unrecognised
    ``*_TOKEN``/``*_ORIGIN``/bare-name key would change this output.
    """
    base_env = _golden_env("spark")
    baseline = lock_toml(build_lock(variation="v", env=base_env))
    intruders = {
        "SOMETHING_NEW_API_KEY": "sk-1",
        "TOTALLY_UNRELATED": "value",
        "AWS_SESSION_TOKEN": "sk-2",
        "primary_model": "lowercase-is-a-different-key",
        "NEXT_QUARTERS_CREDENTIAL": "sk-3",
        "PRIMARY_PEER_ORIGIN_V2": "http://peer:8080",
        "": "empty-key",
    }
    for key, value in intruders.items():
        got = lock_toml(build_lock(variation="v", env={**base_env, key: value}))
        assert got == baseline, f"injecting {key!r} changed the lock — not an allowlist"


def test_allowlist_env_returns_only_allowlisted_keys() -> None:
    env = _deployed_env("thor")
    kept = allowlist_env(env)
    assert set(kept) == set(env) & lock_keys()
    assert all(kept[key] == env[key] for key in kept)


def test_lock_keys_covers_everything_the_renderer_emits() -> None:
    """The allowlist is DERIVED from ``profile_env``, so it cannot fall behind it.

    Every key any built-in profile — or any (shape, card) render — actually
    produces is in the allowlist, apart from the deliberate exclusions
    (:data:`~lobes.runtime._lock.EXCLUDED_RENDERED_KEYS`, i.e.
    ``COMPOSE_PROFILES``, which ``_compose.MERGE_ONLY_FILES`` names as
    operator-typed state a template can never regenerate). Adding a role, a
    knob or a card-level ``host_env`` key without teaching the lock about it
    fails HERE rather than silently dropping it from every future lock.
    """
    emitted: set[str] = set()
    for card in builtin_names():
        emitted |= set(profile_env(resolve_profile(card)))
    for shape_name, card in shape_golden_pairs():
        emitted |= set(render_shape(resolve_shape(shape_name), resolve_profile(card)).env)
    missing = {key for key in emitted if not is_excluded(key)} - lock_keys()
    assert not missing, f"rendered keys the lock would silently drop: {sorted(missing)}"


# --- criterion 3: the filename sits outside the .env family ------------------


def test_lock_filename_matches_neither_gitignore_rule() -> None:
    """Positional rule: a ``.env`` SUFFIX is ignored, a ``.env.`` PREFIX allowed.

    The lock is deliberately NEITHER — an env-shaped committed file is exactly
    where someone would paste a real ``.env`` and blank a few lines by hand.
    """
    assert LOCK_FILENAME == "deployment.lock.toml"
    assert not LOCK_FILENAME.endswith(".env")  # not ignored by the suffix rule
    assert not LOCK_FILENAME.startswith(".env")  # not blessed by the prefix rule
    assert ".env" not in LOCK_FILENAME
    assert LOCK_FILENAME.endswith(".toml")  # the house format


def test_lock_path_joins_the_filename_onto_a_directory(tmp_path: Path) -> None:
    assert lock_path(tmp_path) == tmp_path / LOCK_FILENAME


# --- criterion 4: the lock and the goldens cannot silently disagree ----------


@pytest.mark.parametrize("card", builtin_names())
def test_lock_agrees_with_the_profile_golden_on_every_shared_key(card: str) -> None:
    golden = _golden_env(card)
    lock = build_lock(variation=card, env=_deployed_env(card), profile=card)
    shared = set(lock.env) & set(golden)
    assert shared, f"the lock and tests/goldens/{card}.env share no key at all"
    for key in shared:
        assert lock.env[key] == golden[key]
    # Every rendered knob the golden carries survives into the lock; only the
    # deliberate exclusions may be missing.
    assert {key for key in golden if not is_excluded(key)} - set(lock.env) == set()


@pytest.mark.parametrize("shape_name,card", shape_golden_pairs())
def test_lock_agrees_with_the_shape_golden_on_every_shared_key(shape_name: str, card: str) -> None:
    golden = read_env_file(shape_golden_path(shape_name, card))
    env = {**golden, **_OPERATOR_TYPED}
    lock = build_lock(variation=f"{card}-{shape_name}", env=env, profile=card, shape=shape_name)
    for key in set(lock.env) & set(golden):
        assert lock.env[key] == golden[key]
    assert {key for key in golden if not is_excluded(key)} - set(lock.env) == set()


# --- the file format -------------------------------------------------------


def test_lock_toml_round_trips_through_tomllib() -> None:
    """Values with quotes and backslashes survive — PRIMARY_SPECULATIVE_CONFIG does both."""
    golden = read_env_file(shape_golden_path("spark-lobe", "spark"))
    assert any("\\" in value or '"' in value for value in golden.values()), (
        "this test needs a golden carrying escape-worthy values; "
        "spark-lobe__spark.env no longer has one"
    )
    lock = build_lock(
        variation="spark-lobe",
        env={**golden, **_OPERATOR_TYPED},
        profile="spark",
        shape="spark-lobe",
        lobes_version="0.0.0",
        evidence="docs/evidence/2026-08-25-accept-spark-lobe-dspark-render.txt",
    )
    parsed = tomllib.loads(lock_toml(lock))
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["variation"]["id"] == "spark-lobe"
    assert parsed["variation"]["profile"] == "spark"
    assert parsed["variation"]["shape"] == "spark-lobe"
    assert parsed["variation"]["lobes_version"] == "0.0.0"
    assert parsed["env"] == dict(lock.env)


def test_lock_toml_is_deterministic_and_sorted() -> None:
    env = _deployed_env("orin")
    first = lock_toml(build_lock(variation="orin", env=env))
    second = lock_toml(build_lock(variation="orin", env=dict(reversed(list(env.items())))))
    assert first == second
    keys = [line.split(" = ", 1)[0] for line in first.splitlines() if " = " in line and "_" in line]
    env_keys = [key for key in keys if key in lock_keys()]
    assert env_keys == sorted(env_keys)


def test_load_lock_round_trips_a_written_lock(tmp_path: Path) -> None:
    lock = build_lock(
        variation="thor-worker",
        env=_deployed_env("thor"),
        profile="thor",
        shape="thor-worker",
        lobes_version="0.67.0",
        files={"docker-compose.yml": "sha256:" + "ab" * 32},
        evidence=None,
    )
    written = write_lock(tmp_path, lock)
    assert written == tmp_path / LOCK_FILENAME
    back = load_lock(written)
    assert back == lock
    assert isinstance(back, DeploymentLock)
    assert back.evidence is None


def test_write_lock_refuses_to_write_a_lock_carrying_a_secret(tmp_path: Path) -> None:
    """A hand-built DeploymentLock bypassing build_lock is still gated at write time."""
    rogue = DeploymentLock(
        variation="rogue",
        env={"GATEWAY_API_KEY": "sk-oops"},
        profile=None,
        shape=None,
        lobes_version=None,
        files={},
        evidence=None,
    )
    with pytest.raises(ModelGearError, match="GATEWAY_API_KEY") as excinfo:
        write_lock(tmp_path, rogue)
    assert excinfo.value.code == EXIT_USER_ERROR
    assert not (tmp_path / LOCK_FILENAME).exists()


def test_lock_header_marks_the_file_generated() -> None:
    text = lock_toml(build_lock(variation="v", env=_golden_env("spark")))
    assert text.startswith("#")
    assert "lobes" in text.splitlines()[0]


def test_evidence_and_files_are_optional_and_omitted_when_empty() -> None:
    text = lock_toml(build_lock(variation="v", env=_golden_env("spark")))
    parsed = tomllib.loads(text)
    assert "evidence" not in parsed["variation"]
    assert parsed.get("files", {}) == {}


def test_variation_id_is_an_explicit_parameter_not_detected() -> None:
    """t2 owns identity resolution; the writer only ever records what it is given."""
    lock = build_lock(variation="whatever-the-caller-says", env=_golden_env("base"))
    assert lock.variation == "whatever-the-caller-says"
    source = Path(__file__).resolve().parents[1] / "lobes" / "runtime" / "_lock.py"
    text = source.read_text(encoding="utf-8")
    assert "nvidia-smi" not in text
    assert "socket" not in text
    assert "gethostname" not in text


def test_file_digest_is_content_addressed(tmp_path: Path) -> None:
    one = tmp_path / "docker-compose.yml"
    two = tmp_path / "copy.yml"
    one.write_bytes(b"services: {}\n")
    two.write_bytes(b"services: {}\n")
    assert file_digest(one) == file_digest(two)
    assert file_digest(one).startswith("sha256:")
    two.write_bytes(b"services: {gateway: {}}\n")
    assert file_digest(one) != file_digest(two)


def test_wiring_urls_never_enter_the_lock() -> None:
    """A ``*_BASE_URL`` the renderer DOES emit is still kept out.

    It is a wiring fact, and the same key is retargetable by hand at another
    box — which would put an internal origin into a committed file. A restore
    re-renders it from the shape, so nothing is lost.
    """
    golden = read_env_file(shape_golden_path("thor-worker", "thor"))
    assert "WORKER_BASE_URL" in golden, "this test needs a golden that renders a wiring URL"
    env = {**golden, "WORKER_BASE_URL": "http://thor.tailnet.example:8080"}
    lock = build_lock(variation="thor-worker", env=env)
    assert "WORKER_BASE_URL" not in lock.env
    assert "tailnet" not in lock_toml(lock)
    assert is_excluded("WORKER_BASE_URL")
    assert is_excluded("PRIMARY_URL")


# --- PR #223 review: a malformed lock is untrusted input, not a crash --------

#: ``(label, text)`` pairs — every one of these used to escape ``load_lock`` as
#: a ``tomllib``/``TypeError``/``AttributeError`` traceback out of both
#: ``lobes init --from-lock`` and ``lobes doctor``.
_MALFORMED_LOCKS = [
    ("not-toml", "this is not toml ][\n"),
    ("bad-utf8", None),  # written as bytes below
    ("variation-not-a-table", 'schema_version = 1\nvariation = "spark"\n'),
    ("env-not-a-table", 'schema_version = 1\nenv = "PRIMARY_MODEL=x"\n'),
    ("files-not-a-table", "schema_version = 1\nfiles = 3\n"),
    ("env-value-not-a-string", "schema_version = 1\n[env]\nPRIMARY_MAX_MODEL_LEN = 4096\n"),
    ("files-value-not-a-string", "schema_version = 1\n[files]\n'a.yml' = true\n"),
    ("variation-id-not-a-string", "schema_version = 1\n[variation]\nid = 12\n"),
    ("profile-not-a-string", 'schema_version = 1\n[variation]\nid = "v"\nprofile = 7\n'),
]


@pytest.mark.parametrize("label,text", _MALFORMED_LOCKS, ids=[case[0] for case in _MALFORMED_LOCKS])
def test_a_malformed_lock_raises_a_controlled_user_error(
    tmp_path: Path, label: str, text: str | None
) -> None:
    path = tmp_path / LOCK_FILENAME
    if text is None:
        path.write_bytes(b"schema_version = 1\n[env]\nA = \xff\xfe\n")
    else:
        path.write_text(text, encoding="utf-8")
    with pytest.raises(ModelGearError) as excinfo:
        load_lock(path)
    assert excinfo.value.code == EXIT_USER_ERROR
    assert LOCK_FILENAME in str(excinfo.value)
    assert excinfo.value.remediation


def test_a_well_formed_lock_still_loads(tmp_path: Path) -> None:
    """The validation narrows nothing a real capture produces."""
    lock = build_lock(
        variation="spark",
        env=_golden_env("spark"),
        profile="spark",
        shape="machine-as-brain",
        lobes_version="0.68.0",
        files={"docker-compose.yml": "sha256:" + "0" * 64},
    )
    write_lock(tmp_path, lock)
    assert load_lock(lock_path(tmp_path)) == lock
