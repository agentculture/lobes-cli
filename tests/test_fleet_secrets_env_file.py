"""The secret env family: a second ``env_file`` entry across fleet services
(plan task t4, spec c20/h15, c21/h16).

Two mechanisms already exist in-tree for materialising a secret without an
operator ever typing one by hand: ``scripts/gen-api-key.py`` GENERATES a
bearer key with the stdlib ``secrets`` module and writes it into the
gitignored deployment ``.env``, and ``cf-tunnel.env.example`` is committed
verbatim and copied locally to the gitignored ``.cf-tunnel.env``. This task
generalises the second pattern: every fleet service that already reads
``.env`` via compose's long-form ``env_file:`` list gains a SECOND
``required: false`` entry for ``.secrets.env`` — a sibling secret file an
operator (or another generator) may drop beside ``.env``. Because the entry
is ``required: false``, a deployment with no such file is unaffected: compose
silently skips a missing optional env file, so parsing the template is
sufficient to prove byte-identical behaviour without invoking docker.

This module never eyeballs the template — every assertion parses it with
``yaml.safe_load``, exactly like the sibling ``test_fleet_template_gateway_env.py``
and ``test_compose_chain.py`` do.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"
_AUDIO_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.audio.yml"

_SECRETS_ENV_FILENAME = ".secrets.env"

# Services in the base fleet template that read the deployment .env today
# (every vllm-* generate/embed/rerank lane). ``gateway`` and
# ``llamacpp-primary`` deliberately do NOT use ``env_file`` — the gateway
# reads only scoped, non-secret keys via ``environment:`` interpolation (see
# the comment above its service block: "never the whole file via env_file"),
# and llamacpp-primary serves a local GGUF and never needs HF_TOKEN. A new
# service added to either list without updating this test is exactly the
# drift this suite exists to catch.
_EXPECTED_ENV_FILE_SERVICES = frozenset(
    {
        "vllm-primary",
        "vllm-embed",
        "vllm-embed-deep",
        "vllm-rerank",
        "vllm-minor",
        "vllm-hand",
        "vllm-middle",
        "vllm-multimodal",
        "vllm-multimodal-coder",
        "vllm-muse",
        "vllm-worker",
        "vllm-associate",
    }
)

_EXPECTED_NO_ENV_FILE_SERVICES = frozenset({"gateway", "llamacpp-primary"})

# Audio overlay services that read .env today (chatterbox, stt). ``realtime``
# and the audio overlay's ``gateway`` override deliberately stay env_file-free,
# mirroring the base gateway's scoped-environment-only pattern.
_EXPECTED_AUDIO_ENV_FILE_SERVICES = frozenset({"chatterbox", "stt"})
_EXPECTED_AUDIO_NO_ENV_FILE_SERVICES = frozenset({"realtime", "gateway"})


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _env_file_list(svc: dict) -> list:
    return svc.get("env_file", [])


def test_expected_services_present_in_fleet_template():
    """Sanity: the service inventory this test targets still matches the
    template — a service rename/removal must update this test explicitly
    rather than silently going unchecked."""
    compose = _load(_FLEET_COMPOSE)
    services = set(compose["services"])
    assert _EXPECTED_ENV_FILE_SERVICES <= services
    assert _EXPECTED_NO_ENV_FILE_SERVICES <= services


def test_every_env_file_service_reads_dot_env_first():
    """The pre-existing ``.env`` entry is untouched — same path, same
    ``required: false`` — so a deployment with no ``.secrets.env`` keeps
    reading ``.env`` exactly as before."""
    compose = _load(_FLEET_COMPOSE)
    for name in _EXPECTED_ENV_FILE_SERVICES:
        entries = _env_file_list(compose["services"][name])
        assert entries, f"{name}: expected an env_file list"
        assert entries[0] == {
            "path": ".env",
            "required": False,
        }, f"{name}: first env_file entry must stay .env/required:false"


def test_every_env_file_service_also_reads_secrets_env():
    """Every fleet service that reads ``.env`` also reads the secret sibling
    file, as a second ``required: false`` entry — verified by parsing the
    template, not by eyeballing it."""
    compose = _load(_FLEET_COMPOSE)
    for name in _EXPECTED_ENV_FILE_SERVICES:
        entries = _env_file_list(compose["services"][name])
        assert {
            "path": _SECRETS_ENV_FILENAME,
            "required": False,
        } in entries, f"{name}: missing the second (.secrets.env) env_file entry"


def test_env_file_service_has_exactly_two_entries():
    """No stray third entry, and the two are in the documented order
    (``.env`` then ``.secrets.env``) — a byte-identical, minimal diff from
    the pre-t4 template."""
    compose = _load(_FLEET_COMPOSE)
    for name in _EXPECTED_ENV_FILE_SERVICES:
        entries = _env_file_list(compose["services"][name])
        assert entries == [
            {"path": ".env", "required": False},
            {"path": _SECRETS_ENV_FILENAME, "required": False},
        ], f"{name}: unexpected env_file entries {entries!r}"


def test_services_without_env_file_stay_that_way():
    """gateway and llamacpp-primary deliberately never read the deployment
    .env wholesale; this task must not change that (a scoped-environment-only
    service picking up .secrets.env would silently widen its secret
    exposure, exactly the leak this template comment warns against)."""
    compose = _load(_FLEET_COMPOSE)
    for name in _EXPECTED_NO_ENV_FILE_SERVICES:
        assert (
            "env_file" not in compose["services"][name]
        ), f"{name}: must not gain an env_file block"


def test_no_service_reads_secrets_env_without_also_reading_dot_env():
    """A service must never read .secrets.env in isolation — the secret file
    is always a SECOND entry beside .env, never a replacement, so a
    deployment cannot end up depending on .secrets.env alone."""
    compose = _load(_FLEET_COMPOSE)
    for name, svc in compose["services"].items():
        entries = _env_file_list(svc)
        paths = {e.get("path") for e in entries if isinstance(e, dict)}
        if _SECRETS_ENV_FILENAME in paths:
            assert ".env" in paths, f"{name}: reads .secrets.env without .env"


def test_missing_secrets_env_is_a_no_op_by_construction():
    """Both entries carry required: false, which is compose's own contract
    for 'skip silently if absent' — so a deployment with no .secrets.env
    behaves byte-identically to the pre-t4 template. This is the
    parse-level equivalent of running `docker compose config` against a
    deployment dir with and without the file and diffing the output (not
    done here since CI has no docker)."""
    compose = _load(_FLEET_COMPOSE)
    for name in _EXPECTED_ENV_FILE_SERVICES:
        entries = _env_file_list(compose["services"][name])
        assert all(
            e.get("required") is False for e in entries
        ), f"{name}: every env_file entry must be required: false"


# --- audio overlay: same contract for its two .env-reading services -------


def test_audio_overlay_services_also_read_secrets_env():
    compose = _load(_AUDIO_COMPOSE)
    services = set(compose["services"])
    assert _EXPECTED_AUDIO_ENV_FILE_SERVICES <= services
    for name in _EXPECTED_AUDIO_ENV_FILE_SERVICES:
        entries = _env_file_list(compose["services"][name])
        assert entries == [
            {"path": ".env", "required": False},
            {"path": _SECRETS_ENV_FILENAME, "required": False},
        ], f"{name}: unexpected env_file entries {entries!r}"


def test_audio_overlay_scoped_services_stay_env_file_free():
    compose = _load(_AUDIO_COMPOSE)
    for name in _EXPECTED_AUDIO_NO_ENV_FILE_SERVICES:
        assert (
            "env_file" not in compose["services"][name]
        ), f"{name}: must not gain an env_file block"
