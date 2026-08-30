"""Every scaffold file the fleet compose bind-mounts must actually be scaffolded.

Issue #227 (task t3): the reranker chat template is baked into vllm-rerank as
a read-only bind mount (``./qwen3_reranker.jinja`` -> the container path
``--chat-template`` names), the same pattern ``mg-logwrap.sh`` already uses.
This is a general "mount-matches-scaffold" test, not a reranker-only one: it
parses the fleet compose YAML, collects every ``./<file>`` volume source that
names a file (not a directory mount like ``./logs``), and asserts each one is
either a value ``FLEET_TEMPLATES`` materialises or the tool-parser plugin's
dest name (``PLUGIN_DEST_NAME``) — so a scaffold can never silently omit a
file the compose expects to find on disk.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lobes.runtime import _compose

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"

_RERANK_TEMPLATE_MOUNT_DEST = "/usr/local/share/lobes/qwen3_reranker.jinja"


def _load_fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


def _dot_slash_file_sources(compose: dict) -> list[str]:
    """Every ``./<name>`` volume source across all services that has a file
    extension (i.e. is not a directory mount like ``./logs``)."""
    sources = []
    for service in compose["services"].values():
        for volume in service.get("volumes", []):
            if not isinstance(volume, str) or not volume.startswith("./"):
                continue
            source = volume.split(":", 1)[0]
            name = source[len("./") :]
            if Path(name).suffix:
                sources.append(name)
    return sources


def test_every_dot_slash_file_mount_is_scaffolded():
    compose = _load_fleet()
    sources = _dot_slash_file_sources(compose)
    assert sources, "expected at least one ./<file> volume mount in the fleet compose"

    scaffolded = set(_compose.FLEET_TEMPLATES.values()) | {_compose.PLUGIN_DEST_NAME}
    missing = [name for name in sources if name not in scaffolded]
    assert not missing, (
        f"compose mounts {missing} from disk but no scaffold writes them "
        "(FLEET_TEMPLATES / the plugin writer) — a fresh deployment would 404 "
        "docker's bind-mount step"
    )


def test_vllm_rerank_declares_chat_template_flag():
    compose = _load_fleet()
    command = compose["services"]["vllm-rerank"]["command"]
    assert f"--chat-template={_RERANK_TEMPLATE_MOUNT_DEST}" in command


def test_vllm_rerank_chat_template_immediately_follows_hf_overrides():
    compose = _load_fleet()
    command = compose["services"]["vllm-rerank"]["command"]
    hf_overrides_index = next(
        i
        for i, item in enumerate(command)
        if isinstance(item, str) and item.startswith("--hf-overrides=")
    )
    assert command[hf_overrides_index + 1] == f"--chat-template={_RERANK_TEMPLATE_MOUNT_DEST}"


def test_vllm_rerank_jinja_mount_is_read_only():
    compose = _load_fleet()
    volumes = compose["services"]["vllm-rerank"]["volumes"]
    jinja_mounts = [
        v for v in volumes if isinstance(v, str) and v.startswith("./qwen3_reranker.jinja:")
    ]
    assert len(jinja_mounts) == 1
    assert jinja_mounts[0] == f"./qwen3_reranker.jinja:{_RERANK_TEMPLATE_MOUNT_DEST}:ro"


def test_no_other_service_gains_chat_template_or_rerank_env_key():
    compose = _load_fleet()
    for name, service in compose["services"].items():
        if name == "vllm-rerank":
            continue
        command = service.get("command") or []
        for item in command:
            if isinstance(item, str):
                assert "qwen3_reranker.jinja" not in item
        for volume in service.get("volumes", []):
            if isinstance(volume, str):
                assert "qwen3_reranker.jinja" not in volume
