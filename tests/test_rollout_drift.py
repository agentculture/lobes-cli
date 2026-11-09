"""Compose drift + deployment-lock detection (lobes.rollout._drift)."""

from pathlib import Path

from lobes.rollout import _drift as d

TEMPLATE = """\
services:
  vllm-primary:
    environment:
      - NOT_GATEWAY=1
  gateway:
    container_name: model-gear-gateway
    # comment - FAKE=1
    environment:
      - GATEWAY_PORT=8000
      - AUDIO_URL=${AUDIO_URL:-}
      # - COMMENTED=${COMMENTED:-}
      - STT_HE_BASE_URL=${STT_HE_BASE_URL:-}
      - STT_HE_FEASIBLE=${STT_HE_FEASIBLE:-}
      - STT_EN_BASE_URL=${STT_EN_BASE_URL:-}
      - STT_EN_FEASIBLE=${STT_EN_FEASIBLE:-}
    ports:
      - "8000:8000"
"""

SPARK2 = """\
services:
  gateway:
    environment:
      - GATEWAY_PORT=8000
      - AUDIO_URL=${AUDIO_URL:-}
      - STT_HE_BASE_URL=${STT_HE_BASE_URL:-}
      - STT_HE_FEASIBLE=${STT_HE_FEASIBLE:-}
"""

OVERRIDE = """\
services:
  gateway:
    environment:
      - STT_HE_FEASIBLE=${STT_HE_FEASIBLE:-}
      - EXTRA=1
    restart: always
"""


def test_template_keys_ignore_comments_and_other_services():
    keys = d.gateway_env_keys(TEMPLATE)
    assert "NOT_GATEWAY" not in keys and "COMMENTED" not in keys
    assert keys[:2] == ["GATEWAY_PORT", "AUDIO_URL"]
    assert "STT_EN_BASE_URL" in keys


def test_spark2_missing_stt_en_flagged():
    r = d.detect_drift(TEMPLATE, {"docker-compose.yml": SPARK2})
    assert r.missing == ["STT_EN_BASE_URL", "STT_EN_FEASIBLE"]
    assert r.drifted


def test_override_counts_as_present():
    ov = OVERRIDE.replace("EXTRA=1", "STT_EN_BASE_URL=x\n      - STT_EN_FEASIBLE=y")
    r = d.detect_drift(TEMPLATE, {"docker-compose.yml": SPARK2, "docker-compose.override.yml": ov})
    assert r.missing == []


def test_lock_affected():
    files = {"docker-compose.yml": SPARK2, "deployment.lock.toml": "x=1"}
    assert d.detect_drift(TEMPLATE, files).lock_affected
    assert not d.detect_drift(TEMPLATE, {"docker-compose.yml": SPARK2}).lock_affected


def test_patch_override_adds_only_missing_after_existing_entry():
    p = d.patch_override(
        TEMPLATE,
        {"docker-compose.yml": SPARK2, "docker-compose.override.yml": OVERRIDE},
        now="20261109T120000Z",
    )
    assert p.backup_name == "docker-compose.override.yml.bak.20261109T120000Z"
    assert p.added == ["STT_EN_BASE_URL", "STT_EN_FEASIBLE"]
    lines = p.new_text.splitlines()
    i = lines.index("      - EXTRA=1")
    assert lines[i + 1] == "      - STT_EN_BASE_URL=${STT_EN_BASE_URL:-}"
    assert lines[i + 2] == "      - STT_EN_FEASIBLE=${STT_EN_FEASIBLE:-}"
    assert "    restart: always" in lines
    assert "+      - STT_EN_BASE_URL=${STT_EN_BASE_URL:-}" in p.diff
    assert (
        d.detect_drift(
            TEMPLATE,
            {"docker-compose.yml": SPARK2, "docker-compose.override.yml": p.new_text},
        ).missing
        == []
    )


def test_patch_override_no_override_file_creates_block():
    p = d.patch_override(TEMPLATE, {"docker-compose.yml": SPARK2}, now="T")
    assert p.new_text.startswith("services:\n  gateway:\n    environment:\n")
    assert p.new_text.count("- STT_EN") == 2
    assert p.backup_name is None


def test_patch_override_override_without_gateway_env():
    ov = "services:\n  vllm-primary:\n    restart: always\n"
    p = d.patch_override(
        TEMPLATE, {"docker-compose.yml": SPARK2, "docker-compose.override.yml": ov}, now="T"
    )
    assert "  vllm-primary:\n    restart: always\n" in p.new_text
    assert "  gateway:\n    environment:\n      - STT_EN_BASE_URL" in p.new_text


def test_patch_noop_when_nothing_missing():
    p = d.patch_override(TEMPLATE, {"docker-compose.yml": TEMPLATE}, now="T")
    assert p.added == [] and p.new_text is None and p.diff == ""


def test_real_template_parses():
    t = Path(d.__file__).parents[1] / "templates/fleet/docker-compose.yml"
    keys = d.gateway_env_keys(t.read_text())
    assert "GATEWAY_PORT" in keys and len(keys) > 20
