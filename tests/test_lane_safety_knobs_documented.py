"""Doc-test alignment for the lane-safety knobs (plan lane-safety, t17).

Every ``LOBES_*`` knob in ``docs/contracts/lane-safety-runtime.md`` section 1
must be (a) documented in the fleet ``env.example`` with its off value, (b) read
somewhere under ``lobes/``, and (c) named in the operator docs. The
single-model ``env.example`` carries the compile-cache pair only.

This file does NOT re-prove behaviour. The per-feature tests that show each
off value restores the pre-change behaviour are:

| knob(s)                                             | proving test |
|-----------------------------------------------------|---|
| LOBES_BOOT_DELAY_S, _BOOT_ORDER, _BOOT_STAGGER_S    | tests/test_logwrap_hold.py |
| LOBES_START_MAX, LOBES_MEM_GATE                     | tests/test_logwrap_backoff.py |
| LOBES_VLLM_CACHE_ROOT, LOBES_COMPILE_CACHE_DIR      | tests/test_lane_safety_compose.py |
| LOBES_THERMAL_* (zones, shed, resume)               | tests/test_pressure_policy.py |
| LOBES_WATCHER_MIN_INTERVAL_S                        | tests/test_watcher.py |
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "docs" / "contracts" / "lane-safety-runtime.md"
FLEET_ENV = ROOT / "lobes" / "templates" / "fleet" / "env.example"
SINGLE_ENV = ROOT / "lobes" / "templates" / "env.example"
OPERATOR_DOCS = [
    ROOT / "docs" / "operating-a-deployment.md",
    ROOT / "docs" / "gateway-fleet.md",
    ROOT / "docs" / "machine-profiles.md",
    ROOT / "docs" / "measuring-lane-performance.md",
]

# Knobs whose contract off-value column is a dash: no off value exists.
NO_OFF_VALUE = {
    "LOBES_START_WINDOW_S",
    "LOBES_MEM_GATE_TIMEOUT_S",
    "LOBES_COMPILE_CACHE_DIR",
    "LOBES_WATCHER_MIN_INTERVAL_S",
}
SINGLE_KNOBS = {"LOBES_VLLM_CACHE_ROOT", "LOBES_COMPILE_CACHE_DIR"}


def _contract_knobs() -> dict[str, str | None]:
    """knob -> off value from the section 1 table (None when the cell is a dash)."""
    section = CONTRACT.read_text().split("## 1. Knobs", 1)[1].split("\n## 2.", 1)[0]
    out: dict[str, str | None] = {}
    for line in section.splitlines():
        m = re.match(r"\|\s*`(LOBES_[A-Z_]+)`\s*\|(.*)\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        off = cells[2]
        out[m.group(1)] = None if off in ("—", "-") else off.strip("`") if off != "empty" else ""
    return out


def _env_off_values(path: Path) -> dict[str, str]:
    text = path.read_text()
    return dict(re.findall(r"#\s+off value: (LOBES_[A-Z_]+)=(\S*)", text))


def _documented(path: Path) -> set[str]:
    text = path.read_text()
    return set(re.findall(r"^# (LOBES_[A-Z_]+)=", text, re.M))


def test_contract_table_parses() -> None:
    knobs = _contract_knobs()
    assert len(knobs) == 13
    assert {k for k, v in knobs.items() if v is None} == NO_OFF_VALUE


def test_fleet_env_example_documents_every_knob_with_its_off_value() -> None:
    knobs = _contract_knobs()
    assert _documented(FLEET_ENV) >= set(knobs)
    offs = _env_off_values(FLEET_ENV)
    for knob, off in knobs.items():
        if off is None:
            assert knob not in offs, f"{knob} has no contract off value"
        else:
            assert offs.get(knob) == off, f"{knob}: env.example off value != contract"


def test_single_env_example_documents_the_compile_cache_pair() -> None:
    knobs = _contract_knobs()
    assert _documented(SINGLE_ENV) >= SINGLE_KNOBS
    offs = _env_off_values(SINGLE_ENV)
    for knob in SINGLE_KNOBS:
        if knobs[knob] is not None:
            assert offs.get(knob) == knobs[knob]


def test_every_documented_knob_is_read_in_code() -> None:
    corpus = "\n".join(
        p.read_text(errors="ignore")
        for p in (ROOT / "lobes").rglob("*")
        if p.is_file() and p.suffix in {".py", ".sh", ".yml", ".toml", ".in"}
    )
    for knob in _contract_knobs():
        assert knob in corpus, f"{knob} is documented but not present under lobes/"


def test_operator_docs_name_every_knob() -> None:
    corpus = "\n".join(p.read_text() for p in OPERATOR_DOCS)
    for knob in _contract_knobs():
        assert knob in corpus, f"{knob} is not named in the operator docs"


def test_gateway_receives_the_thermal_keys() -> None:
    """Issue #311: the gateway has no env_file, so each thermal key needs an
    explicit passthrough or the sampler reads nothing in the container."""
    compose = (ROOT / "lobes" / "templates" / "fleet" / "docker-compose.yml").read_text()
    # The gateway service block: from its key to the next 2-space service key.
    gateway = re.split(r"\n  [a-z][a-z0-9_-]*:\n", compose.split("\n  gateway:\n", 1)[1], 1)[0]
    for key in ("LOBES_THERMAL_ZONES", "LOBES_THERMAL_SHED_C", "LOBES_THERMAL_RESUME_C"):
        assert f"- {key}=${{{key}:-}}" in gateway, key
