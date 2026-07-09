"""Tests for the opt-in vllm-minor service and its gateway routing (t9).

Two concerns:
1. Gateway routing — with MINOR_BASE_URL / MINOR_SERVED_NAME configured, a
   ``minor`` Backend is added and ``resolve_model`` / ``order_backends`` route
   correctly; without those env vars the table is exactly as today.
2. Compose template — the fleet compose has a ``vllm-minor`` service under the
   ``minor`` profile, with ``--language-model-only`` and no ``--quantization``
   flag (bf16 serving).
"""

from __future__ import annotations

import pathlib

import yaml

from lobes.gateway._config import build_config
from lobes.gateway._routing import order_backends, resolve_model

_MINOR_SERVED = "Qwen/Qwen3.5-4B"
_PRIMARY_SERVED = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"

# ---------------------------------------------------------------------------
# Gateway routing tests (written before the implementation — TDD)
# ---------------------------------------------------------------------------


def test_minor_backend_not_added_by_default() -> None:
    """Without MINOR_BASE_URL / MINOR_SERVED_NAME, the routing table is unchanged."""
    table, _ = build_config({})
    names = [b.name for b in table.backends]
    assert "minor" not in names
    assert len(table.backends) == 1  # only primary — existing invariant must hold


def test_minor_backend_added_when_minor_url_set() -> None:
    """MINOR_BASE_URL alone triggers the minor backend with the default served name."""
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    names = [b.name for b in table.backends]
    assert "minor" in names
    minor = next(b for b in table.backends if b.name == "minor")
    assert minor.served_name == _MINOR_SERVED
    assert minor.base_url == "http://vllm-minor:8000"
    assert minor.task == "generate"


def test_minor_backend_not_added_when_only_minor_served_name_set() -> None:
    """MINOR_SERVED_NAME alone does NOT wire the minor backend — a served name
    with no URL describes a model, not a reachable backend."""
    table, _ = build_config({"MINOR_SERVED_NAME": _MINOR_SERVED})
    names = [b.name for b in table.backends]
    assert "minor" not in names
    assert names == ["primary"]


def test_minor_backend_added_when_both_minor_url_and_served_name_set() -> None:
    """MINOR_BASE_URL wires the backend; MINOR_SERVED_NAME alongside it just
    customises the served name (mirrors the default-name case above)."""
    table, _ = build_config(
        {"MINOR_BASE_URL": "http://vllm-minor:8000", "MINOR_SERVED_NAME": _MINOR_SERVED}
    )
    names = [b.name for b in table.backends]
    assert "minor" in names
    minor = next(b for b in table.backends if b.name == "minor")
    assert minor.base_url == "http://vllm-minor:8000"
    assert minor.served_name == _MINOR_SERVED


def test_minor_backend_url_stripped() -> None:
    """Trailing slashes are stripped from MINOR_BASE_URL (URL normalisation)."""
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000/"})
    minor = next(b for b in table.backends if b.name == "minor")
    assert minor.base_url == "http://vllm-minor:8000"


def test_resolve_model_routes_minor_served_name_to_itself() -> None:
    """resolve_model returns the minor served name when a minor backend is wired."""
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    assert resolve_model(table, _MINOR_SERVED) == _MINOR_SERVED


def test_resolve_model_primary_unaffected_when_minor_present() -> None:
    """The primary's served name still resolves correctly alongside the minor."""
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    assert resolve_model(table, _PRIMARY_SERVED) == _PRIMARY_SERVED


def test_order_backends_minor_is_owner_with_no_failover() -> None:
    """order_backends: minor is the sole entry for its own served name — INVERTED
    for issue #91 ("advertised implies reachable"): minor no longer fails over
    to primary (or anywhere else). A request naming the minor served name is
    attempted at minor, once, full stop."""
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    result = order_backends(table, _MINOR_SERVED)
    names = [b.name for b in result]
    assert names == ["minor"]
    assert "primary" not in names  # no cross-backend failover, ever


def test_primary_never_failovers_to_minor_when_minor_present() -> None:
    """INVERTED for issue #91: even with minor wired, order_backends for the
    primary served name returns primary alone — minor is never a failover
    candidate for a request that named the primary model."""
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    result = order_backends(table, _PRIMARY_SERVED)
    names = [b.name for b in result]
    assert names == ["primary"]
    assert "minor" not in names


def test_minor_does_not_pollute_embed_failover_chain() -> None:
    """The embed backend must NOT fall over to minor (different task families)."""
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "EMBED_URL": "http://vllm-embed:8000",
            "EMBED_SERVED_NAME": "Qwen/Qwen3-Embedding-0.6B",
        }
    )
    embed_result = order_backends(table, "Qwen/Qwen3-Embedding-0.6B")
    assert all(b.task == "embed" for b in embed_result)
    assert not any(b.name == "minor" for b in embed_result)


def test_existing_gateway_tests_invariant_unchanged() -> None:
    """No-env baseline: exactly one backend (primary) — regression guard for t9."""
    table, _ = build_config({})
    assert len(table.backends) == 1
    assert table.backends[0].name == "primary"


# ---------------------------------------------------------------------------
# Compose template assertion tests
# ---------------------------------------------------------------------------

_COMPOSE_PATH = (
    pathlib.Path(__file__).parent.parent / "lobes" / "templates" / "fleet" / "docker-compose.yml"
)


def _load_compose() -> dict:
    with _COMPOSE_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_fleet_compose_has_vllm_minor_service() -> None:
    """The fleet compose template defines a vllm-minor service."""
    compose = _load_compose()
    assert "vllm-minor" in compose["services"], "vllm-minor service missing from fleet compose"


def test_vllm_minor_has_minor_profile() -> None:
    """vllm-minor is opt-in via the 'minor' compose profile."""
    svc = _load_compose()["services"]["vllm-minor"]
    profiles = svc.get("profiles", [])
    assert "minor" in profiles, f"expected 'minor' in profiles, got {profiles!r}"


def test_vllm_minor_has_language_model_only() -> None:
    """vllm-minor command includes --language-model-only (drops the ViT tower)."""
    svc = _load_compose()["services"]["vllm-minor"]
    cmd = [str(c) for c in svc.get("command", [])]
    assert "--language-model-only" in cmd, "--language-model-only missing from vllm-minor command"


def test_vllm_minor_has_no_quantization_flag() -> None:
    """vllm-minor must NOT have --quantization (bf16 = native precision, no quant)."""
    svc = _load_compose()["services"]["vllm-minor"]
    cmd = [str(c) for c in svc.get("command", [])]
    assert not any(
        c.startswith("--quantization") for c in cmd
    ), "--quantization must not appear in vllm-minor command (bf16 serving)"


def test_vllm_minor_container_name() -> None:
    """vllm-minor container is named model-gear-vllm-minor."""
    svc = _load_compose()["services"]["vllm-minor"]
    assert svc.get("container_name") == "model-gear-vllm-minor"


def test_vllm_minor_has_healthcheck() -> None:
    """vllm-minor has a healthcheck block (same pattern as primary)."""
    svc = _load_compose()["services"]["vllm-minor"]
    assert "healthcheck" in svc, "vllm-minor is missing a healthcheck"


def test_vllm_minor_default_fleet_unchanged() -> None:
    """Without the 'minor' profile, the default services are primary/embed/rerank/gateway."""
    compose = _load_compose()
    svcs = compose["services"]
    # The four always-on services must remain (no profiles field = always on).
    for name in ("vllm-primary", "vllm-embed", "vllm-rerank", "gateway"):
        svc = svcs[name]
        assert (
            "profiles" not in svc
        ), f"{name} must NOT have a profiles key (it must always start by default)"


# ---------------------------------------------------------------------------
# vllm-middle compose template assertion tests (t3)
# ---------------------------------------------------------------------------


def test_fleet_compose_has_vllm_middle_service() -> None:
    """The fleet compose template defines a vllm-middle service."""
    compose = _load_compose()
    assert "vllm-middle" in compose["services"], "vllm-middle service missing from fleet compose"


def test_vllm_middle_has_middle_profile() -> None:
    """vllm-middle is opt-in via the 'middle' compose profile (mirrors vllm-minor pattern)."""
    svc = _load_compose()["services"]["vllm-middle"]
    profiles = svc.get("profiles", [])
    assert "middle" in profiles, f"expected 'middle' in profiles, got {profiles!r}"


def test_vllm_middle_has_quantization_flag() -> None:
    """vllm-middle must have --quantization=modelopt_fp4 (NVFP4 checkpoint)."""
    svc = _load_compose()["services"]["vllm-middle"]
    cmd = [str(c) for c in svc.get("command", [])]
    assert any(
        c.startswith("--quantization") for c in cmd
    ), "--quantization missing from vllm-middle command (NVFP4 must be served quantised)"


def test_vllm_middle_wired_to_middle_env_vars() -> None:
    """vllm-middle command references MIDDLE_MAX_MODEL_LEN and MIDDLE_GPU_MEM_UTIL."""
    svc = _load_compose()["services"]["vllm-middle"]
    cmd = " ".join(str(c) for c in svc.get("command", []))
    assert (
        "MIDDLE_MAX_MODEL_LEN" in cmd
    ), "MIDDLE_MAX_MODEL_LEN not referenced in vllm-middle command"
    assert "MIDDLE_GPU_MEM_UTIL" in cmd, "MIDDLE_GPU_MEM_UTIL not referenced in vllm-middle command"


def test_vllm_middle_container_name() -> None:
    """vllm-middle container is named model-gear-vllm-middle."""
    svc = _load_compose()["services"]["vllm-middle"]
    assert svc.get("container_name") == "model-gear-vllm-middle"


def test_vllm_middle_has_healthcheck() -> None:
    """vllm-middle has a healthcheck block (same pattern as primary/minor)."""
    svc = _load_compose()["services"]["vllm-middle"]
    assert "healthcheck" in svc, "vllm-middle is missing a healthcheck"


def test_primary_max_model_len_default_is_128k() -> None:
    """PRIMARY_MAX_MODEL_LEN default is 131072 (128K, full native context — the
    27B KV cache is util-bound not context-bound, so the always-on Gemma
    multimodal gear frees co-resident headroom by trimming to 32K instead)."""
    svc = _load_compose()["services"]["vllm-primary"]
    cmd = [str(c) for c in svc.get("command", [])]
    max_len_flag = next((c for c in cmd if c.startswith("--max-model-len=")), None)
    assert max_len_flag is not None, "--max-model-len flag missing from vllm-primary command"
    # The default (after :-) must be 131072, not the old 65536 trim.
    assert (
        "131072" in max_len_flag
    ), f"PRIMARY_MAX_MODEL_LEN default should be 131072 (128K), got: {max_len_flag!r}"
    assert (
        "262144" not in max_len_flag
    ), "PRIMARY_MAX_MODEL_LEN default must NOT be 262144 — fleet uses 128K (duo budget)"


def test_gateway_has_middle_env_vars() -> None:
    """Gateway service passes MIDDLE_BASE_URL and MIDDLE_SERVED_NAME to the container."""
    gw = _load_compose()["services"]["gateway"]
    env_lines = [str(e) for e in gw.get("environment", [])]
    assert any(
        "MIDDLE_BASE_URL" in e for e in env_lines
    ), "MIDDLE_BASE_URL missing from gateway environment"
    assert any(
        "MIDDLE_SERVED_NAME" in e for e in env_lines
    ), "MIDDLE_SERVED_NAME missing from gateway environment"
