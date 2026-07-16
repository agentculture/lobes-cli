"""Tests for the 14B-class NVFP4 gear and the tier->role_hint map (issue #68, updated #69).

The 14B (nvidia/Qwen3-14B-NVFP4) was the original 'middle'/normal tier (issue #68).
Issue #69 reframed the tier vocabulary to main/minor/multimodal and demoted the 14B
to a legacy *candidate* (the Gemma 4 12B 'multimodal' gear now serves the normal slot).
This file keeps the 14B-entry coverage (still a valid generate candidate) and tracks
the post-#69 tier map; the new-vocabulary resolution is also covered in test_catalog.py.

Acceptance criteria (post-#69):
- catalog.py keeps the 14B generate gear, now role_hint='candidate' (kept, legacy)
- A tier->role_hint map exists at module level with main/minor/multimodal +
  cheap/normal/hard back-compat: normal->multimodal, hard->primary, cheap->minor
- tool_parser == infer_parser(id) and quantization is non-empty
"""

from __future__ import annotations

import pytest

from lobes.catalog import (
    SUPPORTED_MODELS,
    TIER_ROLE,
    resolve_tier,
)
from lobes.runtime._parser import infer_parser

_MIDDLE_ID = "nvidia/Qwen3-14B-NVFP4"
_MINOR_ID = "Qwen/Qwen3.5-4B"
_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"


# ---------------------------------------------------------------------------
# Middle gear catalog entry
# ---------------------------------------------------------------------------


def test_middle_gear_exists() -> None:
    """The 14B-class NVFP4 middle gear must be present in the catalog."""
    ids = [m.id for m in SUPPORTED_MODELS]
    assert _MIDDLE_ID in ids, f"{_MIDDLE_ID} not found in catalog"


def test_middle_gear_task_is_generate() -> None:
    """The middle gear must be a generate (chat/completion) model."""
    middle = next(m for m in SUPPORTED_MODELS if m.id == _MIDDLE_ID)
    assert middle.task == "generate"


def test_14b_gear_role_hint_is_candidate() -> None:
    """Post-#69 the 14B is demoted from 'middle' to a legacy 'candidate' (kept)."""
    middle = next(m for m in SUPPORTED_MODELS if m.id == _MIDDLE_ID)
    assert middle.role_hint == "candidate"


def test_middle_gear_tool_parser_matches_infer_parser() -> None:
    """tool_parser must agree with what the runtime would auto-select.

    The catalog is the single source of truth for parser values;
    lobes switch uses infer_parser to auto-select — they must match.
    """
    middle = next(m for m in SUPPORTED_MODELS if m.id == _MIDDLE_ID)
    assert infer_parser(middle.id) == middle.tool_parser


def test_middle_gear_quantization_is_nonempty() -> None:
    """Generate gears must have a non-empty quantization field."""
    middle = next(m for m in SUPPORTED_MODELS if m.id == _MIDDLE_ID)
    assert middle.quantization, f"{_MIDDLE_ID}: quantization must be non-empty for a generate gear"


def test_middle_gear_status_is_configured() -> None:
    """The 14B is a candidate not yet load-tested — status must be 'configured'."""
    middle = next(m for m in SUPPORTED_MODELS if m.id == _MIDDLE_ID)
    assert middle.status == "configured"


def test_middle_gear_has_positive_native_max_model_len() -> None:
    """lobes switch relies on native_max_model_len to clamp boot context."""
    middle = next(m for m in SUPPORTED_MODELS if m.id == _MIDDLE_ID)
    assert middle.native_max_model_len > 0


def test_middle_gear_generate_fields_are_zero_dimension_and_empty_hf_overrides() -> None:
    """Generate models must have dimension==0 and empty hf_overrides."""
    middle = next(m for m in SUPPORTED_MODELS if m.id == _MIDDLE_ID)
    assert middle.dimension == 0
    assert middle.hf_overrides == ""


def test_middle_gear_has_no_moe_backend() -> None:
    """The 14B is a dense model — moe_backend must be empty."""
    middle = next(m for m in SUPPORTED_MODELS if m.id == _MIDDLE_ID)
    assert not middle.moe_backend, f"{_MIDDLE_ID}: moe_backend must be empty for a dense model"
    assert not middle.shape.lower().startswith("moe")


def test_middle_gear_has_no_speculative_config() -> None:
    """The 14B checkpoint has no MTP draft head — speculative_config must be empty."""
    middle = next(m for m in SUPPORTED_MODELS if m.id == _MIDDLE_ID)
    assert "MTP" not in middle.id.upper()
    assert middle.speculative_config == ""


# ---------------------------------------------------------------------------
# Tier -> role_hint map
# ---------------------------------------------------------------------------


def test_tier_role_map_exists_and_has_three_tiers() -> None:
    """TIER_ROLE must be a module-level dict with cheap/normal/hard keys."""
    assert isinstance(TIER_ROLE, dict)
    assert set(TIER_ROLE.keys()) >= {"cheap", "normal", "hard"}


def test_tier_role_map_values() -> None:
    """Post-#69 back-compat values: cheap->minor / normal->multimodal / hard->primary."""
    assert TIER_ROLE["cheap"] == "minor"
    assert TIER_ROLE["normal"] == "multimodal"
    assert TIER_ROLE["hard"] == "primary"


# ---------------------------------------------------------------------------
# resolve_tier helper
# ---------------------------------------------------------------------------


def test_resolve_tier_cheap_returns_minor_gear() -> None:
    """resolve_tier('cheap') must return the 4B minor gear."""
    model = resolve_tier("cheap")
    assert model.id == _MINOR_ID
    assert model.role_hint == "minor"


def test_resolve_tier_normal_returns_multimodal_gear() -> None:
    """Post-#69 resolve_tier('normal') resolves to the Gemma 'multimodal' gear."""
    model = resolve_tier("normal")
    assert model.role_hint == "multimodal"
    assert model.task == "generate"


def test_resolve_tier_hard_returns_primary_gear() -> None:
    """resolve_tier('hard') must return the 27B primary generate gear."""
    model = resolve_tier("hard")
    assert model.role_hint == "primary"
    assert model.task == "generate"


# ---------------------------------------------------------------------------
# Capability-ROLE vocabulary (cortex / senses) over the existing backend roles
# ---------------------------------------------------------------------------


def test_tier_role_map_includes_cortex_and_senses() -> None:
    """cortex maps onto the primary backend, senses onto the multimodal backend.

    These are new capability-ROLE names layered over the EXISTING internal roles
    (primary / multimodal) — no internal service/env/container was renamed.
    """
    assert TIER_ROLE["cortex"] == "primary"
    assert TIER_ROLE["senses"] == "multimodal"


def test_resolve_tier_cortex_returns_primary_gear() -> None:
    """resolve_tier('cortex') must return the 27B primary generate gear."""
    model = resolve_tier("cortex")
    assert model.role_hint == "primary"
    assert model.task == "generate"
    assert model.id == _PRIMARY_ID


def test_resolve_tier_senses_returns_multimodal_gear() -> None:
    """resolve_tier('senses') must resolve to the Gemma 'multimodal' gear."""
    model = resolve_tier("senses")
    assert model.role_hint == "multimodal"
    assert model.task == "generate"


def test_cortex_and_senses_resolve_same_gears_as_main_and_multimodal() -> None:
    """cortex is an alias onto the primary backend (same as main/hard) and senses
    onto the multimodal backend (same as multimodal/normal): every existing alias
    keeps working and the new names just re-address the same gears."""
    assert resolve_tier("cortex").id == resolve_tier("main").id == resolve_tier("hard").id
    assert resolve_tier("senses").id == resolve_tier("multimodal").id == resolve_tier("normal").id


def test_tier_role_map_includes_muse_as_its_own_backend() -> None:
    """muse is the first capability-ROLE whose backend name IS the role name —
    there is no pre-#81 internal name to preserve for it."""
    assert TIER_ROLE["muse"] == "muse"


def test_resolve_tier_muse_returns_the_31b_gear() -> None:
    """resolve_tier('muse') must return the Gemma 4 31B muse gear."""
    model = resolve_tier("muse")
    assert model.role_hint == "muse"
    assert model.task == "generate"
    assert model.id == "nvidia/Gemma-4-31B-IT-NVFP4"


def test_tier_role_capability_order_is_ascending_with_muse() -> None:
    """tier_aliases derives ascending capability order from each role's LAST
    occurrence in TIER_ROLE — muse must land between multimodal and primary
    (minor < multimodal < muse < primary), or the upward-fallback ladder
    breaks. Pinned here against the dict's insertion order."""
    last_pos: dict[str, int] = {}
    for i, role in enumerate(TIER_ROLE.values()):
        last_pos[role] = i
    roles_asc = sorted(last_pos, key=last_pos.__getitem__)
    assert roles_asc == ["minor", "multimodal", "muse", "primary"]


def test_resolve_tier_unknown_raises_value_error() -> None:
    """resolve_tier must raise ValueError for an unknown tier name."""
    with pytest.raises(ValueError, match="unknown tier"):
        resolve_tier("ultra")


def test_resolve_tier_returns_supported_model_instance() -> None:
    """resolve_tier must return a SupportedModel, not a string."""
    from lobes.catalog import SupportedModel

    for tier in ("cheap", "normal", "hard"):
        model = resolve_tier(tier)
        assert isinstance(model, SupportedModel), f"resolve_tier({tier!r}) returned {type(model)}"
