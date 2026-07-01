"""Smoke tests for the Gemma 4 12B "duo" integration (issue #69).

Two layers — the split is the CI boundary:

Layer A — Config/routing (always runs, no live model needed):
  * model=main / model=multimodal / model=minor resolve to the correct gear
    served names via the gateway routing helpers (build_config + resolve_model).
  * resolve_tier() from the catalog agrees on role_hint / model id.
  * The legacy 14B (nvidia/Qwen3-14B-NVFP4) sits in the catalog as
    role_hint="candidate" and is placed behind the compose "middle" profile
    (not default-on), while vllm-multimodal has no profile and is default-on.

Layer B — Live gateway (skipped unless LOBES_SMOKE_BASE_URL is set, i.e. on
the DGX Spark where t7 validates the physical hardware):
  * model=main: a plain text prompt returns non-empty assistant content.
  * model=multimodal (Gemma 4 12B): an image+text chat request using the
    OpenAI content-parts shape (type=image_url) returns valid output.
  * model=multimodal: an audio+text chat request using the OpenAI
    content-parts shape (type=input_audio / format=wav) returns valid output.

All live calls use stdlib urllib only (mirroring lobes/assess.py).
"""

from __future__ import annotations

import base64
import json
import os
import struct
import urllib.request
from pathlib import Path

import pytest
import yaml

from lobes.catalog import SUPPORTED_MODELS, resolve_tier
from lobes.gateway._config import (
    _DEFAULT_MINOR,
    _DEFAULT_MULTIMODAL,
    _DEFAULT_PRIMARY,
    build_config,
)
from lobes.gateway._routing import resolve_model

# ---------------------------------------------------------------------------
# Shared paths and constants
# ---------------------------------------------------------------------------

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"

_LEGACY_14B_ID = "nvidia/Qwen3-14B-NVFP4"
# "Support both" (docs/vllm-nightly-migration.md §7, 2026-07-02): the default
# "multimodal" gear is now the NVFP4 base it-model with native MTP wired
# (coolthor/…); the coder fine-tune (sakamakismile/…) is kept but demoted to a
# candidate — see tests/test_catalog.py for the dedicated coder coverage.
_GEMMA_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_MINOR_ID = "Qwen/Qwen3.5-4B"
_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"

# Tiny 1×1 RGB PNG (valid PNG, useful as a minimal image payload).
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQI12P4"
    "z8AAAAACAAHiIbwzAAAAAElFTkSuQmCC"
)


# Tiny silent WAV: 8 kHz, 8-bit, mono, 1 sample.  Built with struct so the
# bytes are self-documenting and won't drift from the actual format.
def _build_tiny_wav_b64() -> str:
    num_channels, sample_rate, bits = 1, 8000, 8
    block_align = num_channels * bits // 8
    byte_rate = sample_rate * block_align
    audio_data = b"\x80\x00"  # 1 silence sample + WAV even-length pad
    fmt = struct.pack("<HHIIHH", 1, num_channels, sample_rate, byte_rate, block_align, bits)
    riff_size = 4 + (8 + 16) + (8 + len(audio_data))
    wav = (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", 1)
        + audio_data
    )
    return base64.b64encode(wav).decode()


_TINY_WAV_B64 = _build_tiny_wav_b64()

# ---------------------------------------------------------------------------
# Helpers shared between CI and live layers
# ---------------------------------------------------------------------------

# A three-tier fleet env (primary always wired; minor + multimodal explicit).
_FULL_FLEET_ENV = {
    "MINOR_BASE_URL": "http://vllm-minor:8000",
    "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
}


def _build_full_fleet():
    """Return a RoutingTable with all three generate tiers wired."""
    table, _ = build_config(_FULL_FLEET_ENV)
    return table


# ---------------------------------------------------------------------------
# Layer A — Config / routing (always runs in CI)
# ---------------------------------------------------------------------------


class TestTierResolutionGateway:
    """Gateway routing helper assertions (no sockets; reuses build_config/resolve_model)."""

    def test_main_resolves_to_primary_via_gateway(self) -> None:
        # model=main must route to the 27B MTP primary served name.
        table, _ = build_config({})
        assert resolve_model(table, "main") == _DEFAULT_PRIMARY

    def test_multimodal_resolves_to_gemma_via_gateway(self) -> None:
        # model=multimodal must route to the Gemma 4 12B served name when wired.
        table, _ = build_config({"MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000"})
        assert resolve_model(table, "multimodal") == _DEFAULT_MULTIMODAL

    def test_minor_resolves_to_4b_via_gateway(self) -> None:
        # model=minor must route to the Qwen 4B served name when wired.
        table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
        assert resolve_model(table, "minor") == _DEFAULT_MINOR

    def test_full_fleet_three_tier_aliases_all_resolve(self) -> None:
        # With both optional backends wired, all six tier aliases resolve correctly.
        table = _build_full_fleet()
        assert resolve_model(table, "main") == _DEFAULT_PRIMARY
        assert resolve_model(table, "multimodal") == _DEFAULT_MULTIMODAL
        assert resolve_model(table, "minor") == _DEFAULT_MINOR
        # Back-compat aliases agree.
        assert resolve_model(table, "hard") == _DEFAULT_PRIMARY
        assert resolve_model(table, "normal") == _DEFAULT_MULTIMODAL
        assert resolve_model(table, "cheap") == _DEFAULT_MINOR

    def test_default_primary_constant_matches_catalog(self) -> None:
        # The gateway's _DEFAULT_PRIMARY constant must agree with the catalog
        # primary gear id so the two don't diverge silently.
        primary = resolve_tier("main")
        assert primary.id == _DEFAULT_PRIMARY

    def test_default_multimodal_constant_matches_catalog(self) -> None:
        gemma = resolve_tier("multimodal")
        assert gemma.id == _DEFAULT_MULTIMODAL

    def test_default_minor_constant_matches_catalog(self) -> None:
        minor = resolve_tier("minor")
        assert minor.id == _DEFAULT_MINOR


class TestCatalogTierResolution:
    """Catalog-level resolve_tier assertions for the duo (main/multimodal/minor)."""

    def test_main_resolve_tier_returns_primary(self) -> None:
        m = resolve_tier("main")
        assert m.role_hint == "primary"
        assert m.task == "generate"
        assert m.id == _PRIMARY_ID

    def test_multimodal_resolve_tier_returns_gemma(self) -> None:
        m = resolve_tier("multimodal")
        assert m.role_hint == "multimodal"
        assert m.task == "generate"
        assert m.id == _GEMMA_ID

    def test_minor_resolve_tier_returns_4b(self) -> None:
        m = resolve_tier("minor")
        assert m.role_hint == "minor"
        assert m.task == "generate"
        assert m.id == _MINOR_ID


class TestLegacy14BProfileSelectability:
    """The legacy 14B is a candidate, not a tier — explicitly selectable, not default-on."""

    def test_14b_in_catalog_with_candidate_role(self) -> None:
        entry = next((m for m in SUPPORTED_MODELS if m.id == _LEGACY_14B_ID), None)
        assert entry is not None, f"{_LEGACY_14B_ID} not found in catalog"
        assert (
            entry.role_hint == "candidate"
        ), f"{_LEGACY_14B_ID}: expected role_hint='candidate', got {entry.role_hint!r}"

    def test_no_tier_alias_resolves_to_14b(self) -> None:
        # No tier alias (main/minor/multimodal or cheap/normal/hard) should land
        # on the legacy 14B — it is demoted and no TIER_ROLE value maps to it.
        table = _build_full_fleet()
        resolved = {
            resolve_model(table, alias)
            for alias in ("main", "multimodal", "minor", "hard", "normal", "cheap")
        }
        assert _LEGACY_14B_ID not in resolved, f"Legacy 14B appeared in tier resolution: {resolved}"

    @pytest.mark.skipif(
        not _FLEET_COMPOSE.is_file(),
        reason="fleet compose template not present in this install",
    )
    def test_14b_compose_service_is_behind_middle_profile(self) -> None:
        # Parse the fleet docker-compose.yml and confirm:
        # (a) vllm-middle (the 14B service) declares profiles: [middle]
        # (b) vllm-multimodal (the Gemma gear) has NO profiles key (default-on)
        compose = yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))
        services = compose.get("services", {})

        middle_svc = services.get("vllm-middle")
        assert middle_svc is not None, "vllm-middle service missing from fleet compose"
        profiles = middle_svc.get("profiles", [])
        assert (
            "middle" in profiles
        ), f"vllm-middle must declare profiles: [middle] (got {profiles!r})"

        # Confirm the default model arg in the command contains the 14B id.
        command = middle_svc.get("command", [])
        cmd_str = " ".join(str(c) for c in command)
        assert (
            _LEGACY_14B_ID in cmd_str
        ), f"vllm-middle command should reference {_LEGACY_14B_ID!r}; got: {cmd_str!r}"

    @pytest.mark.skipif(
        not _FLEET_COMPOSE.is_file(),
        reason="fleet compose template not present in this install",
    )
    def test_multimodal_compose_service_has_no_profile(self) -> None:
        # vllm-multimodal is default-on — it must have no 'profiles' key so that
        # a plain `docker compose up` includes it without any extra flag.
        compose = yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))
        multimodal_svc = compose.get("services", {}).get("vllm-multimodal")
        assert multimodal_svc is not None, "vllm-multimodal service missing from fleet compose"
        assert (
            "profiles" not in multimodal_svc
        ), "vllm-multimodal must NOT declare a compose profile (it is default-on)"
        # Confirm it references the Gemma served name.
        command = multimodal_svc.get("command", [])
        cmd_str = " ".join(str(c) for c in command)
        assert (
            _GEMMA_ID in cmd_str
        ), f"vllm-multimodal command should reference {_GEMMA_ID!r}; got: {cmd_str!r}"


# ---------------------------------------------------------------------------
# Layer B — Live gateway (gated on LOBES_SMOKE_BASE_URL)
# ---------------------------------------------------------------------------

_LIVE_REASON = "live smoke needs a running gateway (set LOBES_SMOKE_BASE_URL)"
_live = pytest.mark.skipif(
    not os.environ.get("LOBES_SMOKE_BASE_URL"),
    reason=_LIVE_REASON,
)


def _post_chat(base_url: str, payload: dict, timeout: int = 120) -> dict:
    """POST to /v1/chat/completions and return the parsed JSON response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # local endpoint only
        return json.load(r)


@_live
def test_live_main_text_returns_nonempty_content() -> None:
    """model=main (27B primary) responds with non-empty text to a plain prompt."""
    base_url = os.environ["LOBES_SMOKE_BASE_URL"]
    resp = _post_chat(
        base_url,
        {
            "model": "main",
            "messages": [{"role": "user", "content": "Reply with the word hello."}],
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    content = resp["choices"][0]["message"].get("content") or ""
    assert content.strip(), f"model=main returned empty content; full response: {resp}"


@_live
def test_live_multimodal_image_text_returns_valid_output() -> None:
    """model=multimodal (Gemma 4 12B) accepts an image+text content-parts request."""
    base_url = os.environ["LOBES_SMOKE_BASE_URL"]
    resp = _post_chat(
        base_url,
        {
            "model": "multimodal",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
        },
    )
    choices = resp.get("choices") or []
    assert choices, f"model=multimodal image request: empty choices; response: {resp}"
    content = choices[0].get("message", {}).get("content") or ""
    assert content.strip(), f"model=multimodal image request: empty content; response: {resp}"


@_live
def test_live_multimodal_audio_text_returns_valid_output() -> None:
    """model=multimodal (Gemma 4 12B) accepts an audio+text content-parts request."""
    base_url = os.environ["LOBES_SMOKE_BASE_URL"]
    resp = _post_chat(
        base_url,
        {
            "model": "multimodal",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you hear in this audio?"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": _TINY_WAV_B64,
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
        },
    )
    choices = resp.get("choices") or []
    assert choices, f"model=multimodal audio request: empty choices; response: {resp}"
    content = choices[0].get("message", {}).get("content") or ""
    assert content.strip(), f"model=multimodal audio request: empty content; response: {resp}"
