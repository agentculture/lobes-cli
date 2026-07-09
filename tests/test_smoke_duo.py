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
  * Perception (ground-truth) checks — these are the ones that can actually
    fail if the model ignores the media, unlike the wire checks below:
    - model=multimodal names the colour of an in-process-generated solid-
      colour PNG (two colours, so a lucky guess/prior can't pass silently).
    - model=multimodal transcribes a known word synthesized by the rig's own
      TTS (Chatterbox, via POST /v1/audio/speech) and sent back as audio.
  * Wire checks (transport only, NOT perception — see the docstring on each):
    - model=multimodal accepts an image_url content-part built from a 1x1
      placeholder PNG and returns non-empty text.
    - model=multimodal accepts an input_audio content-part built from a tiny
      silent WAV and returns non-empty text.

All live calls use stdlib urllib only (mirroring lobes/assess.py).
"""

from __future__ import annotations

import base64
import json
import os
import struct
import time
import urllib.error
import urllib.request
import zlib
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

# Tiny 1×1 RGB PNG (valid PNG, useful as a minimal image payload). Used only by
# the *wire-check* tests below — it proves nothing about perception because a
# 1×1 placeholder carries no verifiable content. See _solid_png() for the
# ground-truth image used by the perception tests.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQI12P4"
    "z8AAAAACAAHiIbwzAAAAAElFTkSuQmCC"
)


def _solid_png(rgb: tuple[int, int, int], size: int = 96) -> bytes:
    """Build a solid-colour PNG in-process, stdlib only (zlib + struct).

    Used by the live image-perception test: a model that actually reads
    pixels should be able to name this colour; a model that ignores the
    image cannot (verified as a negative control — see the task report for
    tests/test_smoke_duo.py's t8 commit).
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit truecolour
    row = b"\x00" + bytes(rgb) * size  # filter byte 0 + RGB pixels
    idat = zlib.compress(row * size)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


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


# The word synthesized by TTS and expected back from the transcription. Common
# enough vocabulary that a model which *can* hear will produce it verbatim.
_AUDIO_WORD = "banana"

# Chatterbox has a recorded poisoned-CUDA-context failure mode: once tripped it
# returns 500 to every request until the container is restarted (issue #89,
# docs/chatterbox-tts.md). A bounded retry rides out a transient blip; if every
# attempt 500s we report "TTS backend unhealthy" rather than silently treating
# it as "senses cannot hear" — those are different findings.
_TTS_RETRIES = 3
_TTS_RETRY_BACKOFF_SECONDS = 2.0


def _synthesize_speech(base_url: str, text: str) -> bytes:
    """POST /v1/audio/speech and return the raw WAV bytes.

    A 404 means the audio facade isn't reachable through the gateway at all —
    ``AUDIO_URL`` never reached the gateway container (issue #96), fixed in the
    base fleet template by task t4 but not yet applied to a running deployment
    that hasn't been re-scaffolded. That is a redeploy problem, not a flaky
    call, so it fails immediately (no retry) with a message naming the issue.

    A 500 (or a connection failure) is retried a bounded number of times before
    failing with a message that says "TTS backend unhealthy" — distinct from a
    senses-cannot-hear finding, which only the transcription assertion in
    ``test_live_multimodal_audio_perception_transcribes_known_word`` can make.
    """
    url = base_url.rstrip("/") + "/v1/audio/speech"
    payload = {
        "model": "chatterbox",
        "input": text,
        "voice": "default",
        "response_format": "wav",
    }
    data = json.dumps(payload).encode()
    last_status: int | str = "no response"
    last_body = b""
    for attempt in range(1, _TTS_RETRIES + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # local endpoint only
                return r.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            if exc.code == 404:
                pytest.fail(
                    "POST /v1/audio/speech returned 404 -- the audio facade is not "
                    "wired into this gateway deployment (AUDIO_URL never reached the "
                    "gateway container, issue #96; fixed in the base fleet template "
                    "by task t4). Re-scaffold this deployment (`lobes init --fleet "
                    "--audio` against the current templates) before the audio "
                    f"perception test can run. Response body: {body!r}"
                )
            last_status, last_body = exc.code, body
        except urllib.error.URLError as exc:
            last_status, last_body = "connection error", str(exc.reason).encode()
        if attempt < _TTS_RETRIES:
            time.sleep(_TTS_RETRY_BACKOFF_SECONDS)
    pytest.fail(
        f"TTS backend unhealthy: POST /v1/audio/speech failed on all "
        f"{_TTS_RETRIES} attempts (last status={last_status}, body={last_body!r}). "
        "This matches Chatterbox's known poisoned-CUDA-context failure mode "
        "(docs/chatterbox-tts.md, issue #89) -- cleared only by restarting the "
        "chatterbox container -- and is a TTS-backend finding, not evidence "
        "that senses cannot hear."
    )


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


# ---------------------------------------------------------------------------
# Layer B — Perception (ground-truth) checks.
#
# Unlike the wire checks below, these send content whose correct answer is
# known in advance and assert the model reports THAT answer -- a model that
# ignores the media (or a broken vision/audio path that returns fluent but
# unrelated text) fails these, not just a request that never got a 200.
# ---------------------------------------------------------------------------

# Two colours, not one: a single colour could be a lucky guess or a language
# prior ("things are often red"). Two independent correct answers is evidence
# the model actually read the pixels.
_COLOUR_CASES = [
    ("red", (255, 0, 0)),
    ("blue", (0, 0, 255)),
]


@_live
@pytest.mark.parametrize("colour_name,rgb", _COLOUR_CASES)
def test_live_multimodal_image_perception_names_colour(colour_name: str, rgb) -> None:
    """model=multimodal (Gemma 4 12B) names the colour of a ground-truth image.

    Proves perception, not just transport: the PNG is generated in-process
    (stdlib zlib/struct, no external asset) as a solid fill of a known colour,
    and the assertion requires that exact colour's name in the reply. A model
    that ignores the image content cannot pass this by accident -- verified as
    a negative control while writing this test (feeding a blue image while
    asserting for "red" fails; see the t8 task report for the transcript).
    Compare test_live_multimodal_accepts_image_content_part_wire_check below,
    which only proves the wire and asserts nothing about correctness.
    """
    base_url = os.environ["LOBES_SMOKE_BASE_URL"]
    png_b64 = base64.b64encode(_solid_png(rgb)).decode()
    resp = _post_chat(
        base_url,
        {
            "model": "multimodal",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What single colour fills this image? Answer with one word.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    choices = resp.get("choices") or []
    assert (
        choices
    ), f"model=multimodal image perception ({colour_name}): empty choices; response: {resp}"
    content = choices[0].get("message", {}).get("content") or ""
    assert colour_name in content.lower(), (
        f"model=multimodal did not name {colour_name!r} for a solid-{colour_name} "
        f"ground-truth image; got {content!r}. This means the model is not reading "
        f"image pixels (or the vision path is broken). Full response: {resp}"
    )


@_live
def test_live_multimodal_audio_perception_transcribes_known_word() -> None:
    """model=multimodal (Gemma 4 12B) transcribes a ground-truth spoken word.

    Proves perception, not just transport: the word is synthesized fresh by
    the rig's own TTS (Chatterbox, via POST /v1/audio/speech on this same
    gateway) so the "correct answer" is known in advance, then sent back to
    model=multimodal as an input_audio content-part. A model that ignores the
    audio (or fabricates fluent-but-wrong text) fails the containment check.
    Compare test_live_multimodal_accepts_audio_content_part_wire_check below,
    which only proves the wire and asserts nothing about correctness.

    This currently fails loudly on deployments where AUDIO_URL hasn't reached
    the gateway container (issue #96, /v1/audio/speech -> 404) -- that is a
    redeploy problem, not a perception failure, and is reported as such by
    _synthesize_speech() rather than silently skipped.
    """
    base_url = os.environ["LOBES_SMOKE_BASE_URL"]
    wav_bytes = _synthesize_speech(base_url, _AUDIO_WORD)
    wav_b64 = base64.b64encode(wav_bytes).decode()

    resp = _post_chat(
        base_url,
        {
            "model": "multimodal",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe the speech in this audio. "
                                "Reply with only the words spoken."
                            ),
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": wav_b64, "format": "wav"},
                        },
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
        },
    )
    choices = resp.get("choices") or []
    assert choices, f"model=multimodal audio perception: empty choices; response: {resp}"
    content = choices[0].get("message", {}).get("content") or ""
    assert _AUDIO_WORD in content.lower(), (
        f"model=multimodal did not transcribe the known word {_AUDIO_WORD!r} "
        f"synthesized by Chatterbox; got {content!r}. This means senses cannot "
        f"hear (or the audio path is broken) -- not a TTS-backend problem, "
        f"which _synthesize_speech() would have reported separately. "
        f"Full response: {resp}"
    )


# ---------------------------------------------------------------------------
# Layer B — Wire checks (transport only, NOT perception).
#
# These assert only that the gateway accepts an image_url / input_audio
# content-part and returns a 200 with non-empty text. The payloads are a 1x1
# placeholder PNG and a near-silent WAV with no verifiable content, so a model
# that completely ignores the media still passes. See the perception tests
# above for assertions the model would actually fail if it weren't looking/
# listening.
# ---------------------------------------------------------------------------


@_live
def test_live_multimodal_accepts_image_content_part_wire_check() -> None:
    """model=multimodal (Gemma 4 12B) accepts an image_url content-part (transport only).

    This asserts the request is well-formed and answered, NOT that the model
    perceived the image -- the payload is a 1x1 placeholder PNG with no
    verifiable content, so a model that ignores the image entirely still
    passes. See test_live_multimodal_image_perception_names_colour for the
    ground-truth perception assertion.
    """
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
def test_live_multimodal_accepts_audio_content_part_wire_check() -> None:
    """model=multimodal (Gemma 4 12B) accepts an input_audio content-part (transport only).

    This asserts the request is well-formed and answered, NOT that the model
    perceived the audio -- the payload is a near-silent single-sample WAV with
    no verifiable content, so a model that ignores the audio entirely still
    passes. See test_live_multimodal_audio_perception_transcribes_known_word
    for the ground-truth perception assertion.
    """
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
