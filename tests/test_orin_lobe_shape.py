"""The ``orin-lobe`` deployment shape + the Tegra iowait declaration (t7).

Two independent regressions, both found on the LIVE Jetson AGX Orin 64GB and
both previously survivable only by hand-patching the deployed box:

1. **The budget clobber.** The Orin ran the ``thor-lobe`` shape over the
   ``orin`` card (docs/orin-profiles.md, "Shape choice"), and a shape's
   ``[overrides.senses]`` WINS over the card profile at render time
   (:func:`lobes.profiles.shape_render._overlay`) — so Thor's measured sm_110
   values (0.30 / 131072) silently replaced the Orin card's own
   (0.45 / 262144), and the operator re-patched the rendered ``.env`` by hand
   after every render. ``orin-lobe`` is the shape that carries the Orin
   budget as its own declaration; the goldens below are the byte-level proof.

2. **The Tegra iowait accounting quirk.** ``/proc/stat`` on this board reports
   ~59% iowait with zero disk I/O (the ``sugov:*`` cpufreq-governor kthreads
   flicker through D state and inflate ``nr_iowait``), so the gateway's
   pressure policy — reading it literally at the shipped default of 50 —
   429-sheds every full-tier request indefinitely, i.e. the whole of what this
   board serves. The fix is declared on the CARD profile
   (:attr:`lobes.profiles.schema.Profile.host_env`), not on this shape,
   because it is a fact about the BOARD: a shape-scoped fix would leave a bare
   ``lobes init`` (machine-as-brain) on the same box shedding.

**HONESTY (#108).** ``orin-lobe`` is DECLARED, UNVALIDATED: no box has booted
it. Its senses budget is inherited from ``builtin/orin.toml``'s own
MEASURED-PENDING hypothesis, itself pending the live boot that backfills it.
Nothing in this module asserts a validation claim — only that the declared
data renders what it says it renders.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

from lobes.gateway import _pressure_policy
from lobes.gateway._pressure_policy import _env_float, decide
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import ROLE_ENV_PREFIX
from lobes.profiles.shape_render import ROLE_SERVICE, render_shape, shape_env, shape_services
from lobes.profiles.shapes import AUDIO_ROLES, builtin_shape_names, resolve_shape
from tests.goldens.regen import FLEET_COMPOSE, shape_golden_path

_GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"

# The knob the gateway pressure policy actually reads (verified against
# lobes/gateway/_pressure_policy.py, not restated from memory).
_IOWAIT_KEY = "LOBES_IOWAIT_DEGRADED_THRESHOLD"

# The live measurement this declaration exists for: ~59% iowait on an idle
# board (2026-07-17, re-confirmed 2026-08-04 — 5 of 8 cores at ~97%, vmstat
# bi/bo ~= 0, GPU 0%).
_TEGRA_PHANTOM_IOWAIT_PERCENT = 59.0


def _shape_toml(name: str) -> str:
    return (
        files("lobes.profiles.builtin_shapes").joinpath(f"{name}.toml").read_text(encoding="utf-8")
    )


# --- 1. the clobber this shape exists to prevent -----------------------------


def test_orin_lobe_renders_the_orin_senses_budget_not_thor_s() -> None:
    """THE regression: orin-lobe on the orin card renders 0.45 / 262144.

    Asserted against the committed golden FILES (not just the render), and
    contrasted with ``thor-lobe__orin.env`` — the pairing the box actually ran,
    where Thor's override wins and the Orin card's own values vanish.
    """
    orin_lobe = shape_golden_path("orin-lobe", "orin").read_text(encoding="utf-8")
    assert "MULTIMODAL_GPU_MEM_UTIL=0.45" in orin_lobe
    assert "MULTIMODAL_MAX_MODEL_LEN=262144" in orin_lobe

    # The clobber, still true for the sibling shape (kept as the contrast, not
    # as a bug to fix -- thor-lobe's values are Thor's, measured, and correct
    # for Thor).
    thor_lobe = shape_golden_path("thor-lobe", "orin").read_text(encoding="utf-8")
    assert "MULTIMODAL_GPU_MEM_UTIL=0.3" in thor_lobe
    assert "MULTIMODAL_MAX_MODEL_LEN=131072" in thor_lobe

    # And the card profile's own rendering agrees with orin-lobe, not thor-lobe.
    card = (_GOLDENS_DIR / "orin.env").read_text(encoding="utf-8")
    assert "MULTIMODAL_GPU_MEM_UTIL=0.45" in card
    assert "MULTIMODAL_MAX_MODEL_LEN=262144" in card


def test_orin_lobe_senses_override_stays_in_lockstep_with_the_orin_card() -> None:
    """The shape's override MUST equal builtin/orin.toml's own senses values.

    The override is a deliberate restatement (that is what makes the shape
    carry the budget as its own declaration), and a restatement is a drift
    hazard: the t9 live boot backfills the card profile's MEASURED-PENDING
    hypothesis — a trimmed context is an explicitly legitimate outcome — and
    this assertion is what fails CI if only one of the two files is updated.
    """
    override = resolve_shape("orin-lobe").override("senses")
    card_senses = resolve_profile("orin").role("senses")
    assert override.gpu_mem_util == card_senses.gpu_mem_util
    assert override.max_model_len == card_senses.max_model_len
    # Budget only -- the shape never restates the model, quantization or
    # backend (those flow through from the card, single-sourced there).
    assert override.model is None
    assert override.quantization is None
    assert override.attention_backend is None
    assert override.feasible is True


def test_orin_lobe_renders_the_card_s_qat_checkpoint_unchanged() -> None:
    env = render_shape(resolve_shape("orin-lobe"), resolve_profile("orin")).env
    assert env["MULTIMODAL_MODEL"] == "unsloth/gemma-4-12B-it-qat-w4a16"
    assert env["MULTIMODAL_SERVED_NAME"] == env["MULTIMODAL_MODEL"]


# --- the hosting decision: senses + pooling, no cortex, no audio -------------


def test_orin_lobe_drops_cortex_and_hosts_no_audio_services() -> None:
    """No stt/tts: the Parakeet base image has no sm_87 kernels (measured live).

    Consequence asserted at the compose level — the audio sidecars and the
    realtime bridge are simply not in the service set, on any card.
    """
    shape = resolve_shape("orin-lobe")
    for card in builtin_names():
        services = set(shape_services(shape, resolve_profile(card)))
        assert ROLE_SERVICE["cortex"] not in services
        for audio_role in AUDIO_ROLES:
            assert ROLE_SERVICE[audio_role] not in services
        assert "realtime" not in services
        assert ROLE_SERVICE["embedder"] in services
        assert ROLE_SERVICE["reranker"] in services


def test_orin_lobe_flags_dropped_cortex_off_and_leaks_no_knobs() -> None:
    env = shape_env(resolve_shape("orin-lobe"), resolve_profile("orin"))
    prefix = ROLE_ENV_PREFIX["cortex"]
    assert env.get(f"{prefix}_FEASIBLE") == "false"
    leaked = [k for k in env if k.startswith(f"{prefix}_") and k != f"{prefix}_FEASIBLE"]
    assert leaked == []


def test_orin_lobe_toml_declares_itself_unvalidated() -> None:
    text = _shape_toml("orin-lobe")
    assert "UNVALIDATED" in text
    # The two live facts the file must not lose: why no audio, and what the
    # override guards against.
    assert "sm_87" in text
    assert "no kernel image is available" in text


# --- 2. the Tegra iowait declaration ----------------------------------------


def test_the_orin_card_declares_the_iowait_threshold_the_policy_reads() -> None:
    host_env = dict(resolve_profile("orin").host_env)
    assert host_env == {_IOWAIT_KEY: "100"}


def test_the_declared_key_is_the_one_the_pressure_policy_actually_reads(monkeypatch) -> None:
    """Name check with teeth: feed the rendered KEY to the policy's own reader.

    ``_env_float`` is exactly how ``_pressure_policy`` resolves the threshold
    at import, so a renamed/typo'd key would surface here as the default
    coming back instead of the declared value.
    """
    declared = resolve_profile("orin").host_env[_IOWAIT_KEY]
    monkeypatch.setenv(_IOWAIT_KEY, declared)
    assert _env_float(_IOWAIT_KEY, 50.0) == 100.0


def test_the_declared_threshold_stops_the_phantom_iowait_shedding(monkeypatch) -> None:
    """The behavioural proof, both directions, at the measured ~59%.

    At the shipped default the board sheds its only lobe; at the declared
    value it serves. (``decide`` reads the module-level threshold at call
    time, so this needs no reload — monkeypatch restores it.)
    """
    declared = float(resolve_profile("orin").host_env[_IOWAIT_KEY])

    monkeypatch.setattr(_pressure_policy, "IOWAIT_DEGRADED_THRESHOLD", 50.0)
    shipped_default = decide(0.0, _TEGRA_PHANTOM_IOWAIT_PERCENT, "multimodal")
    assert shipped_default["mode"] == "busy"
    assert shipped_default["shed"] is True
    assert shipped_default["reason"] == "pressure"

    monkeypatch.setattr(_pressure_policy, "IOWAIT_DEGRADED_THRESHOLD", declared)
    with_declaration = decide(0.0, _TEGRA_PHANTOM_IOWAIT_PERCENT, "multimodal")
    assert with_declaration["mode"] == "warm"
    assert with_declaration["shed"] is False
    assert with_declaration["servable_tier"] == "multimodal"

    # ...and the swap guard is deliberately untouched: real memory pressure
    # still sheds at the shipped 75.
    assert decide(80.0, 0.0, "multimodal")["shed"] is True


def test_the_compose_template_passes_the_key_into_the_gateway() -> None:
    """A declaration only helps if .env reaches the container (mirror check)."""
    text = FLEET_COMPOSE.read_text(encoding="utf-8")
    assert f"- {_IOWAIT_KEY}=${{{_IOWAIT_KEY}:-50}}" in text


def test_the_declaration_survives_every_shape_including_the_default() -> None:
    """Card-scoped, not shape-scoped: machine-as-brain on orin gets it too.

    This is the whole reason the value lives on the card profile — a bare
    ``lobes init`` (which renders machine-as-brain) on this board must not be
    the one path still shedding.
    """
    for shape_name in builtin_shape_names():
        env = shape_env(resolve_shape(shape_name), resolve_profile("orin"))
        assert env.get(_IOWAIT_KEY) == "100", f"{shape_name} lost the orin card's declaration"


@pytest.mark.parametrize("card", [c for c in builtin_names() if c != "orin"])
def test_no_other_card_gains_a_host_env_key_under_any_shape(card: str) -> None:
    """Narrowness: the mechanism moves nothing on base/spark/thor.

    The byte-for-byte guarantee is the goldens'; this is the in-suite echo
    that says WHY they cannot have moved.
    """
    assert dict(resolve_profile(card).host_env) == {}
    for shape_name in builtin_shape_names():
        env = shape_env(resolve_shape(shape_name), resolve_profile(card))
        assert not any(k.startswith("LOBES_") for k in env)


def test_only_the_orin_goldens_carry_a_lobes_key() -> None:
    """Same narrowness, asserted against the committed golden FILES.

    ``template-defaults.env`` is excluded: it is the compose template's own
    ``${VAR:-default}`` surface (where ``LOBES_IOWAIT_DEGRADED_THRESHOLD=50``
    has always lived), not a profile/shape rendering.
    """
    goldens = [p for p in _GOLDENS_DIR.glob("*.env") if p.stem != "template-defaults"]
    goldens += list((_GOLDENS_DIR / "shapes").glob("*.env"))
    for path in goldens:
        has_lobes_key = any(
            line.startswith("LOBES_") for line in path.read_text(encoding="utf-8").splitlines()
        )
        expected = path.stem == "orin" or path.stem.endswith("__orin")
        assert has_lobes_key == expected, f"{path.name}: unexpected LOBES_* key presence"
