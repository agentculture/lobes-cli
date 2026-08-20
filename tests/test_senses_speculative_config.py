"""The senses lane's ``--speculative-config`` is an env-parameterized OFF-switch (t5).

WHY this exists: the senses swap targets ``unsloth/gemma-4-12B-it-qat-w4a16``, a
checkpoint that ships **no MTP draft head**. The recorded operator decision is to
attempt the incumbent ``google/gemma-4-12B-it-assistant`` draft at boot and — if
vLLM refuses it — drop speculative decoding entirely rather than block the swap.
That has to be a one-line ``.env`` edit, not a hand-edit of the packaged compose.

THE MECHANISM (and why it is not a ``command:`` list):

``docker compose`` cannot conditionally omit a ``command:`` **list** item. A list
item that substitutes to the empty string renders as an empty argv element
(``- ""``), and ``vllm serve`` treats that as a second positional and dies::

    usage: vllm serve [model_tag] [options]
    error: unrecognized arguments:

VERIFIED against the served build (``0.23.1rc1.dev672+g93d8f834d``, the image
``lobes/vllm-gemma4:local``) by parsing that exact argv through vLLM's own serve
parser — exit code 2. So the "fold the flag+value into one list item and set it
empty" approach is dead.

A **string** ``command:`` is shell-lexed by compose *after* substitution, so a
variable that expands to nothing leaves no token at all — the flag is omitted
**entirely**, which is the only rendering ``vllm serve`` accepts. That is why
this one lane (and only this one) declares ``command:`` as a folded string.

The default uses ``${VAR-default}`` (no colon) on purpose: ``${VAR:-default}``
treats an *empty* value as "unset" and would substitute the default back in,
making the off-switch unreachable. With ``-``, ``MULTIMODAL_SPECULATIVE_CONFIG=``
in ``.env`` means exactly what it looks like — off.

These tests reimplement the two compose behaviours above (brace-matched
substitution, then shell-lexing) so they run offline in CI. The simulation was
cross-checked against real ``docker compose config`` output on Compose v5.3.1
while the change was written: both the default render and the off render below
are byte-for-byte what Docker produces.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

from lobes.catalog import SUPPORTED_MODELS, speculative_config_item

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"

_SENSES_SERVICE = "vllm-multimodal"
_KNOB = "MULTIMODAL_SPECULATIVE_CONFIG"
_GEMMA_BASE_ID = "coolthor/gemma-4-12B-it-NVFP4A16"

# The exact argv the senses lane rendered BEFORE this task, captured from
# `docker compose config --format json` on the unmodified template with an empty
# .env (Compose v5.3.1). Parameterizing the speculative flag must not move a
# single byte of it — this frozen list is the "unset => byte-identical" proof.
_BASELINE_ARGV = [
    "vllm",
    "serve",
    "coolthor/gemma-4-12B-it-NVFP4A16",
    "--served-model-name=coolthor/gemma-4-12B-it-NVFP4A16",
    "--host=0.0.0.0",
    "--port=8000",
    "--quantization=compressed-tensors",
    "--max-model-len=32768",
    "--gpu-memory-utilization=0.14",
    "--enable-auto-tool-choice",
    "--tool-call-parser=gemma4",
    "--reasoning-parser=gemma4",
    (
        '--speculative-config={"method": "mtp", "model": '
        '"google/gemma-4-12B-it-assistant", "num_speculative_tokens": 1}'
    ),
    "--trust-remote-code",
]

# Same argv with the speculative flag GONE — not blanked, not empty-valued, gone.
_OFF_ARGV = [tok for tok in _BASELINE_ARGV if not tok.startswith("--speculative-config")]


def _load_fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


def _interpolate(text: str, env: dict[str, str]) -> str:
    """``docker compose``'s ``${VAR-default}`` / ``${VAR:-default}`` substitution.

    Brace-MATCHED, not regex-greedy: the default legitimately contains a JSON
    object, so the substitution's closing ``}`` is found by counting brace depth
    over every ``{``/``}`` — which is what Compose itself does (verified live:
    a default of ``--speculative-config={"method": "mtp", ...}`` round-trips
    whole, rather than being cut at the JSON's own closing brace).
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        start = text.find("${", i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        depth = 1
        j = start + 2
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        inner = text[start + 2 : j - 1]
        if ":-" in inner:
            name, _, default = inner.partition(":-")
            value = env.get(name) or default
        elif "-" in inner:
            name, _, default = inner.partition("-")
            value = env[name] if name in env else default
        else:
            name, value = inner, env.get(inner, "")
        out.append(value)
        i = j
    return "".join(out)


def _render_argv(env: dict[str, str] | None = None) -> list[str]:
    """The senses lane's argv as ``docker compose`` would render it for *env*."""
    command = _load_fleet()["services"][_SENSES_SERVICE]["command"]
    assert isinstance(command, str), (
        f"{_SENSES_SERVICE} command: must be a STRING, not a list — a list item "
        "cannot be conditionally omitted (see this module's docstring)"
    )
    return shlex.split(_interpolate(command, env or {}))


# --- the two acceptance criteria -------------------------------------------


def test_unset_renders_byte_identical_to_the_pre_parameterization_argv() -> None:
    """Acceptance 1: with the knob unset, nothing about the lane changed."""
    assert _render_argv() == _BASELINE_ARGV


def test_off_value_omits_the_flag_entirely() -> None:
    """Acceptance 2: the off value drops the flag — it does not blank its value."""
    argv = _render_argv({_KNOB: ""})

    assert argv == _OFF_ARGV
    assert not any(tok.startswith("--speculative-config") for tok in argv), (
        "the off value must remove the flag from argv entirely; an empty "
        "--speculative-config= is a different (and unintended) contract"
    )
    assert "" not in argv, (
        "an empty argv element is fatal: `vllm serve` reads it as a second "
        "positional and exits 2 with 'unrecognized arguments' (verified against "
        "the served build 0.23.1rc1.dev672+g93d8f834d)"
    )
    # Everything else survives untouched — the off-switch is surgical.
    assert argv == [tok for tok in _BASELINE_ARGV if not tok.startswith("--speculative-config")]


# --- the knob's shape -------------------------------------------------------


def test_knob_uses_the_unset_only_default_operator() -> None:
    """``${VAR-...}``, never ``${VAR:-...}``.

    ``:-`` substitutes the default for an EMPTY value too, which would make the
    empty off-switch above silently re-enable MTP — the exact bug this pins.
    """
    text = _FLEET_COMPOSE.read_text(encoding="utf-8")
    assert f"${{{_KNOB}-" in text, f"{_KNOB} must use the ${{VAR-default}} form"
    assert f"${{{_KNOB}:-" not in text, (
        f"{_KNOB} must NOT use ${{VAR:-default}}: an empty value would fall back "
        "to the default and the off-switch would never engage"
    )


def test_knob_carries_the_current_literal_as_its_default() -> None:
    """The shipped default is the incumbent MTP config, byte for byte."""
    gemma = next(m for m in SUPPORTED_MODELS if m.id == _GEMMA_BASE_ID)
    assert speculative_config_item(gemma) in _BASELINE_ARGV
    assert _render_argv()[-2] == speculative_config_item(gemma)


def test_custom_value_replaces_the_default() -> None:
    """An operator can also RETARGET the draft, not just switch it off.

    The value here is what compose sees AFTER its dotenv layer has parsed the
    ``.env`` line — the inner single quotes must survive that far, or the JSON's
    spaces split it into several argv tokens. The ``.env`` encoding that produces
    exactly this (verified live on Compose v5.3.1) is::

        MULTIMODAL_SPECULATIVE_CONFIG="'--speculative-config={\\"method\\": ...}'"

    i.e. double quotes for dotenv, single quotes for the shell-lexer, and the
    JSON's own quotes escaped. Writing it with bare single quotes instead yields
    ``--speculative-config={method:`` + a stray token — mangled, not rejected.
    """
    custom = '\'--speculative-config={"method": "ngram", "num_speculative_tokens": 2}\''
    argv = _render_argv({_KNOB: custom})
    assert '--speculative-config={"method": "ngram", "num_speculative_tokens": 2}' in argv
    assert not any("gemma-4-12B-it-assistant" in tok for tok in argv)
    assert len(argv) == len(_BASELINE_ARGV), "a custom value must stay ONE argv token"


def test_knob_is_scoped_to_the_senses_lane_only() -> None:
    """The MULTIMODAL_SPECULATIVE_CONFIG knob name itself is senses-only — the
    cortex/worker lanes gained their OWN off-switch knobs instead
    (PRIMARY_SPECULATIVE_CONFIG / WORKER_SPECULATIVE_CONFIG, spec-knobs task),
    never a shared name. The coder/muse lanes gained neither and keep MTP
    hardcoded (or absent) with no off-switch at all."""
    text = _FLEET_COMPOSE.read_text(encoding="utf-8")

    # The senses knob is SUBSTITUTED exactly once — in its own command.
    # (Prose mentions of the name in the comment block above it don't count.)
    assert (
        text.count(f"${{{_KNOB}") == 1
    ), f"${{{_KNOB} must be substituted exactly once in the fleet template"

    for name in ("vllm-multimodal-coder", "vllm-muse"):
        command = _load_fleet()["services"][name]["command"]
        assert isinstance(
            command, list
        ), f"{name}: command must stay a list (no off-switch was added to this lane)"


@pytest.mark.parametrize(
    "lane,knob",
    [
        ("vllm-primary", "PRIMARY_SPECULATIVE_CONFIG"),
        ("vllm-worker", "WORKER_SPECULATIVE_CONFIG"),
    ],
)
def test_primary_and_worker_gained_their_own_off_switch(lane: str, knob: str) -> None:
    """The spec-knobs task extended the senses lane's off-switch mechanism to
    cortex (vllm-primary) and worker (vllm-worker), each with ITS OWN knob name
    — never MULTIMODAL_SPECULATIVE_CONFIG shared across lanes. Both lanes'
    ``command:`` converted from a YAML list to the same shell-lexed STRING form
    for the identical reason the senses lane's did (t5): a list item cannot be
    conditionally omitted."""
    command = _load_fleet()["services"][lane]["command"]
    assert isinstance(command, str), f"{lane}: command must be a STRING, not a list"
    assert f"${{{knob}-" in command, f"{lane}: must default via ${{{knob}-...}}"
    assert f"${{{knob}:-" not in command, (
        f"{lane}: {knob} must NOT use ${{{knob}:-default}} — an empty value "
        "would fall back to the default and the off-switch would never engage"
    )


def test_muse_keeps_its_own_speculative_flag() -> None:
    """muse gained no off-switch — its MTP stays exactly as it was before this
    task (multimodal-coder carries no speculative-config at all, by its own
    long-standing design — see that service's own comment)."""
    command = _load_fleet()["services"]["vllm-muse"]["command"]
    assert any(
        str(tok).startswith("--speculative-config=") for tok in command
    ), "vllm-muse: its own --speculative-config must survive untouched"
