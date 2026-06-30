"""
Static assertions for lobes/templates/fleet/Dockerfile.vllm-gemma4.

No docker daemon, no network, no image builds — pure file-content checks.
"""

import re
from pathlib import Path

DOCKERFILE = (
    Path(__file__).parent.parent
    / "lobes"
    / "templates"
    / "fleet"
    / "Dockerfile.vllm-gemma4"
)


def _lines():
    return DOCKERFILE.read_text().splitlines()


# ---------------------------------------------------------------------------
# 1. File exists
# ---------------------------------------------------------------------------


def test_dockerfile_exists():
    assert DOCKERFILE.exists(), f"Expected {DOCKERFILE} to exist"


# ---------------------------------------------------------------------------
# 2. FROM the 26.05.post1 NGC vLLM base
# ---------------------------------------------------------------------------


def test_from_base():
    lines = _lines()
    from_lines = [l for l in lines if l.strip().startswith("FROM")]
    assert from_lines, "No FROM instruction found"
    assert any(
        "nvcr.io/nvidia/vllm:26.05.post1-py3" in l for l in from_lines
    ), f"Expected FROM nvcr.io/nvidia/vllm:26.05.post1-py3, got: {from_lines}"


# ---------------------------------------------------------------------------
# 3. TRANSFORMERS_REF build ARG is declared
# ---------------------------------------------------------------------------


def test_transformers_ref_arg_declared():
    text = DOCKERFILE.read_text()
    assert "ARG TRANSFORMERS_REF" in text, (
        "Expected 'ARG TRANSFORMERS_REF' build argument in Dockerfile"
    )


# ---------------------------------------------------------------------------
# 4. Transformers installed via uv pip install --system (not bare pip install)
# ---------------------------------------------------------------------------


def test_uses_uv_pip_install_system():
    text = DOCKERFILE.read_text()
    assert "uv pip install --system" in text, (
        "Expected 'uv pip install --system' to install transformers, not bare pip"
    )


def test_no_bare_pip_install_transformers():
    """Bare 'pip install transformers' without uv is disallowed."""
    text = DOCKERFILE.read_text()
    # Allow "pip install uv" (bootstrapping uv itself) but not a bare
    # "pip install transformers" line.
    bare_pip_pattern = re.compile(
        r"^\s*(?:python3?\S*\s+-m\s+)?pip\s+install\b.*\btransformers\b",
        re.MULTILINE,
    )
    matches = bare_pip_pattern.findall(text)
    assert not matches, (
        f"Found bare 'pip install transformers' (should use uv pip install --system): {matches}"
    )


# ---------------------------------------------------------------------------
# 5. Build-stage verification RUN is present
# ---------------------------------------------------------------------------


def test_verification_run_checks_gemma4_unified_in_config_mapping():
    text = DOCKERFILE.read_text()
    assert "gemma4_unified" in text, (
        "Expected a build-stage RUN that references 'gemma4_unified' "
        "(asserting it's registered in CONFIG_MAPPING)"
    )


def test_verification_run_checks_vllm_import():
    text = DOCKERFILE.read_text()
    assert "import vllm" in text, (
        "Expected the verification RUN to assert 'import vllm' succeeds"
    )


def test_verification_run_checks_gemma4_arch():
    text = DOCKERFILE.read_text()
    # The verification should reference ModelRegistry / get_supported_archs
    # and check for a Gemma4 architecture entry.
    has_registry = (
        "ModelRegistry" in text or "get_supported_archs" in text
    )
    has_gemma4_arch = "Gemma4" in text
    assert has_registry and has_gemma4_arch, (
        "Expected verification RUN to check vllm.ModelRegistry for a 'Gemma4' arch. "
        f"has_registry={has_registry}, has_gemma4_arch={has_gemma4_arch}"
    )
