"""Every lane-touching module imports on its own, in a fresh interpreter (no hidden cycle)."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["lobes.embed_lanes", "lobes.gateway.server", "lobes.runtime._lanes", "lobes.cli"],
)
def test_module_imports_cold(module: str) -> None:
    result = subprocess.run(  # nosec B603 — fixed interpreter + literal module names
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
