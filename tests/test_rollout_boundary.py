"""Static boundary check for the gradual fleet rollout (gradual-fleet-rollout, t19).

The rollout may touch only: the CLI install, the ``MODEL_GEAR_VERSION`` pin,
the ``.env`` backup, the gateway image tag/rebuild, the member lock, the
journal, ``residents.toml`` and (opt-in ``--patch-drift``)
``docker-compose.override.yml``. It must never write vLLM lane images,
checkpoints, shape or profile files, or run ``lobes init``.

The check is static (AST over the string constants in each module, docstrings
excluded) so a future edit that adds such a command fails here, before any
member is touched. It proves the code does not *name* these operations; it is
not a sandbox.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = sorted((ROOT / "lobes" / "rollout").glob("*.py")) + [
    ROOT / "lobes" / "cli" / "_commands" / "rollout.py",
    ROOT / "lobes" / "cli" / "_commands" / "lobe.py",
]

# lobes verbs that change what a box serves (anything but the gateway).
# "fleet" is forbidden except the read-only "fleet files" (handled below).
FORBIDDEN_VERBS = {
    "init",
    "switch",
    "serve",
    "start",
    "stop",
    "fleet",
    "tunnel",
    "benchmark",
    "tensorfold",
}
# Names of the things the rollout must never write or build.
FORBIDDEN_TEXT = re.compile(
    r"profiles/|builtin_shapes|/shapes?\b|\.shape\.|docker-compose\.shape|"
    r"\bsafetensors\b|\bhuggingface\b|\bHF_HOME\b|"
    r"\bdocker\s+(pull|build|rmi|commit|push)\b|\bdocker\s+compose\s+(up|build|pull)\b|"
    r"\blobes\s+(init|switch|serve|stop|tunnel|fleet(?!\s+files))\b",
    re.IGNORECASE,
)


def _docstring_nodes(tree: ast.AST) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def violations(source: str) -> list[str]:
    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in skip:
                continue
            m = FORBIDDEN_TEXT.search(node.value)
            if m:
                found.append(f"line {node.lineno}: forbidden text {m.group(0)!r}")
        if isinstance(node, (ast.List, ast.Tuple)):
            elts = node.elts
            for i, e in enumerate(elts):
                if not (isinstance(e, ast.Constant) and isinstance(e.value, str)):
                    continue
                nxt1 = elts[i + 1] if i + 1 < len(elts) else None
                read_only_fleet = (
                    e.value == "fleet" and isinstance(nxt1, ast.Constant) and nxt1.value == "files"
                )
                if (
                    e.value in FORBIDDEN_VERBS
                    and not read_only_fleet
                    and (i == 0 or _is_lobes_argv(elts[:i]))
                ):
                    found.append(f"line {e.lineno}: lobes verb {e.value!r} in an argv")
                if e.value == "up":
                    nxt = elts[i + 1] if i + 1 < len(elts) else None
                    if not (isinstance(nxt, ast.Constant) and nxt.value == "gateway"):
                        found.append(f"line {e.lineno}: 'up' with a target other than gateway")
    return found


def _is_lobes_argv(prefix: list) -> bool:
    """argv prefix that could be ``lobes``: only a name/constant/BoolOp/attribute head."""
    head = prefix[-1] if len(prefix) == 1 else prefix[0]
    if isinstance(head, ast.Constant):
        return head.value in ("lobes", "uv")
    return len(prefix) == 1  # e.g. [lobes_path, "init"] / (t.lobes_path or "lobes", "init")


def test_targets_exist():
    assert len(TARGETS) >= 15
    assert all(p.is_file() for p in TARGETS)


@pytest.mark.parametrize("path", TARGETS, ids=lambda p: p.name)
def test_module_stays_inside_the_boundary(path: Path):
    assert violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "bad",
    [
        'run(["lobes", "init", "--apply"])',
        'run((t.lobes_path or "lobes", "switch", "x"))',
        'run([lobes, "up", "vllm-primary"])',
        'run([lobes, "up"])',
        'run_lobes(["init", "--apply"])',
        'run_lobes(["switch", "model"])',
        'cmd = "lobes init --shape spark-lobe"',
        'p = base / "profiles/spark.toml"',
        'p = "builtin_shapes/spark-lobe.toml"',
        's = "docker pull vllm/vllm-openai:v0.31.0"',
        's = "docker compose up -d vllm-primary"',
    ],
)
def test_checker_catches_forbidden_operations(bad: str):
    assert violations(bad) != []


@pytest.mark.parametrize(
    "good",
    [
        'run((t.lobes_path or "lobes", "up", "gateway", "--apply"))',
        'run_lobes(["up", "gateway", "--build", "--apply"])',
        'run_lobes(["doctor", "--repin-version", "--apply"])',
        'run_lobes(target, ["fleet", "files", "--json"])',
        'run(["docker", "tag", image_id, tag])',
        '"""Never involves profiles/ or shapes."""\nx = 1',
    ],
)
def test_checker_allows_the_rollout_surface(good: str):
    assert violations(good) == []
