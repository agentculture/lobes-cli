"""Build corpus.jsonl: function/class-level chunks of .py files at pinned SHAs.

Reads from git objects (``git show <sha>:<path>``) so the corpus is reproducible.
Usage: python build_corpus.py   (writes corpus.jsonl next to this file)
"""

from __future__ import annotations

import ast
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_LINES = 4  # skip trivial chunks
MAX_CHARS = 2400  # truncate chunk text (document prompts stay bounded)
MAX_CHUNKS_PER_REPO = 1500  # cap, documented in README.md

REPOS = {
    "lobes-cli": {
        "git_dir": "/home/spark/git/.worktrees.lobes-cli/oes-t10",
        "sha": "971117b1cb79a8f242bdb056ae0601f9d8a9a348",
        "prefixes": [
            "lobes/gateway/",
            "lobes/runtime/",
            "lobes/cli/",
            "lobes/assess.py",
            "lobes/profiles/",
            "lobes/realtime/",
        ],
    },
    "culture": {
        "git_dir": "/home/spark/git/culture",
        "sha": "ff2581578b4614a4b0b2023bbc4b1a5c41630710",
        "prefixes": [
            "culture_core/clients/",
            "culture_core/cli/",
            "culture_core/protocol/",
            "culture_core/bots/",
            "culture_core/doctor/",
            "culture_core/overview/",
        ],
    },
}


def _git(git_dir: str, *args: str) -> str:
    return subprocess.run(  # nosec B603 B607
        ["git", "-C", git_dir, *args], capture_output=True, text=True, check=True
    ).stdout


def chunk_source(src: str, path: str) -> list[dict]:
    """Top-level functions/classes and methods of classes, with line spans."""
    tree = ast.parse(src)
    lines = src.splitlines()
    out: list[dict] = []

    def emit(node, qual):
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        end = node.end_lineno
        if end - start + 1 < MIN_LINES:
            return
        out.append(
            {
                "path": path,
                "name": qual,
                "start": start,
                "end": end,
                "text": "\n".join(lines[start - 1 : end])[:MAX_CHARS],
            }
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node, node.name)
        elif isinstance(node, ast.ClassDef):
            methods = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                for m in methods:
                    emit(m, f"{node.name}.{m.name}")
            else:
                emit(node, node.name)
    return out


def main() -> int:
    rows = []
    for repo, cfg in REPOS.items():
        files = _git(cfg["git_dir"], "ls-tree", "-r", "--name-only", cfg["sha"]).split()
        files = [
            f for f in files if f.endswith(".py") and any(f.startswith(p) for p in cfg["prefixes"])
        ]
        n = 0
        for f in sorted(files):
            src = _git(cfg["git_dir"], "show", f"{cfg['sha']}:{f}")
            try:
                chunks = chunk_source(src, f)
            except SyntaxError:
                continue
            for c in chunks:
                if n >= MAX_CHUNKS_PER_REPO:
                    break
                n += 1
                c.update(repo=repo, sha=cfg["sha"], id=f"{repo}:{c['path']}:{c['name']}")
                rows.append(c)
    with open(HERE / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} chunks", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
