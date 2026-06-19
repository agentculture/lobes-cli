"""Boundary guard: model-gear is SERVE-ONLY (issue #44).

model-gear vectorizes and scores over HTTP — it is the serving half of the
embedding/reranker gear split described in issue #44.  The storage/retrieval
half (vector stores, indexes, chunkers, retrievers) belongs to eidetic-cli.
These tests enforce that boundary at the package level so the split never
silently collapses:

1. No vector-DB or embedding-storage library appears in any dependency group.
2. No module inside model_gear/ has a stem that names a storage or retrieval
   concept (exact stem match — serve concepts like "embeddings" or "rerank"
   are explicitly allowed).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_PKG = _ROOT / "model_gear"

# Libraries that live on the storage/retrieval side of the boundary.
_VECTOR_DB_MARKERS = {
    "faiss",
    "chromadb",
    "qdrant",
    "pinecone",
    "weaviate",
    "pgvector",
    "milvus",
    "lancedb",
    "annoy",
    "hnswlib",
    "sentence-transformers",
}

# Module stems that belong to eidetic-cli, not model-gear.
_STORAGE_RETRIEVAL_STEMS = {
    "vector_store",
    "vectorstore",
    "index",
    "indexer",
    "retrieval",
    "retriever",
    "chunk",
    "chunker",
    "embed_store",
}


def test_no_vector_db_dependencies() -> None:
    """pyproject.toml must not pull in any vector-DB / embedding-store library."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})

    all_deps: list[str] = list(project.get("dependencies", []))
    for group_deps in project.get("optional-dependencies", {}).values():
        all_deps.extend(group_deps)

    for dep in all_deps:
        dep_lower = dep.lower()
        for marker in _VECTOR_DB_MARKERS:
            assert marker not in dep_lower, (
                f"Boundary violation: dependency {dep!r} contains forbidden "
                f"marker {marker!r} — vector-DB/embedding-store libraries "
                f"belong in eidetic-cli, not model-gear (issue #44)."
            )


def test_no_storage_or_retrieval_modules() -> None:
    """model_gear/ must not contain any module named after a storage/retrieval concept."""
    for py_file in _PKG.rglob("*.py"):
        stem = py_file.stem
        assert stem not in _STORAGE_RETRIEVAL_STEMS, (
            f"Boundary violation: {py_file} has stem {stem!r} which is a "
            f"storage/retrieval concept that belongs in eidetic-cli, not "
            f"model-gear (issue #44)."
        )
