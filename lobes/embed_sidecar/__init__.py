"""EmbeddingGemma 2 sidecar — a sentence-transformers ``/v1/embeddings`` server.

The pure request/response layer lives in :mod:`lobes.embed_sidecar.server` and
is unit-tested with an injected fake encoder; the FastAPI shell and the real
SentenceTransformer load only run inside the sidecar container.
"""
