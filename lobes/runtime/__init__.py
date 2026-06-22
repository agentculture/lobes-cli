"""Runtime layer: drive the local vLLM deployment (docker compose + .env).

The command handlers in :mod:`lobes.cli._commands` stay thin; the testable
logic (``.env`` read/write, deployment-dir resolution, scaffolding, health
polling) lives here. Everything is stdlib-only — subprocess with fixed argv
lists (no shell) and ``urllib`` for HTTP — so the published wheel carries no
runtime dependencies.
"""
