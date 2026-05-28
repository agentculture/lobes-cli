"""``python -m model_gear.gateway`` — the gateway container entrypoint.

Builds the routing table + server config from the environment (set by the fleet
compose ``gateway`` service) and serves forever.
"""

from __future__ import annotations

from model_gear.gateway._config import build_config
from model_gear.gateway.server import serve


def main() -> None:  # pragma: no cover - process entrypoint
    table, cfg = build_config()
    serve(table, cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
