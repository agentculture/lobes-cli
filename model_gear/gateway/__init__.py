"""model-gear gateway — a stdlib OpenAI-compatible reverse proxy for the fleet.

Fronts two always-warm vLLM backends on one port: routes each request by its
``model`` field, defaults unknown/missing names to the primary, and fails over to
the other backend when the chosen one is down. Runs as the ``gateway`` container
in a ``model init --fleet`` deployment (``python -m model_gear.gateway``).

Public surface:

- :func:`build_config` — env → ``(RoutingTable, ServerConfig)``
- :func:`serve` — bind and serve forever
"""

from __future__ import annotations

from model_gear.gateway._config import build_config
from model_gear.gateway.server import serve

__all__ = ["build_config", "serve"]
