"""lobes gateway — a stdlib OpenAI-compatible reverse proxy for the fleet.

Fronts the fleet's vLLM backend(s) on one port: routes each request by its
``model`` field, defaults unknown/missing names to the primary, and (when an
opt-in fallback is configured) fails over to it when the chosen one is down. Runs
as the ``gateway`` container in a ``lobes init --fleet`` deployment
(``python -m lobes.gateway``).

Public surface:

- :func:`build_config` — env → ``(RoutingTable, ServerConfig)``
- :func:`serve` — bind and serve forever
"""

from __future__ import annotations

from lobes.gateway._config import build_config
from lobes.gateway.server import serve

__all__ = ["build_config", "serve"]
