"""``python -m model_gear.realtime`` — the realtime container entrypoint.

Builds nothing here: :mod:`model_gear.realtime.app` reads its config from the
environment (set by the ``realtime`` fleet service) at import. Needs the
``[realtime]`` extra; only ever invoked inside the container.
"""

from __future__ import annotations

from model_gear.realtime.app import main

if __name__ == "__main__":  # pragma: no cover
    main()
