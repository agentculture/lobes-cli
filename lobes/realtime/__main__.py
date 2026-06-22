"""``python -m lobes.realtime`` — the realtime container entrypoint.

Builds nothing here: :mod:`lobes.realtime.app` reads its config from the
environment (set by the ``realtime`` fleet service) at import. Needs the
``[realtime]`` extra; only ever invoked inside the container.
"""

from __future__ import annotations

from lobes.realtime.app import main

if __name__ == "__main__":  # pragma: no cover
    main()
