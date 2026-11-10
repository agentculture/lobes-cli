"""Which mesh member is THIS box.

``LOBES_MESH_NAME`` in ``.env`` is only what the next gateway recreate would
use; the running gateway container carries the name the mesh actually knows.
On 2026-11-10 the main gateway box (mesh name ``core``) had a ``.env`` whose
``LOBES_MESH_NAME`` read ``spark`` — another member's name — so reading ``.env``
alone made the site and the rollout treat the real spark as this box.

Rules, in order:

1. the running gateway container's ``LOBES_MESH_NAME`` wins over ``.env``
   (a disagreement is reported, never silently papered over);
2. a name that appears as a PEER in this box's own mesh roster is not this box
   (a gateway never lists itself), so it is refused rather than used.
"""

from __future__ import annotations

import subprocess  # nosec B404 - fixed argv, read-only docker inspect
from typing import Callable, Iterable, Optional

GATEWAY_CONTAINER = "model-gear-gateway"
_KEY = "LOBES_MESH_NAME="


class IdentityError(ValueError):
    """This box's mesh name cannot be determined safely."""


def running_gateway_mesh_name(
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    container: str = GATEWAY_CONTAINER,
) -> Optional[str]:
    """``LOBES_MESH_NAME`` from the running gateway container's env, or None.

    Only that one key is extracted; no other env value is kept or returned.
    """
    try:
        p = run(  # nosec B603 B607 - fixed argv
            [
                "docker",
                "inspect",
                container,
                "--format",
                "{{range .Config.Env}}{{println .}}{{end}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    for line in (p.stdout or "").splitlines():
        if line.startswith(_KEY):
            value = line[len(_KEY) :].strip()
            return value or None
    return None


def resolve_local_name(
    env_name: Optional[str],
    roster_peers: Iterable[str] = (),
    *,
    explicit: Optional[str] = None,
    inspect: Callable[[], Optional[str]] = running_gateway_mesh_name,
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(name, warning)`` for this box. Raises :class:`IdentityError`.

    ``explicit`` (an operator ``--local``) is taken as given, but still refused
    when it is a roster peer.
    """
    peers = {str(p) for p in roster_peers}
    env_name = (env_name or "").strip() or None
    warning = None
    if explicit and explicit.strip():
        name = explicit.strip()
    else:
        live = inspect()
        name = live or env_name
        if live and env_name and live != env_name:
            warning = (
                f".env LOBES_MESH_NAME={env_name!r} disagrees with the running gateway "
                f"({live!r}); using {live!r}. A gateway recreate from this .env would "
                f"rename this box."
            )
    if name and name in peers:
        raise IdentityError(
            f"{name!r} is a peer in this box's own mesh roster, so it is not this box; "
            "pass --local <this box's mesh name>"
        )
    return name, warning
