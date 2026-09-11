"""Schema-versioned Announcement wire format for mesh federation.

Pure dataclasses + JSON, stdlib only, no I/O.

Wire contract (JSON)
--------------------
* ``Announcement`` carries ``name``, ``origin``, ``schema_version``,
  ``roles`` (``{role -> RoleInfo}``), ``fingerprint``, ``capacity`` and
  ``private`` (default ``False`` — the announcement-level flag, kept for
  future use).
* ``RoleInfo`` mirrors the catalog :class:`~lobes.roles.RoleInfo` vocabulary
  and adds a ``private`` flag so callers can mark a role so it never leaves
  the box.  Every field in the payload has a home in the role registry — no
  parallel field vocabulary.

Fingerprint
-----------
The four **DISQUALIFYING_FIELDS** from ``_replicas.py`` gate replica
compatibility (served_id, quantization, max_model_len, runtime).  This module
reuses that exact set at import time so the wire schema and the replica pool
stay in lock-step (spec c5 / c45 / h36).

Versioning
----------
``schema_version`` is a dotted-major string (e.g. ``"1.0.0"``).  During
decode the major segment is compared against ``SCHEMA_MAJOR`` — a mismatch
raises :exc:`MeshSchemaIncompatible`.  Unknown JSON fields are silently
ignored (forward-compatible).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

from lobes.gateway._replicas import DISQUALIFYING_FIELDS

# Current wire schema major version.  Bump whenever the Announcement or
# RoleInfo schema is **not** forward-compatible (extra fields are ignored, so
# additive changes do NOT require a bump).
SCHEMA_MAJOR: int = 1

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fingerprint:
    """Serving fingerprint for one replica — mirrors DISQUALIFYING_FIELDS.

    Field order matches ``_replicas.DISQUALIFYING_FIELDS`` so the two stay in
    lock-step without an assertion.
    """

    served_id: str
    quantization: str
    max_model_len: int  # tokens
    runtime: str  # e.g. "vllm", "llamacpp"


# Runtime guard — the Fingerprint field order must stay in lock-step with the
# replica pool so fingerprint comparison cannot silently diverge.  The test
# suite also checks this, but a module-load assertion catches breakage before
# any test runs.
_Fingerprint__FIELD_ORDER: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Fingerprint))
assert (
    _Fingerprint__FIELD_ORDER == DISQUALIFYING_FIELDS
), f"Fingerprint fields {_Fingerprint__FIELD_ORDER} != DISQUALIFYING_FIELDS {DISQUALIFYING_FIELDS}"


@dataclass(frozen=True)
class RoleInfo:
    """Per-role metadata on the wire.

    Mirrors :class:`~lobes.roles.RoleInfo` so every field in the payload has a
    home in the role registry — no parallel field vocabulary.  The
    ``private`` flag, when ``True``, causes the role to be dropped by
    :meth:`Announcement.public`.
    """

    model: str
    runtime: str
    context: int  # max_model_len (tokens); 0 when N/A
    quant: str  # quantization label
    responsibilities: tuple[str, ...]
    forbidden_responsibilities: tuple[str, ...]
    private: bool = False  # when True this role never leaves the box


@dataclass(frozen=True)
class Announcement:
    """A versioned announcement that flows between mesh nodes.

    ``public()`` returns a new instance with every role whose ``private`` flag
    is ``True`` dropped — the announcement never leaves the box.
    """

    name: str
    origin: str
    schema_version: str
    roles: dict[str, RoleInfo]  # role -> metadata
    fingerprint: Fingerprint  # what this replica serves
    capacity: float  # max active requests for this replica
    private: bool = False  # legacy / announcement-level flag

    def public(self) -> Announcement:
        """Return a new Announcement with all private roles dropped.

        This method never mutates *self* — it returns a fresh instance so the
        original can still circulate on the wire.
        """
        clean_roles = {name: info for name, info in self.roles.items() if not info.private}
        return dataclasses.replace(self, roles=clean_roles)


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------


def encode(a: Announcement) -> bytes:
    """Serialise *a* to JSON bytes."""
    obj: dict[str, Any] = {
        "name": a.name,
        "origin": a.origin,
        "schema_version": a.schema_version,
        "roles": {name: dataclasses.asdict(info) for name, info in a.roles.items()},
        "fingerprint": dataclasses.asdict(a.fingerprint),
        "capacity": a.capacity,
        "private": a.private,
    }
    return json.dumps(obj).encode("utf-8")


def decode(data: bytes) -> Announcement:
    """Deserialise an ``Announcement`` from JSON bytes.

    Raises :exc:`MeshSchemaIncompatible` when the major schema version in the
    payload does not match :data:`SCHEMA_MAJOR`.  Unknown JSON fields are
    silently ignored (forward-compatible).
    """
    obj = json.loads(data)

    # --- schema version ---------------------------------------------------
    try:
        major = int(obj["schema_version"].split(".")[0])
    except (KeyError, ValueError, AttributeError) as exc:
        raise MeshSchemaIncompatible(
            f"expected major {SCHEMA_MAJOR}, cannot parse schema version"
        ) from exc
    if major != SCHEMA_MAJOR:
        actual_major = obj.get("schema_version", "<?>")
        if isinstance(actual_major, str) and "." in actual_major:
            label = actual_major
        else:
            label = str(actual_major)
        raise MeshSchemaIncompatible(f"schema version {SCHEMA_MAJOR} expected, got {label}")

    # --- role info --------------------------------------------------------
    roles: dict[str, RoleInfo] = {}
    for role_name, role_obj in obj.get("roles", {}).items():
        roles[role_name] = RoleInfo(
            model=role_obj["model"],
            runtime=role_obj["runtime"],
            context=role_obj["context"],
            quant=role_obj["quant"],
            responsibilities=tuple(role_obj["responsibilities"]),
            forbidden_responsibilities=tuple(role_obj["forbidden_responsibilities"]),
            private=bool(role_obj.get("private", False)),
        )

    # --- fingerprint ------------------------------------------------------
    fp_obj = obj["fingerprint"]
    fingerprint = Fingerprint(
        served_id=fp_obj["served_id"],
        quantization=fp_obj["quantization"],
        max_model_len=fp_obj["max_model_len"],
        runtime=fp_obj["runtime"],
    )

    return Announcement(
        name=obj["name"],
        origin=obj["origin"],
        schema_version=obj["schema_version"],
        roles=roles,
        fingerprint=fingerprint,
        capacity=float(obj["capacity"]),
        private=bool(obj.get("private", False)),
    )


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class MeshSchemaIncompatible(ValueError):
    """Schema-version mismatch between mesh nodes.

    The message always includes both the **expected** major version and the
    **actual** major version so the operator can tell them apart without a
    traceback.
    """

    pass
