"""Schema-versioned Announcement wire format for mesh federation.

Pure dataclasses + JSON, stdlib only, no I/O.

Wire contract (JSON)
--------------------
* ``Announcement`` carries ONLY ``name``, ``origin``, ``schema_version`` and
  ``roles`` (``{role -> RoleInfo}``).
* ``RoleInfo`` mirrors the catalog :class:`~lobes.roles.RoleInfo` vocabulary
  (model / runtime / context / quant / responsibilities /
  forbidden_responsibilities) and adds the per-role-lane ``fingerprint`` and
  ``capacity`` — the fingerprint and the capacity belong to each ROLE LANE,
  not to the announcement: a member serves several lanes (e.g. cortex on
  vLLM NVFP4 at 262144 and embed on a different model), and the pool/suffix
  logic compares fingerprints PER ROLE.  ``capacity=None`` means the lane is
  uncalibrated.  A ``private`` flag lets callers mark a role so it never
  leaves the box.  Every shared field in the payload has a home in the role
  registry — no parallel field vocabulary.

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


class MeshSchemaIncompatible(ValueError):
    """Schema-version mismatch between mesh nodes.

    The message always includes both the **expected** major version and the
    **actual** major version so the operator can tell them apart without a
    traceback.
    """


# Current wire schema major version.  Bump whenever the Announcement or
# RoleInfo schema is **not** forward-compatible (extra fields are ignored, so
# additive changes do NOT require a bump).
SCHEMA_MAJOR: int = 1

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fingerprint:
    """Serving fingerprint for one role lane — mirrors DISQUALIFYING_FIELDS.

    Field order matches ``_replicas.DISQUALIFYING_FIELDS`` so the two stay in
    lock-step without an assertion.
    """

    served_id: str
    quantization: str
    max_model_len: int  # tokens
    runtime: str  # e.g. "vllm", "llamacpp"


# Runtime guard — the Fingerprint field order must stay in lock-step with the
# replica pool so per-role fingerprint comparison cannot silently diverge.
# The test suite also checks this, but a module-load check catches breakage
# before any test runs.
_FINGERPRINT_FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Fingerprint))
if _FINGERPRINT_FIELDS != DISQUALIFYING_FIELDS:
    raise RuntimeError(
        f"Fingerprint fields {_FINGERPRINT_FIELDS} != "
        f"DISQUALIFYING_FIELDS {DISQUALIFYING_FIELDS}"
    )


@dataclass(frozen=True)
class RoleInfo:
    """Per-role metadata on the wire — one entry per ROLE LANE.

    Mirrors :class:`~lobes.roles.RoleInfo` so every shared field in the
    payload has a home in the role registry — no parallel field vocabulary.
    The ``fingerprint`` and ``capacity`` belong to the lane itself, not to
    the announcement: a member serves several lanes (cortex on vLLM NVFP4 at
    262144 and embed on a different model) and the pool/suffix logic compares
    fingerprints PER ROLE.  ``capacity=None`` means the lane is uncalibrated.
    The ``private`` flag, when ``True``, causes the role to be dropped by
    :meth:`Announcement.public`.
    """

    model: str
    runtime: str
    context: int  # max_model_len (tokens); 0 when N/A
    quant: str  # quantization label
    responsibilities: tuple[str, ...]
    forbidden_responsibilities: tuple[str, ...]
    fingerprint: Fingerprint  # what this role's lane serves (per-lane, not per-announcement)
    capacity: float | None = None  # max active requests for this lane; None = uncalibrated
    private: bool = False  # when True this role never leaves the box


@dataclass(frozen=True)
class Announcement:
    """A versioned announcement that flows between mesh nodes.

    Carries only identity + roles: ``name``, ``origin``, ``schema_version``
    and ``roles`` — the fingerprint and the capacity live on each role lane
    (see :class:`RoleInfo`), not here.

    ``public()`` returns a new instance with every role whose ``private`` flag
    is ``True`` dropped — the announcement never leaves the box.
    """

    name: str
    origin: str
    schema_version: str
    roles: dict[str, RoleInfo]  # role -> metadata

    def public(self) -> Announcement:
        """Return a new Announcement with all private roles dropped.

        This method never mutates *self* — it returns a fresh instance so the
        original can still circulate on the wire.  Built via the explicit
        constructor (S5886) rather than :func:`dataclasses.replace` — the
        latter is typed to return ``DataclassInstance``, not ``Announcement``.
        """
        clean_roles = {name: info for name, info in self.roles.items() if not info.private}
        return Announcement(
            name=self.name,
            origin=self.origin,
            schema_version=self.schema_version,
            roles=clean_roles,
        )


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
    }
    return json.dumps(obj).encode("utf-8")


def _check_schema_major(obj: dict[str, Any]) -> None:
    """Raise :exc:`MeshSchemaIncompatible` unless *obj* declares our major."""
    try:
        major = int(obj["schema_version"].split(".")[0])
    except (KeyError, ValueError, AttributeError) as exc:
        raise MeshSchemaIncompatible(
            f"expected major {SCHEMA_MAJOR}, cannot parse schema version"
        ) from exc
    if major != SCHEMA_MAJOR:
        raise MeshSchemaIncompatible(
            f"schema version {SCHEMA_MAJOR} expected, got {obj.get('schema_version', '<?>')}"
        )


def _decode_role_info(role_name: str, role_obj: Any) -> RoleInfo:
    """Decode one ``roles`` entry into a :class:`RoleInfo`.

    Every failure — a missing field, a non-dict ``fingerprint``, a bad type —
    is normalized to :exc:`ValueError` naming the offending role.
    """
    if not isinstance(role_obj, dict):
        raise ValueError(f"malformed role {role_name!r}: not a JSON object")
    try:
        fp_obj = role_obj["fingerprint"]
        if not isinstance(fp_obj, dict):
            raise ValueError(f"malformed role {role_name!r}: 'fingerprint' not a JSON object")
        capacity = role_obj.get("capacity")
        return RoleInfo(
            model=role_obj["model"],
            runtime=role_obj["runtime"],
            context=role_obj["context"],
            quant=role_obj["quant"],
            responsibilities=tuple(role_obj["responsibilities"]),
            forbidden_responsibilities=tuple(role_obj["forbidden_responsibilities"]),
            fingerprint=Fingerprint(
                served_id=fp_obj["served_id"],
                quantization=fp_obj["quantization"],
                max_model_len=fp_obj["max_model_len"],
                runtime=fp_obj["runtime"],
            ),
            capacity=float(capacity) if capacity is not None else None,
            private=bool(role_obj.get("private", False)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"malformed role {role_name!r}: {exc}") from exc


def decode(data: bytes) -> Announcement:
    """Deserialise an ``Announcement`` from JSON bytes.

    Raises :exc:`MeshSchemaIncompatible` when the major schema version in the
    payload does not match :data:`SCHEMA_MAJOR`.  Unknown JSON fields are
    silently ignored (forward-compatible) — including the announcement-level
    ``fingerprint`` / ``capacity`` / ``private`` fields the wire carried
    before the per-role-lane reshape.
    """
    obj = json.loads(data)

    # Finding 15 (review #252): validate the top-level shape BEFORE indexing
    # into it. A bare `obj["schema_version"]` / `obj["roles"].items()` /
    # `obj["name"]` on a structurally-wrong-but-valid-JSON body (a list, a
    # string, `roles` present but not a mapping, `name`/`origin` missing)
    # raised an uncaught TypeError/AttributeError/KeyError that `announce()`'s
    # `except (json.JSONDecodeError, ValueError, TypeError)` never fully
    # covered (AttributeError and a bare KeyError were not in that set),
    # crashing the request handler instead of reaching the intended 400.
    # Every failure here is normalized to ValueError so callers have exactly
    # one non-schema failure type to catch.
    if not isinstance(obj, dict):
        raise ValueError("announcement body must be a JSON object")

    _check_schema_major(obj)

    name = obj.get("name")
    origin = obj.get("origin")
    if not isinstance(name, str) or not name:
        raise ValueError("announcement 'name' must be a non-empty string")
    if not isinstance(origin, str):
        raise ValueError("announcement 'origin' must be a string")

    roles_obj = obj.get("roles", {})
    if not isinstance(roles_obj, dict):
        raise ValueError("announcement 'roles' must be a JSON object")

    roles = {
        role_name: _decode_role_info(role_name, role_obj)
        for role_name, role_obj in roles_obj.items()
    }

    return Announcement(
        name=name,
        origin=origin,
        schema_version=obj["schema_version"],
        roles=roles,
    )
