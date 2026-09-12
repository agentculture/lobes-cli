"""Tests for lobes.gateway._mesh_wire — Announcement wire format (plan t2).

Pure dataclass + JSON, stdlib only, no I/O.  Every test runs offline.

Acceptance criteria covered
----------------------------
1. decode(encode(a)) == a; unknown extra fields are ignored; a different
   major schema version raises MeshSchemaIncompatible with both versions in
   the message.  (c5, c45)
2. A role marked private=True is dropped by Announcement.public() so it
   never leaves the box; the flag defaults to False.  (h36)
3. The per-role payload reuses lobes/roles.py RoleInfo fields
   (model/runtime/context/quant/responsibilities/forbidden_responsibilities) —
   no parallel field vocabulary.  (c32)

Per-role-lane contract (t2 follow-up)
-------------------------------------
The fingerprint and the capacity belong to each ROLE LANE, not to the
announcement: a member serves several lanes (cortex on vLLM NVFP4 at 262144
and embed on a different model) and the pool/suffix logic compares
fingerprints PER ROLE.  ``Announcement`` therefore carries ONLY name, origin,
schema_version and roles; ``RoleInfo`` carries ``fingerprint`` (Fingerprint)
and ``capacity`` (float | None, None = uncalibrated) on top of the catalog
fields plus the private flag.

Fingerprint discipline
----------------------
The fingerprint's four fields MUST match
``_replicas.DISQUALIFYING_FIELDS`` (served_id, quantization, max_model_len,
runtime).
"""

from __future__ import annotations

import dataclasses
import json
from typing import get_type_hints

import pytest

from lobes.gateway._mesh_wire import (
    _FINGERPRINT_FIELDS,
    SCHEMA_MAJOR,
    Announcement,
    Fingerprint,
    MeshSchemaIncompatible,
    RoleInfo,
    decode,
    encode,
)
from lobes.gateway._replicas import DISQUALIFYING_FIELDS
from lobes.roles import RoleInfo as CatalogRoleInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Two DIFFERENT lanes on one member: the per-role-lane contract means the
# fingerprints may (and here do) differ — cortex serves the 27B at 262144,
# the embed lane serves a different model with its own window.
_CORTEX_FP = Fingerprint(
    served_id="unsloth/Qwen3.8-27B-NVFP4",
    quantization="NVFP4",
    max_model_len=262144,
    runtime="vllm",
)
_EMBED_FP = Fingerprint(
    served_id="coolthor/gemma-4-12B-it-NVFP4A16",
    quantization="NVFP4",
    max_model_len=32768,
    runtime="vllm",
)

_ROLES_DATA = {
    "cortex": RoleInfo(
        model="unsloth/Qwen3.8-27B-NVFP4",
        runtime="vllm",
        context=262144,
        quant="NVFP4",
        responsibilities=("reasoning", "deciding", "planning"),
        forbidden_responsibilities=(),
        fingerprint=_CORTEX_FP,
        capacity=8.0,
    ),
    "embed": RoleInfo(
        model="coolthor/gemma-4-12B-it-NVFP4A16",
        runtime="vllm",
        context=32768,
        quant="NVFP4",
        responsibilities=("vectorization",),
        forbidden_responsibilities=("final_decision",),
        fingerprint=_EMBED_FP,
        # uncalibrated lane — capacity None must round-trip as JSON null
        capacity=None,
    ),
}


def _minimal() -> Announcement:
    """A single-lane Announcement for the focused tests."""
    return Announcement(
        name="test",
        origin="http://x",
        schema_version="1.0.0",
        roles={
            "cortex": RoleInfo(
                model="x",
                runtime="vllm",
                context=0,
                quant="",
                responsibilities=(),
                forbidden_responsibilities=(),
                fingerprint=_CORTEX_FP,
                capacity=1.0,
            )
        },
    )


def _private_role() -> RoleInfo:
    """A private=True role lane for the .public() tests."""
    return RoleInfo(
        model="s",
        runtime="v",
        context=0,
        quant="",
        responsibilities=(),
        forbidden_responsibilities=(),
        fingerprint=_CORTEX_FP,
        private=True,
    )


@pytest.fixture()
def announcement() -> Announcement:
    """An Announcement with two public lanes (one calibrated, one not)."""
    return Announcement(
        name="spark-lobe",
        origin="http://localhost:8000",
        schema_version="1.0.0",
        roles=_ROLES_DATA,
    )


# ---------------------------------------------------------------------------
# AC-1: encode / decode round-trip, unknown fields, schema mismatch
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """AC-1: decode(encode(a)) == a."""

    def test_roundtrip(self, announcement: Announcement) -> None:
        decoded = decode(encode(announcement))
        assert decoded == announcement

    def test_per_lane_fingerprint_and_capacity(self, announcement: Announcement) -> None:
        """Fingerprint/capacity travel on the role lane, not the announcement."""
        decoded = decode(encode(announcement))
        assert decoded.roles["cortex"].fingerprint == _CORTEX_FP
        assert decoded.roles["cortex"].capacity == 8.0
        assert decoded.roles["embed"].fingerprint == _EMBED_FP
        assert decoded.roles["embed"].capacity is None

    def test_announcement_carries_no_lane_fields(self) -> None:
        """The per-lane contract: the announcement has no fingerprint/capacity."""
        fields = {f.name for f in dataclasses.fields(Announcement)}
        assert "fingerprint" not in fields
        assert "capacity" not in fields

    def test_roundtrip_empty_roles(self) -> None:
        a = Announcement(
            name="test",
            origin="http://x",
            schema_version="1.0.0",
            roles={},
        )
        assert decode(encode(a)) == a


class TestUnknownFields:
    """AC-1: unknown extra fields are ignored on decode."""

    def test_unknown_fields_at_top_level(self) -> None:
        a = _minimal()
        obj = json.loads(encode(a))
        # Unknown keys the decoder does not know about are ignored.
        obj["x_nope"] = 42
        # The announcement-level lane fields the wire carried before the
        # per-role-lane reshape are unknown too — decoded, not read.
        obj["fingerprint"] = dataclasses.asdict(_CORTEX_FP)
        obj["capacity"] = 8.0
        obj["private"] = True
        assert decode(json.dumps(obj).encode()) == a

    def test_unknown_fields_in_role_payload(self) -> None:
        a = _minimal()
        obj = json.loads(encode(a))
        obj["roles"]["cortex"]["extra_role_key"] = "gone"
        assert decode(json.dumps(obj).encode()) == a

    def test_unknown_fields_in_fingerprint_payload(self) -> None:
        a = _minimal()
        obj = json.loads(encode(a))
        obj["roles"]["cortex"]["fingerprint"]["extra_fp_key"] = "gone"
        assert decode(json.dumps(obj).encode()) == a


class TestSchemaIncompatible:
    """AC-1: different major schema version raises MeshSchemaIncompatible."""

    def test_major_mismatch(self, announcement: Announcement) -> None:
        obj = json.loads(encode(announcement))
        obj["schema_version"] = "2.0.0"  # different major
        with pytest.raises(MeshSchemaIncompatible) as exc_info:
            decode(json.dumps(obj).encode())
        # Both versions must appear in the message.
        msg = str(exc_info.value)
        assert str(SCHEMA_MAJOR) in msg  # expected major
        assert "2" in msg  # actual major from payload

    def test_minor_upgrade_ignored(self, announcement: Announcement) -> None:
        """Patch and minor bumps must NOT raise — major is what gates."""
        obj = json.loads(encode(announcement))
        obj["schema_version"] = "1.1.0"
        result = decode(json.dumps(obj).encode())
        assert result.schema_version == "1.1.0"
        assert result == dataclasses.replace(announcement, schema_version="1.1.0")


# ---------------------------------------------------------------------------
# AC-2: private role filtering via Announcement.public()
# ---------------------------------------------------------------------------


class TestPrivate:
    """AC-2: private=True roles are dropped by .public(); flag defaults False."""

    def test_private_flag_defaults_false(self) -> None:
        """When private is omitted, it must default to False (capacity too)."""
        r = RoleInfo(
            model="x",
            runtime="v",
            context=0,
            quant="",
            responsibilities=(),
            forbidden_responsibilities=(),
            fingerprint=_CORTEX_FP,
        )
        assert r.private is False
        assert r.capacity is None

    def test_public_drops_private_roles(self) -> None:
        """A role with private=True is absent from the public view."""
        priv_role = RoleInfo(
            model="secret/model",
            runtime="llamacpp",
            context=8192,
            quant="Q4",
            responsibilities=("hidden",),
            forbidden_responsibilities=(),
            fingerprint=Fingerprint("secret/model", "Q4", 8192, "llamacpp"),
            capacity=2.0,
            private=True,
        )
        a = Announcement(
            name="secret-box",
            origin="http://localhost:9000",
            schema_version="1.0.0",
            roles={"cortex": _ROLES_DATA["cortex"], "shadow": priv_role},
        )
        public = a.public()

        # Public role survives.
        assert "cortex" in public.roles
        assert public.roles["cortex"].model == _ROLES_DATA["cortex"].model

        # Private role is dropped.
        assert "shadow" not in public.roles

    def test_public_returns_new_instance(self) -> None:
        """public() must not mutate the original Announcement."""
        a = Announcement(
            name="test",
            origin="http://x",
            schema_version="1.0.0",
            roles={"s": _private_role()},
        )
        public = a.public()
        # Original still has the role.
        assert "s" in a.roles
        # Public copy does not.
        assert "s" not in public.roles

    def test_public_roundtrip(self) -> None:
        """public() output must encode/decode back identically."""
        a = Announcement(
            name="test",
            origin="http://x",
            schema_version="1.0.0",
            roles={"s": _private_role(), "c": _ROLES_DATA["cortex"]},
        )
        public = a.public()
        decoded = decode(encode(public))
        assert decoded == public

    def test_private_role_still_encodes(self) -> None:
        """A non-public announcement with private=True roles still round-trips."""
        a = Announcement(
            name="test",
            origin="http://x",
            schema_version="1.0.0",
            roles={"s": _private_role()},
        )
        assert decode(encode(a)) == a

    def test_private_encoded_in_json(self) -> None:
        """private=True roles render as JSON booleans."""
        a = Announcement(
            name="test",
            origin="http://x",
            schema_version="1.0.0",
            roles={"s": _private_role()},
        )
        raw = encode(a)
        assert b'"private": true' in raw


# ---------------------------------------------------------------------------
# AC-3: RoleInfo field vocabulary mirrors lobes/roles.py
# ---------------------------------------------------------------------------


class TestRoleInfoFields:
    """AC-3: per-role payload reuses lobes/roles.py RoleInfo fields."""

    def test_role_info_has_same_fields(self) -> None:
        """Wire RoleInfo must carry every catalog RoleInfo field it advertises."""
        wire_fields = {f.name for f in dataclasses.fields(RoleInfo)}
        catalog_fields = {f.name for f in dataclasses.fields(CatalogRoleInfo)}
        # The wire RoleInfo must cover the catalog fields used on the wire.
        required = {
            "model",
            "runtime",
            "context",
            "quant",
            "responsibilities",
            "forbidden_responsibilities",
        }
        assert required.issubset(wire_fields), f"Wire RoleInfo missing: {required - wire_fields}"
        # No parallel vocabulary: every shared wire field has a home in the
        # catalog; the only sanctioned wire additions are the per-lane
        # fingerprint/capacity and the private flag.
        extra = wire_fields - catalog_fields
        assert extra <= {
            "private",
            "fingerprint",
            "capacity",
        }, f"Wire RoleInfo has parallel fields: {extra}"

    def test_role_info_field_types(self) -> None:
        """Each field must have the expected wire type."""
        hints = get_type_hints(RoleInfo)
        assert hints["model"] is str
        assert hints["runtime"] is str
        assert hints["context"] is int
        assert hints["quant"] is str
        assert hints["responsibilities"] == tuple[str, ...]
        assert hints["forbidden_responsibilities"] == tuple[str, ...]
        assert hints["fingerprint"] is Fingerprint
        assert hints["capacity"] == float | None
        assert hints["private"] is bool


# ---------------------------------------------------------------------------
# Fingerprint / DISQUALIFYING_FIELDS discipline
# ---------------------------------------------------------------------------


class TestFingerprintDiscipline:
    """The fingerprint's four fields must match _replicas.DISQUALIFYING_FIELDS."""

    def test_field_names_match_disqualifying(self) -> None:
        """Wire Fingerprint field order == DISQUALIFYING_FIELDS (per ROLE)."""
        fp_fields = tuple(f.name for f in dataclasses.fields(Fingerprint))
        assert fp_fields == DISQUALIFYING_FIELDS

    def test_fingerprint_has_exactly_four_fields(self) -> None:
        assert len(dataclasses.fields(Fingerprint)) == 4

    def test_module_level_guard_uses_same_set(self) -> None:
        """The import-time guard constant matches the disqualifying set."""
        assert _FINGERPRINT_FIELDS == DISQUALIFYING_FIELDS


# ---------------------------------------------------------------------------
# Announcement field inventory
# ---------------------------------------------------------------------------


class TestAnnouncementFields:
    """Announcement must carry ONLY name, origin, schema_version, roles."""

    def test_announcement_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(Announcement)}
        assert fields == {"name", "origin", "schema_version", "roles"}


# ---------------------------------------------------------------------------
# D8: decode() must never leak a bare KeyError on a malformed role
# ---------------------------------------------------------------------------


class TestMalformedRole:
    """D8: a malformed role object (missing fingerprint/model/…) is a 400
    (ValueError), never a 500 (an escaping KeyError)."""

    def test_decode_malformed_role_raises_value_error(self) -> None:
        a = _minimal()
        obj = json.loads(encode(a))
        del obj["roles"]["cortex"]["fingerprint"]
        with pytest.raises(ValueError):
            decode(json.dumps(obj).encode())

    def test_decode_malformed_role_is_not_bare_key_error(self) -> None:
        """The raised ValueError must not itself BE a KeyError — a caller
        catching (JSONDecodeError, ValueError, TypeError) must catch this."""
        a = _minimal()
        obj = json.loads(encode(a))
        del obj["roles"]["cortex"]["model"]
        try:
            decode(json.dumps(obj).encode())
        except ValueError as exc:
            assert not isinstance(exc, KeyError)
        else:
            pytest.fail("expected ValueError")


# ---------------------------------------------------------------------------
# MeshSchemaIncompatible exception
# ---------------------------------------------------------------------------


class TestMeshSchemaIncompatible:
    """MeshSchemaIncompatible must be a ValueError and carry both versions."""

    def test_is_value_error(self) -> None:
        assert issubclass(MeshSchemaIncompatible, ValueError)

    def test_message_contains_both_versions(self) -> None:
        raw = json.dumps(
            {
                "name": "test",
                "origin": "http://x",
                "schema_version": "3.0.0",
                "roles": {},
            }
        ).encode()
        with pytest.raises(MeshSchemaIncompatible) as exc_info:
            decode(raw)
        msg = str(exc_info.value)
        assert str(SCHEMA_MAJOR) in msg
        assert "3" in msg  # actual major from payload


# ---------------------------------------------------------------------------
# review #252 finding 15: decode() must validate structure before indexing
# ---------------------------------------------------------------------------


class TestMalformedTopLevel:
    """A structurally-wrong-but-valid-JSON body must raise ValueError, never
    an uncaught TypeError/AttributeError/KeyError."""

    def test_non_object_body_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            decode(json.dumps([1, 2, 3]).encode())

    def test_roles_not_a_mapping_raises_value_error(self) -> None:
        a = _minimal()
        obj = json.loads(encode(a))
        obj["roles"] = ["not", "a", "mapping"]
        with pytest.raises(ValueError):
            decode(json.dumps(obj).encode())

    def test_missing_name_raises_value_error(self) -> None:
        a = _minimal()
        obj = json.loads(encode(a))
        del obj["name"]
        with pytest.raises(ValueError):
            decode(json.dumps(obj).encode())

    def test_missing_origin_raises_value_error(self) -> None:
        a = _minimal()
        obj = json.loads(encode(a))
        del obj["origin"]
        with pytest.raises(ValueError):
            decode(json.dumps(obj).encode())

    def test_role_entry_not_a_mapping_raises_value_error(self) -> None:
        a = _minimal()
        obj = json.loads(encode(a))
        obj["roles"]["cortex"] = "not a mapping"
        with pytest.raises(ValueError):
            decode(json.dumps(obj).encode())

    def test_none_of_these_raise_a_bare_attribute_or_key_error(self) -> None:
        """Every malformed body above must raise ValueError specifically —
        not let AttributeError/KeyError/TypeError escape uncaught, which is
        what crashed the /mesh/announce handler before this fix."""
        a = _minimal()
        base = json.loads(encode(a))
        bad_bodies = []
        for mutate in (
            lambda o: o.__setitem__("roles", ["x"]),
            lambda o: o.pop("name"),
            lambda o: o.pop("origin"),
        ):
            obj = json.loads(json.dumps(base))
            mutate(obj)
            bad_bodies.append(obj)
        for obj in bad_bodies:
            try:
                decode(json.dumps(obj).encode())
            except ValueError:
                pass
            except (AttributeError, KeyError, TypeError) as exc:
                pytest.fail(f"expected ValueError, escaped {type(exc).__name__}: {exc}")
            else:
                pytest.fail("expected ValueError to be raised")
