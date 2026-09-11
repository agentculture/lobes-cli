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
    SCHEMA_MAJOR,
    Announcement,
    Fingerprint,
    MeshSchemaIncompatible,
    RoleInfo,
    decode,
    encode,
)
from lobes.gateway._replicas import DISQUALIFYING_FIELDS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ROLES_DATA = {
    "cortex": RoleInfo(
        model="unsloth/Qwen3.8-27B-NVFP4",
        runtime="vllm",
        context=131072,
        quant="NVFP4",
        responsibilities=("reasoning", "deciding", "planning"),
        forbidden_responsibilities=(),
    ),
    "senses": RoleInfo(
        model="coolthor/gemma-4-12B-it-NVFP4A16",
        runtime="vllm",
        context=32768,
        quant="NVFP4",
        responsibilities=("intake", "normalize_input"),
        forbidden_responsibilities=("final_decision",),
    ),
}

_FINGERPRINT = Fingerprint(
    served_id="unsloth/Qwen3.8-27B-NVFP4",
    quantization="NVFP4",
    max_model_len=131072,
    runtime="vllm",
)


@pytest.fixture()
def announcement() -> Announcement:
    """A minimal Announcement with two public roles."""
    return Announcement(
        name="spark-lobe",
        origin="http://localhost:8000",
        schema_version="1.0.0",
        roles=_ROLES_DATA,
        fingerprint=_FINGERPRINT,
        capacity=8.0,
        private=False,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _wire(announcement: Announcement) -> bytes:
    """Round-trip encode -> raw JSON so we can inject unknown fields."""
    return encode(announcement)


# ---------------------------------------------------------------------------
# AC-1: encode / decode round-trip, unknown fields, schema mismatch
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """AC-1: decode(encode(a)) == a."""

    def test_roundtrip(self, announcement: Announcement) -> None:
        decoded = decode(encode(announcement))
        assert decoded == announcement

    def test_roundtrip_defaults(self) -> None:
        """Round-trip without an explicit private field (defaults to False)."""
        a = Announcement(
            name="test",
            origin="http://localhost:8000",
            schema_version="1.0.0",
            roles={},
            fingerprint=_FINGERPRINT,
            capacity=1.0,
        )
        assert decode(encode(a)) == a


class TestUnknownFields:
    """AC-1: unknown extra fields are ignored on decode."""

    def test_unknown_fields_at_top_level(self) -> None:
        raw = _wire(
            Announcement(
                name="test",
                origin="http://x",
                schema_version="1.0.0",
                roles={},
                fingerprint=_FINGERPRINT,
                capacity=1.0,
                private=False,
            )
        )
        # Inject a bogus field the decoder does not know about.
        modified = raw.replace(
            b'"private": false',
            b'"private": false, "x_nope": 42',
        )
        a = decode(modified)
        assert a.name == "test"
        assert a.private is False

    def test_unknown_fields_in_role_payload(self) -> None:
        raw = _wire(
            Announcement(
                name="test",
                origin="http://x",
                schema_version="1.0.0",
                roles={
                    "cortex": RoleInfo(
                        model="x",
                        runtime="v",
                        context=0,
                        quant="",
                        responsibilities=(),
                        forbidden_responsibilities=(),
                    )
                },
                fingerprint=_FINGERPRINT,
                capacity=1.0,
                private=False,
            )
        )
        # Inject extra keys into the role object.
        modified = raw.replace(
            b'"forbidden_responsibilities": []',
            b'"forbidden_responsibilities": [], "extra_role_key": "gone"',
        )
        a = decode(modified)
        assert a.roles["cortex"].model == "x"


class TestSchemaIncompatible:
    """AC-1: different major schema version raises MeshSchemaIncompatible."""

    def test_major_mismatch(self) -> None:
        raw = json.dumps(
            {
                "name": "test",
                "origin": "http://x",
                "schema_version": "2.0.0",  # different major
                "roles": {},
                "fingerprint": dataclasses.asdict(_FINGERPRINT),
                "capacity": 1.0,
                "private": False,
            }
        ).encode()
        with pytest.raises(MeshSchemaIncompatible) as exc_info:
            decode(raw)
        # Both versions must appear in the message.
        msg = str(exc_info.value)
        assert str(SCHEMA_MAJOR) in msg  # expected major
        assert "2" in msg  # actual major from payload

    def test_minor_upgrade_ignored(self) -> None:
        """Patch and minor bumps must NOT raise — major is what gates."""
        raw = json.dumps(
            {
                "name": "test",
                "origin": "http://x",
                "schema_version": "1.1.0",
                "roles": {},
                "fingerprint": dataclasses.asdict(_FINGERPRINT),
                "capacity": 1.0,
                "private": False,
            }
        ).encode()
        result = decode(raw)
        assert result.schema_version == "1.1.0"


# ---------------------------------------------------------------------------
# AC-2: private role filtering via Announcement.public()
# ---------------------------------------------------------------------------


class TestPrivate:
    """AC-2: private=True roles are dropped by .public(); flag defaults False."""

    def test_private_flag_defaults_false(self) -> None:
        """When private is omitted, it must default to False."""
        r = RoleInfo(
            model="x",
            runtime="v",
            context=0,
            quant="",
            responsibilities=(),
            forbidden_responsibilities=(),
        )
        assert r.private is False

    def test_public_drops_private_roles(self) -> None:
        """A role with private=True is absent from the public view."""
        pub_role = RoleInfo(
            model="unsloth/Qwen3.8-27B-NVFP4",
            runtime="vllm",
            context=131072,
            quant="NVFP4",
            responsibilities=("reasoning",),
            forbidden_responsibilities=(),
        )
        priv_role = RoleInfo(
            model="secret/model",
            runtime="llamacpp",
            context=8192,
            quant="Q4",
            responsibilities=("hidden",),
            forbidden_responsibilities=(),
            private=True,
        )
        a = Announcement(
            name="secret-box",
            origin="http://localhost:9000",
            schema_version="1.0.0",
            roles={"cortex": pub_role, "shadow": priv_role},
            fingerprint=_FINGERPRINT,
            capacity=4.0,
        )
        public = a.public()

        # Public role survives.
        assert "cortex" in public.roles
        assert public.roles["cortex"].model == pub_role.model

        # Private role is dropped.
        assert "shadow" not in public.roles

    def test_public_returns_new_instance(self) -> None:
        """public() must not mutate the original Announcement."""
        priv_role = RoleInfo(
            model="s",
            runtime="v",
            context=0,
            quant="",
            responsibilities=(),
            forbidden_responsibilities=(),
            private=True,
        )
        a = Announcement(
            name="test",
            origin="http://x",
            schema_version="1.0.0",
            roles={"s": priv_role},
            fingerprint=_FINGERPRINT,
            capacity=1.0,
        )
        public = a.public()
        # Original still has the role.
        assert "s" in a.roles
        # Public copy does not.
        assert "s" not in public.roles

    def test_public_with_private_announcement(self) -> None:
        """Even a private announcement's private roles are dropped."""
        priv_role = RoleInfo(
            model="s",
            runtime="v",
            context=0,
            quant="",
            responsibilities=(),
            forbidden_responsibilities=(),
            private=True,
        )
        pub_role = RoleInfo(
            model="unsloth/Qwen3.8-27B-NVFP4",
            runtime="vllm",
            context=131072,
            quant="NVFP4",
            responsibilities=("reasoning",),
            forbidden_responsibilities=(),
        )
        a = Announcement(
            name="secret",
            origin="http://x",
            schema_version="1.0.0",
            roles={"s": priv_role, "cortex": pub_role},
            fingerprint=_FINGERPRINT,
            capacity=1.0,
            private=True,
        )
        public = a.public()
        assert "cortex" in public.roles
        assert "s" not in public.roles

    def test_public_roundtrip(self) -> None:
        """public() output must encode/decode back identically."""
        priv_role = RoleInfo(
            model="s",
            runtime="v",
            context=0,
            quant="",
            responsibilities=(),
            forbidden_responsibilities=(),
            private=True,
        )
        a = Announcement(
            name="test",
            origin="http://x",
            schema_version="1.0.0",
            roles={"s": priv_role, "c": _ROLES_DATA["cortex"]},
            fingerprint=_FINGERPRINT,
            capacity=1.0,
        )
        public = a.public()
        decoded = decode(encode(public))
        assert decoded == public

    def test_private_role_still_encodes(self) -> None:
        """A non-public announcement with private=True roles still round-trips."""
        priv_role = RoleInfo(
            model="s",
            runtime="v",
            context=0,
            quant="",
            responsibilities=(),
            forbidden_responsibilities=(),
            private=True,
        )
        a = Announcement(
            name="test",
            origin="http://x",
            schema_version="1.0.0",
            roles={"s": priv_role},
            fingerprint=_FINGERPRINT,
            capacity=1.0,
        )
        assert decode(encode(a)) == a

    def test_private_encoded_in_json(self) -> None:
        """private=True roles render as JSON booleans."""
        priv_role = RoleInfo(
            model="s",
            runtime="v",
            context=0,
            quant="",
            responsibilities=(),
            forbidden_responsibilities=(),
            private=True,
        )
        a = Announcement(
            name="test",
            origin="http://x",
            schema_version="1.0.0",
            roles={"s": priv_role},
            fingerprint=_FINGERPRINT,
            capacity=1.0,
        )
        raw = encode(a)
        assert b'"private": true' in raw


# ---------------------------------------------------------------------------
# AC-3: RoleInfo field vocabulary mirrors lobes/roles.py
# ---------------------------------------------------------------------------


class TestRoleInfoFields:
    """AC-3: per-role payload reuses lobes/roles.py RoleInfo fields."""

    def test_role_info_has_same_fields(self) -> None:
        """Wire RoleInfo must carry every catalog RoleInfo field."""
        wire_fields = {f.name for f in dataclasses.fields(RoleInfo)}
        catalog_fields = {
            f.name
            for f in dataclasses.fields(  # type: ignore[arg-type]
                __import__("lobes.roles", fromlist=["RoleInfo"]).RoleInfo,
            )
        }
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
        # No parallel vocabulary: wire fields must be a subset of catalog.
        extra = wire_fields - catalog_fields
        # Allow the wire to add `private` (not in catalog RoleInfo) but nothing
        # else that overlaps catalog fields.
        assert extra <= {"private"}, f"Wire RoleInfo has parallel fields: {extra}"

    def test_role_info_field_types(self) -> None:
        """Each field must have the expected wire type."""
        hints = get_type_hints(RoleInfo)
        assert hints["model"] is str
        assert hints["runtime"] is str
        assert hints["context"] is int
        assert hints["quant"] is str
        assert hints["responsibilities"] == tuple[str, ...]
        assert hints["forbidden_responsibilities"] == tuple[str, ...]


# ---------------------------------------------------------------------------
# Fingerprint / DISQUALIFYING_FIELDS discipline
# ---------------------------------------------------------------------------


class TestFingerprintDiscipline:
    """The fingerprint's four fields must match _replicas.DISQUALIFYING_FIELDS."""

    def test_field_names_match_disqualifying(self) -> None:
        """Wire Fingerprint field order == DISQUALIFYING_FIELDS."""
        fp_fields = tuple(f.name for f in dataclasses.fields(Fingerprint))
        assert fp_fields == DISQUALIFYING_FIELDS

    def test_fingerprint_has_exactly_four_fields(self) -> None:
        assert len(dataclasses.fields(Fingerprint)) == 4


# ---------------------------------------------------------------------------
# Announcement field inventory
# ---------------------------------------------------------------------------


class TestAnnouncementFields:
    """Announcement must carry name, origin, schema_version, roles,
    fingerprint, capacity, and the private flag."""

    def test_announcement_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(Announcement)}
        required = {
            "name",
            "origin",
            "schema_version",
            "roles",
            "fingerprint",
            "capacity",
            "private",
        }
        assert required == fields

    def test_private_default_is_false(self) -> None:
        defaults = {
            f.name: f.default
            for f in dataclasses.fields(Announcement)
            if f.default is not dataclasses.MISSING
        }
        defaults.update(
            {
                f.name: f.default_factory  # type: ignore[operator]
                for f in dataclasses.fields(Announcement)
                if f.default_factory is not dataclasses.MISSING
            }
        )
        assert defaults.get("private") is False


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
                "fingerprint": dataclasses.asdict(_FINGERPRINT),
                "capacity": 1.0,
                "private": False,
            }
        ).encode()
        with pytest.raises(MeshSchemaIncompatible) as exc_info:
            decode(raw)
        msg = str(exc_info.value)
        assert str(SCHEMA_MAJOR) in msg
        assert "3" in msg  # actual major from payload
