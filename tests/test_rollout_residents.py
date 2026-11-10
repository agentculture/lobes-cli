"""Residents file: generate, regenerate, load, save."""

import pytest

from lobes.rollout import _residents as r

ROSTER = {
    "members": [
        {"name": "spark", "origin": "http://spark:8001", "verified": True, "roles": ["cortex"]},
        {"name": "orin-nano", "origin": "http://orin-nano:8000", "verified": True, "roles": []},
        {"name": "ghost", "origin": "http://ghost:8000", "verified": False, "roles": []},
    ]
}
FULL = {
    "ssh": "spark-f8a9",
    "user": "arduino",
    "port": 8001,
    "lobes_path": "/home/arduino/.local/bin/lobes",
    "gateway_key": True,
}


def test_generate_marks_every_field():
    res = r.generate(ROSTER, {"spark": FULL, "orin-nano": {"ssh": "orin-nano"}})
    spark = res.residents["spark"]
    assert spark["ssh"].state == r.DETECTED and spark["ssh"].value == "spark-f8a9"
    assert spark["gateway_key"].value is True
    nano = res.residents["orin-nano"]
    assert nano["ssh"].state == r.DETECTED
    for f in ("user", "port", "lobes_path", "gateway_key"):
        assert nano[f].state == r.NEEDS_YOU and nano[f].value is None
    assert "ghost" not in res.residents  # unverified not listed


def test_mesh_name_not_hostname():
    res = r.generate(ROSTER, {"spark": {"user": "a"}})
    assert res.residents["spark"]["ssh"].state == r.NEEDS_YOU


def test_regenerate_keeps_edits_and_lists_guests():
    first = r.generate({"members": ROSTER["members"][:1]}, {"spark": FULL})
    first.residents["spark"]["user"] = r.Field("operator", r.DETECTED)
    res = r.regenerate(first, ROSTER, {"spark": FULL, "orin-nano": {"ssh": "orin-nano"}})
    assert res.residents["spark"]["user"].value == "operator"
    assert list(res.residents) == ["spark"]
    assert list(res.guests) == ["orin-nano"]


def test_round_trip(tmp_path):
    res = r.generate(ROSTER, {"spark": FULL, "orin-nano": {"ssh": "orin-nano"}})
    p = r.save(res, tmp_path, "residents.toml")
    assert p.read_text().count("[resident.spark]") == 1
    back = r.load(tmp_path, "residents.toml")
    assert back.residents == res.residents


def test_refuses_escape(tmp_path):
    res = r.generate(ROSTER, {})
    for bad in ("../x.toml", "/etc/x.toml", "a/../../x.toml"):
        with pytest.raises(ValueError):
            r.save(res, tmp_path, bad)
        with pytest.raises(ValueError):
            r.load(tmp_path, bad)


def test_load_missing_is_empty(tmp_path):
    assert r.load(tmp_path, "residents.toml").residents == {}
