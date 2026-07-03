"""Tests for ``lobes capabilities`` / ``lobes endpoint`` (issue #81, task t5).

These verbs are the CLI-side view of the six first-class Colleague-facing
roles (``cortex``/``senses``/``embedder``/``reranker``/``stt``/``tts``), built
by the ONE canonical registry builder in :mod:`lobes.roles`. Both are strictly
read-only — no compose/docker call, no ``--apply``.
"""

from __future__ import annotations

import json

import pytest

from lobes.cli import main
from lobes.roles import ROLES
from lobes.runtime import _compose

_ROLE_INFO_FIELDS = {
    "role",
    "model",
    "runtime",
    "endpoint",
    "path",
    "context",
    "quant",
    "mtp",
    "responsibilities",
    "forbidden_responsibilities",
    "ready",
    "loaded",
}


def _scaffold_fleet(path):
    """Write the packaged fleet templates verbatim — the SAME .env `lobes init
    --fleet` would scaffold, so the served-context overlay assertions below
    exercise the real shipped defaults (PRIMARY_MAX_MODEL_LEN=131072,
    MULTIMODAL_MAX_MODEL_LEN=32768, ...), not a hand-rolled fixture."""
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


# ---------------------------------------------------------------------------
# lobes capabilities
# ---------------------------------------------------------------------------


def test_capabilities_json_returns_all_six_roles_with_full_metadata(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(ROLES)
    for role in ROLES:
        info = payload[role]
        assert _ROLE_INFO_FIELDS <= set(info)
        assert info["role"] == role
        assert info["model"]  # never blank


def test_capabilities_json_reports_served_context_not_catalog_native(tmp_path, capsys) -> None:
    """The #81 contract: context is the SERVED --max-model-len from the
    deployment env, not the catalog native (t5's core behaviour change)."""
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cortex"]["context"] == 131072
    assert payload["senses"]["context"] == 32768
    assert payload["cortex"]["loaded"] is True
    assert payload["senses"]["loaded"] is True
    assert payload["embedder"]["loaded"] is True
    assert payload["reranker"]["loaded"] is True
    # Audio overlay not scaffolded here (no --audio) → present, unloaded.
    assert payload["stt"]["loaded"] is False
    assert payload["tts"]["loaded"] is False
    assert payload["stt"]["context"] == 0
    assert payload["tts"]["context"] == 0


def test_capabilities_json_endpoint_is_gateway_base_url(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # VLLM_PORT=8000 in the packaged fleet env.example.
    assert payload["cortex"]["endpoint"] == "http://localhost:8000"
    assert payload["embedder"]["endpoint"] == "http://localhost:8000"


def test_capabilities_non_json_renders_readable_table_with_all_six_roles(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    for role in ROLES:
        assert role in out
    assert "responsibilities:" in out
    assert "131072" in out  # served cortex context visible in the table


def test_capabilities_unscaffolded_still_answers_all_six_roles(capsys) -> None:
    """Read-only: with nothing scaffolded, capabilities degrades gracefully to
    catalog defaults (all unloaded except the always-present cortex) instead of
    erroring — mirrors 'lobes overview --live' on an empty deployment."""
    rc = main(["capabilities", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(ROLES)
    assert payload["cortex"]["loaded"] is True  # primary is always wired
    assert payload["senses"]["loaded"] is False
    assert payload["embedder"]["loaded"] is False
    assert payload["reranker"]["loaded"] is False
    # No overlay env available either → catalog native.
    from lobes.catalog import SUPPORTED_MODELS

    primary_native = next(
        m.native_max_model_len for m in SUPPORTED_MODELS if m.id == payload["cortex"]["model"]
    )
    assert payload["cortex"]["context"] == primary_native


def test_capabilities_never_touches_docker(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)

    def boom(*a, **k):
        raise AssertionError("capabilities must never invoke docker/compose")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    monkeypatch.setattr(_compose, "compose_down", boom)
    monkeypatch.setattr(_compose, "_run", boom)
    monkeypatch.setattr(_compose, "_probe", boom)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0


def test_capabilities_has_no_apply_flag(capsys) -> None:
    """Read-only verb: no --apply, unlike switch/serve/stop/init/fleet/tunnel."""
    with pytest.raises(SystemExit) as exc:
        main(["capabilities", "--apply"])
    assert exc.value.code == 1  # EXIT_USER_ERROR via the structured argparse error


# ---------------------------------------------------------------------------
# lobes endpoint <role>
# ---------------------------------------------------------------------------


def test_endpoint_prints_gateway_base_url_for_cortex(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["endpoint", "cortex", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "http://localhost:8000"


def test_endpoint_json_shape(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["endpoint", "embedder", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"role": "embedder", "endpoint": "http://localhost:8000"}


def test_endpoint_works_for_every_role(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    # The four gateway-fronted roles resolve to the reachable gateway URL; the
    # audio roles (stt/tts) are unwired here (no --audio overlay) → blank, but
    # 'lobes endpoint' still exits 0 for every known role, wired or not.
    expected = {
        "cortex": "http://localhost:8000",
        "senses": "http://localhost:8000",
        "embedder": "http://localhost:8000",
        "reranker": "http://localhost:8000",
        "stt": "",
        "tts": "",
    }
    assert set(expected) == set(ROLES)
    for role in ROLES:
        rc = main(["endpoint", role, "--compose-dir", str(tmp_path)])
        assert rc == 0
        assert capsys.readouterr().out.strip() == expected[role]


def test_endpoint_unknown_role_exits_user_error_with_hint(capsys) -> None:
    rc = main(["endpoint", "bogus"])
    assert rc == 1  # EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err
    for role in ROLES:
        assert role in err


def test_endpoint_unknown_role_json_error_shape(capsys) -> None:
    rc = main(["endpoint", "bogus", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == 1
    assert "bogus" in payload["message"]
    for role in ROLES:
        assert role in payload["remediation"]


def test_endpoint_never_touches_docker(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)

    def boom(*a, **k):
        raise AssertionError("endpoint must never invoke docker/compose")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    monkeypatch.setattr(_compose, "compose_down", boom)
    monkeypatch.setattr(_compose, "_run", boom)
    monkeypatch.setattr(_compose, "_probe", boom)
    rc = main(["endpoint", "cortex", "--compose-dir", str(tmp_path)])
    assert rc == 0


# ---------------------------------------------------------------------------
# Registration — both verbs show up in --help / overview, don't break either
# ---------------------------------------------------------------------------


def test_capabilities_and_endpoint_appear_in_top_level_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "capabilities" in out
    assert "endpoint" in out


def test_overview_still_works_and_lists_the_new_verbs(capsys) -> None:
    rc = main(["overview", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    verbs_section = next(s for s in payload["sections"] if s["title"] == "Verbs")
    joined = " ".join(verbs_section["items"])
    assert "capabilities" in joined
    assert "endpoint" in joined
