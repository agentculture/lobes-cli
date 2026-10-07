"""Per-lane lifecycle + probe for the specialist embed/rerank lanes (oes t9).

No docker, no network: ``up`` tests assert the (monkeypatched) compose argv,
``assess`` tests patch the HTTP primitive. The behavioural obligation (o10):
``lobes up <lane>`` starts or stops only that lane's service, dry-run default.
"""

from __future__ import annotations

import json
import types

import pytest

from lobes import assess as _assess
from lobes.cli import main
from lobes.cli._commands import up as up_cmd
from lobes.embed_lanes import EMBED_LANES, TASK_SCORE
from lobes.runtime import _compose, _health, _lanes

LANE_NAMES = [lane.name for lane in EMBED_LANES]


def _ok() -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _deploy(path, *, lanes: bool = True):
    """A fleet scaffold; ``lanes`` adds a local override defining every lane service."""
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    if lanes:
        body = "services:\n" + "".join(f"  embed-{n}:\n    image: x\n" for n in LANE_NAMES)
        (path / _compose.LOCAL_OVERRIDE).write_text(body, encoding="utf-8")
    return path


def test_lane_targets_come_from_registry_and_never_collide() -> None:
    lanes = set(LANE_NAMES)
    assert lanes <= set(up_cmd.TARGETS)
    assert len(up_cmd.TARGETS) == len(set(up_cmd.TARGETS))
    assert not lanes & (set(up_cmd.ROLE_SERVICE) | {"colleague-stack", "gateway"})
    assert up_cmd.LANE_SERVICE == {n: f"embed-{n}" for n in LANE_NAMES}


@pytest.mark.parametrize("lane", LANE_NAMES)
def test_up_lane_dry_run_touches_only_that_service(tmp_path, capsys, monkeypatch, lane) -> None:
    _deploy(tmp_path)
    monkeypatch.setattr(
        _compose, "run_compose", lambda *a: pytest.fail("dry-run must not run compose")
    )
    rc = main(["up", lane, "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["services"] == [f"embed-{lane}"]
    assert payload["command"].endswith(f"up -d --no-deps embed-{lane}")


@pytest.mark.parametrize("lane", LANE_NAMES)
def test_up_lane_apply_and_down_run_one_service(tmp_path, monkeypatch, lane) -> None:
    _deploy(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(_compose, "run_compose", lambda d, argv: (calls.append(argv), _ok())[1])
    assert main(["up", lane, "--compose-dir", str(tmp_path), "--apply"]) == 0
    assert main(["up", lane, "--compose-dir", str(tmp_path), "--apply", "--down"]) == 0
    svc = f"embed-{lane}"
    assert calls == [
        ["docker", "compose", "up", "-d", "--no-deps", svc],
        ["docker", "compose", "stop", svc],
    ]


def test_up_lane_refused_when_compose_set_lacks_service(tmp_path, capsys) -> None:
    _deploy(tmp_path, lanes=False)
    rc = main(["up", "gemma2-embed", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "embed-gemma2-embed" in err
    assert "does not define" in err


def test_up_lane_rejects_replace(tmp_path) -> None:
    _deploy(tmp_path)
    assert main(["up", "gemma2-embed", "--compose-dir", str(tmp_path), "--replace"]) == 1


def test_status_lists_defined_lanes_only(tmp_path, capsys, monkeypatch) -> None:
    _deploy(tmp_path)
    monkeypatch.setattr(_compose, "inspect_state", lambda name="x": "running")
    monkeypatch.setattr(_health, "is_healthy", lambda port, timeout=3.0: True)
    assert main(["status", "--json", "--compose-dir", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [lane["name"] for lane in payload["lanes"]] == LANE_NAMES
    assert all(lane["state"] == "running" for lane in payload["lanes"])


def test_status_omits_lanes_key_without_overlay(tmp_path, capsys, monkeypatch) -> None:
    _deploy(tmp_path, lanes=False)
    monkeypatch.setattr(_compose, "inspect_state", lambda name="x": "running")
    monkeypatch.setattr(_health, "is_healthy", lambda port, timeout=3.0: True)
    assert main(["status", "--json", "--compose-dir", str(tmp_path)]) == 0
    assert "lanes" not in json.loads(capsys.readouterr().out)


def _assess_deploy(tmp_path, lane: str, url: str = "http://lane:8000"):
    _deploy(tmp_path, lanes=False)
    with (tmp_path / _compose.ENV_FILE).open("a", encoding="utf-8") as fh:
        fh.write(f"\n{lane.upper().replace('-', '_')}_BASE_URL={url}\n")
    return tmp_path


def test_assess_embed_lane_uses_negative_control(tmp_path, capsys, monkeypatch) -> None:
    """Related text must score above unrelated text; the probe sees real vectors."""
    _assess_deploy(tmp_path, "gemma2-embed")
    seen: dict = {}

    def fake_post(url, payload, timeout=300, *, path=""):
        seen.update(url=url, path=path, model=payload["model"])
        vecs = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]}

    monkeypatch.setattr(_assess, "_post", fake_post)
    rc = main(["assess", "gemma2-embed", "--compose-dir", str(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["passed"] is True
    assert seen == {
        "url": "http://lane:8000",
        "path": "/v1/embeddings",
        "model": "google/embeddinggemma-2",
    }
    assert out["probes"]["gemma2-embed"]["role"] == "gemma2-embed"

    def inverted(url, payload, timeout=300, *, path=""):
        vecs = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
        return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]}

    monkeypatch.setattr(_assess, "_post", inverted)
    assert main(["assess", "gemma2-embed", "--compose-dir", str(tmp_path), "--json"]) != 0


def test_assess_score_lane_uses_rerank_endpoint(tmp_path, capsys, monkeypatch) -> None:
    lane = next(lane for lane in EMBED_LANES if lane.task == TASK_SCORE)
    _assess_deploy(tmp_path, lane.name)
    seen: dict = {}

    def fake_post(url, payload, timeout=300, *, path=""):
        seen["path"] = path
        n = len(payload["documents"])
        idx = _assess._RERANK_PROBE_RELEVANT_INDEX
        return {
            "results": [{"index": i, "relevance_score": 1.0 if i == idx else 0.1} for i in range(n)]
        }

    monkeypatch.setattr(_assess, "_post", fake_post)
    assert main(["assess", lane.name, "--compose-dir", str(tmp_path), "--json"]) == 0
    assert seen["path"] == "/v1/rerank"
    capsys.readouterr()


def test_assess_lane_without_endpoint_fails_without_network(tmp_path, capsys, monkeypatch) -> None:
    _deploy(tmp_path, lanes=False)
    monkeypatch.setattr(_assess, "_post", lambda *a, **k: pytest.fail("no endpoint, no call"))
    rc = main(["assess", "nemotron-embed", "--compose-dir", str(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc != 0
    assert out["passed"] is False
    assert "NEMOTRON_EMBED_BASE_URL" in out["probes"]["nemotron-embed"]["error"]


def test_assess_unknown_lane_is_user_error(tmp_path) -> None:
    assert main(["assess", "bogus-lane", "--compose-dir", str(tmp_path)]) == 1


def test_lane_helpers_chain_default(tmp_path) -> None:
    _deploy(tmp_path)
    assert set(_lanes.defined_lane_services(tmp_path, [])) == set(LANE_NAMES)
