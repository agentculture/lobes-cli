"""Tests for the per-role CORRECTNESS probes (issue #81, t7).

Two layers, mirroring tests/test_cli_measure.py's split:

* :mod:`lobes.assess` — the pure probe functions (``probe_cortex_correctness``,
  ``probe_embed_correctness``, ``probe_rerank_correctness``) plus the
  ``run_role_probes``/``render_role_probes`` glue. All monkeypatch the module's
  ``_post`` transport — no network, no GPU, no live model.
* ``lobes assess --probes`` (the CLI verb) — deployment/role-registry
  resolution, ``--json``/text rendering, ``--role`` filtering, and the
  read-only contract.

Unlike lobes.roles_measure (RUNTIME-ONLY — never a correctness claim), these
probes exist specifically to catch a service that is /health-healthy but
semantically WRONG, so several tests here assert exactly that: a stub that
looks fine at the transport level but returns the WRONG answer/ordering must
FAIL its probe.
"""

from __future__ import annotations

import json

import pytest

import lobes.assess as A
from lobes.cli import main
from lobes.cli._commands import assess as cli_assess
from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_SUCCESS
from lobes.runtime import _compose

# ---------------------------------------------------------------------------
# Fake transports (monkeypatched onto lobes.assess._post) — no network at all
# ---------------------------------------------------------------------------


def _fake_post_all_correct(url, payload, timeout=300, path="/v1/chat/completions"):
    if path == "/v1/chat/completions":
        return {
            "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 10},
        }
    if path == "/v1/embeddings":
        # sentence, paraphrase (near), unrelated (orthogonal) — paraphrase wins.
        return {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.99, 0.01]},
                {"index": 2, "embedding": [0.0, 1.0]},
            ]
        }
    if path == "/v1/rerank":
        return {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.2},
                {"index": 2, "relevance_score": 0.1},
            ]
        }
    raise AssertionError(f"unexpected path {path!r}")


def _fake_post_cortex_wrong(url, payload, timeout=300, path="/v1/chat/completions"):
    assert path == "/v1/chat/completions"
    return {
        "choices": [{"message": {"content": "London"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1, "prompt_tokens": 10},
    }


def _fake_post_embed_wrong(url, payload, timeout=300, path="/v1/chat/completions"):
    assert path == "/v1/embeddings"
    # A "healthy" (200 OK, well-shaped) response where the UNRELATED string
    # scores higher than the paraphrase — the semantically-wrong case.
    return {
        "data": [
            {"index": 0, "embedding": [1.0, 0.0]},
            {"index": 1, "embedding": [0.0, 1.0]},  # paraphrase: orthogonal, low sim
            {"index": 2, "embedding": [0.99, 0.01]},  # unrelated: near, high sim
        ]
    }


def _fake_post_rerank_wrong(url, payload, timeout=300, path="/v1/chat/completions"):
    assert path == "/v1/rerank"
    # A "healthy" 200 that ranks an IRRELEVANT document first.
    return {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 2, "relevance_score": 0.1},
        ]
    }


def _fake_post_timeout(url, payload, timeout=300, path="/v1/chat/completions"):
    raise TimeoutError("timed out")


def _fake_post_malformed(url, payload, timeout=300, path="/v1/chat/completions"):
    return {"unexpected": "shape"}


# 200-but-wrong-shape payloads (Qodo Finding 2): the JSON parses fine and the
# top-level keys are present, but a nested field the probe expects to be a
# dict is a bare string/list instead — `.get()` on it raises AttributeError,
# which the except tuples did NOT originally catch (an uncaught exception
# would abort `lobes assess --probes` instead of reporting a structured FAIL).


def _fake_post_cortex_message_is_string(url, payload, timeout=300, path="/v1/chat/completions"):
    assert path == "/v1/chat/completions"
    # "message" should be a dict ({"content": ...}); a 200 with it as a bare
    # string makes `d["choices"][0]["message"].get("content")` raise
    # AttributeError instead of KeyError/TypeError.
    return {"choices": [{"message": "not-a-dict", "finish_reason": "stop"}]}


def _fake_post_embed_items_are_strings(url, payload, timeout=300, path="/v1/chat/completions"):
    assert path == "/v1/embeddings"
    # Each "data" item should be a dict ({"index": ..., "embedding": ...}); a
    # 200 with bare strings makes `item.get("index", 0)` raise AttributeError
    # instead of KeyError/TypeError.
    return {"data": ["not-a-dict-0", "not-a-dict-1", "not-a-dict-2"]}


def _fake_post_rerank_items_are_strings(url, payload, timeout=300, path="/v1/chat/completions"):
    assert path == "/v1/rerank"
    # Each "results" item should be a dict ({"index": ..., "relevance_score":
    # ...}); a 200 with bare strings raises inside the sort key too (TypeError
    # here, since this probe never calls `.get()` — see the audit note on its
    # except tuple), and must still FAIL the probe, not raise.
    return {"results": ["not-a-dict-0", "not-a-dict-1"]}


# ---------------------------------------------------------------------------
# _cosine_similarity — pure math, no transport at all
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert A._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert A._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_degenerate_zero_vector_is_zero() -> None:
    assert A._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# probe_cortex_correctness — known-answer probe
# ---------------------------------------------------------------------------


def test_probe_cortex_correctness_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)
    r = A.probe_cortex_correctness("http://x", "model")
    assert r["role"] == "cortex"
    assert r["probe"] == "known_answer"
    assert r["ok"] is True
    assert r["evidence"] == {"expected": "paris", "content": "paris"}
    assert r["error"] is None
    assert r["latency_ms"] >= 0.0


def test_probe_cortex_correctness_fails_on_wrong_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Healthy transport (a clean 200), but the WRONG answer — must FAIL, not skip.
    monkeypatch.setattr(A, "_post", _fake_post_cortex_wrong)
    r = A.probe_cortex_correctness("http://x", "model")
    assert r["ok"] is False
    assert r["error"] is None  # ran fine — this is a semantic failure, not a transport one
    assert r["evidence"]["content"] == "london"


def test_probe_cortex_correctness_timeout_fails_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_timeout)
    r = A.probe_cortex_correctness("http://x", "model", timeout=0.01)
    assert r["ok"] is False
    assert "timed out" in r["error"]


def test_probe_cortex_correctness_malformed_response_fails_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_malformed)
    r = A.probe_cortex_correctness("http://x", "model")
    assert r["ok"] is False
    assert "unexpected response shape" in r["error"]


def test_probe_cortex_correctness_message_is_string_fails_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Qodo Finding 2: a 200 whose "message" is a bare string (not a dict) must
    # yield a structured FAIL, not an uncaught AttributeError from `.get()`.
    monkeypatch.setattr(A, "_post", _fake_post_cortex_message_is_string)
    r = A.probe_cortex_correctness("http://x", "model")
    assert r["ok"] is False
    assert r["role"] == "cortex"
    assert "unexpected response shape" in r["error"]
    assert "AttributeError" in r["error"]


# ---------------------------------------------------------------------------
# probe_embed_correctness — paraphrase-vs-unrelated cosine similarity probe
# ---------------------------------------------------------------------------


def test_probe_embed_correctness_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)
    r = A.probe_embed_correctness("http://x", "model")
    assert r["ok"] is True
    assert r["role"] == "embedder"
    assert r["probe"] == "embed_similarity"
    assert r["evidence"]["sim_paraphrase"] > r["evidence"]["sim_unrelated"]


def test_probe_embed_correctness_fails_when_unrelated_scores_higher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is exactly the case that would have caught the sm_110 FLASH_ATTN
    # hang's *semantic* cousin: a healthy-looking 200 whose embeddings are
    # simply wrong (unrelated string closer than the paraphrase).
    monkeypatch.setattr(A, "_post", _fake_post_embed_wrong)
    r = A.probe_embed_correctness("http://x", "model")
    assert r["ok"] is False
    assert r["error"] is None
    assert r["evidence"]["sim_unrelated"] > r["evidence"]["sim_paraphrase"]


def test_probe_embed_correctness_timeout_fails_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The sm_110 FLASH_ATTN hang: request accepted, never answered. The probe
    # must enforce a hard timeout and FAIL, never hang or silently skip.
    monkeypatch.setattr(A, "_post", _fake_post_timeout)
    r = A.probe_embed_correctness("http://x", "model", timeout=0.01)
    assert r["ok"] is False
    assert "timed out" in r["error"]


def test_probe_embed_correctness_malformed_response_fails_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_malformed)
    r = A.probe_embed_correctness("http://x", "model")
    assert r["ok"] is False
    assert "unexpected response shape" in r["error"]


def test_probe_embed_correctness_items_are_strings_fails_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Qodo Finding 2: a 200 whose "data" items are bare strings (not dicts)
    # must yield a structured FAIL, not an uncaught AttributeError from the
    # `item.get("index", 0)` sort key.
    monkeypatch.setattr(A, "_post", _fake_post_embed_items_are_strings)
    r = A.probe_embed_correctness("http://x", "model")
    assert r["ok"] is False
    assert r["role"] == "embedder"
    assert "unexpected response shape" in r["error"]
    assert "AttributeError" in r["error"]


# ---------------------------------------------------------------------------
# probe_rerank_correctness — relevant-doc-ranks-first probe
# ---------------------------------------------------------------------------


def test_probe_rerank_correctness_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)
    r = A.probe_rerank_correctness("http://x", "model")
    assert r["ok"] is True
    assert r["role"] == "reranker"
    assert r["probe"] == "rerank_relevance"
    assert r["evidence"]["top_index"] == A._RERANK_PROBE_RELEVANT_INDEX


def test_probe_rerank_correctness_fails_when_irrelevant_doc_ranks_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Context: on the Jetson Thor box this currently FAILS for real (#105/#106)
    # — wrong ordering. This test proves the probe code reports that faithfully.
    monkeypatch.setattr(A, "_post", _fake_post_rerank_wrong)
    r = A.probe_rerank_correctness("http://x", "model")
    assert r["ok"] is False
    assert r["error"] is None
    assert r["evidence"]["top_index"] != A._RERANK_PROBE_RELEVANT_INDEX


def test_probe_rerank_correctness_timeout_fails_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_timeout)
    r = A.probe_rerank_correctness("http://x", "model", timeout=0.01)
    assert r["ok"] is False
    assert "timed out" in r["error"]


def test_probe_rerank_correctness_malformed_response_fails_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_malformed)
    r = A.probe_rerank_correctness("http://x", "model")
    assert r["ok"] is False
    assert "unexpected response shape" in r["error"]


def test_probe_rerank_correctness_items_are_strings_fails_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Qodo Finding 2 audit: this probe never calls `.get()` (see the except
    # tuple's audit comment), so this malformed shape currently raises
    # TypeError, already caught — but it must still yield a structured FAIL,
    # never an uncaught exception, exactly like its cortex/embedder siblings.
    monkeypatch.setattr(A, "_post", _fake_post_rerank_items_are_strings)
    r = A.probe_rerank_correctness("http://x", "model")
    assert r["ok"] is False
    assert r["role"] == "reranker"
    assert "unexpected response shape" in r["error"]


# ---------------------------------------------------------------------------
# run_role_probes / render_role_probes — dispatch + rendering
# ---------------------------------------------------------------------------


def test_run_role_probes_unloaded_role_fails_without_a_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("must not call _post for a role with no endpoint")

    monkeypatch.setattr(A, "_post", boom)
    results = A.run_role_probes({"cortex": None, "embedder": None, "reranker": None})
    assert set(results) == set(A.PROBE_ROLES)
    for role in A.PROBE_ROLES:
        assert results[role]["ok"] is False
        assert results[role]["error"] == "role not loaded / no endpoint"
        assert results[role]["role"] == role


def test_run_role_probes_missing_endpoint_key_also_fails_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("must not call _post for a role missing from endpoints")

    monkeypatch.setattr(A, "_post", boom)
    results = A.run_role_probes({})
    assert set(results) == set(A.PROBE_ROLES)
    assert all(r["ok"] is False for r in results.values())


def test_run_role_probes_all_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)
    endpoints = {role: ("http://x", "model") for role in A.PROBE_ROLES}
    results = A.run_role_probes(endpoints)
    assert set(results) == set(A.PROBE_ROLES)
    assert all(r["ok"] for r in results.values())


def test_run_role_probes_role_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)
    endpoints = {role: ("http://x", "model") for role in A.PROBE_ROLES}
    results = A.run_role_probes(endpoints, roles=("embedder",))
    assert set(results) == {"embedder"}


def test_render_role_probes_shows_pass_and_overall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)
    endpoints = {role: ("http://x", "model") for role in A.PROBE_ROLES}
    results = A.run_role_probes(endpoints)
    md = A.render_role_probes(results)
    assert "PASS" in md
    assert "**Overall: PASS**" in md
    for role in A.PROBE_ROLES:
        assert role in md


def test_render_role_probes_shows_fail_and_overall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "_post", _fake_post_cortex_wrong)
    endpoints = {"cortex": ("http://x", "model")}
    results = A.run_role_probes(endpoints, roles=("cortex",))
    md = A.render_role_probes(results)
    assert "FAIL" in md
    assert "**Overall: FAIL**" in md


# ---------------------------------------------------------------------------
# CLI — lobes assess --probes
# ---------------------------------------------------------------------------


def _scaffold_fleet(path):
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


def test_cli_assess_probes_json_all_pass(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)
    rc = main(["assess", "--probes", "--compose-dir", str(tmp_path), "--port", "8000", "--json"])
    assert rc == EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert set(payload["probes"]) == set(A.PROBE_ROLES)
    for role in A.PROBE_ROLES:
        assert payload["probes"][role]["ok"] is True


def test_cli_assess_probes_json_reports_fail_for_wrong_role(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(A, "_post", _fake_post_rerank_wrong)
    rc = main(
        [
            "assess",
            "--probes",
            "--role",
            "reranker",
            "--compose-dir",
            str(tmp_path),
            "--port",
            "8000",
            "--json",
        ]
    )
    # A failing probe now differentiates the exit code (S3516) — EXIT_ENV_ERROR,
    # mirroring `lobes tunnel`'s status-driven exit code — while the --json
    # payload contract is unchanged (passed: false, same shape as a pass).
    assert rc == EXIT_ENV_ERROR
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert set(payload["probes"]) == {"reranker"}
    assert payload["probes"]["reranker"]["ok"] is False


def test_cli_assess_probes_text_mode_exit_code_reflects_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(A, "_post", _fake_post_rerank_wrong)
    rc = main(
        [
            "assess",
            "--probes",
            "--role",
            "reranker",
            "--compose-dir",
            str(tmp_path),
            "--port",
            "8000",
        ]
    )
    assert rc == EXIT_ENV_ERROR
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_cli_assess_probes_role_filter(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)
    rc = main(
        [
            "assess",
            "--probes",
            "--role",
            "embedder",
            "--compose-dir",
            str(tmp_path),
            "--port",
            "8000",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["probes"]) == {"embedder"}


def test_cli_assess_probes_unknown_role_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["assess", "--probes", "--role", "bogus"])
    assert exc.value.code == 1  # EXIT_USER_ERROR via the structured argparse error


def test_cli_assess_probes_text_mode_renders_markdown(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)
    rc = main(["assess", "--probes", "--compose-dir", str(tmp_path), "--port", "8000"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Per-role correctness probes" in out
    assert "PASS" in out


def test_cli_assess_probes_custom_timeout_reaches_the_probe(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    seen_timeouts: list[float] = []

    def _capture(url, payload, timeout=300, path="/v1/chat/completions"):
        seen_timeouts.append(timeout)
        return _fake_post_all_correct(url, payload, timeout=timeout, path=path)

    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(A, "_post", _capture)
    rc = main(
        [
            "assess",
            "--probes",
            "--timeout",
            "7.5",
            "--compose-dir",
            str(tmp_path),
            "--port",
            "8000",
            "--json",
        ]
    )
    assert rc == 0
    assert seen_timeouts and all(t == 7.5 for t in seen_timeouts)


def test_cli_assess_probes_never_touches_docker(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(A, "_post", _fake_post_all_correct)

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("assess --probes must never invoke docker/compose")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    monkeypatch.setattr(_compose, "compose_down", boom)
    monkeypatch.setattr(_compose, "_run", boom)
    rc = main(["assess", "--probes", "--compose-dir", str(tmp_path), "--port", "8000", "--json"])
    assert rc == 0


def test_cli_assess_probes_has_no_apply_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["assess", "--apply"])
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Qodo Finding 1 — --probes must send the stable role/capability alias
# ("cortex") to the gateway, not the concrete served model id, because every
# probe request travels THROUGH THE GATEWAY (RoleInfo.endpoint is always the
# gateway's own URL for the four gateway-fronted roles — see
# lobes.roles._gateway_role, `endpoint = gateway`). embedder/reranker have no
# such gateway alias (lobes.catalog.TIER_ROLE only covers the generate lane),
# so sending "embedder"/"reranker" literally would 404 as an unknown model
# (lobes.gateway._routing.is_unknown_model) — those two must keep sending the
# concrete served name.
# ---------------------------------------------------------------------------


def test_probe_model_sends_role_alias_for_cortex_only() -> None:
    from types import SimpleNamespace

    cortex_info = SimpleNamespace(model="sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP")
    embed_info = SimpleNamespace(model="Qwen/Qwen3-Embedding-0.6B")
    rerank_info = SimpleNamespace(model="Qwen/Qwen3-Reranker-0.6B")

    assert cli_assess._probe_model("cortex", cortex_info) == "cortex"
    assert cli_assess._probe_model("embedder", embed_info) == "Qwen/Qwen3-Embedding-0.6B"
    assert cli_assess._probe_model("reranker", rerank_info) == "Qwen/Qwen3-Reranker-0.6B"


def test_cli_assess_probes_cortex_uses_gateway_alias_embed_rerank_use_served_name(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    seen_models: dict[str, object] = {}

    def _capture(url, payload, timeout=300, path="/v1/chat/completions"):
        seen_models[path] = payload.get("model")
        return _fake_post_all_correct(url, payload, timeout=timeout, path=path)

    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(A, "_post", _capture)
    rc = main(["assess", "--probes", "--compose-dir", str(tmp_path), "--port", "8000", "--json"])
    assert rc == EXIT_SUCCESS
    # cortex: the stable gateway alias, resolved through the tier-alias table.
    assert seen_models["/v1/chat/completions"] == "cortex"
    # embedder/reranker: the concrete served id — the fleet scaffold default.
    assert seen_models["/v1/embeddings"] == "Qwen/Qwen3-Embedding-0.6B"
    assert seen_models["/v1/rerank"] == "Qwen/Qwen3-Reranker-0.6B"
