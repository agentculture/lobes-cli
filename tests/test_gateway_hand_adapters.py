"""Tests for the ``hand`` lobe's LoRA-adapter honesty surface.

The rule these all serve is #92 — *advertised implies reachable* — applied to
adapters rather than roles. An adapter is only ever advertised once the ENGINE
has confirmed it serves it, and an adapter nobody declared is refused rather
than quietly answered by the base weights.

Four collaborating pieces, one per section below:

* :func:`~lobes.gateway._config._hand_adapter_names` parses the operator's
  ``HAND_LORA_MODULES`` declaration — the single source both the engine's
  ``--lora-modules`` and the gateway's ``hand:<domain>`` aliases read, so the
  two cannot disagree.
* :func:`~lobes.gateway._readiness.probe_backend_adapters` asks the backend's
  own ``/v1/models`` which of those declared adapters actually loaded.
* :class:`~lobes.gateway._readiness.ReadinessCache` folds that probe into the
  background refresh and hands it back through ``current_adapters()``.
* :func:`~lobes.gateway._routing.list_models_payload` lists only the confirmed
  ones.

The failure mode worth guarding is a *silent* one: an adapter that vLLM refused
(unreadable file, rank above ``--max-lora-rank``, a checkpoint it rejected) is
absent from the engine's model list, and every one of these pieces must let
that absence propagate rather than paper over it with the declaration.

Stdlib only, mirroring the gateway's dependency-free discipline.
"""

from __future__ import annotations

import http.client
import json
import threading
import time

import pytest

from lobes.gateway import _config as C
from lobes.gateway import _readiness as R
from lobes.gateway._routing import Backend, RoutingTable, list_models_payload

# --- _hand_adapter_names: parsing the operator's declaration ----------------


def test_hand_adapter_names_parses_a_comma_separated_declaration() -> None:
    env = {C.HAND_LORA_MODULES_ENV: "legal=/adapters/legal,sql=/adapters/sql"}
    assert C._hand_adapter_names(env) == ("legal", "sql")


def test_hand_adapter_names_is_empty_when_undeclared_or_blank() -> None:
    # v1 ships zero adapters, so the EMPTY inventory is the default path, not an
    # edge case. It must be an empty tuple, never a one-element tuple of "".
    assert C._hand_adapter_names({}) == ()
    assert C._hand_adapter_names({C.HAND_LORA_MODULES_ENV: ""}) == ()
    assert C._hand_adapter_names({C.HAND_LORA_MODULES_ENV: "   "}) == ()


def test_hand_adapter_names_keeps_a_path_containing_equals() -> None:
    # partition("=") not split("="): a path with a query string or a padded
    # base64 segment keeps its full value, and the NAME is still just "legal".
    env = {C.HAND_LORA_MODULES_ENV: "legal=/adapters/legal?rev=v2"}
    assert C._hand_adapter_names(env) == ("legal",)


def test_hand_adapter_names_skips_malformed_segments() -> None:
    # A segment with no "=", no name, or no path is dropped rather than
    # producing a nameless alias that could never resolve.
    env = {
        C.HAND_LORA_MODULES_ENV: "good=/a,noequals,=/orphanpath,emptypath=,  ,ok=/b",
    }
    assert C._hand_adapter_names(env) == ("good", "ok")


def test_hand_adapter_names_dedupes_preserving_first_position() -> None:
    env = {C.HAND_LORA_MODULES_ENV: "legal=/a,sql=/b,legal=/c"}
    assert C._hand_adapter_names(env) == ("legal", "sql")


def test_hand_adapter_names_tolerates_surrounding_whitespace() -> None:
    env = {C.HAND_LORA_MODULES_ENV: "  legal = /adapters/legal , sql=/adapters/sql "}
    assert C._hand_adapter_names(env) == ("legal", "sql")


# --- probe_backend_adapters: ask the ENGINE, and fail closed ----------------


def _models_opener(status: int, ids: list[str]):
    """A ``PeerOpener`` returning a vLLM-shaped ``/v1/models`` body."""
    body = json.dumps({"object": "list", "data": [{"id": i} for i in ids]})

    def opener(_url: str, _timeout: float, _api_key: str | None):
        return status, body

    return opener


def test_probe_backend_adapters_returns_the_intersection() -> None:
    # Declared three, engine loaded two -> only the two that loaded.
    got = R.probe_backend_adapters(
        "http://vllm-hand:8000",
        ("legal", "sql", "refused"),
        opener=_models_opener(200, ["LiquidAI/LFM2.5-1.2B-Instruct", "legal", "sql"]),
    )
    assert got == frozenset({"legal", "sql"})


def test_probe_backend_adapters_ignores_ids_lobes_never_declared() -> None:
    # An engine listing an id we never declared cannot inject it into this
    # box's advertised surface — including the base checkpoint itself.
    got = R.probe_backend_adapters(
        "http://vllm-hand:8000",
        ("legal",),
        opener=_models_opener(200, ["legal", "smuggled", "LiquidAI/LFM2.5-1.2B-Instruct"]),
    )
    assert got == frozenset({"legal"})


def test_probe_backend_adapters_empty_declaration_opens_no_socket() -> None:
    calls: list[str] = []

    def opener(url: str, _t: float, _k: str | None):
        calls.append(url)
        return 200, "{}"

    assert R.probe_backend_adapters("http://vllm-hand:8000", (), opener=opener) == frozenset()
    assert calls == [], "an empty declaration must short-circuit before dialling"


def test_probe_backend_adapters_hits_the_models_path() -> None:
    seen: list[str] = []

    def opener(url: str, _t: float, _k: str | None):
        seen.append(url)
        return 200, json.dumps({"data": [{"id": "legal"}]})

    R.probe_backend_adapters("http://vllm-hand:8000/", ("legal",), opener=opener)
    assert seen == ["http://vllm-hand:8000" + R._MODELS_PATH]


def test_probe_backend_adapters_sends_no_api_key() -> None:
    # A co-resident fleet backend on the internal compose network, not a
    # cross-box peer: no credential should be presented.
    seen: list[object] = []

    def opener(_url: str, _t: float, api_key: str | None):
        seen.append(api_key)
        return 200, json.dumps({"data": [{"id": "legal"}]})

    R.probe_backend_adapters("http://vllm-hand:8000", ("legal",), opener=opener)
    assert seen == [None]


def test_probe_backend_adapters_fails_closed_on_non_200() -> None:
    got = R.probe_backend_adapters(
        "http://vllm-hand:8000", ("legal",), opener=_models_opener(503, ["legal"])
    )
    assert got == frozenset(), "a warming engine must advertise nothing, not its declaration"


def test_probe_backend_adapters_fails_closed_when_unreachable() -> None:
    for exc in (OSError("refused"), http.client.HTTPException("bad"), ValueError("port")):

        def boom(_u, _t, _k, _exc=exc):
            raise _exc

        assert R.probe_backend_adapters("http://vllm-hand:8000", ("legal",), opener=boom) == (
            frozenset()
        )


def test_probe_backend_adapters_fails_closed_on_malformed_body() -> None:
    # Not JSON, JSON of the wrong shape, and entries that are not dicts — none
    # may raise into the caller that folds this into /v1/models.
    for body in ("not json at all", '{"data": "not-a-list"}', "null", '{"data": [1, 2, 3]}'):

        def opener(_u, _t, _k, _b=body):
            return 200, _b

        assert R.probe_backend_adapters("http://vllm-hand:8000", ("legal",), opener=opener) == (
            frozenset()
        )


# --- ReadinessCache: the background adapter refresh -------------------------


def test_cache_seeds_adapters_empty_and_construction_opens_no_socket() -> None:
    calls: list[str] = []

    def probe(base_url: str, _declared: tuple[str, ...]) -> frozenset[str]:
        calls.append(base_url)
        return frozenset({"legal"})

    cache = R.ReadinessCache(
        {},
        adapter_targets={"hand": ("http://vllm-hand:8000", ("legal",))},
        adapter_probe=probe,
        start=False,
    )
    assert cache.current_adapters() == {"hand": frozenset()}
    assert calls == [], "construction must not probe"


def test_current_adapters_returns_a_copy_isolated_from_caller_mutation() -> None:
    cache = R.ReadinessCache(
        {},
        adapter_targets={"hand": ("http://vllm-hand:8000", ("legal",))},
        adapter_probe=lambda _u, _d: frozenset({"legal"}),
        start=False,
    )
    snapshot = cache.current_adapters()
    snapshot["hand"] = frozenset({"tampered"})
    snapshot["injected"] = frozenset()
    assert cache.current_adapters() == {"hand": frozenset()}


def test_background_refresh_populates_confirmed_adapters() -> None:
    seen = threading.Event()

    def probe(_base_url: str, declared: tuple[str, ...]) -> frozenset[str]:
        seen.set()
        return frozenset(set(declared) & {"legal"})

    cache = R.ReadinessCache(
        {},
        adapter_targets={"hand": ("http://vllm-hand:8000", ("legal", "refused"))},
        adapter_probe=probe,
        interval=0.01,
        start=True,
    )
    try:
        assert seen.wait(timeout=5.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if cache.current_adapters().get("hand") == frozenset({"legal"}):
                break
            time.sleep(0.01)
        assert cache.current_adapters() == {"hand": frozenset({"legal"})}
    finally:
        cache.stop()


def test_one_adapter_probe_raising_degrades_to_empty_and_spares_the_others() -> None:
    # Per-backend try: a misbehaving probe advertises nothing rather than
    # aborting the pass, so a second adapter-bearing lane still refreshes.
    done = threading.Event()

    def probe(base_url: str, _declared: tuple[str, ...]) -> frozenset[str]:
        if "broken" in base_url:
            raise RuntimeError("probe exploded")
        done.set()
        return frozenset({"ok"})

    cache = R.ReadinessCache(
        {},
        adapter_targets={
            "hand": ("http://broken:8000", ("legal",)),
            "other": ("http://good:8000", ("ok",)),
        },
        adapter_probe=probe,
        interval=0.01,
        start=True,
    )
    try:
        assert done.wait(timeout=5.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if cache.current_adapters().get("other") == frozenset({"ok"}):
                break
            time.sleep(0.01)
        snapshot = cache.current_adapters()
        assert snapshot["hand"] == frozenset(), "a raising probe must fail closed"
        assert snapshot["other"] == frozenset({"ok"}), "and must not abort the pass"
    finally:
        cache.stop()


def test_no_adapter_targets_leaves_current_adapters_empty() -> None:
    # The overwhelmingly common deployment: no adapters declared anywhere. The
    # adapter refresh must be skipped entirely, not run against nothing.
    cache = R.ReadinessCache({}, interval=0.01, start=True)
    try:
        time.sleep(0.05)
        assert cache.current_adapters() == {}
    finally:
        cache.stop()


# --- list_models_payload: only CONFIRMED adapters are listed ----------------


def _hand_table() -> RoutingTable:
    return RoutingTable(
        backends=(
            Backend(
                "hand",
                "http://vllm-hand:8000",
                "LiquidAI/LFM2.5-1.2B-Instruct",
                adapters=("legal", "sql"),
            ),
        ),
        default_model="LiquidAI/LFM2.5-1.2B-Instruct",
        aliases={},
    )


def _ids(payload: dict) -> list[str]:
    return [entry["id"] for entry in payload["data"]]


def test_list_models_omits_adapters_when_none_are_confirmed() -> None:
    # Declared but unconfirmed is INVISIBLE. This is the empty-inventory shape
    # v1 actually ships, and the live Spark/Orin probes match it: exactly one
    # entry, the base.
    assert _ids(list_models_payload(_hand_table())) == ["LiquidAI/LFM2.5-1.2B-Instruct"]
    assert _ids(list_models_payload(_hand_table(), loaded_adapters={})) == [
        "LiquidAI/LFM2.5-1.2B-Instruct"
    ]
    assert _ids(list_models_payload(_hand_table(), loaded_adapters={"hand": frozenset()})) == [
        "LiquidAI/LFM2.5-1.2B-Instruct"
    ]


def test_list_models_lists_only_the_confirmed_adapters() -> None:
    # "sql" was declared and did NOT load — it must not appear.
    payload = list_models_payload(_hand_table(), loaded_adapters={"hand": frozenset({"legal"})})
    assert _ids(payload) == ["LiquidAI/LFM2.5-1.2B-Instruct", "legal"]


def test_list_models_ignores_a_confirmation_for_an_undeclared_adapter() -> None:
    # Confirmation is an intersection, not a source of truth: an adapter the
    # routing table never declared cannot be listed even if a probe reports it.
    payload = list_models_payload(
        _hand_table(), loaded_adapters={"hand": frozenset({"legal", "smuggled"})}
    )
    assert _ids(payload) == ["LiquidAI/LFM2.5-1.2B-Instruct", "legal"]


def test_hand_adapter_aliases_are_derived_from_the_same_declaration() -> None:
    # The point of deriving both from HAND_LORA_MODULES: the gateway's
    # ``hand:<domain>`` aliases and the engine's ``--lora-modules`` cannot drift
    # apart, because there is only one declaration to read.
    table, _ = C.build_config(
        {
            "HAND_BASE_URL": "http://vllm-hand:8000",
            C.HAND_LORA_MODULES_ENV: "legal=/adapters/legal,sql=/adapters/sql",
        }
    )
    hand = next(b for b in table.backends if b.name == "hand")
    assert hand.adapters == ("legal", "sql")
    assert table.aliases[f"hand{C.HAND_ADAPTER_SEP}legal"] == "legal"
    assert table.aliases[f"hand{C.HAND_ADAPTER_SEP}sql"] == "sql"


def test_an_undeclared_hand_adapter_gets_no_alias() -> None:
    # No alias is what produces the ``model_not_found`` 404 rather than a
    # silent fall-back to the base weights — a caller who asked for the legal
    # specialist must never be handed a generalist answer and told it worked.
    table, _ = C.build_config(
        {
            "HAND_BASE_URL": "http://vllm-hand:8000",
            C.HAND_LORA_MODULES_ENV: "legal=/adapters/legal",
        }
    )
    assert f"hand{C.HAND_ADAPTER_SEP}nonexistent" not in table.aliases


def test_empty_inventory_yields_no_adapter_aliases_but_still_serves_the_base() -> None:
    table, _ = C.build_config({"HAND_BASE_URL": "http://vllm-hand:8000"})
    hand = next(b for b in table.backends if b.name == "hand")
    assert hand.adapters == ()
    assert not [a for a in table.aliases if a.startswith(f"hand{C.HAND_ADAPTER_SEP}")]
    assert table.aliases.get("hand") is not None, "model=hand must still resolve to the base"


# --- collision + shadowing warnings (Qodo review, PR #184) ------------------


def test_an_adapter_colliding_with_another_backends_served_name_warns(capsys) -> None:
    # _backend_for matches `requested in backend.adapters` as well as
    # served_name, so an adapter name is an ownership claim exactly like a
    # served name — and the collision resolves by backend order, silently.
    C._warn_on_served_name_collisions(
        [
            Backend("primary", "http://p:8000", "some/model"),
            Backend(
                "hand", "http://h:8000", "LiquidAI/LFM2.5-1.2B-Instruct", adapters=("some/model",)
            ),
        ]
    )
    err = capsys.readouterr().err
    assert "'some/model'" in err
    assert "order-dependent" in err
    assert "HAND_LORA_MODULES" in err, "the remedy must name the knob the duplicate came from"


def test_a_served_name_collision_still_recommends_served_name(capsys) -> None:
    # The pre-existing message must not regress into adapter advice when no
    # adapter is involved.
    C._warn_on_served_name_collisions(
        [
            Backend("embed", "http://e:8000", "same/id", task="embed"),
            Backend("embed_deep", "http://d:8000", "same/id", task="embed"),
        ]
    )
    err = capsys.readouterr().err
    assert "*_SERVED_NAME" in err
    assert "HAND_LORA_MODULES" not in err
    assert "WRONG VECTOR SPACE" in err, "the embed-specific detail must survive"


def test_no_warning_when_every_claimed_id_is_distinct(capsys) -> None:
    C._warn_on_served_name_collisions(
        [
            Backend("primary", "http://p:8000", "some/model"),
            Backend("hand", "http://h:8000", "base/id", adapters=("legal", "sql")),
        ]
    )
    assert capsys.readouterr().err == ""


def test_an_adapter_shadowed_by_an_alias_warns(capsys) -> None:
    # resolve_model checks aliases FIRST, so this adapter is unreachable by its
    # own name — a total shadow, not an order-dependent race.
    C._warn_on_adapter_alias_shadowing(
        [Backend("hand", "http://h:8000", "base/id", adapters=("cortex",))],
        {"cortex": "unsloth/Qwen3.6-27B-NVFP4"},
    )
    err = capsys.readouterr().err
    assert "'cortex'" in err
    assert "unreachable by name" in err


def test_the_hand_adapter_alias_itself_is_not_reported_as_shadowing(capsys) -> None:
    # `hand:legal -> legal` is the alias we deliberately mint; it must not warn
    # about the adapter it exists to reach.
    backends = [Backend("hand", "http://h:8000", "base/id", adapters=("legal",))]
    C._warn_on_adapter_alias_shadowing(backends, C._hand_adapter_aliases(backends))
    assert capsys.readouterr().err == ""


def test_hand_adapter_aliases_helper_is_empty_without_a_hand_backend() -> None:
    assert C._hand_adapter_aliases([]) == {}
    assert C._hand_adapter_aliases([Backend("primary", "http://p:8000", "m")]) == {}
    assert C._hand_adapter_aliases(
        [Backend("hand", "http://h:8000", "base/id", adapters=("legal",))]
    ) == {f"hand{C.HAND_ADAPTER_SEP}legal": "legal"}


def test_default_adapter_probe_uses_the_local_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default probe binds the LOCAL timeout, never the peer thread's
    # cross-box budget: these are co-resident lanes on the compose network.
    seen: dict[str, object] = {}

    def fake_probe(base_url, declared, *, timeout, opener=None):
        seen.update(base_url=base_url, declared=tuple(declared), timeout=timeout)
        return frozenset({"legal"})

    cache = R.ReadinessCache({}, timeout=1.25, start=False)
    monkeypatch.setattr(R, "probe_backend_adapters", fake_probe)
    got = cache._default_adapter_probe("http://vllm-hand:8000", ("legal",))
    assert got == frozenset({"legal"})
    assert seen == {
        "base_url": "http://vllm-hand:8000",
        "declared": ("legal",),
        "timeout": 1.25,
    }


def test_list_models_drops_adapters_of_an_unready_backend() -> None:
    # A backend filtered out by readiness takes its adapters with it — an
    # adapter cannot outlive the lane that serves it.
    payload = list_models_payload(
        _hand_table(),
        ready={"hand": False},
        loaded_adapters={"hand": frozenset({"legal"})},
    )
    assert _ids(payload) == []
