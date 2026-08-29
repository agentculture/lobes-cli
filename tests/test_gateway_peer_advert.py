"""A proxied role advertises the HOSTING PEER's own ready and context (#220).

Observed on the DGX Spark 2026-08-27: ``GET /capabilities`` reported
``associate -> ready:false, loaded:false, context:1048576`` while the Orin that
actually hosts the seat reported ``ready:true, context:128000`` and served a
request in 0.6 s. Two independent causes, both fixed here:

* **ready** — the peer probe asserted the peer's ``/v1/models`` lists the id
  this box forwards. That is unsatisfiable whenever the box forwards an ALIAS
  rather than a raw checkpoint id, which the ``associate`` lane must do (the
  alias is the only address that survives sharing a checkpoint with ``worker``).
* **context** — a role this box does not host has no ``<PREFIX>_MAX_MODEL_LEN``
  in this box's ``.env``, so the local computation fell through to the
  CATALOG's native ceiling: the checkpoint's maximum, not the window the peer
  chose to serve.

The fix relays the peer's own ``GET /capabilities`` entry — the surface the
audio lanes have always used (issue #129) — for BOTH facts, keeping the
``/v1/models`` check as a fallback for a peer that has no such entry.
"""

from __future__ import annotations

import json

from lobes.gateway import _readiness as R
from lobes.gateway._config import build_config
from lobes.gateway.server import capabilities_payload, peer_specs_from_table

_ORIGIN = "http://orin.tailnet:8000"
_LIGHTNING = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"


def _caps_opener(payload: dict, *, status: int = 200):
    """An opener that answers /capabilities and 404s everything else."""

    def opener(url: str, _timeout: float, _api_key: str | None = None):
        if url.endswith("/capabilities"):
            return status, json.dumps(payload).encode()
        return 404, b"{}"

    return opener


def _models_opener(ids: list[str]):
    def opener(url: str, _timeout: float, _api_key: str | None = None):
        if url.endswith("/v1/models"):
            return 200, json.dumps({"data": [{"id": i} for i in ids]}).encode()
        return 404, b"{}"  # no /capabilities entry — force the fallback

    return opener


# --- the probe --------------------------------------------------------------


def test_advert_relays_ready_and_context_from_the_peer() -> None:
    advert = R.probe_peer_advert(
        _ORIGIN,
        "associate",
        # NOTE: the served name is the ALIAS, which the peer's /v1/models will
        # never list — the exact case the old probe could not satisfy.
        "associate",
        opener=_caps_opener({"roles": {"associate": {"ready": True, "context": 128000}}}),
    )
    assert advert == R.PeerAdvert(ready=True, context=128000)


def test_a_peer_that_says_not_ready_is_relayed_as_not_ready() -> None:
    advert = R.probe_peer_advert(
        _ORIGIN,
        "associate",
        "associate",
        opener=_caps_opener({"roles": {"associate": {"ready": False, "context": 128000}}}),
    )
    assert advert.ready is False
    # ...but its context is still known, and still the peer's.
    assert advert.context == 128000


def test_a_peer_with_no_entry_for_the_role_falls_back_to_the_models_check() -> None:
    """An older lobes gateway, or a bare vLLM someone pointed a peer origin at:
    no working deployment loses its readiness signal on this change."""
    advert = R.probe_peer_advert(_ORIGIN, "senses", _LIGHTNING, opener=_models_opener([_LIGHTNING]))
    assert advert.ready is True
    assert advert.context is None  # unknown — the caller keeps its local answer


def test_the_fallback_still_refuses_a_peer_that_does_not_serve_the_id() -> None:
    advert = R.probe_peer_advert(
        _ORIGIN, "senses", _LIGHTNING, opener=_models_opener(["some/other-model"])
    )
    assert advert == R.PeerAdvert(ready=False, context=None)


def test_every_degradation_is_not_ready_and_never_raises() -> None:
    def boom(_url: str, _timeout: float, _key: str | None = None):
        raise OSError("connection refused")

    assert R.probe_peer_advert(_ORIGIN, "associate", "x", opener=boom) == R.PeerAdvert(False, None)
    assert R.probe_peer_advert(
        _ORIGIN, "associate", "x", opener=_caps_opener({}, status=503)
    ) == R.PeerAdvert(False, None)


def test_a_garbage_context_is_unknown_not_zero() -> None:
    """A 0/negative/non-numeric context is an ABSENT fact, not a served window
    — advertising 0 would be worse than falling back to the local answer."""
    for bad in (0, -1, "nonsense", None, True):
        advert = R.probe_peer_advert(
            _ORIGIN,
            "associate",
            "associate",
            opener=_caps_opener({"roles": {"associate": {"ready": True, "context": bad}}}),
        )
        assert advert.context is None, bad
    # ...and a numeric string IS a fact (compose/env values arrive as strings).
    advert = R.probe_peer_advert(
        _ORIGIN,
        "associate",
        "associate",
        opener=_caps_opener({"roles": {"associate": {"ready": True, "context": "128000"}}}),
    )
    assert advert.context == 128000


# --- the cache carries both halves -----------------------------------------


def test_cache_exposes_ready_in_current_and_context_in_its_own_accessor() -> None:
    spec = R.PeerSpec(name="associate", origin=_ORIGIN, served_name="associate", role="associate")
    cache = R.ReadinessCache(
        {},
        peer_specs=[spec],
        peer_probe=lambda _s: R.PeerAdvert(ready=True, context=128000),
        start=False,
    )
    cache.refresh()
    assert cache.current()["associate"] is True
    assert cache.current_peer_context()["associate"] == 128000


def test_a_pre_220_bool_probe_still_works() -> None:
    """An injected probe (or a third-party one) written against the old
    plain-bool contract keeps working and simply says nothing about context."""
    spec = R.PeerSpec(name="multimodal", origin=_ORIGIN, served_name="x", role="senses")
    cache = R.ReadinessCache({}, peer_specs=[spec], peer_probe=lambda _s: True, start=False)
    cache.refresh()
    assert cache.current()["multimodal"] is True
    assert cache.current_peer_context()["multimodal"] is None


def test_an_unprobed_peer_is_unknown_in_both_channels() -> None:
    spec = R.PeerSpec(name="multimodal", origin=_ORIGIN, served_name="x", role="senses")
    cache = R.ReadinessCache({}, peer_specs=[spec], peer_probe=lambda _s: True, start=False)
    assert cache.current()["multimodal"] is None
    assert cache.current_peer_context()["multimodal"] is None


def test_a_raising_probe_degrades_to_not_ready_without_killing_the_pass() -> None:
    def boom(_spec):
        raise RuntimeError("peer probe exploded")

    spec = R.PeerSpec(name="multimodal", origin=_ORIGIN, served_name="x", role="senses")
    cache = R.ReadinessCache({}, peer_specs=[spec], peer_probe=boom, start=False)
    cache.refresh()
    assert cache.current()["multimodal"] is False
    assert cache.current_peer_context()["multimodal"] is None


# --- the spec knows the ROLE name, not just the backend name ----------------


def test_peer_specs_carry_the_role_name_a_peers_capabilities_is_keyed_by() -> None:
    """`multimodal` is the BACKEND; `senses` is the role a peer's
    /capabilities lists. Getting this wrong would look up a key that is never
    present and silently fall back forever."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "some/cortex",
        "MULTIMODAL_FEASIBLE": "false",
        "MULTIMODAL_PEER_ORIGIN": _ORIGIN,
        "MULTIMODAL_PEER_PROXY": "true",
        "MULTIMODAL_SERVED_NAME": "some/senses",
    }
    table, _ = build_config(env)
    specs = peer_specs_from_table(table, env)
    assert specs["multimodal"].role == "senses"
    assert specs["multimodal"].role_name() == "senses"


def test_role_name_defaults_to_the_backend_name() -> None:
    """Every role whose backend shares its name (muse/worker/associate/hand/
    stt/tts) needs no declaration — and a hand-built spec must not break."""
    assert R.PeerSpec(name="worker", origin=_ORIGIN, served_name="x").role_name() == "worker"


# --- the payload ------------------------------------------------------------


def _proxied_associate_env() -> dict[str, str]:
    return {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "some/cortex",
        "PRIMARY_MAX_MODEL_LEN": "262144",
        "ASSOCIATE_FEASIBLE": "false",
        "ASSOCIATE_PEER_ORIGIN": _ORIGIN,
        "ASSOCIATE_PEER_PROXY": "true",
        "ASSOCIATE_SERVED_NAME": "associate",
        "ASSOCIATE_BASE_URL": "http://vllm-associate:8000",
    }


def test_capabilities_advertises_the_peers_context_for_a_proxied_role() -> None:
    env = _proxied_associate_env()
    table, cfg = build_config(env)
    payload = capabilities_payload(
        table,
        cfg,
        env,
        gateway_url="http://spark.tailnet:8000",
        backend_ready={"associate": True},
        peer_context={"associate": 128000},
    )
    assert payload["associate"]["ready"] is True
    assert payload["associate"]["context"] == 128000
    # `feasible: false` is unchanged and correct — it means "this box does not
    # HOST it", which is what `hosted_by` + `proxied` then qualify.
    assert payload["associate"]["feasible"] is False
    assert payload["associate"]["proxied"] is True
    assert payload["associate"]["hosted_by"] == _ORIGIN


def test_a_peer_that_said_nothing_keeps_the_local_answer() -> None:
    env = _proxied_associate_env()
    table, cfg = build_config(env)
    with_none = capabilities_payload(
        table, cfg, env, backend_ready={"associate": True}, peer_context={"associate": None}
    )
    without = capabilities_payload(table, cfg, env, backend_ready={"associate": True})
    assert with_none == without


def test_a_hosted_role_never_takes_a_peers_context() -> None:
    """Only a role in ``peer_proxied`` reads the peer channel — a box that
    hosts a lane always computes that lane's context from its own .env."""
    env = _proxied_associate_env()
    table, cfg = build_config(env)
    payload = capabilities_payload(
        table,
        cfg,
        env,
        backend_ready={"primary": True},
        # A malicious/confused peer mapping naming a HOSTED backend changes nothing.
        peer_context={"primary": 999, "associate": 128000},
    )
    assert payload["cortex"]["context"] == 262144


def test_no_peer_context_leaves_the_payload_byte_identical() -> None:
    """The h1 discipline: a deployment with no proxied roles is unaffected."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "some/cortex",
    }
    table, cfg = build_config(env)
    assert capabilities_payload(table, cfg, env, peer_context={}) == capabilities_payload(
        table, cfg, env
    )
