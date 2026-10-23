"""Speech-to-text lanes on the gateway (stt-lanes): ``stt-he`` / ``stt-en``.

``POST /v1/audio/transcriptions`` picks a LANE when the multipart ``model``
field (or, failing that, the ``X-Lobes-Model`` header) names one. Every other
request keeps the pre-lane path byte for byte: the path-keyed ``stt`` role on
``AUDIO_URL``. A selected lane is served locally behind ``<LANE>_BASE_URL``,
else forwarded ONE hop to the verified mesh member announcing it, else 404
``role_infeasible``; a wired lane that is down is a 503 — never another engine.
"""

from __future__ import annotations

import json
from email import policy
from email.parser import BytesParser

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._routing import TRANSCRIBE_TASK, model_routed_backends, resolve_model
from lobes.stt_lanes import READY_PATH, STT_LANES, lane_by_name

PATH = "/v1/audio/transcriptions"
AUDIO_URL = "http://realtime:8080"
HE_URL = "http://stt-he:9002"
EN_URL = "http://stt:9002"
BOUNDARY = "----lobesSttLaneBoundary"
CT = f"multipart/form-data; boundary={BOUNDARY}"
AUDIO = b"RIFF" + bytes(range(256)) * 16
HE = lane_by_name("stt-he")
EN = lane_by_name("stt-en")


def _body(**fields: str) -> bytes:
    b = BOUNDARY.encode()
    out = (
        b"--" + b + b'\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n" + AUDIO + b"\r\n"
    )
    for name, value in fields.items():
        out += (
            b"--"
            + b
            + f'\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            + value.encode()
            + b"\r\n"
        )
    return out + b"--" + b + b"--\r\n"


def _fields(body: bytes) -> dict[str, bytes]:
    msg = BytesParser(policy=policy.HTTP).parsebytes(f"Content-Type: {CT}\r\n\r\n".encode() + body)
    return {
        p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
        for p in msg.iter_parts()
    }


class _Up:
    def __init__(self, status=200, body=b'{"text": "shalom"}'):
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body

    def read_all(self):
        return self._body

    def read(self, _n=8192):
        data, self._body = self._body, b""
        return data

    def close(self):
        pass


class _Rec:
    """``open_upstream`` stub recording every dial; names in ``down`` refuse."""

    def __init__(self, down=(), status=200):
        self.calls: list[dict] = []
        self.down = set(down)
        self.status = status

    def __call__(self, backend, path, body, headers, *, connect_timeout, read_timeout):
        self.calls.append(
            {
                "name": backend.name,
                "base_url": backend.base_url,
                "path": path,
                "body": body,
                "headers": list(headers),
            }
        )
        if backend.name in self.down:
            raise S.UpstreamError(f"{backend.name}: connection refused")
        return _Up(self.status)


def _env(*lanes: str, **extra: str) -> dict[str, str]:
    env = {"AUDIO_URL": AUDIO_URL}
    urls = {"stt-he": HE_URL, "stt-en": EN_URL}
    for name in lanes:
        env[lane_by_name(name).base_url_env] = urls[name]
    env.update(extra)
    return env


def _post(env, body, *, headers=(), rec=None, snap=None, audio_ready=True):
    table, cfg = build_config(env)
    rec = rec or _Rec()
    resp = S.handle_audio_request(
        table,
        cfg,
        {},
        PATH,
        [("Content-Type", CT), *headers],
        body,
        rec,
        audio_ready_probe=lambda: audio_ready,
        mesh_snapshot=snap,
    )
    return resp, rec


def _hdr(resp, name):
    return [v for k, v in resp.headers if k.lower() == name.lower()]


# --- config ------------------------------------------------------------------


def test_unwired_lanes_add_nothing_to_the_routing_table():
    base, _ = build_config({"AUDIO_URL": AUDIO_URL})
    assert not any(b.task == TRANSCRIBE_TASK for b in base.backends)
    assert "stt-he" not in base.aliases and "stt-en" not in base.aliases


def test_wired_lanes_become_path_routed_backends():
    table, _ = build_config(_env("stt-he", "stt-en", STT_MODEL="ivrit-ai/whisper-large-v3-turbo"))
    lanes = {b.name: b for b in table.backends if b.task == TRANSCRIBE_TASK}
    assert lanes["stt-he"].base_url == HE_URL
    assert lanes["stt-he"].served_name == HE.catalog_id
    assert lanes["stt-en"].base_url == EN_URL
    assert lanes["stt-en"].served_name == "ivrit-ai/whisper-large-v3-turbo"
    # Never addressable through a JSON `model` field, nor via an alias.
    assert not any(b.task == TRANSCRIBE_TASK for b in model_routed_backends(table))
    assert "stt-he" not in table.aliases
    assert resolve_model(table, "stt-he") == table.default_model


def test_declared_infeasible_lane_is_in_table_infeasible():
    table, _ = build_config(_env("stt-he", STT_HE_FEASIBLE="false"))
    assert "stt-he" in table.infeasible
    assert "stt-en" not in table.infeasible


def test_readiness_probes_lanes_on_the_sidecar_ready_path():
    table, _ = build_config(_env("stt-he", "stt-en"))
    paths = S.readiness_paths(table)
    assert paths["stt-he"] == READY_PATH == "/v1/health/ready"
    assert paths["stt-en"] == READY_PATH
    assert "primary" not in paths  # every other backend keeps the shared /health


def test_endpoints_advertise_transcriptions_for_a_wired_lane_only_once():
    table, _ = build_config({"STT_HE_BASE_URL": HE_URL})
    eps = S._endpoints_for(table, audio=False)
    assert eps.count("POST /v1/audio/transcriptions") == 1
    table, _ = build_config({"STT_HE_BASE_URL": HE_URL})
    assert S._endpoints_for(table, audio=True).count("POST /v1/audio/transcriptions") == 1


# --- selection ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "headers", "expected"),
    [
        ({"model": "stt-he"}, (), "stt-he"),
        ({"model": "stt-en"}, (), "stt-en"),
        ({}, (("X-Lobes-Model", "stt-he"),), "stt-he"),
        ({"model": "whisper-1"}, (("x-lobes-model", "stt-en"),), "stt-en"),
        # Both name a lane and disagree: the form field (the OpenAI-native
        # selector, part of the signed body) wins.
        ({"model": "stt-en"}, (("X-Lobes-Model", "stt-he"),), "stt-en"),
        ({}, (), None),
        ({"model": "stt"}, (), None),
        ({"model": "whisper-1"}, (), None),
        ({"model": "ivrit-ai/whisper-large-v3-turbo"}, (), None),
        ({"model": "stt-fr"}, (), None),
        ({}, (("X-Lobes-Model", "decider:general"),), None),
    ],
)
def test_lane_selection(fields, headers, expected):
    headers = [("Content-Type", CT), *headers]
    lane = S.select_stt_lane(PATH, headers, _body(**fields))
    assert (lane.name if lane else None) == expected


def test_selection_applies_to_transcriptions_only():
    headers = [("Content-Type", CT), ("X-Lobes-Model", "stt-he")]
    assert S.select_stt_lane("/v1/audio/speech", headers, _body(model="stt-he")) is None
    assert S.select_stt_lane(PATH + "?x=1", headers, _body()) is not None


# --- the legacy path stays byte-identical ----------------------------------------


@pytest.mark.parametrize("fields", [{}, {"model": "whisper-1"}, {"model": "stt"}])
def test_non_lane_requests_take_the_stt_role_path_unchanged(fields):
    body = _body(**fields)
    resp, rec = _post(_env("stt-he", "stt-en"), body)
    assert resp.status == 200
    assert [(c["name"], c["base_url"], c["path"]) for c in rec.calls] == [
        ("audio", AUDIO_URL, PATH)
    ]
    assert rec.calls[0]["body"] == body  # verbatim: no language injected
    assert not _hdr(resp, S.LANE_HEADER)


# --- local lanes -----------------------------------------------------------------


def test_local_lane_gets_its_language_injected():
    resp, rec = _post(_env("stt-he"), _body(model="stt-he", response_format="json"))
    assert resp.status == 200
    assert [(c["name"], c["base_url"], c["path"]) for c in rec.calls] == [("stt-he", HE_URL, PATH)]
    sent = _fields(rec.calls[0]["body"])
    assert sent["language"] == b"he"
    assert sent["model"] == b"stt-he"
    assert sent["response_format"] == b"json"
    assert sent["file"] == AUDIO
    assert _hdr(resp, S.LANE_HEADER) == ["stt-he"]


def test_stt_en_is_the_local_stt_container_asked_for_english():
    resp, rec = _post(_env("stt-en"), _body(model="stt-en"))
    assert resp.status == 200
    assert rec.calls[0]["base_url"] == EN_URL
    assert _fields(rec.calls[0]["body"])["language"] == b"en"


def test_header_selected_lane_injects_too():
    resp, rec = _post(_env("stt-he"), _body(), headers=[("X-Lobes-Model", "stt-he")])
    assert resp.status == 200
    assert _fields(rec.calls[0]["body"])["language"] == b"he"


def test_caller_language_is_kept_verbatim():
    body = _body(model="stt-en", language="he")
    resp, rec = _post(_env("stt-en"), body)
    assert resp.status == 200
    assert rec.calls[0]["body"] == body


def test_upstream_status_is_relayed():
    resp, rec = _post(_env("stt-he"), _body(model="stt-he"), rec=_Rec(status=415))
    assert resp.status == 415


def test_down_lane_is_503_and_never_falls_back():
    rec = _Rec(down={"stt-he"})
    resp, rec = _post(_env("stt-he"), _body(model="stt-he"), rec=rec)
    assert resp.status == 503
    assert _hdr(resp, "Retry-After")
    assert json.loads(resp.body)["error"]["type"] == "backend_unavailable"
    assert [c["name"] for c in rec.calls] == ["stt-he"]  # not AUDIO_URL


def test_lane_not_hosted_is_404_role_infeasible_even_with_audio_wired():
    resp, rec = _post(_env("stt-en"), _body(model="stt-he"))
    assert resp.status == 404
    err = json.loads(resp.body)["error"]
    assert err["type"] == err["code"] == "role_infeasible"
    assert "stt-he" in err["message"]
    assert rec.calls == []


def test_declared_infeasible_lane_is_404_and_never_dialed():
    resp, rec = _post(_env("stt-he", STT_HE_FEASIBLE="false"), _body(model="stt-he"))
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "role_infeasible"
    assert rec.calls == []


def test_a_lane_is_served_even_when_the_stt_role_is_declared_off():
    resp, rec = _post(_env("stt-he", STT_FEASIBLE="false"), _body(model="stt-he"))
    assert resp.status == 200
    assert rec.calls[0]["name"] == "stt-he"


def test_hop_marked_arrival_is_served_locally():
    resp, rec = _post(_env("stt-he"), _body(model="stt-he"), headers=[(S.PROXIED_HEADER, "stt-he")])
    assert resp.status == 200
    assert rec.calls[0]["name"] == "stt-he"


# --- /capabilities -----------------------------------------------------------------


def _caps(env, ready=None, snap=None):
    table, cfg = build_config(env)
    return S.capabilities_payload(
        table, cfg, env, gateway_url="http://gw:8000", backend_ready=ready, mesh_snapshot=snap
    )


def test_unwired_lanes_have_no_capabilities_key():
    payload = _caps({"AUDIO_URL": AUDIO_URL})
    assert "stt-he" not in payload and "stt-en" not in payload


def test_wired_lane_capabilities_entry():
    env = _env("stt-he", "stt-en", STT_MODEL="ivrit-ai/whisper-large-v3-turbo")
    env["STT_RUNTIME"] = "transformers"
    payload = _caps(env, ready={"stt-he": True, "stt-en": False})
    he = payload["stt-he"]
    assert he["lane"] is True
    assert he["task"] == "transcribe"
    assert he["model"] == HE.catalog_id
    assert he["runtime"] == "sherpa-onnx"
    assert he["language"] == "he"
    assert he["revision"] == HE.revision
    assert he["licence"] == "CC-BY-4.0"
    assert he["path"] == PATH
    assert he["endpoint"] == "http://gw:8000"
    assert he["feasible"] is True and he["ready"] is True
    assert he["fingerprint"]["served_id"] == HE.catalog_id
    assert he["fingerprint"]["runtime"] == "sherpa-onnx"
    en = payload["stt-en"]
    assert en["model"] == "ivrit-ai/whisper-large-v3-turbo"
    assert en["runtime"] == "transformers"
    assert en["language"] == "en"
    assert en["ready"] is False
    assert "revision" not in en and "licence" not in en


def test_infeasible_lane_is_advertised_not_ready():
    payload = _caps(_env("stt-he", STT_HE_FEASIBLE="false"), ready={"stt-he": True})
    assert payload["stt-he"]["feasible"] is False
    assert payload["stt-he"]["ready"] is False


def test_lane_ids_listed_on_v1_models_only_when_servable():
    table, _ = build_config(_env("stt-he", "stt-en"))
    assert S.stt_lane_model_ids(table, {"stt-he": True, "stt-en": None}, None) == ("stt-he",)
    assert S.stt_lane_model_ids(table, None, None) == ("stt-he", "stt-en")
    table, _ = build_config(_env("stt-he", STT_HE_FEASIBLE="false"))
    assert S.stt_lane_model_ids(table, {"stt-he": True}, None) == ()


def test_every_registry_lane_is_covered_here():
    assert {lane.name for lane in STT_LANES} == {"stt-he", "stt-en"}
