"""Live capabilities gate — the executable "advertised implies reachable" check.

This is the local pre-PR gate the plan of the same name exists to create. It is
**not** a CI job: CI has no GPU and no fleet. It is run by a developer, against a
real running deployment, before opening a PR — the single trigger is
``scripts/live-check.sh`` (which resolves the port the way the CLI does, arms
this module via ``LOBES_SMOKE_BASE_URL``, and returns a pass/fail exit code).

Why it exists
-------------
Its absence is precisely why the #87 reachability fix shipped in 0.38.0 while the
reference rig kept running 0.36.0 for five days, and why #92 was filed against
lobes as a *code* regression when the code was already correct and merely
undeployed. A green unit suite says the code is right; it says nothing about
whether the code a caller actually dials is the code that shipped. This gate
closes that gap by dialing the deployment and asserting that everything the
deployment *advertises* it can actually *reach*.

Fail, never skip, when armed
----------------------------
The module is gated on ``LOBES_SMOKE_BASE_URL`` (reused from
``tests/test_smoke_duo.py`` for consistency). When it is **unset**, every test
here skips cleanly, so the offline suite stays green. When it is **set**, an
operator has explicitly asked for the gate — so an unreachable deployment,
a 404 on an advertised path, or a version skew must **FAIL the run**, never
degrade to ``pytest.skip``. A gate that skips when things are broken is worse
than no gate: it is the exact silent-pass that let the drift above happen. There
is therefore no runtime ``pytest.skip`` anywhere below the module gate.

The five checks, and the issues each enforces
---------------------------------------------
1. ``test_advertised_ready_roles_are_reachable`` — for every role in
   ``GET /capabilities`` whose ``ready`` is true, dial ``endpoint + path`` and
   assert the response proves the path is reachably served. A **404** on an
   advertised, ready path is the defect (#92, #96). A dead backend that a
   pre-honest-readiness gateway relays as a bare **5xx** without ``Retry-After``
   (e.g. the poisoned-CUDA-context STT sidecar, #89) is also caught — it is
   "advertised ready but not honestly reachable".
2. ``test_advertised_models_are_reachable`` — for every id in ``GET /v1/models``,
   dial it on its own task lane and assert it is never a 404 "model does not
   exist" (#91). A model the fleet lists must never read as "will never exist".
   The lane is resolved from the live contract (pooling models go to
   ``/v1/embeddings`` / ``/v1/rerank``, everything else to
   ``/v1/chat/completions``) so a listed *generate* model that 404s is the
   genuine #91 defect, while a pooling model is never falsely dialed on a chat
   route it was never meant to answer.
3. ``test_cli_and_gateway_capabilities_agree`` — ``lobes capabilities --json``
   and ``GET /capabilities`` must agree on ``endpoint``, ``ready`` and
   ``loaded`` for all six roles (#95, folded into #92). Since t7 the CLI is a
   *client* of the gateway, so agreement is by construction whenever the gateway
   answers — this check proves the deployment actually implements that (a pre-t7
   CLI/gateway pair would diverge) and that ``lobes capabilities`` reaches the
   gateway at all (``source == "gateway"``, not the degraded offline fallback).
4. ``test_deployed_gateway_version_matches_cli`` — the gateway's ``GET /health``
   reports ``{"version": ...}`` (#99). Compare it to the CLI's
   ``lobes.__version__`` and FAIL on mismatch. ``Dockerfile.gateway`` runs
   ``pip install "lobes-cli==${MODEL_GEAR_VERSION}"``, ``lobes init`` writes that
   pin once, and no verb ever re-bumps it — so merged fixes never reach a
   deployment unless someone redeploys. A ``/health`` with **no** ``version``
   field at all means the gateway image predates this work: that is skew, not a
   pass.
5. ``test_colleague_discovers_and_dials_cortex_and_senses`` — reproduce a
   Colleague's discovery path (#81, #87): given ONLY the gateway origin and no
   ``COLLEAGUE_*_BASE_URL`` override, resolve ``cortex`` and ``senses`` from the
   contract alone and get an answer. No model id is hardcoded — the model, the
   endpoint and the path all come from ``GET /capabilities``.

Reachability classification — 404 vs 429 vs 503 vs 5xx
------------------------------------------------------
A busy or warming box must not turn the gate red; only a genuinely unreachable
or dishonest one must. So :func:`_classify` treats a response as **reachable**
when it is any 2xx (served), any non-404 4xx (the endpoint parsed and rejected
our deliberately-minimal request, so the path exists), a **429** ``server_busy``
(the pressure-shed policy working, #88 — the reference rig currently misreads
sticky swap occupancy, #100, which is exactly why 429 must not be a failure), or
a **503 that carries ``Retry-After``** (the honest "owner dead / backend warming"
answer this plan introduced, #14/#89). It treats a response as **unreachable**
only when it is a **404** (advertised path absent), a **connection failure**, a
**503 without ``Retry-After``**, or any **other 5xx** (a bare relay of a dead
backend with no honest retry signal — the STT 502 case). On a 429 the probe
retries once with the documented ``X-Lobes-Override: 1`` header, which forces the
request past the shed so a real 404 hiding behind pressure is still surfaced.

Everything uses the standard library only (``urllib`` + ``json`` + ``struct``),
mirroring ``lobes/assess.py`` and ``tests/test_smoke_duo.py``.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess  # nosec B404 — invokes this repo's own `python -m lobes`, no shell
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import pytest

from lobes import __version__ as LOBES_VERSION
from lobes.roles import ROLES

# ---------------------------------------------------------------------------
# Module gate — armed iff LOBES_SMOKE_BASE_URL is set (fail-not-skip below it).
# ---------------------------------------------------------------------------
_BASE_URL = (os.environ.get("LOBES_SMOKE_BASE_URL") or "").rstrip("/")

pytestmark = pytest.mark.skipif(
    not _BASE_URL,
    reason=(
        "live capabilities gate needs a running deployment — set "
        "LOBES_SMOKE_BASE_URL=http://localhost:<port> (use scripts/live-check.sh, "
        "which resolves the port from .env and arms this gate)"
    ),
)

# Generate lanes (cortex/senses, and any un-roled generate model) can be slow
# under load and cortex is a thinking model — be generous. Pooling/audio lanes
# answer fast. Neither ever blocks forever: a stalled socket must FAIL the armed
# gate, not hang an operator's pre-PR run.
_GENERATE_TIMEOUT = 120
_LANE_TIMEOUT = 60
_META_TIMEOUT = 15

_OVERRIDE_HEADER = "X-Lobes-Override"


# ---------------------------------------------------------------------------
# Stdlib HTTP with a structured result (no requests, mirroring lobes/assess.py).
# ---------------------------------------------------------------------------
@dataclass
class _Probe:
    """The outcome of one dial: an HTTP status (or None on connection failure)."""

    status: int | None
    retry_after: str | None
    body: bytes
    error: str | None


def _url(path: str) -> str:
    return _BASE_URL + path


def _http(method: str, url: str, *, headers=None, data=None, timeout: int) -> _Probe:
    """One request. Never raises — a connection failure folds into ``status=None``."""
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # local deployment only
            return _Probe(resp.status, resp.headers.get("Retry-After"), resp.read(), None)
    except urllib.error.HTTPError as exc:  # a real HTTP status (4xx/5xx) — reachable wire
        return _Probe(exc.code, exc.headers.get("Retry-After"), exc.read(), None)
    except (urllib.error.URLError, OSError) as exc:  # refused / DNS / timeout — no wire
        reason = getattr(exc, "reason", exc)
        return _Probe(None, None, b"", str(reason))


def _classify(p: _Probe) -> tuple[bool, str]:
    """Map a probe to ``(reachable, human_verdict)`` — the heart of the gate.

    See the module docstring for the full 404/429/503/5xx rationale.
    """
    if p.status is None:
        return False, f"connection failure ({p.error})"
    if p.status == 404:
        return False, "404 not-found"
    if p.status == 429:
        return True, "429 server_busy (pressure shed #88 — reachable)"
    if p.status == 503:
        if p.retry_after is not None:
            return True, "503 + Retry-After (honest warming/dead-owner — reachable)"
        return False, "503 without Retry-After (bare relay, not honest)"
    if 200 <= p.status < 300:
        return True, f"{p.status} served"
    if 400 <= p.status < 500:
        return True, f"{p.status} (request rejected — path exists, reachable)"
    return False, f"{p.status} backend error (no honest Retry-After — dishonest relay)"


def _dial(method: str, url: str, *, data=None, content_type=None, timeout: int) -> _Probe:
    """Dial once; on a 429 pressure-shed, force-serve once via ``X-Lobes-Override``.

    Returns the probe the gate should classify. A 429 already means "reachable",
    but forcing the request past the shed surfaces the true underlying status
    (so a 404 hiding behind host pressure is still caught). If the override retry
    is itself shed again or blips on the wire, keep the original 429 (still
    reachable — a busy box must not turn the gate red).
    """
    headers = {"Content-Type": content_type} if content_type else {}
    p = _http(method, url, headers=headers, data=data, timeout=timeout)
    if p.status == 429:
        forced = _http(
            method,
            url,
            headers={**headers, _OVERRIDE_HEADER: "1"},
            data=data,
            timeout=timeout,
        )
        if forced.status is not None and forced.status != 429:
            return forced
    return p


# ---------------------------------------------------------------------------
# Payload builders — minimal, but real enough to EXERCISE the backend so a dead
# one reveals itself (an empty STT request 422s at the facade without ever
# reaching the poisoned sidecar; a real file 502s — the fault the gate exists
# to catch).
# ---------------------------------------------------------------------------
def _tiny_wav() -> bytes:
    """A 1-sample 8 kHz/8-bit mono WAV, built with struct so it can't drift."""
    num_channels, sample_rate, bits = 1, 8000, 8
    block_align = num_channels * bits // 8
    byte_rate = sample_rate * block_align
    audio = b"\x80\x00"  # one silence sample + even-length pad
    fmt = struct.pack("<HHIIHH", 1, num_channels, sample_rate, byte_rate, block_align, bits)
    riff_size = 4 + (8 + 16) + (8 + len(audio))
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", 1)
        + audio
    )


def _stt_multipart(model: str) -> tuple[bytes, str]:
    """A multipart/form-data body carrying a tiny WAV + model field for STT."""
    boundary = "----lobeslivecheckboundary7f3a"
    parts = [
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n' f"{model}\r\n"
        ).encode(),
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="probe.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        ).encode(),
        _tiny_wav() + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _role_dial(role: str, info: dict) -> _Probe:
    """Dial a role's advertised ``endpoint + path`` with a backend-exercising payload."""
    url = (info.get("endpoint") or "").rstrip("/") + (info.get("path") or "")
    model = info.get("model") or ""
    if role == "stt":
        data, ctype = _stt_multipart(model or "whisper-1")
        return _dial("POST", url, data=data, content_type=ctype, timeout=_LANE_TIMEOUT)
    if role == "tts":
        payload = {"model": model or "chatterbox", "input": "ping", "response_format": "wav"}
        timeout = _LANE_TIMEOUT
    elif role == "embedder":
        payload = {"model": model, "input": "ping"}
        timeout = _LANE_TIMEOUT
    elif role == "reranker":
        payload = {"model": model, "query": "ping", "documents": ["alpha", "beta"]}
        timeout = _LANE_TIMEOUT
    else:  # cortex, senses, or any other generate role
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        timeout = _GENERATE_TIMEOUT
    return _dial(
        "POST",
        url,
        data=json.dumps(payload).encode(),
        content_type="application/json",
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Fixtures — the live contract, fetched once and shared. A failure to read the
# contract when the gate is ARMED is itself a hard FAIL (not a skip).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def caps() -> dict:
    """``GET /capabilities`` — the six-role contract, keyed by role name."""
    p = _http("GET", _url("/capabilities"), timeout=_META_TIMEOUT)
    if p.status is None:
        pytest.fail(
            f"GATE ARMED but the gateway is unreachable at {_url('/capabilities')} "
            f"({p.error}). LOBES_SMOKE_BASE_URL is set, so this is a FAIL, not a skip: "
            "an operator asked for the gate and there is nothing serving. Start the "
            "deployment (`lobes serve --apply` / `lobes fleet up --apply`) or point "
            "LOBES_SMOKE_BASE_URL at the running gateway."
        )
    if p.status != 200:
        pytest.fail(f"GET /capabilities returned {p.status}, expected 200. Body: {p.body[:400]!r}")
    try:
        data = json.loads(p.body)
    except ValueError as exc:
        pytest.fail(f"GET /capabilities returned a non-JSON body: {exc}; body={p.body[:400]!r}")
    missing = [r for r in ROLES if r not in data]
    if missing:
        pytest.fail(f"GET /capabilities is missing roles {missing}; got keys {sorted(data)}")
    return data


@pytest.fixture(scope="module")
def model_ids() -> list[str]:
    """The ids advertised by ``GET /v1/models`` (the fleet's dialable models)."""
    p = _http("GET", _url("/v1/models"), timeout=_META_TIMEOUT)
    if p.status is None:
        pytest.fail(f"GATE ARMED but GET /v1/models is unreachable ({p.error}) — FAIL, not skip.")
    if p.status != 200:
        pytest.fail(f"GET /v1/models returned {p.status}, expected 200. Body: {p.body[:400]!r}")
    data = json.loads(p.body)
    return [entry["id"] for entry in data.get("data", [])]


# ---------------------------------------------------------------------------
# Check 1 — advertised-ready roles are reachable (#92, #96, #89).
# ---------------------------------------------------------------------------
def test_advertised_ready_roles_are_reachable(caps: dict) -> None:
    """Every role advertised ``ready=true`` must answer its own ``endpoint+path``."""
    reachable: list[str] = []
    faults: list[str] = []
    for role in ROLES:
        info = caps[role]
        if not info.get("ready"):
            continue  # only ready roles carry the "advertised implies reachable" promise
        url = (info.get("endpoint") or "").rstrip("/") + (info.get("path") or "")
        ok, verdict = _classify(_role_dial(role, info))
        line = f"  {role:9} {url:42} -> {verdict}"
        (reachable if ok else faults).append(line)
    assert not faults, (
        "advertised-ready roles that are NOT reachably served "
        "(a 404 is #92/#96; a bare 5xx without Retry-After is a dishonestly relayed "
        "dead backend, e.g. the poisoned STT sidecar #89):\n"
        + "\n".join(faults)
        + ("\n\nreachable:\n" + "\n".join(reachable) if reachable else "")
    )


# ---------------------------------------------------------------------------
# Check 2 — every advertised model is reachable on its lane (#91).
# ---------------------------------------------------------------------------
def test_advertised_models_are_reachable(caps: dict, model_ids: list[str]) -> None:
    """No id in ``/v1/models`` may read as "model does not exist" on its lane."""
    assert model_ids, "GET /v1/models advertised no models at all — nothing is reachable."
    embed_model = caps["embedder"].get("model")
    rerank_model = caps["reranker"].get("model")
    reachable: list[str] = []
    faults: list[str] = []
    for mid in model_ids:
        # Resolve the id to its correct task lane from the live contract, so a
        # pooling model is never falsely dialed on a chat route it can't answer.
        if mid == embed_model:
            probe = _dial(
                "POST",
                _url("/v1/embeddings"),
                data=json.dumps({"model": mid, "input": "ping"}).encode(),
                content_type="application/json",
                timeout=_LANE_TIMEOUT,
            )
            lane = "/v1/embeddings"
        elif mid == rerank_model:
            probe = _dial(
                "POST",
                _url("/v1/rerank"),
                data=json.dumps({"model": mid, "query": "ping", "documents": ["a", "b"]}).encode(),
                content_type="application/json",
                timeout=_LANE_TIMEOUT,
            )
            lane = "/v1/rerank"
        else:  # cortex/senses and any un-roled generate candidate → completion
            probe = _dial(
                "POST",
                _url("/v1/chat/completions"),
                data=json.dumps(
                    {
                        "model": mid,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                    }
                ).encode(),
                content_type="application/json",
                timeout=_GENERATE_TIMEOUT,
            )
            lane = "/v1/chat/completions"
        ok, verdict = _classify(probe)
        line = f"  {mid:58} [{lane}] -> {verdict}"
        (reachable if ok else faults).append(line)
    report = (
        "#91: GET /v1/models advertises models that read as 'model does not exist' "
        "when dialed on their own lane (a listed model must never be undialable):\n"
        + "\n".join(faults)
        + ("\n\nreachable:\n" + "\n".join(reachable) if reachable else "")
    )
    assert not faults, report


# ---------------------------------------------------------------------------
# Check 3 — `lobes capabilities --json` agrees with GET /capabilities (#95/#92).
# ---------------------------------------------------------------------------
def test_cli_and_gateway_capabilities_agree(caps: dict) -> None:
    """The CLI's contract view and the gateway's must agree for all six roles."""
    port = urllib.parse.urlsplit(_BASE_URL).port or 80
    # Run the REAL CLI in a fresh subprocess: tests/conftest.py's autouse
    # `offline_runtime` neutralises the in-process gateway probe, and the point
    # here is to exercise the actual `lobes capabilities` client end to end.
    proc = subprocess.run(  # nosec B603 — fixed argv, this repo's own module, no shell
        [sys.executable, "-m", "lobes", "capabilities", "--json", "--port", str(port)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        proc.returncode == 0
    ), f"`lobes capabilities --json --port {port}` exited {proc.returncode}: {proc.stderr.strip()}"
    cli = json.loads(proc.stdout)
    # `lobes capabilities --json` no longer carries a top-level "source" key
    # (Qodo action-required finding on PR #102 — the gateway's own
    # GET /capabilities returns exactly the six role keys, so the CLI's
    # gateway-mode JSON must match byte-for-byte, with nothing extra mixed
    # in). The offline/gateway distinction now lives out-of-band: the CLI
    # writes a one-line notice to stderr only when it degrades to the
    # offline .env fallback, and stays silent on stderr when it actually
    # reached the gateway. So "stderr is empty" is the live signal here.
    assert not proc.stderr.strip(), (
        f"`lobes capabilities --json --port {port}` wrote to stderr: "
        f"{proc.stderr.strip()!r} — that only happens on the offline-fallback path, "
        f"meaning the CLI could not reach the gateway on port {port} and fell back to "
        "its offline .env view. The CLI and the gateway can only be proven to agree "
        "when the CLI actually reaches the gateway (#96); a degraded fallback is "
        "itself a reachability fault."
    )
    disagreements: list[str] = []
    for role in ROLES:
        for field in ("endpoint", "ready", "loaded"):
            cli_val = cli.get(role, {}).get(field)
            gw_val = caps[role].get(field)
            if cli_val != gw_val:
                disagreements.append(f"  {role}.{field}: cli={cli_val!r} != gateway={gw_val!r}")
    assert not disagreements, (
        "#95/#92: `lobes capabilities --json` and GET /capabilities disagree on "
        "endpoint/ready/loaded (they must be one source of truth):\n" + "\n".join(disagreements)
    )


# ---------------------------------------------------------------------------
# Check 4 — deployed gateway version matches the CLI (#99 version skew).
# ---------------------------------------------------------------------------
def test_deployed_gateway_version_matches_cli() -> None:
    """The gateway's ``/health`` version must equal ``lobes.__version__``."""
    p = _http("GET", _url("/health"), timeout=_META_TIMEOUT)
    if p.status is None:
        pytest.fail(f"GATE ARMED but GET /health is unreachable ({p.error}) — FAIL, not skip.")
    assert p.status == 200, f"GET /health returned {p.status}, expected 200. Body: {p.body[:400]!r}"
    health = json.loads(p.body)
    gw_version = health.get("version")
    assert gw_version is not None, (
        "#99 version skew: GET /health reports NO 'version' field, so this gateway "
        "image PREDATES the honest-version work. Dockerfile.gateway pins "
        "'lobes-cli==${MODEL_GEAR_VERSION}' once (at `lobes init` time) and no verb "
        "re-pins it, so a container can silently run a stale release for days after "
        f"the host CLI ({LOBES_VERSION}) and PyPI moved on. A missing version is skew, "
        f"not a pass: redeploy the gateway to at least {LOBES_VERSION}. Body: {health!r}"
    )
    # Dev lane: a gateway built from a TestPyPI pre-release of THIS version
    # (MODEL_GEAR_VERSION=X.Y.Z.devN + GATEWAY_PIP_EXTRA_INDEX_URL) is this
    # branch's own code, not drift — accept when the public base matches.
    gw_base = gw_version.split(".dev", 1)[0]
    assert gw_version == LOBES_VERSION or (".dev" in gw_version and gw_base == LOBES_VERSION), (
        f"#99 version skew: the gateway /health reports {gw_version!r} but this CLI is "
        f"{LOBES_VERSION!r}. A merged fix has not reached this deployment — redeploy "
        f"(re-run `lobes init` / rebuild the gateway image) so Dockerfile.gateway "
        f"re-pins lobes-cli=={LOBES_VERSION}. This is the exact drift (#99) that made "
        "#92 look like a code regression when the fix was already published."
    )


# ---------------------------------------------------------------------------
# Check 5 — a Colleague can discover + dial cortex/senses from the contract alone
# (#81, #87). No model id is hardcoded — everything comes from /capabilities.
# ---------------------------------------------------------------------------
def test_colleague_discovers_and_dials_cortex_and_senses(caps: dict) -> None:
    """Given only the gateway origin, resolve cortex/senses and get an answer."""
    answers: list[str] = []
    faults: list[str] = []
    for role in ("cortex", "senses"):
        info = caps[role]
        if not info.get("ready"):
            faults.append(f"  {role}: /capabilities reports ready=false — cannot be discovered")
            continue
        endpoint = (info.get("endpoint") or "").rstrip("/")
        path = info.get("path") or ""
        model = info.get("model") or ""
        if not endpoint or not path or not model:
            faults.append(
                f"  {role}: contract incomplete (endpoint={endpoint!r} path={path!r} "
                f"model={model!r}) — a Colleague cannot dial it"
            )
            continue
        payload = {
            "model": model,  # discovered, never hardcoded
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 8,
            "temperature": 0,
        }
        probe = _dial(
            "POST",
            endpoint + path,
            data=json.dumps(payload).encode(),
            content_type="application/json",
            timeout=_GENERATE_TIMEOUT,
        )
        ok, verdict = _classify(probe)
        if not ok:
            faults.append(
                f"  {role}: discovered {endpoint + path} but it is not reachable: {verdict}"
            )
            continue
        if probe.status is not None and 200 <= probe.status < 300:
            try:
                choices = json.loads(probe.body).get("choices")
            except ValueError:
                choices = None
            if not choices:
                faults.append(
                    f"  {role}: dialed {endpoint + path} and got {probe.status} but no "
                    f"'choices' — not a usable answer. Body: {probe.body[:200]!r}"
                )
                continue
            answers.append(
                f"  {role}: discovered via contract, dialed {endpoint + path}, got an answer"
            )
        else:
            # Reachable but shed/warming (429/503+Retry-After). Discovery still
            # worked — the endpoint was resolved from the contract and answered.
            answers.append(f"  {role}: discovered {endpoint + path}, reachable ({verdict})")
    report = (
        "#81/#87 Colleague discovery path failed — a peer given ONLY the gateway "
        "origin could not resolve-and-dial these roles from the contract:\n"
        + "\n".join(faults)
        + ("\n\nsucceeded:\n" + "\n".join(answers) if answers else "")
    )
    assert not faults, report
