"""Live per-role health probe — ``lobes doctor --role <role>`` (issue #234).

The consumer-facing half of #234. Colleague armed the `associate` seat from
the gateway's `/capabilities` advert alone, and lost a measured run to three
facts the advert could not tell it apart:

* the advertised `context` was the checkpoint ceiling, not the served window
  (fixed in :func:`lobes.roles._resolve_context`, but a consumer still wants
  to *see* the served number rather than trust a field);
* a live proxied lane read `ready: false`, so anything gating on `ready`
  refused a working seat;
* a request naming the served id was refused `role_infeasible`, because the
  raw id resolves to a different local backend — the alias is the only
  routable address, and nothing said so.

Every check here asks the deployment a question whose answer is a FACT about
this box, not a restatement of its own advert: the served window comes from
`/tokenize` (which the engine answers from its real `max_model_len`), and the
routing checks issue actual requests. That is the point — an advert-only check
would have passed on every box that produced the failures above.

**This lane issues real inference.** One bounded completion through the alias,
and one through the served id. Both are tiny and `stream: false`; the verb is
still read-only in the sense every other read-only verb is (it changes nothing),
but it is not free, which is why it lives behind an explicit `--role` and never
runs as part of the default `lobes doctor` sweep.

**The generate probe deliberately disables thinking.** Measured on the live
Orin associate lane 2026-08-30: with thinking on (this checkpoint's template
default) a 4096-token budget spent the whole budget reasoning and returned
HTTP 200 with EMPTY content on 8 of 12 tasks. A health check that sent a
small budget with thinking on would report a healthy 200 carrying nothing —
so it sends `enable_thinking: false`, and :func:`_generate_check` names an
empty 200 as its own failure rather than passing it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

# A bounded, cheap generate probe. Small on purpose: this is a liveness and
# routing question, not a quality one.
_PROBE_MESSAGES = [{"role": "user", "content": "Reply with the single word: ok"}]
_PROBE_MAX_TOKENS = 16
_DEFAULT_TIMEOUT = 30.0


def _request(
    url: str,
    *,
    payload: Mapping[str, Any] | None,
    headers: Mapping[str, str] | None,
    timeout: float,
) -> tuple[int, dict | None, float]:
    """``(status, parsed_body_or_None, elapsed_seconds)``. Never raises.

    A non-2xx is a RESULT here, not an exception: the served-id check exists
    precisely to observe a 404, and a probe that raised on it could not report
    the thing it was written to find. Connection failures surface as status 0
    so a caller can tell "nothing answered" from "answered badly".
    """
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs)  # noqa: S310 - http only
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:  # a real answer with a non-2xx status
        raw, status = exc.read(), exc.code
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None, round(time.monotonic() - started, 3)
    elapsed = round(time.monotonic() - started, 3)
    try:
        return status, json.loads(raw.decode("utf-8")), elapsed
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, None, elapsed


def _check(id_: str, passed: bool, severity: str, message: str, remediation: str = "") -> dict:
    """Doctor's own check shape, so `--role` renders through the same path."""
    out = {"id": id_, "passed": passed, "severity": severity, "message": message}
    if remediation:
        out["remediation"] = remediation
    return out


def advert_check(entry: Mapping[str, Any] | None, role: str) -> list[dict]:
    """What `/capabilities` claims for ``role`` — model, window, ready, host."""
    if entry is None:
        return [
            _check(
                "role_advert",
                False,
                "error",
                f"the gateway advertises no `{role}` entry",
                "check the gateway is current ('lobes up gateway --build --apply'); "
                "a pre-0.69 image predates several role fields",
            )
        ]
    proxied = bool(entry.get("proxied"))
    hosted_by = entry.get("hosted_by")
    where = f"proxied to {hosted_by}" if proxied and hosted_by else "hosted locally"
    context = entry.get("context")
    window = "no window advertised" if context is None else f"advertises {context} tokens"
    checks = [
        _check(
            "role_advert",
            True,
            "info",
            f"{role}: {entry.get('model') or '(no model)'} — {where}, {window}",
        )
    ]
    # `feasible:false` on a PROXIED role is by design ("this box does not HOST
    # it"), not a fault — #234 ask 2 was partly a misreading of that field. Say
    # so here rather than letting a consumer read the flag as broken.
    if proxied:
        checks.append(
            _check(
                "role_proxied",
                bool(entry.get("ready")),
                "warn",
                (
                    f"peer {hosted_by} reports ready"
                    if entry.get("ready")
                    else f"peer {hosted_by} does not report ready"
                ),
                (
                    ""
                    if entry.get("ready")
                    else "check the peer is up and serving this role; `feasible:false` "
                    "is expected for a proxied role and is not the problem"
                ),
            )
        )
    return checks


def window_check(
    base_url: str, role: str, advertised: int | None, headers: Mapping[str, str], timeout: float
) -> dict:
    """The SERVED window, from `/tokenize` — the engine's own `max_model_len`.

    This is the check that would have caught #234's first fact directly: the
    engine answers from what it is actually serving, so a disagreement with
    the advert is visible rather than inferred.
    """
    status, body, _ = _request(
        f"{base_url}/tokenize",
        payload={"model": role, "prompt": "hello"},
        headers=headers,
        timeout=timeout,
    )
    if status != 200 or not isinstance(body, dict):
        return _check(
            "served_window",
            False,
            "warn",
            f"/tokenize did not answer for `{role}` (HTTP {status or 'no response'})",
            "a proxied role has no local engine to ask — this is expected off-box; "
            "for a hosted role, check the lane is up ('lobes status')",
        )
    served = body.get("max_model_len")
    if served is None:
        return _check("served_window", False, "warn", "/tokenize reported no max_model_len")
    if advertised is not None and served != advertised:
        return _check(
            "served_window",
            False,
            "error",
            f"advert says {advertised} tokens, engine serves {served}",
            "a consumer sizing work from the advert will overrun the real window; "
            "re-render and restart the gateway so it reads the lane's own value",
        )
    return _check("served_window", True, "info", f"engine serves {served} tokens")


def _generate(
    base_url: str, model: str, headers: Mapping[str, str], timeout: float
) -> tuple[int, dict | None, float]:
    return _request(
        f"{base_url}/v1/chat/completions",
        payload={
            "model": model,
            "messages": _PROBE_MESSAGES,
            "max_tokens": _PROBE_MAX_TOKENS,
            "temperature": 0,
            # See the module docstring: with thinking on, a small budget returns
            # a healthy-looking 200 with empty content.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers=headers,
        timeout=timeout,
    )


def _content_of(body: dict | None) -> str | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    return message.get("content") if isinstance(message, dict) else None


def alias_check(base_url: str, role: str, headers: Mapping[str, str], timeout: float) -> dict:
    """Does `model=<role>` actually answer? The documented address must work."""
    status, body, elapsed = _generate(base_url, role, headers, timeout)
    if status != 200:
        kind = ""
        if isinstance(body, dict):
            kind = str((body.get("error") or {}).get("code") or "")
        return _check(
            "alias_routes",
            False,
            "error",
            f"model={role} refused: HTTP {status or 'no response'}"
            + (f" ({kind})" if kind else ""),
            "this is the documented address for the role — if it refuses, no "
            "consumer can reach the lane by contract",
        )
    content = _content_of(body)
    if content is not None and not content.strip():
        # The empty-200 hazard, named rather than passed (issue #234).
        return _check(
            "alias_routes",
            False,
            "error",
            f"model={role} answered HTTP 200 in {elapsed}s with EMPTY content",
            "a bounded request returned no text — send a larger max_tokens, or "
            "chat_template_kwargs={'enable_thinking': false}, and treat "
            "finish_reason=length with empty content as a failure, not a success",
        )
    return _check("alias_routes", True, "info", f"model={role} answered 200 in {elapsed}s")


def served_id_check(
    base_url: str,
    role: str,
    served_id: str | None,
    headers: Mapping[str, str],
    timeout: float,
) -> dict | None:
    """Does the raw checkpoint id route, or is the alias the only address?

    Not a pass/fail on the deployment: a served id that resolves to a different
    local backend is a known ambiguity when two roles share a checkpoint (the
    `worker`/`associate` collision). The check exists so a consumer LEARNS which
    addresses work before arming a seat, instead of discovering it mid-run.
    """
    if not served_id or served_id == role:
        return None
    status, body, _ = _generate(base_url, served_id, headers, timeout)
    if status == 200:
        return _check("served_id", True, "info", f"the served id also routes ({served_id})")
    kind = ""
    if isinstance(body, dict):
        kind = str((body.get("error") or {}).get("code") or "")
    return _check(
        "served_id",
        True,  # informational: the alias is the contract, not this
        "warn",
        f"the served id does NOT route (HTTP {status or 'no response'}"
        + (f", {kind}" if kind else "")
        + f") — address this role as `{role}`",
        f"use model={role}; the raw checkpoint id is ambiguous when two roles "
        "share a checkpoint and resolves to whichever backend owns it locally",
    )


def probe_role(
    base_url: str,
    role: str,
    entry: Mapping[str, Any] | None,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[dict]:
    """Every check for one role, in the order a consumer would ask them."""
    hdrs = dict(headers or {})
    checks = list(advert_check(entry, role))
    advertised = entry.get("context") if isinstance(entry, dict) else None
    served_id = entry.get("model") if isinstance(entry, dict) else None
    checks.append(window_check(base_url, role, advertised, hdrs, timeout))
    checks.append(alias_check(base_url, role, hdrs, timeout))
    served = served_id_check(base_url, role, served_id, hdrs, timeout)
    if served is not None:
        checks.append(served)
    return checks
