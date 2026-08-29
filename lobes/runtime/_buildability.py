"""Buildability guard for a variation's pinned ``MODEL_GEAR_VERSION`` (t10).

``lobes/templates/fleet/Dockerfile.gateway`` installs the gateway as
``pip install lobes-cli==${MODEL_GEAR_VERSION}`` — from PyPI for an ordinary
release, or from a TestPyPI dev index for a ``.devN`` PR build (see that
Dockerfile's own comment block). Per this repo's publish convention, only a PR
publishes a ``.devN`` wheel, to TestPyPI — nothing guarantees that build still
resolves once the PR merges or closes.

A committed variation (``deployment.lock.toml``, :mod:`lobes.runtime._lock`)
records the ``lobes_version`` it was captured at. If that pin has since gone
stale, the honest place to say so is a preflight check that runs BEFORE
``docker build`` — not a ``pip install`` traceback nested three layers deep in
a build log. That is the "fails early" half of t10's acceptance criteria; this
module is the "detects and reports clearly" half.

**Two layers, deliberately separated:**

1. **Offline, always available — version-shape risk.** :func:`is_dev_version`
   classifies a version string as PEP 440 dev-release shaped with no network
   at all. A ``.devN`` pin is inherently the high-risk shape (ephemeral
   TestPyPI); an ordinary release version is the low-risk shape. This layer
   never needs an index and is fully covered by the hermetic test suite.
2. **Optional, injectable — live installability.** Determining whether a
   pinned version *actually* resolves right now requires asking a package
   index, which needs the network. :data:`IndexQuery` is the seam: callers
   pass a callable ``(package, version) -> bool``; the test suite passes a
   fake, and only a caller wiring this into a live command
   (``lobes doctor``, a ``--from-lock`` restore, …) should pass
   :func:`default_pypi_index_query`, which is the one function here that
   touches the network — and only when actually called, never at import time
   or as a module-level side effect.

Stdlib only.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.runtime._lock import DeploymentLock

#: The package the gateway Dockerfiles pin (``Dockerfile.gateway``,
#: ``Dockerfile.realtime``, ``Dockerfile.chatterbox`` all install this name).
PYPI_PACKAGE = "lobes-cli"

#: PyPI's per-release JSON endpoint. A 404 means that exact version does not
#: (or no longer does) resolve on this index; any other status is a genuine
#: query failure, not a clean "not found".
PYPI_JSON_URL = "https://pypi.org/pypi/{package}/{version}/json"

#: TestPyPI's per-release JSON endpoint, same shape as :data:`PYPI_JSON_URL`.
#: Per this repo's publish convention (CLAUDE.md: "PRs publish a `.devN` to
#: TestPyPI"), a dev-shaped pin was never published to PyPI at all — querying
#: PYPI_JSON_URL for one always 404s, even when the build genuinely resolves.
TESTPYPI_JSON_URL = "https://test.pypi.org/pypi/{package}/{version}/json"

#: PEP 440's dev-release segment: a trailing ``.devN`` (or bare ``devN``
#: immediately after the release segment, e.g. ``0.67.0dev12``). This repo's
#: own publish convention (CLAUDE.md: "PRs publish a ``.devN`` to TestPyPI")
#: always emits the dotted form, but the bare form is valid PEP 440 too, so
#: both are matched.
_DEV_VERSION_RE = re.compile(r"\.?dev(\d+)?$", re.IGNORECASE)


def is_dev_version(version: str) -> bool:
    """Whether *version* is PEP 440 dev-release shaped, e.g. ``"0.67.0.dev12"``.

    Offline, no network. Only a PR publishes a version with this shape
    (per this repo's ``publish.yml``), and only to TestPyPI — an index whose
    retention is not guaranteed the way PyPI's is. A version with no such
    segment (``"0.67.0"``) is treated as an ordinary release: low risk by
    shape, though :func:`check_buildability` can still verify it live.

    An empty/``None`` version is not a dev version — see
    :func:`check_buildability`, which classifies that case separately
    (``"unversioned"``), since a missing pin is its own, more basic, failure.
    """
    if not version:
        return False
    return bool(_DEV_VERSION_RE.search(version.strip()))


#: A live index query: ``(package, version) -> True`` if that exact version
#: resolves on the configured index right now, ``False`` if it cleanly does
#: not (e.g. a 404). Implementations should *raise* on a genuine query
#: failure (network outage, non-404 HTTP error) rather than returning
#: ``False`` — that distinction matters: "the index says no" and "we could
#: not ask the index" must never be reported identically.
IndexQuery = Callable[[str, str], bool]


def pypi_index_url_for(package: str, version: str) -> str:
    """The JSON endpoint to query for *package*==*version* — PyPI or TestPyPI.

    Pure URL-selection, no network: a dev-shaped *version* (see
    :func:`is_dev_version`) routes to :data:`TESTPYPI_JSON_URL` because that
    is the only index this repo's publish convention ever puts a `.devN`
    build on; anything else routes to :data:`PYPI_JSON_URL`. Split out from
    :func:`default_pypi_index_query` so the routing decision itself is
    testable without making a request (PR #223 review, defect 2: the guard
    previously always asked PyPI, so an available dev pin was reported
    unbuildable).
    """
    template = TESTPYPI_JSON_URL if is_dev_version(version) else PYPI_JSON_URL
    return template.format(package=package, version=version)


def default_pypi_index_query(package: str, version: str, *, timeout: float = 5.0) -> bool:
    """Real network check against PyPI's (or TestPyPI's) JSON API.

    NOT exercised by the committed test suite (which is hermetic — no
    network) and NOT called anywhere by default: it is only ever invoked by a
    caller that explicitly passes it as *index_query* to
    :func:`check_buildability` / :func:`check_lock_buildability`. See the
    module docstring's "two layers" note.

    Queries whichever index :func:`pypi_index_url_for` selects for *version*
    — TestPyPI for a `.devN`-shaped pin, PyPI otherwise, since this repo's
    publish convention only ever puts a dev build on TestPyPI. Returns
    ``True`` if ``package==version`` resolves on that index, ``False`` on a
    clean 404 (that exact version does not exist there). Any other failure
    (network error, non-404 HTTP status) is re-raised rather than silently
    reported as "unbuildable" — an outage is not proof of unbuildability.
    """
    url = pypi_index_url_for(package, version)
    request = urllib.request.Request(url, headers={"User-Agent": "lobes-buildability-guard"})
    try:
        with urllib.request.urlopen(
            request, timeout=timeout
        ) as response:  # nosec B310 - fixed https pypi.org/test.pypi.org URL
            response.read()
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


@dataclass(frozen=True)
class BuildabilityResult:
    """The outcome of checking one ``package==version`` pin.

    :param version: The version string checked (possibly empty).
    :param risk: ``"unversioned"`` (no pin at all — the Dockerfile build arg
        would be empty), ``"ephemeral_dev"`` (PEP 440 dev-release shaped —
        see :func:`is_dev_version`), or ``"released"`` (ordinary shape).
        Computed offline, always populated.
    :param installable: ``True``/``False`` if an *index_query* was supplied
        and answered; ``None`` if no live check was performed (the offline
        layer only).
    :param message: A single clear sentence naming the Dockerfile/pin this
        traces back to — meant to be surfaced verbatim at the CLI boundary,
        not a ``pip install`` traceback found three layers into a build log.
    """

    version: str
    risk: str
    installable: Optional[bool]
    message: str


def check_buildability(
    version: str | None,
    *,
    package: str = PYPI_PACKAGE,
    index_query: IndexQuery | None = None,
) -> BuildabilityResult:
    """Classify *version*'s buildability risk, optionally verified live.

    Always available (no *index_query*): returns the offline ``risk``
    classification with ``installable=None``. Pass *index_query* — a fake in
    tests, :func:`default_pypi_index_query` in a live caller — to also ask
    whether the pin resolves right now; a ``False`` answer is the strongest,
    clearest signal this function can produce, and its message spells out
    exactly which Dockerfile/pip-install step would otherwise fail.
    """
    version = (version or "").strip()

    if not version:
        return BuildabilityResult(
            version="",
            risk="unversioned",
            installable=None,
            message=(
                "no MODEL_GEAR_VERSION recorded for this variation — "
                f"Dockerfile.gateway runs `pip install {package}==${{MODEL_GEAR_VERSION}}`, "
                "which fails immediately on an empty pin; re-capture the lock "
                "after `lobes init` has written MODEL_GEAR_VERSION into .env"
            ),
        )

    risk = "ephemeral_dev" if is_dev_version(version) else "released"

    installable: Optional[bool] = None
    if index_query is not None:
        installable = index_query(package, version)

    if installable is False:
        message = (
            f"{package}=={version} is not installable from the configured index — "
            f"`docker build` of this variation's gateway would fail at "
            f"`pip install {package}=={version}` (Dockerfile.gateway); "
            "re-capture this variation's lock against a released version, or "
            "publish a fresh dev build before restoring it"
        )
    elif risk == "ephemeral_dev":
        message = (
            f"{package}=={version} is a TestPyPI `.devN` build — only a PR "
            "publishes one (see CLAUDE.md's publish convention), and nothing "
            "guarantees it still resolves once that PR merges or closes; "
            "this variation's build stays fragile even though "
            + ("it currently resolves" if installable else "it was not checked live")
        )
    else:
        message = f"{package}=={version} is an ordinary release pin"

    return BuildabilityResult(version=version, risk=risk, installable=installable, message=message)


def check_lock_buildability(
    lock: DeploymentLock,
    *,
    package: str = PYPI_PACKAGE,
    index_query: IndexQuery | None = None,
) -> BuildabilityResult:
    """:func:`check_buildability` over a captured :class:`~lobes.runtime._lock.DeploymentLock`.

    Convenience wrapper for the natural call site — a ``--from-lock`` restore
    or a dedicated buildability check reads ``lock.lobes_version`` (the
    version :mod:`lobes.runtime._lock` records at capture time) rather than
    threading the raw string through by hand.
    """
    return check_buildability(lock.lobes_version, package=package, index_query=index_query)


def assert_buildable(result: BuildabilityResult) -> None:
    """Raise a clear, early error if *result* proves the pin unbuildable.

    This is what makes the failure "early" per t10's acceptance criteria: a
    caller runs this — and the live :func:`default_pypi_index_query` behind
    it — as a preflight, before invoking ``docker build`` at all. The
    resulting :class:`~lobes.cli._errors.ModelGearError` carries the same
    message :attr:`BuildabilityResult.message` already spells out, so the
    operator sees "lobes-cli==0.67.0.dev12 is not installable ... would fail
    at `pip install` (Dockerfile.gateway)" at the CLI boundary, not a raw pip
    traceback nested inside a build log.

    Only raises when *installable* is definitively ``False`` or the pin is
    ``"unversioned"`` — a bare ``"ephemeral_dev"`` classification with no
    live check (``installable is None``) is a warning-shaped fact, not a
    proven failure, so it does not raise here; callers that want to warn on
    that case can inspect :attr:`BuildabilityResult.risk` themselves.
    """
    if result.installable is False or result.risk == "unversioned":
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=result.message,
            remediation=(
                "run `lobes init --apply` to re-render MODEL_GEAR_VERSION against "
                "the version actually installed, or point this variation's lock "
                "at a released lobes-cli version"
            ),
        )
