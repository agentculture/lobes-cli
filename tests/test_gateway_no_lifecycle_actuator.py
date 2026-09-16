"""t10 (issue #82), AC3 — the gateway package must never import the lifecycle
actuator (:mod:`lobes.runtime._compose`).

**Why this test exists, not just what it checks.** The operator decided (frame
claim c16, confirmed) that v1 ships MANUAL lifecycle only: ``lobes`` owns
start/stop via ``lobes up`` / ``lobes fleet``, and a request that reaches a
cold backend (e.g. ``innereye`` before an operator has run
``lobes up innereye --apply``) gets an honest, retryable 503 — never a silent
boot triggered by the request itself. That contract is currently a
STRUCTURAL fact, not a convention: every ``docker compose`` call site in this
codebase lives under ``lobes/cli/_commands/*.py``, and the data plane
(``lobes/gateway/``) imports nothing from ``lobes.runtime._compose`` at all.
Without a test pinning that absence, a future refactor — e.g. "let the
gateway auto-start a cold backend on demand" (explicitly rejected for v1,
see the ``up.py`` module docstring and c18) — could wire a compose call into
a request handler and nobody would notice until it shipped. This test is the
tripwire.

**Static AST scan, not a runtime import-graph check, and not a substring
grep.** Three ways to look for this were considered:

* **Runtime module-graph check** (assert ``lobes.runtime._compose`` is not in
  ``sys.modules`` after importing the gateway package) — REJECTED: under a
  shared pytest process (this suite runs ``-n auto``, and even single-process
  the module cache is process-wide), an unrelated test file (e.g.
  ``tests/test_cli_up.py``) may import ``lobes.runtime._compose`` earlier in
  the same worker, poisoning ``sys.modules`` and producing a false failure
  that has nothing to do with what the gateway itself imports. This check
  would be import-ORDER-dependent, which is exactly the kind of flake this
  guard must not have.
* **Plain substring grep** for ``"_compose"`` over the gateway source text —
  REJECTED as the sole method: too blunt (a future identifier that merely
  CONTAINS "_compose", e.g. a hypothetical ``recompose_headers`` helper,
  would false-positive) and too easy to defeat trivially (a
  ``import lobes.runtime._compose as _c`` re-export, or an
  ``importlib.import_module("lobes.runtime" + "._compose")`` string build,
  would slip past a naive grep that only matches the literal substring
  ``_compose`` — though note the AST approach below has the identical blind
  spot for the ``importlib.import_module`` case, since that call is not a
  static ``import``/``from ... import`` statement at all).
* **AST parse of every ``.py`` file under ``lobes/gateway/`` for ``Import`` /
  ``ImportFrom`` nodes** — CHOSEN. It only fires on an actual Python import
  statement (so it does not need the gateway package to import cleanly, or
  its heavy runtime deps to be installed, to run), is immune to import order
  and to other test files' side effects, and names the offending FILE
  directly in the assertion message.

**Known blind spot (documented, not fixed here).** This test does not, and
cannot, catch every way a lifecycle actuator could sneak into the data
plane — only a static ``import``/``from`` of a module whose dotted path
contains ``_compose``. It would NOT catch: (1) an ``importlib.import_module``
string-built import of ``lobes.runtime._compose`` (no static import node to
find); (2) the gateway shelling out to ``docker compose`` directly via
``subprocess``, bypassing the ``_compose`` module entirely — a hand check
under "Sanity: no gateway file shells out to docker either" below covers this
specific codebase's CURRENT absence of that pattern, but is a weaker,
substring-only check with the same limits as the rejected grep above, so it
is a best-effort companion, not an equivalent guarantee. Both gaps are
acceptable for what this test is FOR: it is a tripwire against an ordinary,
in-repo refactor quietly crossing the c16 line, not a defense against
deliberate obfuscation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import lobes.gateway as _gateway_pkg

_GATEWAY_DIR = Path(_gateway_pkg.__file__).parent
_REPO_ROOT = _GATEWAY_DIR.parent.parent


def _imported_dotted_names(source: str) -> set[str]:
    """Every module dotted-path named by an ``import``/``from ... import`` in ``source``."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                names.add(module)
                for alias in node.names:
                    names.add(f"{module}.{alias.name}")
    return names


def _gateway_source_files() -> list[Path]:
    files = sorted(_GATEWAY_DIR.rglob("*.py"))
    assert files, f"expected .py files under {_GATEWAY_DIR}, found none (test setup is broken)"
    return files


def test_no_gateway_module_statically_imports_the_compose_actuator() -> None:
    offenders: dict[str, set[str]] = {}
    for path in _gateway_source_files():
        hits = {
            name
            for name in _imported_dotted_names(path.read_text(encoding="utf-8"))
            if "_compose" in name
        }
        if hits:
            offenders[str(path.relative_to(_REPO_ROOT))] = hits
    assert not offenders, (
        "lobes/gateway/ (the data plane) imported lobes.runtime._compose (the "
        "lifecycle actuator) in: "
        f"{offenders!r}. This breaks the v1 manual-lifecycle contract (frame "
        "claim c16, confirmed): the data plane must never start or stop a "
        "backend as a side effect of serving a request -- a cold backend must "
        "always get an honest, retryable warming response instead. Move the "
        "lifecycle call back into lobes/cli/_commands/ (where every other "
        "'docker compose' call site lives), or get an explicit, reviewed "
        "decision to change c16 before adding this import."
    )


def test_the_compose_module_itself_is_reachable_only_from_cli_and_runtime() -> None:
    """Sanity control: confirm the AST scan above is actually exercised.

    ``lobes/cli/_commands/up.py`` and ``lobes/cli/_commands/fleet.py`` DO
    import ``lobes.runtime._compose`` -- if this control ever went red, it
    would mean the scan itself stopped finding real hits (e.g. a helper
    refactor broke :func:`_imported_dotted_names`), which would silently
    defang the gateway guard above without that test itself failing.
    """
    up_py = _REPO_ROOT / "lobes" / "cli" / "_commands" / "up.py"
    hits = {
        name
        for name in _imported_dotted_names(up_py.read_text(encoding="utf-8"))
        if "_compose" in name
    }
    assert hits, (
        "expected lobes/cli/_commands/up.py to import lobes.runtime._compose; "
        "if it no longer does, _imported_dotted_names() may be broken, which "
        "would mean the gateway-absence test above is not actually testing "
        "anything"
    )


def test_no_gateway_module_shells_out_to_docker_directly() -> None:
    """Best-effort companion check (documented blind spot, see module docstring):

    a data-plane module could bypass ``_compose`` entirely and still start or
    stop a container by shelling out to ``docker compose``/``docker`` itself.
    This is a plain substring scan, not an AST scan -- it looks for the
    literal string ``"docker compose"`` and an actual ``subprocess`` usage
    (``import subprocess`` or a ``subprocess.`` call), not just the word
    "subprocess" appearing anywhere -- ``_pressure_policy.py`` truthfully
    DOCUMENTS that it does no subprocess calls, and an earlier, cruder version
    of this check flagged that very disclaimer as a false positive. A hit here
    is a prompt to go read the file, not an automatic proof of a violation.
    Today there are zero hits; this pins that as a known-good baseline.
    """
    offenders = []
    for path in _gateway_source_files():
        text = path.read_text(encoding="utf-8")
        if "docker compose" in text or "import subprocess" in text or "subprocess." in text:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        "lobes/gateway/ mentions 'docker compose' or uses 'subprocess' in: "
        f"{offenders!r} -- read the hit(s) to confirm this is not the data "
        "plane shelling out to start/stop a container directly (c16 forbids "
        "this exactly like a _compose import would)."
    )
