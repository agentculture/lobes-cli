"""lobes site front end (t18): the packaged page and its vendored fonts.

Static checks over ``lobes/site/web`` plus a loopback server on port 0 serving the
packaged web root. No browser; t20 runs the page live with Playwright.
"""

from __future__ import annotations

import http.client
import re
from importlib import resources

import pytest

from lobes.rollout._journal import Journal
from lobes.rollout._observe import MemberObservation
from lobes.rollout._residents import Residents
from lobes.site import server as site

WEB = resources.files("lobes.site").joinpath("web")
FONTS = tuple(f"lato-latin-{w}.woff2" for w in (300, 400, 700, 900))
API_ROUTES = ("/api/mesh", "/api/rollout", "/api/residents", "/api/lobe-export")


def _html() -> str:
    return WEB.joinpath("index.html").read_text(encoding="utf-8")


def _vendor_refs(html: str) -> set:
    return set(re.findall(r"/vendor/([^\s\"')]+)", html))


@pytest.fixture
def srv(tmp_path):
    local = "http://127.0.0.1:1"
    sources = site.Sources(
        local_origin=local,
        local_name="here",
        observe=lambda origin: MemberObservation(origin=origin),
        residents=Residents,
        journal=lambda: Journal(tmp_path),
        export_listing=lambda role, member: {},
    )
    s = site.SiteServer(sources, host="127.0.0.1", port=0)  # the packaged web root
    s.start()
    yield s
    s.stop()


def _get(s, path):
    host, port = s.address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        conn.close()


def test_packaged_page_is_served_under_the_csp(srv):
    code, headers, body = _get(srv, "/")
    assert code == 200
    assert headers["Content-Type"].startswith("text/html")
    csp = headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert body.decode() == _html()
    assert "Placeholder page" not in body.decode()


def test_page_makes_no_external_request():
    html = _html()
    assert not re.search(r"https?://", html), "absolute URL in index.html"
    assert not re.search(r"""(src|href)\s*=\s*["']//""", html), "protocol-relative URL"
    assert not re.search(r"url\(\s*['\"]?//", html), "protocol-relative url()"
    assert "fonts.googleapis" not in html
    assert "cdn" not in html.lower()
    # every src/href is same-origin: a fragment, a root path, or an inline data: icon
    for attr in re.findall(r"""(?:src|href)\s*=\s*["']([^"']*)["']""", html):
        assert attr.startswith(("#", "/", "data:")), attr


def test_every_referenced_vendor_file_exists_and_is_allow_listed(srv):
    refs = _vendor_refs(_html())
    assert set(FONTS) <= refs
    for name in refs:
        assert WEB.joinpath("vendor", name).is_file(), name
        code, headers, body = _get(srv, f"/vendor/{name}")
        assert code == 200, name
        assert headers["Content-Type"] == "font/woff2"
        assert body == WEB.joinpath("vendor", name).read_bytes()


def test_fonts_and_licence_ship_in_the_package(srv):
    for name in FONTS:
        data = WEB.joinpath("vendor", name).read_bytes()
        assert data[:4] == b"wOF2", name
    code, headers, body = _get(srv, "/vendor/lato-OFL.txt")
    assert code == 200
    assert headers["Content-Type"].startswith("text/plain")
    assert b"SIL Open Font License" in body


def test_every_vendor_file_is_servable():
    for entry in WEB.joinpath("vendor").iterdir():
        ext = "." + entry.name.rsplit(".", 1)[-1]
        assert ext in site._VENDOR_TYPES, entry.name
        assert site._VENDOR_NAME.match(entry.name), entry.name


def test_page_fetches_every_api_route():
    html = _html()
    for route in API_ROUTES:
        assert f'"{route}"' in html, route


def test_page_has_the_designed_views_and_controls():
    html = _html()
    for needle in (
        'href="#mesh"',
        'href="#rollout"',
        'href="#residents"',
        'role="tablist"',
        '"Graph"',
        '"Table"',
        'id="theme"',
        "prefers-color-scheme",
        "prefers-reduced-motion",
        ":focus-visible",
        'html[data-theme="dark"]',
    ):
        assert needle in html, needle
    # role families are signs, never colours: colour is kept for health and
    # one neutral "spans boxes" blue (light, dark)
    assert "var SIGN = {" in html
    for fam in ("gen:", "dec:", "emb:", "sp:"):
        assert fam in html.split("var SIGN = {", 1)[1].split("};", 1)[0], fam
    for colour in ("--link:#2a78d6", "--link:#3987e5"):
        assert colour in html, colour
    for retired in ("--f-gen", "--f-dec", "--f-emb", "--f-sp"):
        assert retired not in html, retired


def test_page_hard_codes_no_fleet_member():
    html = _html().lower()
    for name in ("spark2", "cultureos", "thor", "orin", "glm-5.3", "parakeet", "clef"):
        assert name not in html, name


def test_page_never_invents_memory_or_a_hub():
    html = _html()
    # memory ring and size only from a measured member; hub only from /api/mesh's hub
    assert "memory not measured" in html
    assert "if (mem)" in html
    assert "mesh.hub && mesh.hub.member" in html


def test_page_has_no_declared_hub_label():
    # deviation d2: the hub is only ever the member running lobes-cloudflared,
    # read from the running system; nothing is labelled declared any more
    html = _html()
    assert "declared" not in html
    assert "holds the outside tunnel" in html
