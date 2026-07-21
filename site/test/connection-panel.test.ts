import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeEach, describe, expect, it } from "vitest";

import {
  PANEL_HOOKS,
  REALTIME_GLOBAL_KEY,
  REALTIME_NOTICE_EVENT,
  REALTIME_READY_EVENT,
  mountConnectionPanel,
} from "../src/scripts/connection-panel.ts";
import { createRealtimeConnection } from "../src/scripts/realtime-connection.ts";
import type { ConnectionNotice, SocketLike } from "../src/scripts/realtime-connection.ts";

// Resolved from the vitest root (site/), not from `import.meta.url`: under
// the jsdom environment that is an http: URL, not a file: one.
const COMPONENT_SOURCE = readFileSync(
  resolve(process.cwd(), "src/components/ConnectionPanel.astro"),
  "utf8",
);

/**
 * A fixture mirroring the component's markup closely enough to drive the
 * wiring. The `PANEL_HOOKS` scan below is what keeps this honest: it asserts
 * the real component still provides every hook this fixture does, so a
 * renamed attribute fails the suite rather than only the browser.
 */
function panelFixture(): HTMLElement {
  const root = document.createElement("div");
  root.setAttribute("data-connection-panel", "");
  root.innerHTML = `
    <div data-connection-state data-state="disconnected" role="status" aria-live="polite">
      <span data-connection-glyph>○</span>
      <strong data-connection-label>Disconnected</strong>
      <span data-connection-detail></span>
    </div>
    <code data-connection-url>—</code>
    <input type="text" value="/v1/realtime" data-connection-endpoint />
    <select data-connection-rate>
      <option value="24000">24000 Hz</option>
      <option value="16000">16000 Hz</option>
    </select>
    <select data-connection-aec>
      <option value="none">none</option>
      <option value="aec">aec</option>
    </select>
    <label>
      <input type="checkbox" data-connection-conversation />
    </label>
    <span data-conversation-state data-armed="false" data-live="false"></span>
    <button data-connection-connect disabled>Connect</button>
    <button data-connection-disconnect disabled>Disconnect</button>
    <button data-connection-check disabled>Check gateway</button>
    <p data-connection-check-result data-verdict="idle"></p>
  `;
  document.body.append(root);
  return root;
}

class FakeSocket implements SocketLike {
  readyState = 0;
  binaryType = "blob";
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event: { code: number; reason: string; wasClean: boolean }) => void) | null = null;
  sent: unknown[] = [];
  constructor(readonly url: string) {}
  send(data: string): void {
    this.sent.push(data);
  }
  closedWith: number | null = null;
  close(code?: number): void {
    // Records only. A real browser socket sits in CLOSING until the peer
    // acknowledges, which is exactly the state the panel must render — a
    // fake that closed synchronously would hide it.
    this.closedWith = code ?? null;
  }
  open(): void {
    this.readyState = 1;
    this.onopen?.();
  }
  shut(code: number, wasClean = true): void {
    this.readyState = 3;
    this.onclose?.({ code, reason: "", wasClean });
  }
}

function mounted() {
  const sockets: FakeSocket[] = [];
  const connection = createRealtimeConnection({
    baseHref: "http://localhost:4321/dev-connection",
    socketFactory: (url) => {
      const socket = new FakeSocket(url);
      sockets.push(socket);
      return socket;
    },
  });
  const root = panelFixture();
  const panel = mountConnectionPanel(root, { connection });
  return { root, panel, sockets, connection };
}

const query = <T extends Element>(root: ParentNode, hook: string) =>
  root.querySelector<T>(`[${hook}]`)!;

beforeEach(() => {
  document.body.innerHTML = "";
});

describe("ConnectionPanel.astro markup", () => {
  it("provides every hook the wiring requires", () => {
    for (const hook of PANEL_HOOKS) {
      expect(COMPONENT_SOURCE, `missing ${hook}`).toContain(hook);
    }
  });

  it("has no field that could hold a credential", () => {
    // The browser holds no key. A password field, or an input whose name or
    // label mentions a key/token, would be an invitation to paste one into a
    // page that then puts it on the wire as a query param.
    expect(COMPONENT_SOURCE).not.toMatch(/type="password"/);
    expect(COMPONENT_SOURCE).not.toMatch(/<input[^>]*\b(api[_-]?key|token|secret|bearer)\b/i);
  });

  it("ships its buttons disabled so a JS-less visit cannot pretend to work", () => {
    expect(COMPONENT_SOURCE).toMatch(/data-connection-connect disabled/);
    expect(COMPONENT_SOURCE).toMatch(/data-connection-check disabled/);
  });

  it("announces state changes to assistive technology", () => {
    expect(COMPONENT_SOURCE).toContain('role="status"');
    expect(COMPONENT_SOURCE).toContain('aria-live="polite"');
  });
});

describe("mountConnectionPanel", () => {
  it("renders each state with a distinct glyph AND word, not colour alone", () => {
    const { root, sockets } = mounted();
    const glyph = query<HTMLElement>(root, "data-connection-glyph");
    const label = query<HTMLElement>(root, "data-connection-label");
    const region = query<HTMLElement>(root, "data-connection-state");

    const seen: Array<[string, string, string]> = [];
    const snapshot = () =>
      seen.push([region.dataset["state"]!, glyph.textContent!, label.textContent!]);

    snapshot(); // disconnected
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    snapshot(); // connecting
    sockets[0]!.open();
    snapshot(); // open
    query<HTMLButtonElement>(root, "data-connection-disconnect").click();
    snapshot(); // closing
    sockets[0]!.shut(1000);

    // A second attempt that never opens: a close before `open` is a FAILED
    // handshake, not an ended session, and must look different.
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    sockets[1]!.shut(1006, false);
    snapshot(); // failed

    expect(seen.map((entry) => entry[0])).toEqual([
      "disconnected",
      "connecting",
      "open",
      "closing",
      "failed",
    ]);
    // Every state is distinguishable without seeing any colour at all.
    expect(new Set(seen.map((entry) => entry[1])).size).toBe(5);
    expect(new Set(seen.map((entry) => entry[2])).size).toBe(5);
  });

  it("shows the session URL and it carries no credential", () => {
    const { root } = mounted();
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    const shown = query<HTMLElement>(root, "data-connection-url").textContent!;
    expect(shown).toContain("ws://localhost:4321/v1/realtime");
    expect(shown).not.toMatch(/token|key|authorization|bearer/i);
  });

  it("sends the operator's chosen sample rate on the connect URL", () => {
    const { root, sockets } = mounted();
    query<HTMLSelectElement>(root, "data-connection-rate").value = "16000";
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    expect(sockets[0]!.url).toContain("input_sample_rate=16000");
  });

  it("locks the config controls while a session is live and frees them after", () => {
    const { root, sockets } = mounted();
    const rate = query<HTMLSelectElement>(root, "data-connection-rate");
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    sockets[0]!.open();
    expect(rate.disabled).toBe(true);
    expect(query<HTMLButtonElement>(root, "data-connection-connect").disabled).toBe(true);
    expect(query<HTMLButtonElement>(root, "data-connection-disconnect").disabled).toBe(false);
    sockets[0]!.shut(1000);
    expect(rate.disabled).toBe(false);
  });

  it("broadcasts every notice on document for the sibling islands", () => {
    // t11 (mic) and t12 (event log) are built in parallel and may load in
    // any order; this seam needs no import and no load-order agreement.
    const heard: ConnectionNotice[] = [];
    document.addEventListener(REALTIME_NOTICE_EVENT, (event) => {
      heard.push((event as CustomEvent<ConnectionNotice>).detail);
    });
    const { root, sockets } = mounted();
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    sockets[0]!.open();
    sockets[0]!.onmessage?.({ data: '{"type":"session.created"}' });
    expect(heard.map((notice) => notice.kind)).toEqual(["connecting", "open"].map(() => "state").concat("event"));
  });

  it("publishes the connection both ways so load order cannot matter", () => {
    let fromEvent: unknown = null;
    document.addEventListener(REALTIME_READY_EVENT, (event) => {
      fromEvent = (event as CustomEvent<{ connection: unknown }>).detail.connection;
    });
    const { panel } = mounted();
    expect(fromEvent).toBe(panel.connection);
    expect((globalThis as Record<string, unknown>)[REALTIME_GLOBAL_KEY]).toBe(panel.connection);
    panel.destroy();
    expect((globalThis as Record<string, unknown>)[REALTIME_GLOBAL_KEY]).toBeUndefined();
  });

  it("reports a rejected credential from the preflight without echoing it", async () => {
    const root = panelFixture();
    mountConnectionPanel(root, {
      connection: createRealtimeConnection({
        baseHref: "http://localhost:4321/",
        socketFactory: (url) => new FakeSocket(url),
      }),
      fetchImpl: async () => ({
        ok: false,
        status: 401,
        json: async () => ({}),
      }),
    });
    query<HTMLButtonElement>(root, "data-connection-check").click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    const result = query<HTMLElement>(root, "data-connection-check-result");
    expect(result.dataset["verdict"]).toBe("rejected");
    expect(result.textContent).toContain("401");
  });

  it("throws loudly when the markup is missing a hook", () => {
    const broken = document.createElement("div");
    expect(() => mountConnectionPanel(broken)).toThrow(/missing required element/);
  });
});

// ---------------------------------------------------------------------------
// Conversation arming — issue #151 t19.
//
// The toggle defaults OFF (unchecked in the markup, see panelFixture()
// above) so a session is ears-only unless an operator deliberately opts in.
// The chosen shape is ARM-AT-CONNECT: the checkbox is read once, when
// Connect is pressed, and — if checked — exactly one `response.create`
// event is sent the moment the socket reaches "open". That is the whole
// contract these tests pin: OFF sends nothing extra ever (byte-identical to
// the pre-#151-t19 wire from the client's side), ON sends exactly one frame
// at a known point, and the control locks while a session is live so a
// mid-session flip can never leave the operator guessing which behaviour is
// in effect.
// ---------------------------------------------------------------------------
describe("mountConnectionPanel — conversation arming (issue #151 t19)", () => {
  it("ships unchecked, and the state text says so before anything connects", () => {
    const { root } = mounted();
    const checkbox = query<HTMLInputElement>(root, "data-connection-conversation");
    const state = query<HTMLElement>(root, "data-conversation-state");
    expect(checkbox.checked).toBe(false);
    expect(state.dataset["armed"]).toBe("false");
    expect(state.dataset["live"]).toBe("false");
    expect(state.textContent).toMatch(/ears-only/i);
  });

  it("off: connecting and opening a session sends no response.create — byte-identical to ears-only", () => {
    const { root, sockets } = mounted();
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    sockets[0]!.open();
    expect(sockets[0]!.sent).toEqual([]);
    const state = query<HTMLElement>(root, "data-conversation-state");
    expect(state.dataset["armed"]).toBe("false");
    expect(state.dataset["live"]).toBe("true");
  });

  it("on: checking the box before Connect sends exactly one response.create right after open", () => {
    const { root, sockets } = mounted();
    query<HTMLInputElement>(root, "data-connection-conversation").checked = true;
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    // Nothing is sent while still connecting — arming happens at "open".
    expect(sockets[0]!.sent).toEqual([]);
    sockets[0]!.open();
    expect(sockets[0]!.sent).toEqual(['{"type":"response.create"}']);
    // Opening again (defensively) must not send a second one.
    sockets[0]!.onopen?.();
    expect(sockets[0]!.sent).toEqual(['{"type":"response.create"}']);
    const state = query<HTMLElement>(root, "data-conversation-state");
    expect(state.dataset["armed"]).toBe("true");
    expect(state.dataset["live"]).toBe("true");
    expect(state.textContent).toMatch(/armed/i);
  });

  it("reads the checkbox fresh on each Connect click, not a stale value from before", () => {
    const { root, sockets } = mounted();
    const checkbox = query<HTMLInputElement>(root, "data-connection-conversation");

    // Checked, then unchecked again before ever pressing Connect.
    checkbox.checked = true;
    checkbox.checked = false;
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    sockets[0]!.open();
    expect(sockets[0]!.sent).toEqual([]);
    sockets[0]!.shut(1000);

    // Now check it and connect again — a fresh connect must arm.
    checkbox.checked = true;
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    sockets[1]!.open();
    expect(sockets[1]!.sent).toEqual(['{"type":"response.create"}']);
  });

  it("locks the toggle while a session is live, and frees it again after disconnect", () => {
    const { root, sockets } = mounted();
    const checkbox = query<HTMLInputElement>(root, "data-connection-conversation");
    expect(checkbox.disabled).toBe(false);
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    expect(checkbox.disabled).toBe(true);
    sockets[0]!.open();
    expect(checkbox.disabled).toBe(true);
    sockets[0]!.shut(1000);
    expect(checkbox.disabled).toBe(false);
  });

  it("toggling while idle updates the at-a-glance state text immediately, with no connection at all", () => {
    const { root } = mounted();
    const checkbox = query<HTMLInputElement>(root, "data-connection-conversation");
    const state = query<HTMLElement>(root, "data-conversation-state");

    checkbox.checked = true;
    checkbox.dispatchEvent(new Event("change"));
    expect(state.dataset["armed"]).toBe("true");
    expect(state.dataset["live"]).toBe("false");
    expect(state.textContent).toMatch(/will reply/i);

    checkbox.checked = false;
    checkbox.dispatchEvent(new Event("change"));
    expect(state.dataset["armed"]).toBe("false");
    expect(state.textContent).toMatch(/ears-only/i);
  });

  it("a session that failed to open (never armed) resets sessionArmed for the next attempt", () => {
    const { root, sockets } = mounted();
    query<HTMLInputElement>(root, "data-connection-conversation").checked = true;
    query<HTMLButtonElement>(root, "data-connection-connect").click();
    sockets[0]!.shut(1006, false); // handshake failure — never opened, never armed
    const state = query<HTMLElement>(root, "data-conversation-state");
    expect(state.dataset["live"]).toBe("false");
    expect(state.dataset["armed"]).toBe("true"); // checkbox is still checked — will retry armed

    query<HTMLButtonElement>(root, "data-connection-connect").click();
    sockets[1]!.open();
    expect(sockets[1]!.sent).toEqual(['{"type":"response.create"}']);
  });
});
