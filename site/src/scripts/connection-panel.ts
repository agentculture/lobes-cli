/**
 * The connection panel's DOM binding — issue #151 t13.
 *
 * The markup is server-rendered by `ConnectionPanel.astro`; this module finds
 * it and makes it live. That split is the site's progressive-enhancement
 * stance (a no-JS visit still renders a composed, honestly-inert panel) and
 * it keeps every query in one place a test can drive with a jsdom fragment.
 *
 * The panel deliberately has NO field for a key, and never will: the browser
 * holds no credential. The only thing it can say about authentication is what
 * the gateway told the local proxy — see the "check gateway" preflight.
 */
import {
  AEC_MODES,
  DEFAULT_ENDPOINT,
  SAMPLE_RATES,
  createRealtimeConnection,
  probeGateway,
} from "./realtime-connection.ts";
import type {
  AecMode,
  ConnectionNotice,
  ConnectionState,
  FetchLike,
  RealtimeConnection,
  SampleRate,
} from "./realtime-connection.ts";

/**
 * Every notice this panel hears is re-dispatched on `document` under this
 * name, with the {@link ConnectionNotice} as `detail`.
 *
 * It is the ZERO-IMPORT seam for the sibling islands (t11 mic/playback, t12
 * event log): they can render the whole session without importing this module
 * or knowing when it loaded. The typed module export is the better seam when
 * import order is under someone's control; this one always works.
 */
export const REALTIME_NOTICE_EVENT = "lobes:realtime";

/**
 * Fired once on `document` when the panel mounts, with
 * `detail: { connection }`. An island that loads FIRST listens for this; an
 * island that loads LATER reads `window.lobesRealtime`. Both are populated,
 * because three islands built in parallel have no guaranteed load order.
 */
export const REALTIME_READY_EVENT = "lobes:realtime-ready";

/** Property the mounted connection is published on, for late-loading islands. */
export const REALTIME_GLOBAL_KEY = "lobesRealtime";

interface StatePresentation {
  label: string;
  /** Shape, not colour: the state must survive a monochrome screen. */
  glyph: string;
}

/**
 * Colour is never the only signal (WCAG 1.4.1). Each state carries a distinct
 * GLYPH and a distinct WORD; the stylesheet adds a distinct border treatment
 * on top, and the live region announces the change to a screen reader.
 */
const STATE_PRESENTATION: Record<ConnectionState, StatePresentation> = {
  disconnected: { label: "Disconnected", glyph: "○" },
  connecting: { label: "Connecting…", glyph: "◐" },
  open: { label: "Open", glyph: "●" },
  closing: { label: "Closing…", glyph: "◑" },
  failed: { label: "Failed", glyph: "✕" },
};

/**
 * Every hook `mountConnectionPanel` requires the markup to provide.
 *
 * The markup lives in `ConnectionPanel.astro` and the wiring lives here, so
 * the two can drift apart silently — a renamed attribute would only surface
 * as a thrown error in a browser nobody opened. This list is what
 * `test/connection-panel.test.ts` scans the component source against, so the
 * drift is caught by `npm test` instead.
 */
export const PANEL_HOOKS = [
  "data-connection-endpoint",
  "data-connection-rate",
  "data-connection-aec",
  "data-connection-connect",
  "data-connection-disconnect",
  "data-connection-check",
  "data-connection-state",
  "data-connection-glyph",
  "data-connection-label",
  "data-connection-detail",
  "data-connection-url",
  "data-connection-check-result",
] as const;

export interface MountConnectionPanelOptions {
  /** Injected in tests; defaults to a fresh connection over the real socket. */
  connection?: RealtimeConnection;
  /** Injected in tests; defaults to the global `fetch`. */
  fetchImpl?: FetchLike;
  /** Where notices are broadcast. `null` disables broadcasting. */
  broadcastTarget?: EventTarget | null;
  /** Object the connection is published on. `null` disables publishing. */
  globalTarget?: Record<string, unknown> | null;
}

export interface MountedConnectionPanel {
  connection: RealtimeConnection;
  destroy(): void;
}

function requireElement<T extends Element>(root: ParentNode, selector: string): T {
  const found = root.querySelector<T>(selector);
  if (found === null) {
    throw new Error(`connection panel: missing required element ${selector}`);
  }
  return found;
}

/**
 * Wire a server-rendered `[data-mount="connection"]` panel.
 *
 * Returns the connection so the page (or the coordinator wiring
 * `index.astro`) can hand the SAME session to the mic island and the event
 * log — one socket, three islands.
 */
export function mountConnectionPanel(
  root: HTMLElement,
  options: MountConnectionPanelOptions = {},
): MountedConnectionPanel {
  const endpointInput = requireElement<HTMLInputElement>(root, "[data-connection-endpoint]");
  const rateSelect = requireElement<HTMLSelectElement>(root, "[data-connection-rate]");
  const aecSelect = requireElement<HTMLSelectElement>(root, "[data-connection-aec]");
  const connectButton = requireElement<HTMLButtonElement>(root, "[data-connection-connect]");
  const disconnectButton = requireElement<HTMLButtonElement>(root, "[data-connection-disconnect]");
  const checkButton = requireElement<HTMLButtonElement>(root, "[data-connection-check]");
  const stateRegion = requireElement<HTMLElement>(root, "[data-connection-state]");
  const stateGlyph = requireElement<HTMLElement>(root, "[data-connection-glyph]");
  const stateLabel = requireElement<HTMLElement>(root, "[data-connection-label]");
  const detailOut = requireElement<HTMLElement>(root, "[data-connection-detail]");
  const urlOut = requireElement<HTMLElement>(root, "[data-connection-url]");
  const checkOut = requireElement<HTMLElement>(root, "[data-connection-check-result]");

  const connection = options.connection ?? createRealtimeConnection();
  const broadcastTarget =
    options.broadcastTarget === undefined ? globalThis.document : options.broadcastTarget;
  const globalTarget =
    options.globalTarget === undefined
      ? (globalThis as unknown as Record<string, unknown>)
      : options.globalTarget;

  function broadcast(notice: ConnectionNotice): void {
    broadcastTarget?.dispatchEvent(
      new CustomEvent<ConnectionNotice>(REALTIME_NOTICE_EVENT, { detail: notice }),
    );
  }

  function renderState(state: ConnectionState, detail: string, url: string | null): void {
    const presentation = STATE_PRESENTATION[state];
    stateRegion.dataset["state"] = state;
    stateGlyph.textContent = presentation.glyph;
    stateLabel.textContent = presentation.label;
    detailOut.textContent = detail;
    // The URL is shown because it is the one place a mistyped endpoint
    // becomes obvious. It never contains a credential — there is none to put
    // in it, and putting one in a query string would land it in every log
    // between here and the gateway.
    urlOut.textContent = url ?? "—";
    const busy = state === "connecting" || state === "open" || state === "closing";
    connectButton.disabled = busy;
    disconnectButton.disabled = !busy;
    endpointInput.disabled = busy;
    rateSelect.disabled = busy;
    aecSelect.disabled = busy;
  }

  const unsubscribe = connection.subscribe((notice) => {
    if (notice.kind === "state") {
      renderState(notice.state, notice.detail, notice.url);
    }
    broadcast(notice);
  });

  function onConnect(): void {
    connection.setEndpoint(endpointInput.value);
    connection.updateSettings({
      inputSampleRate: Number(rateSelect.value) as SampleRate,
      aecMode: aecSelect.value as AecMode,
    });
    connection.connect();
  }

  function onDisconnect(): void {
    connection.disconnect();
  }

  async function onCheck(): Promise<void> {
    checkButton.disabled = true;
    checkOut.dataset["verdict"] = "pending";
    checkOut.textContent = "checking…";
    try {
      const fetchImpl: FetchLike = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
      const probe = await probeGateway(fetchImpl);
      checkOut.dataset["verdict"] = probe.reachable
        ? probe.authorized === false
          ? "rejected"
          : "ok"
        : "unreachable";
      checkOut.textContent = probe.detail || "no verdict";
    } catch (error) {
      checkOut.dataset["verdict"] = "unreachable";
      checkOut.textContent = `check failed: ${String(error)}`;
    } finally {
      checkButton.disabled = false;
    }
  }

  connectButton.addEventListener("click", onConnect);
  disconnectButton.addEventListener("click", onDisconnect);
  checkButton.addEventListener("click", () => void onCheck());

  // The markup ships the buttons disabled so a JS-less visit cannot pretend
  // to work; taking over is what enables them. `renderState` below owns
  // connect/disconnect from here on, so only the check button is enabled
  // here.
  checkButton.disabled = false;
  if (endpointInput.value.trim() === "") {
    endpointInput.value = DEFAULT_ENDPOINT;
  }
  renderState(connection.state, "not connected yet", connection.url);

  if (globalTarget !== null) {
    globalTarget[REALTIME_GLOBAL_KEY] = connection;
  }
  broadcastTarget?.dispatchEvent(
    new CustomEvent(REALTIME_READY_EVENT, { detail: { connection } }),
  );

  return {
    connection,
    destroy() {
      unsubscribe();
      connectButton.removeEventListener("click", onConnect);
      disconnectButton.removeEventListener("click", onDisconnect);
      if (globalTarget !== null && globalTarget[REALTIME_GLOBAL_KEY] === connection) {
        delete globalTarget[REALTIME_GLOBAL_KEY];
      }
    },
  };
}

/** The option values the panel's two selects offer, for the .astro markup. */
export const PANEL_CHOICES = {
  sampleRates: SAMPLE_RATES,
  aecModes: AEC_MODES,
} as const;
