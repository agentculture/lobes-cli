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

/**
 * The conversation opt-in trigger — issue #151 t19. Mirrors
 * `lobes/realtime/_conversation.py`'s `RESPONSE_CREATE_EVENT_TYPE`
 * ("response.create"). Sent with no other fields: `_conversation.py`'s
 * `is_response_create` only checks `payload["type"]`, and this panel adopts
 * the ARM-AT-CONNECT shape the server explicitly supports ("send it once,
 * at connect, and get a reply to every committed turn thereafter") rather
 * than the OpenAI-style per-transcript shape — one checkbox checked before
 * pressing Connect is the whole interaction, which is what makes the site
 * simplest to drive for a live acceptance run: no per-turn control to
 * remember, and the toggle can never be forgotten mid-conversation because
 * it is read once, at connect time.
 */
const RESPONSE_CREATE_EVENT_TYPE = "response.create";

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
  "data-connection-conversation",
  "data-connection-connect",
  "data-connection-disconnect",
  "data-connection-check",
  "data-connection-state",
  "data-connection-glyph",
  "data-connection-label",
  "data-connection-detail",
  "data-connection-url",
  "data-connection-check-result",
  "data-conversation-state",
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
  const conversationCheckbox = requireElement<HTMLInputElement>(
    root,
    "[data-connection-conversation]",
  );
  const connectButton = requireElement<HTMLButtonElement>(root, "[data-connection-connect]");
  const disconnectButton = requireElement<HTMLButtonElement>(root, "[data-connection-disconnect]");
  const checkButton = requireElement<HTMLButtonElement>(root, "[data-connection-check]");
  const stateRegion = requireElement<HTMLElement>(root, "[data-connection-state]");
  const stateGlyph = requireElement<HTMLElement>(root, "[data-connection-glyph]");
  const stateLabel = requireElement<HTMLElement>(root, "[data-connection-label]");
  const detailOut = requireElement<HTMLElement>(root, "[data-connection-detail]");
  const urlOut = requireElement<HTMLElement>(root, "[data-connection-url]");
  const checkOut = requireElement<HTMLElement>(root, "[data-connection-check-result]");
  const conversationStateOut = requireElement<HTMLElement>(root, "[data-conversation-state]");

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

  // -- conversation arming (issue #151 t19) --------------------------------
  //
  // Two booleans, two different questions:
  //   armIntent    — what the checkbox said the LAST time Connect was
  //                  pressed. Read once, at that moment (`onConnect`), so a
  //                  toggle flipped mid-session cannot retroactively change
  //                  what an already-open socket did — the design brief's
  //                  "do not leave the user guessing" requirement, answered
  //                  by making the toggle itself uneditable while live (see
  //                  `renderState`'s busy-disable, below) rather than by
  //                  silently ignoring a change no one could make anyway.
  //   sessionArmed — did THIS live session actually get its response.create
  //                  sent. False until the socket reaches "open" with
  //                  armIntent true; reset on every disconnect/failure so a
  //                  reconnect starts the question over.
  // `renderConversationState` is the one place both collapse into the
  // always-visible text the design brief asked for.
  let armIntent = false;
  let sessionArmed = false;

  function renderConversationState(state: ConnectionState): void {
    const live = state === "connecting" || state === "open" || state === "closing";
    const armed = live ? sessionArmed : conversationCheckbox.checked;
    conversationStateOut.dataset["armed"] = String(armed);
    conversationStateOut.dataset["live"] = String(live);
    conversationStateOut.textContent = live
      ? armed
        ? "Armed — response.create sent; every committed turn on this session gets a spoken reply"
        : "Ears-only — this live session will not reply (the toggle applies at the next Connect)"
      : armed
        ? "Will reply — response.create sends automatically right after Connect"
        : "Ears-only (default) — this session will not reply";
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
    // Locked while live: arming is a connect-time decision (see the module
    // doc above), and a control that visibly does nothing while disabled is
    // the honest way to say so — matching how the other three session-config
    // fields already lock for the same reason.
    conversationCheckbox.disabled = busy;
  }

  const unsubscribe = connection.subscribe((notice) => {
    if (notice.kind === "state") {
      renderState(notice.state, notice.detail, notice.url);
      if (notice.state === "open" && armIntent && !sessionArmed) {
        sessionArmed = true;
        connection.sendEvent({ type: RESPONSE_CREATE_EVENT_TYPE });
      } else if (notice.state === "disconnected" || notice.state === "failed") {
        sessionArmed = false;
      }
      renderConversationState(notice.state);
    }
    broadcast(notice);
  });

  function onConnect(): void {
    connection.setEndpoint(endpointInput.value);
    connection.updateSettings({
      inputSampleRate: Number(rateSelect.value) as SampleRate,
      aecMode: aecSelect.value as AecMode,
    });
    armIntent = conversationCheckbox.checked;
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

  function onConversationToggle(): void {
    renderConversationState(connection.state);
  }

  connectButton.addEventListener("click", onConnect);
  disconnectButton.addEventListener("click", onDisconnect);
  checkButton.addEventListener("click", () => void onCheck());
  conversationCheckbox.addEventListener("change", onConversationToggle);

  // The markup ships the buttons disabled so a JS-less visit cannot pretend
  // to work; taking over is what enables them. `renderState` below owns
  // connect/disconnect from here on, so only the check button is enabled
  // here.
  checkButton.disabled = false;
  if (endpointInput.value.trim() === "") {
    endpointInput.value = DEFAULT_ENDPOINT;
  }
  // The checkbox itself ships unchecked in the markup (no `checked`
  // attribute) — nothing here changes that. Only the always-visible text
  // state needs an explicit first render, to match whatever the browser
  // restored the control to (a reloaded tab can restore form state even
  // with JS disabled-then-enabled mid-session).
  renderState(connection.state, "not connected yet", connection.url);
  renderConversationState(connection.state);

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
      conversationCheckbox.removeEventListener("change", onConversationToggle);
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
