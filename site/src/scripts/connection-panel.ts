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
  DEFAULT_LANGUAGE,
  LANGUAGE_PRESETS,
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
import { DEMO_TOOLS, createNotesStore, executeDemoTool } from "./demo-tools.ts";
import type { NotesStore } from "./demo-tools.ts";

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

/**
 * The two extra client-sent event types this panel now speaks, mirroring
 * `lobes/realtime/_session.py`'s `SESSION_UPDATE_EVENT_TYPE` /
 * `CONVERSATION_ITEM_CREATE_EVENT_TYPE` / `FUNCTION_CALL_OUTPUT_ITEM_TYPE`
 * (hebrew-realtime). This module has no Python import path into `lobes/`,
 * so the literal strings are re-declared here — exactly the trade
 * `realtime-events.ts`'s own header comment already documents for the
 * server-origin event vocabulary.
 */
const SESSION_UPDATE_EVENT_TYPE = "session.update";
const CONVERSATION_ITEM_CREATE_EVENT_TYPE = "conversation.item.create";
const FUNCTION_CALL_OUTPUT_ITEM_TYPE = "function_call_output";

/**
 * Dispatched on `window` with `detail: {callId, output}` the moment this
 * panel sends a demo tool's `function_call_output` — the OUTBOUND half of a
 * tool round trip, which `conversation-view.ts` cannot observe by watching
 * inbound server events alone. See `ConversationView.astro`'s mount script.
 */
export const TOOL_RESULT_EVENT = "lobes:tool-result";

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
  "data-connection-language",
  "data-connection-conversation",
  "data-connection-tools",
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
  "data-tools-state",
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
  const languageInput = requireElement<HTMLInputElement>(root, "[data-connection-language]");
  const conversationCheckbox = requireElement<HTMLInputElement>(
    root,
    "[data-connection-conversation]",
  );
  const toolsCheckbox = requireElement<HTMLInputElement>(root, "[data-connection-tools]");
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
  const toolsStateOut = requireElement<HTMLElement>(root, "[data-tools-state]");

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
    languageInput.disabled = busy;
    // Locked while live: arming is a connect-time decision (see the module
    // doc above), and a control that visibly does nothing while disabled is
    // the honest way to say so — matching how the other three session-config
    // fields already lock for the same reason.
    conversationCheckbox.disabled = busy;
    toolsCheckbox.disabled = busy;
  }

  // -- tools declaration (hebrew-realtime) ---------------------------------
  //
  // A SECOND connect-time intent, read once at Connect exactly like
  // `armIntent` above, and a fresh `NotesStore` per session so remembered
  // notes never leak across a reconnect. `toolsReady` becomes true once the
  // server's `session.updated` echoes the declaration back — see
  // `scripts/realtime-he-accept.py`'s own reference shape: session.created
  // -> session.update(tools) -> session.updated -> THEN response.create.
  //
  // This keeps exactly ONE arming path (`maybeArm`, below), the single place
  // in this module that ever sends `response.create`: with tools declared,
  // arming waits for `toolsReady`; without them, it fires at "open" exactly
  // as before issue #151 t19 shipped it. A second, independent send —
  // `response.create` after a tool result — is not "arming" a session, it is
  // continuing an already-armed one (the reference script does the same),
  // so it is not routed through `maybeArm`.
  let toolsIntent = false;
  let toolsUpdateSent = false;
  let toolsReady = false;
  let notesStore: NotesStore = createNotesStore();

  function renderToolsState(): void {
    if (!toolsIntent) {
      toolsStateOut.textContent = "No tools declared — a plain ears-or-talks session.";
      return;
    }
    const names = DEMO_TOOLS.map((tool) => tool.name).join(", ");
    toolsStateOut.textContent = toolsReady
      ? `Declared — the server echoed tool_choice=auto for: ${names}`
      : `Will declare via session.update right after session.created: ${names}`;
  }

  function maybeArm(): void {
    if (!armIntent || sessionArmed || connection.state !== "open") return;
    if (toolsIntent && !toolsReady) return; // wait for the session.update round trip
    sessionArmed = true;
    connection.sendEvent({ type: RESPONSE_CREATE_EVENT_TYPE });
    renderConversationState(connection.state);
  }

  /** Run one demo tool call and answer it — the client half of the tool
   * round trip (`conversation.item.create(function_call_output)`, then
   * `response.create` to continue the SAME response; see
   * `lobes/realtime/_conversation.py`'s module doc, "A tool turn is the
   * ordinary turn with one extra leg"). */
  function handleToolCall(event: Record<string, unknown>): void {
    const callId = event["call_id"];
    const name = event["name"];
    if (typeof callId !== "string" || typeof name !== "string") return;
    const rawArguments = typeof event["arguments"] === "string" ? (event["arguments"] as string) : "";
    const output = executeDemoTool(name, rawArguments, notesStore);
    connection.sendEvent({
      type: CONVERSATION_ITEM_CREATE_EVENT_TYPE,
      item: { type: FUNCTION_CALL_OUTPUT_ITEM_TYPE, call_id: callId, output },
    });
    connection.sendEvent({ type: RESPONSE_CREATE_EVENT_TYPE });
    // The outbound half of the round trip the conversation view cannot see
    // arrive over the wire (it only sees the inbound call). Zero-import seam
    // on `window`, mirroring `mic-island.ts`'s `emitMuteEvent` — a no-op if
    // nothing is listening, and guarded so this file stays importable outside
    // a browser (the Astro build, a test with no jsdom `window`).
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent(TOOL_RESULT_EVENT, { detail: { callId, output } }));
    }
  }

  const unsubscribe = connection.subscribe((notice) => {
    if (notice.kind === "state") {
      renderState(notice.state, notice.detail, notice.url);
      if (notice.state === "open") {
        maybeArm();
      } else if (notice.state === "disconnected" || notice.state === "failed") {
        sessionArmed = false;
        toolsUpdateSent = false;
        toolsReady = false;
        renderToolsState();
      }
      renderConversationState(notice.state);
    } else if (notice.kind === "event") {
      const type = notice.event["type"];
      if (type === "session.created" && toolsIntent && !toolsUpdateSent) {
        toolsUpdateSent = true;
        connection.sendEvent({
          type: SESSION_UPDATE_EVENT_TYPE,
          session: { tools: DEMO_TOOLS, tool_choice: "auto" },
        });
      } else if (type === "session.updated" && toolsIntent && !toolsReady) {
        toolsReady = true;
        renderToolsState();
        maybeArm();
      } else if (type === "response.function_call_arguments.done") {
        handleToolCall(notice.event);
      }
    }
    broadcast(notice);
  });

  function onConnect(): void {
    connection.setEndpoint(endpointInput.value);
    connection.updateSettings({
      inputSampleRate: Number(rateSelect.value) as SampleRate,
      aecMode: aecSelect.value as AecMode,
      language: languageInput.value,
    });
    armIntent = conversationCheckbox.checked;
    toolsIntent = toolsCheckbox.checked;
    toolsUpdateSent = false;
    toolsReady = false;
    notesStore = createNotesStore();
    renderToolsState();
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

  function onToolsToggle(): void {
    // Only meaningful before Connect: `toolsIntent` itself is read fresh in
    // `onConnect`, so flipping this while idle just previews what the next
    // connect will do (mirrors `onConversationToggle`'s idle-only preview).
    if (connection.state === "disconnected" || connection.state === "failed") {
      toolsIntent = toolsCheckbox.checked;
      renderToolsState();
    }
  }

  connectButton.addEventListener("click", onConnect);
  disconnectButton.addEventListener("click", onDisconnect);
  checkButton.addEventListener("click", () => void onCheck());
  conversationCheckbox.addEventListener("change", onConversationToggle);
  toolsCheckbox.addEventListener("change", onToolsToggle);

  // The markup ships the buttons disabled so a JS-less visit cannot pretend
  // to work; taking over is what enables them. `renderState` below owns
  // connect/disconnect from here on, so only the check button is enabled
  // here.
  checkButton.disabled = false;
  if (endpointInput.value.trim() === "") {
    endpointInput.value = DEFAULT_ENDPOINT;
  }
  // Default to Hebrew — this harness's own default (see
  // `realtime-connection.ts`'s `DEFAULT_LANGUAGE`), never the server's own
  // "en" default. Free text: an operator can type any short code
  // (`parse_language` is a shape check, not a registry).
  if (languageInput.value.trim() === "") {
    languageInput.value = DEFAULT_LANGUAGE;
  }
  // The checkbox itself ships unchecked in the markup (no `checked`
  // attribute) — nothing here changes that. Only the always-visible text
  // state needs an explicit first render, to match whatever the browser
  // restored the control to (a reloaded tab can restore form state even
  // with JS disabled-then-enabled mid-session).
  renderState(connection.state, "not connected yet", connection.url);
  renderConversationState(connection.state);
  renderToolsState();

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
      toolsCheckbox.removeEventListener("change", onToolsToggle);
      if (globalTarget !== null && globalTarget[REALTIME_GLOBAL_KEY] === connection) {
        delete globalTarget[REALTIME_GLOBAL_KEY];
      }
    },
  };
}

/** The option values the panel's selects/datalist offer, for the .astro markup. */
export const PANEL_CHOICES = {
  sampleRates: SAMPLE_RATES,
  aecModes: AEC_MODES,
  languagePresets: LANGUAGE_PRESETS,
} as const;
