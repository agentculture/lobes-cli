/**
 * The realtime session transport — issue #151 t13.
 *
 * One WebSocket to `/v1/realtime`, opened and closed by the operator, with
 * its state observable. Everything about the session's CONTENT — capturing
 * mic audio into it (t11) and rendering the events out of it (t12) — belongs
 * to sibling islands; this module owns only the socket, the state machine
 * around it, and the URL that addresses it.
 *
 * NO CREDENTIAL LIVES HERE, and none can. The page dials its OWN origin; the
 * dev server's proxy attaches `Authorization: Bearer <key>` on the way to the
 * gateway (`site/proxy/gateway-proxy.mjs`). That indirection is not a
 * convenience — a browser `WebSocket` cannot set request headers at all, and
 * query-parameter / subprotocol authentication is out of scope by operator
 * decision, so a same-origin proxy is the only way a browser reaches a
 * header-authenticated gateway without being handed a key.
 *
 * DOM-free on purpose: no `document`, no `window`, `WebSocket` injectable.
 * The state machine is then testable under vitest without a browser, which
 * is where its transitions are actually asserted.
 */

/** Sample rates the bridge accepts (`parse_session_config` rejects any other). */
export const SAMPLE_RATES = [24000, 16000] as const;
export type SampleRate = (typeof SAMPLE_RATES)[number];

/**
 * Server-side AEC modes. `none` is the shipped default and the right answer
 * for this site: the browser gets echo cancellation from `getUserMedia`
 * constraints (t11), so the server passthrough stays off.
 */
export const AEC_MODES = ["none", "aec"] as const;
export type AecMode = (typeof AEC_MODES)[number];

/**
 * The default endpoint is a PATH, not an origin — it resolves against
 * whatever origin served the page, which is what keeps the request
 * same-origin and therefore proxied. An absolute `ws://…` override is
 * accepted (useful when a proxy runs on another port) but reaching the
 * gateway directly cannot work: the browser has no credential to send.
 */
export const DEFAULT_ENDPOINT = "/v1/realtime";

/** Where the connection panel preflights: keyed, so it proves the credential. */
export const MODELS_PATH = "/v1/models";

/** Keyless on the gateway, but still proxied — it sends no CORS headers. */
export const CAPABILITIES_PATH = "/capabilities";

export type ConnectionState = "disconnected" | "connecting" | "open" | "closing" | "failed";

export interface SessionSettings {
  inputSampleRate: SampleRate;
  aecMode: AecMode;
}

export const DEFAULT_SESSION_SETTINGS: SessionSettings = {
  inputSampleRate: 24000,
  aecMode: "none",
};

/**
 * What subscribers hear. A discriminated union rather than several callbacks
 * so a consumer (t12's event log especially) can render an ordered stream
 * that interleaves transport facts with server events — the order of "socket
 * opened" against "first boundary event" is exactly the kind of thing this
 * site exists to make visible.
 */
export type ConnectionNotice =
  | { kind: "state"; state: ConnectionState; url: string | null; detail: string }
  /** A well-formed JSON text frame — the server event schema. */
  | { kind: "event"; event: Record<string, unknown>; raw: string }
  /** A text frame that was not JSON. Surfaced, never swallowed. */
  | { kind: "malformed"; raw: string; detail: string }
  /** A binary frame. The #151 wire is base64-in-JSON, so this is unexpected. */
  | { kind: "binary"; byteLength: number };

export type ConnectionListener = (notice: ConnectionNotice) => void;

/**
 * The slice of `WebSocket` this module uses. Declared structurally so a test
 * can hand in a fake without a DOM and without `any`.
 */
export interface SocketLike {
  readonly readyState: number;
  binaryType?: string;
  send(data: string | ArrayBufferLike | ArrayBufferView | Blob): void;
  close(code?: number, reason?: string): void;
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: (() => void) | null;
  onclose: ((event: { code: number; reason: string; wasClean: boolean }) => void) | null;
}

export type SocketFactory = (url: string) => SocketLike;

export interface RealtimeConnectionOptions {
  /** Path or absolute URL. Defaults to {@link DEFAULT_ENDPOINT}. */
  endpoint?: string;
  settings?: Partial<SessionSettings>;
  /** The document's own URL; relative endpoints resolve against it. */
  baseHref?: string;
  /** Injected in tests; defaults to the global `WebSocket`. */
  socketFactory?: SocketFactory;
}

export interface RealtimeConnection {
  readonly state: ConnectionState;
  /** The URL of the current or most recent attempt; `null` before the first. */
  readonly url: string | null;
  getSettings(): SessionSettings;
  /** Rejected while a session is live — the config is fixed at connect time. */
  updateSettings(patch: Partial<SessionSettings>): boolean;
  setEndpoint(endpoint: string): boolean;
  getEndpoint(): string;
  connect(): void;
  disconnect(): void;
  /** Raw frame out. `false` when the socket is not open. */
  send(data: string | ArrayBufferLike | ArrayBufferView | Blob): boolean;
  /** A JSON event out — the #151 wire (t11's `input_audio_buffer.append`). */
  sendEvent(event: Record<string, unknown>): boolean;
  subscribe(listener: ConnectionListener): () => void;
}

const OPEN_READY_STATE = 1;

/**
 * Turn an endpoint + settings into the URL to dial.
 *
 * Exported and pure because the scheme mapping is the one piece of this that
 * silently breaks a deployment when wrong: a page served over HTTPS (the
 * mkcert path in the README) MUST use `wss:`, or the browser blocks the
 * connection as mixed content and reports nothing useful.
 */
export function resolveSessionUrl(
  endpoint: string,
  settings: SessionSettings,
  baseHref: string,
): string {
  const url = new URL(endpoint, baseHref);
  if (url.protocol === "http:") {
    url.protocol = "ws:";
  } else if (url.protocol === "https:") {
    url.protocol = "wss:";
  }
  // Only the two knobs this panel exposes are sent. Every other key
  // (`input_audio_format`, `input_channels`, `turn_detection`) is left to the
  // server's default, and the negotiated result comes back on
  // `session.created` — so what the session actually agreed to is read off
  // the wire, never assumed from what we typed.
  url.searchParams.set("input_sample_rate", String(settings.inputSampleRate));
  url.searchParams.set("aec_mode", settings.aecMode);
  return url.toString();
}

/**
 * Human-readable close-code detail.
 *
 * `wasOpen` matters more than the code: a browser reports EVERY failed
 * handshake as 1006 with no status, so a 401 (wrong key), a 404
 * (`role_infeasible` — the `stt` lane is declared off) and "no dev server at
 * all" are indistinguishable from inside the page. Saying so, and pointing at
 * the preflight that CAN see the status, beats inventing a cause.
 */
export function describeClose(code: number, reason: string, wasOpen: boolean): string {
  const tail = reason ? ` — ${reason}` : "";
  if (!wasOpen) {
    return (
      `handshake failed (code ${code})${tail}. The browser WebSocket API hides the ` +
      `HTTP status of a rejected upgrade, so this looks identical for "no local proxy", ` +
      `"gateway unreachable", "401 wrong or missing key" and "404 stt lane declared off". ` +
      `Run the gateway check above — it sees the status the socket cannot.`
    );
  }
  if (code === 1008) {
    return `server closed the session: policy violation (1008)${tail} — see the error event above for the named code`;
  }
  if (code === 1000) {
    return `closed cleanly (1000)${tail}`;
  }
  return `connection closed (code ${code})${tail}`;
}

export function createRealtimeConnection(
  options: RealtimeConnectionOptions = {},
): RealtimeConnection {
  const listeners = new Set<ConnectionListener>();
  let endpoint = options.endpoint ?? DEFAULT_ENDPOINT;
  let settings: SessionSettings = { ...DEFAULT_SESSION_SETTINGS, ...options.settings };
  const baseHref = options.baseHref ?? globalThis.location?.href ?? "http://localhost/";
  const socketFactory: SocketFactory =
    options.socketFactory ?? ((url) => new WebSocket(url) as unknown as SocketLike);

  let socket: SocketLike | null = null;
  let state: ConnectionState = "disconnected";
  let url: string | null = null;
  let everOpen = false;

  function emit(notice: ConnectionNotice): void {
    for (const listener of [...listeners]) {
      listener(notice);
    }
  }

  function setState(next: ConnectionState, detail: string): void {
    state = next;
    emit({ kind: "state", state: next, url, detail });
  }

  function handleMessage(data: unknown): void {
    if (typeof data === "string") {
      let parsed: unknown;
      try {
        parsed = JSON.parse(data);
      } catch (error) {
        emit({ kind: "malformed", raw: data, detail: String(error) });
        return;
      }
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        emit({ kind: "malformed", raw: data, detail: "expected a JSON object" });
        return;
      }
      emit({ kind: "event", event: parsed as Record<string, unknown>, raw: data });
      return;
    }
    // Binary is not part of the #151 wire in either direction (audio-out
    // arrives base64-encoded inside JSON deltas). Reporting the size rather
    // than dropping it means a wire regression shows up in the event log
    // instead of as unexplained silence.
    const byteLength =
      data instanceof ArrayBuffer
        ? data.byteLength
        : ArrayBuffer.isView(data)
          ? data.byteLength
          : -1;
    emit({ kind: "binary", byteLength });
  }

  function connect(): void {
    if (state === "connecting" || state === "open") {
      return;
    }
    url = resolveSessionUrl(endpoint, settings, baseHref);
    everOpen = false;
    setState("connecting", `dialling ${url}`);
    let created: SocketLike;
    try {
      created = socketFactory(url);
    } catch (error) {
      // A malformed endpoint throws synchronously — a typo in the override
      // field should say so, not look like an unreachable server.
      socket = null;
      setState("failed", `could not open a socket: ${String(error)}`);
      return;
    }
    socket = created;
    created.binaryType = "arraybuffer";
    created.onopen = () => {
      everOpen = true;
      setState("open", "session open — waiting for session.created");
    };
    created.onmessage = (event) => handleMessage(event.data);
    created.onerror = () => {
      // The browser deliberately gives no detail here (it would leak
      // cross-origin information). The close event that follows carries what
      // little there is, so nothing is reported twice.
    };
    created.onclose = (event) => {
      socket = null;
      const detail = describeClose(event.code, event.reason, everOpen);
      setState(everOpen ? "disconnected" : "failed", detail);
    };
  }

  function disconnect(): void {
    if (socket === null) {
      if (state !== "disconnected") {
        setState("disconnected", "closed locally");
      }
      return;
    }
    setState("closing", "closing the session");
    socket.close(1000, "client closed the session");
  }

  // Closures, not object methods: a sibling island is free to destructure
  // (`const { sendEvent } = connection`) without losing `this`.
  function send(data: string | ArrayBufferLike | ArrayBufferView | Blob): boolean {
    if (socket === null || socket.readyState !== OPEN_READY_STATE) {
      return false;
    }
    socket.send(data);
    return true;
  }

  function idle(): boolean {
    return state === "disconnected" || state === "failed";
  }

  return {
    get state() {
      return state;
    },
    get url() {
      return url;
    },
    getSettings: () => ({ ...settings }),
    getEndpoint: () => endpoint,
    updateSettings(patch) {
      if (!idle()) {
        return false;
      }
      settings = { ...settings, ...patch };
      return true;
    },
    setEndpoint(next) {
      if (!idle()) {
        return false;
      }
      endpoint = next.trim() === "" ? DEFAULT_ENDPOINT : next.trim();
      return true;
    },
    connect,
    disconnect,
    send,
    sendEvent: (event) => send(JSON.stringify(event)),
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

/* -------------------------------------------------------------------------
 * Gateway preflight
 * ---------------------------------------------------------------------- */

export interface GatewayProbe {
  /** Did anything answer at all? */
  reachable: boolean;
  /** `true` accepted, `false` rejected (401), `null` not determined. */
  authorized: boolean | null;
  /** Is the `stt` lane feasible on this box? `null` when unknown. */
  sttFeasible: boolean | null;
  /** Peer named by an infeasible lane's honest referral, when declared. */
  hostedBy: string | null;
  detail: string;
}

/**
 * The slice of `fetch` the preflight uses. Narrow enough that the real
 * `globalThis.fetch` satisfies it without a cast, and that a test can supply
 * a three-property object.
 */
export type FetchLike = (input: string) => Promise<{
  ok: boolean;
  status: number;
  json: () => Promise<unknown>;
}>;

/**
 * Ask the gateway two questions the WebSocket cannot answer for itself.
 *
 * `GET /v1/models` goes through the same proxy entry with the same injected
 * credential as the session, so a 401 here is a 401 there — it is the only
 * way the page can tell "wrong key" from "nothing listening". `GET
 * /capabilities` is keyless and reports whether the `stt` lane (which owns
 * the realtime route) is feasible on this box at all, plus the `hosted_by`
 * peer when a mesh shape dropped it.
 */
export async function probeGateway(
  fetchImpl: FetchLike,
  baseHref = globalThis.location?.href ?? "http://localhost/",
): Promise<GatewayProbe> {
  const probe: GatewayProbe = {
    reachable: false,
    authorized: null,
    sttFeasible: null,
    hostedBy: null,
    detail: "",
  };
  const notes: string[] = [];

  try {
    const response = await fetchImpl(new URL(MODELS_PATH, baseHref).toString());
    probe.reachable = true;
    if (response.status === 401) {
      probe.authorized = false;
      notes.push(
        "gateway answered 401 — its GATEWAY_API_KEY does not match LOBES_GATEWAY_API_KEY in the dev server's environment",
      );
    } else if (response.ok) {
      probe.authorized = true;
      notes.push("credential accepted");
    } else {
      notes.push(`gateway answered ${response.status} on ${MODELS_PATH}`);
    }
  } catch (error) {
    notes.push(
      `no answer on ${MODELS_PATH} (${String(error)}) — is \`npm run dev\` serving this page, and is the gateway up at LOBES_GATEWAY_URL?`,
    );
  }

  try {
    const response = await fetchImpl(new URL(CAPABILITIES_PATH, baseHref).toString());
    if (response.ok) {
      probe.reachable = true;
      const payload = (await response.json()) as Record<string, unknown> | null;
      const stt = payload?.["stt"];
      if (stt !== null && typeof stt === "object") {
        const role = stt as Record<string, unknown>;
        probe.sttFeasible = role["feasible"] === true;
        const hostedBy = role["hosted_by"];
        probe.hostedBy = typeof hostedBy === "string" && hostedBy !== "" ? hostedBy : null;
        notes.push(
          probe.sttFeasible
            ? "stt lane feasible"
            : `stt lane declared off — /v1/realtime will 404 role_infeasible${
                probe.hostedBy ? `; hosted by ${probe.hostedBy}` : ""
              }`,
        );
      }
    }
  } catch {
    // The first probe already reported unreachability; a second identical
    // complaint would be noise.
  }

  probe.detail = notes.join("; ");
  return probe;
}
