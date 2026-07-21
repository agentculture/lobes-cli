import { describe, expect, it, vi } from "vitest";

import {
  DEFAULT_ENDPOINT,
  createRealtimeConnection,
  describeClose,
  probeGateway,
  resolveSessionUrl,
} from "../src/scripts/realtime-connection.ts";
import type {
  ConnectionNotice,
  SessionSettings,
  SocketLike,
} from "../src/scripts/realtime-connection.ts";

const SETTINGS: SessionSettings = { inputSampleRate: 24000, aecMode: "none" };

/** A hand-driven stand-in for the browser's WebSocket. */
class FakeSocket implements SocketLike {
  readyState = 0;
  binaryType = "blob";
  sent: unknown[] = [];
  closedWith: [number | undefined, string | undefined] | null = null;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event: { code: number; reason: string; wasClean: boolean }) => void) | null = null;

  constructor(readonly url: string) {}

  send(data: string | ArrayBufferLike | ArrayBufferView | Blob): void {
    this.sent.push(data);
  }

  close(code?: number, reason?: string): void {
    this.closedWith = [code, reason];
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.();
  }

  message(data: unknown): void {
    this.onmessage?.({ data });
  }

  shut(code: number, reason = "", wasClean = true): void {
    this.readyState = 3;
    this.onclose?.({ code, reason, wasClean });
  }
}

function harness(endpoint = DEFAULT_ENDPOINT) {
  const sockets: FakeSocket[] = [];
  const notices: ConnectionNotice[] = [];
  const connection = createRealtimeConnection({
    endpoint,
    baseHref: "http://localhost:4321/dev-connection",
    socketFactory: (url) => {
      const socket = new FakeSocket(url);
      sockets.push(socket);
      return socket;
    },
  });
  connection.subscribe((notice) => notices.push(notice));
  return { connection, sockets, notices };
}

const states = (notices: ConnectionNotice[]) =>
  notices.filter((n) => n.kind === "state").map((n) => n.state);

describe("resolveSessionUrl", () => {
  it("resolves a path against the page origin and swaps http for ws", () => {
    const url = resolveSessionUrl("/v1/realtime", SETTINGS, "http://localhost:4321/dev-connection");
    expect(url).toBe("ws://localhost:4321/v1/realtime?input_sample_rate=24000&aec_mode=none");
  });

  it("uses wss when the page is served over HTTPS (the mkcert flow)", () => {
    // Getting this wrong is silent: the browser blocks a ws: socket from an
    // https: page as mixed content and reports nothing actionable.
    const url = resolveSessionUrl("/v1/realtime", SETTINGS, "https://lobes.local:4321/");
    expect(url.startsWith("wss://lobes.local:4321/v1/realtime")).toBe(true);
  });

  it("carries the chosen sample rate and AEC mode as query params", () => {
    const url = resolveSessionUrl(
      "/v1/realtime",
      { inputSampleRate: 16000, aecMode: "aec" },
      "http://localhost:4321/",
    );
    expect(url).toContain("input_sample_rate=16000");
    expect(url).toContain("aec_mode=aec");
  });

  it("honours an absolute override without rewriting its origin", () => {
    const url = resolveSessionUrl("ws://127.0.0.1:5199/v1/realtime", SETTINGS, "http://localhost:4321/");
    expect(url.startsWith("ws://127.0.0.1:5199/v1/realtime")).toBe(true);
  });

  it("never puts anything credential-shaped in the URL", () => {
    const url = resolveSessionUrl("/v1/realtime", SETTINGS, "http://localhost:4321/");
    // A token in a query string lands in every access log on the path. The
    // panel has no key to leak, and this asserts the shape stays that way.
    expect(url).not.toMatch(/token|key|authorization|bearer/i);
  });
});

describe("connection state machine", () => {
  it("walks disconnected -> connecting -> open -> disconnected on a clean close", () => {
    const { connection, sockets, notices } = harness();
    connection.connect();
    expect(connection.state).toBe("connecting");
    sockets[0]!.open();
    expect(connection.state).toBe("open");
    connection.disconnect();
    expect(connection.state).toBe("closing");
    expect(sockets[0]!.closedWith?.[0]).toBe(1000);
    sockets[0]!.shut(1000);
    expect(connection.state).toBe("disconnected");
    expect(states(notices)).toEqual(["connecting", "open", "closing", "disconnected"]);
  });

  it("reports a never-opened socket as FAILED, not merely disconnected", () => {
    // This is the whole diagnostic value of the state: "the proxy is down"
    // and "the session ended" must not look the same.
    const { connection, sockets } = harness();
    connection.connect();
    sockets[0]!.shut(1006, "", false);
    expect(connection.state).toBe("failed");
  });

  it("does not open a second socket while one is live", () => {
    const { connection, sockets } = harness();
    connection.connect();
    connection.connect();
    sockets[0]!.open();
    connection.connect();
    expect(sockets).toHaveLength(1);
  });

  it("parses JSON text frames into event notices", () => {
    const { connection, sockets, notices } = harness();
    connection.connect();
    sockets[0]!.open();
    sockets[0]!.message('{"type":"session.created","session":{"id":"sess_1"}}');
    const event = notices.find((n) => n.kind === "event");
    expect(event?.kind === "event" && event.event["type"]).toBe("session.created");
  });

  it("surfaces a non-JSON text frame instead of swallowing it", () => {
    const { connection, sockets, notices } = harness();
    connection.connect();
    sockets[0]!.open();
    sockets[0]!.message("not json at all");
    expect(notices.some((n) => n.kind === "malformed")).toBe(true);
  });

  it("reports a binary frame — the #151 wire is base64-in-JSON both ways", () => {
    const { connection, sockets, notices } = harness();
    connection.connect();
    sockets[0]!.open();
    sockets[0]!.message(new ArrayBuffer(320));
    const binary = notices.find((n) => n.kind === "binary");
    expect(binary?.kind === "binary" && binary.byteLength).toBe(320);
  });

  it("refuses to send before the socket is open, and sends after", () => {
    const { connection, sockets } = harness();
    connection.connect();
    expect(connection.sendEvent({ type: "input_audio_buffer.append" })).toBe(false);
    sockets[0]!.open();
    expect(connection.sendEvent({ type: "input_audio_buffer.append", audio: "AAA=" })).toBe(true);
    expect(sockets[0]!.sent[0]).toBe('{"type":"input_audio_buffer.append","audio":"AAA="}');
  });

  it("freezes the session config while a session is live", () => {
    // The bridge fixes the config at connect time from the connect URL; a
    // control that appeared to change it mid-session would be a lie.
    const { connection, sockets } = harness();
    connection.connect();
    sockets[0]!.open();
    expect(connection.updateSettings({ inputSampleRate: 16000 })).toBe(false);
    expect(connection.getSettings().inputSampleRate).toBe(24000);
    sockets[0]!.shut(1000);
    expect(connection.updateSettings({ inputSampleRate: 16000 })).toBe(true);
  });

  it("reports a synchronously-thrown socket construction as failed", () => {
    const connection = createRealtimeConnection({
      baseHref: "http://localhost:4321/",
      socketFactory: () => {
        throw new Error("bad url");
      },
    });
    connection.connect();
    expect(connection.state).toBe("failed");
  });
});

describe("describeClose", () => {
  it("says a failed handshake hides its HTTP status", () => {
    const detail = describeClose(1006, "", false);
    expect(detail).toContain("handshake failed");
    expect(detail).toContain("401");
    expect(detail).toContain("404");
  });

  it("names a 1008 policy close as a server decision", () => {
    expect(describeClose(1008, "invalid config", true)).toContain("policy violation");
  });
});

describe("probeGateway", () => {
  const response = (status: number, body: unknown = {}) => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  });

  it("reads a 401 as a credential rejection, naming the env var to fix", async () => {
    const fetchImpl = vi.fn(async (url: string) =>
      url.includes("/v1/models") ? response(401) : response(500),
    );
    const probe = await probeGateway(fetchImpl, "http://localhost:4321/");
    expect(probe.reachable).toBe(true);
    expect(probe.authorized).toBe(false);
    expect(probe.detail).toContain("LOBES_GATEWAY_API_KEY");
  });

  it("reports an infeasible stt lane and the peer hosting it", async () => {
    const fetchImpl = vi.fn(async (url: string) =>
      url.includes("/capabilities")
        ? response(200, { stt: { feasible: false, hosted_by: "http://thor:8000" } })
        : response(200, { data: [] }),
    );
    const probe = await probeGateway(fetchImpl, "http://localhost:4321/");
    expect(probe.authorized).toBe(true);
    expect(probe.sttFeasible).toBe(false);
    expect(probe.hostedBy).toBe("http://thor:8000");
    expect(probe.detail).toContain("role_infeasible");
  });

  it("reports nothing listening as unreachable, not as unauthorized", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    const probe = await probeGateway(fetchImpl, "http://localhost:4321/");
    expect(probe.reachable).toBe(false);
    expect(probe.authorized).toBeNull();
    expect(probe.detail).toContain("npm run dev");
  });
});
