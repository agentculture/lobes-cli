import { describe, expect, it } from "vitest";

import {
  DEFAULT_GATEWAY_URL,
  GATEWAY_API_KEY_VAR,
  GATEWAY_URL_VAR,
  PROXY_CONTEXTS,
  applyCredential,
  buildProxyConfig,
  describeProxy,
  readProxyEnvironment,
} from "../proxy/gateway-proxy.mjs";

/** Records the header operations a proxy hook performs, in order. */
function recordingRequest() {
  const calls: Array<[string, string, string?]> = [];
  return {
    calls,
    removeHeader(name: string) {
      calls.push(["remove", name]);
    },
    setHeader(name: string, value: string) {
      calls.push(["set", name, value]);
    },
  };
}

describe("readProxyEnvironment", () => {
  it("falls back to the local gateway when no URL is declared", () => {
    expect(readProxyEnvironment({})).toEqual({
      gatewayUrl: DEFAULT_GATEWAY_URL,
      apiKey: "",
    });
  });

  it("trims and normalises a declared URL", () => {
    const env = { [GATEWAY_URL_VAR]: "  http://127.0.0.1:9000/  " };
    expect(readProxyEnvironment(env).gatewayUrl).toBe("http://127.0.0.1:9000");
  });

  it("treats an empty key as no key — the gateway's own gate is opt-in", () => {
    // A bare local fleet runs with GATEWAY_API_KEY unset, so demanding a
    // credential here would break the commonest setup.
    expect(readProxyEnvironment({ [GATEWAY_API_KEY_VAR]: "   " }).apiKey).toBe("");
  });

  it("carries a declared key through verbatim", () => {
    expect(readProxyEnvironment({ [GATEWAY_API_KEY_VAR]: " s3cret " }).apiKey).toBe("s3cret");
  });
});

describe("applyCredential", () => {
  it("strips inbound credentials BEFORE attaching the operator's", () => {
    const request = recordingRequest();
    applyCredential(request, "s3cret");
    // Order is the assertion: setting first and stripping second would ship
    // an unauthenticated handshake and 401 every session.
    expect(request.calls).toEqual([
      ["remove", "cookie"],
      ["remove", "authorization"],
      ["set", "Authorization", "Bearer s3cret"],
    ]);
  });

  it("attaches nothing when no key is configured, but still strips", () => {
    const request = recordingRequest();
    applyCredential(request, "");
    expect(request.calls).toEqual([
      ["remove", "cookie"],
      ["remove", "authorization"],
    ]);
  });
});

describe("buildProxyConfig", () => {
  it("claims the session route and the preflight routes, with ws enabled", () => {
    const config = buildProxyConfig({ gatewayUrl: "http://127.0.0.1:8000", apiKey: "k" });
    expect(Object.keys(config).sort()).toEqual([...PROXY_CONTEXTS].sort());
    for (const entry of Object.values(config)) {
      expect(entry.ws).toBe(true);
      expect(entry.target).toBe("http://127.0.0.1:8000");
      expect(entry.changeOrigin).toBe(true);
    }
    // /v1/ must be a claimed context or the WebSocket route is unreachable.
    expect(Object.keys(config)).toContain("/v1/");
  });

  it("wires BOTH the HTTP and the upgrade hook", () => {
    // Wiring only one is the silent half-failure: the preflight would
    // authenticate and the session would 401, or the reverse.
    const config = buildProxyConfig({ gatewayUrl: "http://127.0.0.1:8000", apiKey: "k" });
    const events: string[] = [];
    const requests: ReturnType<typeof recordingRequest>[] = [];
    const proxy = {
      on(event: string, handler: (req: unknown) => void) {
        events.push(event);
        const request = recordingRequest();
        requests.push(request);
        handler(request);
      },
    };
    config["/v1/"]?.configure?.(proxy as never, {} as never);
    expect(events).toEqual(["proxyReq", "proxyReqWs"]);
    for (const request of requests) {
      expect(request.calls).toContainEqual(["set", "Authorization", "Bearer k"]);
    }
  });

  it("never puts the key in the config anywhere a serialiser could reach", () => {
    // The credential lives in a closure, not in a serialisable field. If it
    // ever moves into `headers`, a config dump (`astro dev --verbose`, a
    // crash report, a bug-report paste) would print it.
    const config = buildProxyConfig({ gatewayUrl: "http://127.0.0.1:8000", apiKey: "TOPSECRET" });
    const dumped = JSON.stringify(config, (_key, value) =>
      typeof value === "function" ? "[fn]" : value,
    );
    expect(dumped).not.toContain("TOPSECRET");
  });
});

describe("describeProxy", () => {
  it("reports key PRESENCE and never key material", () => {
    const line = describeProxy({ gatewayUrl: "http://127.0.0.1:8000", apiKey: "TOPSECRET" });
    expect(line).toContain("http://127.0.0.1:8000");
    expect(line).toContain("injecting Authorization");
    // A dev-server banner is a scrollback buffer and sometimes a pasted bug
    // report. Not a prefix, not a length, not a hash — presence only.
    expect(line).not.toContain("TOPSECRET");
    expect(line).not.toContain("TOPSEC");
  });

  it("says plainly when no credential is attached", () => {
    const line = describeProxy({ gatewayUrl: DEFAULT_GATEWAY_URL, apiKey: "" });
    expect(line).toContain("no credential");
  });
});
