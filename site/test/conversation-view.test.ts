import { beforeEach, describe, expect, it } from "vitest";

import { mountConversationView } from "../src/scripts/conversation-view.ts";
import { EVENT_FIXTURES } from "../src/scripts/event-fixtures.ts";

function createRoot(): HTMLElement {
  const root = document.createElement("div");
  root.innerHTML = `
    <ol data-conversation-list aria-live="polite"></ol>
    <p data-conversation-empty>Nothing heard or said yet.</p>
    <table>
      <tbody data-latency-body></tbody>
    </table>
    <p data-latency-median></p>
  `;
  document.body.append(root);
  return root;
}

beforeEach(() => {
  document.body.innerHTML = "";
});

describe("mountConversationView", () => {
  it("starts empty", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    expect(view.turns).toHaveLength(0);
    expect(root.querySelector<HTMLElement>("[data-conversation-empty]")!.hidden).toBe(false);
  });

  it("renders a heard turn from a transcription-completed event, with dir=auto", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({
      type: "conversation.item.input_audio_transcription.completed",
      text: "מה השעה עכשיו?",
    });
    expect(view.turns).toHaveLength(1);
    expect(view.turns[0]!.kind).toBe("heard");
    const textEl = root.querySelector<HTMLElement>(".cv-text")!;
    expect(textEl.getAttribute("dir")).toBe("auto");
    expect(textEl.textContent).toBe("מה השעה עכשיו?");
  });

  it("renders no bubble for a blank transcript (noise the STT gate dropped)", () => {
    // Seen in a real browser, 2026-09-18: a column of empty "YOU" bubbles.
    const root = createRoot();
    const view = mountConversationView(root);
    for (const text of ["", "   ", undefined]) {
      view.pushEvent({ type: "conversation.item.input_audio_transcription.completed", text });
    }
    expect(view.turns).toHaveLength(0);
  });

  it("renders a said turn from response.text.done", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({ type: "response.text.done", text: "hello" });
    expect(view.turns[0]!.kind).toBe("said");
  });

  it("renders a tool turn with its name, call id and arguments, then fills in the result", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({
      type: "response.function_call_arguments.done",
      call_id: "call_1",
      name: "roll_dice",
      arguments: '{"sides": 6}',
    });
    expect(view.turns[0]!.kind).toBe("tool");
    expect(view.turns[0]!.toolName).toBe("roll_dice");
    const resultEl = root.querySelector<HTMLElement>('[data-call-id="call_1"]')!;
    expect(resultEl.textContent).toContain("no answer seen");

    view.recordToolResult("call_1", '{"sides":6,"result":4}');
    expect(resultEl.textContent).toContain('"result":4');
  });

  it("renders an interrupted marker on response.interrupted", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({ type: "response.interrupted", response_id: "resp_1", truncated: true });
    expect(view.turns[0]!.kind).toBe("interrupted");
  });

  it("renders a named error turn", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({ type: "error", code: "generate_failed", message: "gateway 404" });
    expect(view.turns[0]!.kind).toBe("error");
    expect(view.turns[0]!.text).toContain("generate_failed");
  });

  it("ignores boundary/lifecycle events it has no turn shape for", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({ type: "session.created" });
    view.pushEvent({ type: "input_audio_buffer.speech_started" });
    expect(view.turns).toHaveLength(0);
  });

  it("never throws on a malformed payload", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    expect(() => view.pushEvent("nope")).not.toThrow();
    expect(() => view.pushEvent(null)).not.toThrow();
    expect(() => view.pushEvent({ no_type: true })).not.toThrow();
  });

  it("clear() empties both the transcript and the latency table", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({ type: "response.text.done", text: "hi" });
    view.pushEvent({ type: "response.done", response_id: "resp_1", timings: { first_delta: 100 } });
    view.clear();
    expect(view.turns).toHaveLength(0);
    expect(view.latencyRows).toHaveLength(0);
    expect(root.querySelectorAll(".cv-turn")).toHaveLength(0);
  });

  it("replays the full hebrew-realtime fixture story without throwing", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    for (const event of EVENT_FIXTURES) {
      expect(() => view.pushEvent(event)).not.toThrow();
    }
    expect(view.turns.length).toBeGreaterThan(0);
    // The fixture's tool turn is in there.
    expect(view.turns.some((t) => t.kind === "tool" && t.toolName === "get_current_time")).toBe(true);
  });
});

describe("latency table", () => {
  it("renders only the timings stages present, omitting the rest as em-dashes", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({
      type: "response.done",
      response_id: "resp_1",
      timings: { stt: 100, generate: 200, first_delta: 450 },
    });
    expect(view.latencyRows).toHaveLength(1);
    const row = root.querySelector("tbody tr")!;
    const cells = Array.from(row.querySelectorAll("td")).map((td) => td.textContent);
    expect(cells).toContain("100 ms");
    expect(cells).toContain("200 ms");
    expect(cells).toContain("450 ms");
    expect(cells.filter((c) => c === "—").length).toBeGreaterThan(0);
  });

  it("ignores a response.done with no timings — no row added", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({ type: "response.done", response_id: "resp_1" });
    expect(view.latencyRows).toHaveLength(0);
  });

  it("newest response is prepended, so it reads first", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    view.pushEvent({ type: "response.done", response_id: "resp_1", timings: { first_delta: 100 } });
    view.pushEvent({ type: "response.done", response_id: "resp_2", timings: { first_delta: 200 } });
    const rows = Array.from(root.querySelectorAll<HTMLElement>("tbody tr"));
    expect(rows[0]!.dataset["responseId"]).toBe("resp_2");
    expect(rows[1]!.dataset["responseId"]).toBe("resp_1");
  });

  it("shows a running median of first_delta", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    const medianEl = root.querySelector<HTMLElement>("[data-latency-median]")!;
    expect(medianEl.textContent).toContain("no response.done");

    view.pushEvent({ type: "response.done", response_id: "resp_1", timings: { first_delta: 100 } });
    view.pushEvent({ type: "response.done", response_id: "resp_2", timings: { first_delta: 300 } });
    expect(medianEl.textContent).toContain("200");
  });

  it("tolerates an unknown extra stage key in the row data without crashing rendering", () => {
    const root = createRoot();
    const view = mountConversationView(root);
    expect(() =>
      view.pushEvent({
        type: "response.done",
        response_id: "resp_1",
        timings: { first_sentence: 150, first_delta: 400 },
      }),
    ).not.toThrow();
    expect(view.latencyRows[0]!.timings["first_sentence"]).toBe(150);
  });
});
