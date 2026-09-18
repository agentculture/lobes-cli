/**
 * The chat-style conversation view + latency table — hebrew-realtime.
 *
 * `event-log.ts` renders the RAW wire, one row per event, which is exactly
 * right for debugging the protocol but is not what an operator actually
 * watches during a live Hebrew session: they want to know what was HEARD,
 * what was SAID, what TOOL ran and with what result, and whether a reply
 * was INTERRUPTED — a turn-shaped view built by folding several raw events
 * together, not a 1:1 mirror of them.
 *
 * Fed the exact same way `EventStream.astro` is (the zero-import `window`
 * CustomEvent seam `connection-panel.ts` already broadcasts on
 * `lobes:realtime-event`), so this view needs no new wiring anywhere else —
 * see `src/pages/index.astro`'s bridge script.
 *
 * RTL: `dir="auto"` on every text node holding transcript/reply text lets
 * the Unicode Bidi Algorithm choose per-row direction from the row's own
 * first strong character — a Hebrew turn renders right-to-left, an English
 * one is untouched, and a mixed Hebrew+Latin turn (e.g. a tool's JSON
 * arguments) does not break the surrounding layout because direction is
 * scoped to that one text node, never the whole page.
 */

import type { RawEvent } from "./realtime-events";

export type TurnKind = "heard" | "said" | "tool" | "interrupted" | "error";

export interface ConversationTurn {
  id: string;
  kind: TurnKind;
  text: string;
  /** Only present on a `tool` turn. */
  toolName?: string;
  toolCallId?: string;
  toolArguments?: string;
  /** Set once the tool result is known (this browser never withholds one —
   * see demo-tools.ts — but a turn a page reloaded mid-call would show
   * this as undefined, honestly). */
  toolResult?: string;
}

export interface LatencyRow {
  responseId: string;
  timings: Record<string, number>;
}

export interface ConversationViewController {
  readonly root: HTMLElement;
  pushEvent(raw: unknown): void;
  /**
   * Record the OUTBOUND tool result the browser itself computed and sent
   * (see `connection-panel.ts`'s `handleToolCall`) — the one half of a tool
   * round trip this view cannot see arrive over the wire, because it never
   * goes out over `pushEvent`'s inbound server-event path. A no-op if
   * *callId* does not match any rendered tool turn (e.g. this view was
   * mounted, or cleared, after the call already rendered).
   */
  recordToolResult(callId: string, output: string): void;
  clear(): void;
  readonly turns: readonly ConversationTurn[];
  readonly latencyRows: readonly LatencyRow[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const STAGE_ORDER = ["stt", "generate", "tool_wait", "phonikud", "tts", "first_delta"] as const;

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
}

export function mountConversationView(root: HTMLElement): ConversationViewController {
  const listEl = root.querySelector<HTMLElement>("[data-conversation-list]");
  const emptyEl = root.querySelector<HTMLElement>("[data-conversation-empty]");
  const latencyBodyEl = root.querySelector<HTMLElement>("[data-latency-body]");
  const latencyMedianEl = root.querySelector<HTMLElement>("[data-latency-median]");
  if (!listEl) {
    throw new Error("mountConversationView: root is missing a [data-conversation-list] element");
  }

  const turns: ConversationTurn[] = [];
  const latencyRows: LatencyRow[] = [];
  const firstDeltas: number[] = [];
  let counter = 0;

  function nextId(prefix: string): string {
    counter += 1;
    return `${prefix}-${counter}`;
  }

  function updateEmpty(): void {
    if (emptyEl) emptyEl.hidden = turns.length > 0;
  }

  function renderTurn(turn: ConversationTurn): void {
    const row = document.createElement("li");
    row.className = `cv-turn cv-${turn.kind}`;
    row.dataset.turnKind = turn.kind;

    const speaker = document.createElement("span");
    speaker.className = "cv-speaker";
    speaker.textContent =
      turn.kind === "heard"
        ? "You"
        : turn.kind === "said"
          ? "lobes"
          : turn.kind === "tool"
            ? `tool · ${turn.toolName ?? "?"}`
            : turn.kind === "interrupted"
              ? "interrupted"
              : "error";
    row.append(speaker);

    const text = document.createElement("span");
    text.className = "cv-text";
    // The RTL contract this module exists to prove: per-row automatic
    // direction, never a page-wide assumption about which language a
    // session is speaking.
    text.setAttribute("dir", "auto");
    text.textContent = turn.text;
    row.append(text);

    if (turn.kind === "tool") {
      const argsEl = document.createElement("span");
      argsEl.className = "cv-tool-args";
      argsEl.setAttribute("dir", "auto");
      argsEl.textContent = `args: ${turn.toolArguments ?? "{}"}`;
      row.append(argsEl);

      const resultEl = document.createElement("span");
      resultEl.className = "cv-tool-result";
      resultEl.setAttribute("dir", "auto");
      resultEl.dataset.callId = turn.toolCallId ?? "";
      resultEl.textContent =
        turn.toolResult !== undefined
          ? `result: ${turn.toolResult}`
          : "result: (no answer seen on this connection — the operator may be answering from a different tab)";
      row.append(resultEl);
    }

    listEl!.append(row);
  }

  function addTurn(turn: ConversationTurn): void {
    turns.push(turn);
    renderTurn(turn);
    updateEmpty();
  }

  function findOpenToolTurn(callId: string): ConversationTurn | undefined {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const turn = turns[i]!;
      if (turn.kind === "tool" && turn.toolCallId === callId) return turn;
    }
    return undefined;
  }

  function updateToolResult(callId: string, result: string): void {
    const turn = findOpenToolTurn(callId);
    if (!turn) return;
    turn.toolResult = result;
    // A plain attribute-equality scan rather than `CSS.escape` in a
    // selector string — call ids are server-generated opaque tokens, and
    // this avoids depending on a jsdom/browser CSS.escape polyfill for
    // something a linear scan of (at most a few hundred) rows does just as
    // well.
    for (const el of Array.from(listEl!.querySelectorAll<HTMLElement>("[data-call-id]"))) {
      if (el.dataset["callId"] === callId) {
        el.textContent = `result: ${result}`;
        break;
      }
    }
  }

  function renderLatencyRow(row: LatencyRow): void {
    if (!latencyBodyEl) return;
    const tr = document.createElement("tr");
    tr.dataset.responseId = row.responseId;
    const idCell = document.createElement("td");
    idCell.textContent = row.responseId;
    tr.append(idCell);
    for (const stage of STAGE_ORDER) {
      const cell = document.createElement("td");
      const value = row.timings[stage];
      cell.textContent = value === undefined ? "—" : `${value} ms`;
      if (stage === "first_delta" && value !== undefined) {
        cell.className = "cv-latency-highlight";
      }
      tr.append(cell);
    }
    // Newest first.
    latencyBodyEl.prepend(tr);
  }

  function updateLatencyMedian(): void {
    if (!latencyMedianEl) return;
    const m = median(firstDeltas);
    latencyMedianEl.textContent =
      m === null ? "no response.done with a first_delta timing yet" : `median first_delta: ${m} ms`;
  }

  function pushEvent(raw: unknown): void {
    if (!isRecord(raw) || typeof raw.type !== "string") return;
    const event = raw as RawEvent;

    switch (event.type) {
      case "conversation.item.input_audio_transcription.completed": {
        addTurn({
          id: nextId("heard"),
          kind: "heard",
          text: typeof event.text === "string" ? event.text : "",
        });
        break;
      }
      case "response.text.done": {
        addTurn({
          id: nextId("said"),
          kind: "said",
          text: typeof event.text === "string" ? event.text : "",
        });
        break;
      }
      case "response.function_call_arguments.done": {
        addTurn({
          id: nextId("tool"),
          kind: "tool",
          text: `called ${typeof event.name === "string" ? event.name : "?"}`,
          toolName: typeof event.name === "string" ? event.name : undefined,
          toolCallId: typeof event.call_id === "string" ? event.call_id : undefined,
          toolArguments: typeof event.arguments === "string" ? event.arguments : "{}",
        });
        break;
      }
      case "response.interrupted": {
        addTurn({ id: nextId("interrupted"), kind: "interrupted", text: "reply interrupted (barge-in)" });
        break;
      }
      case "error": {
        const code = typeof event.code === "string" ? event.code : "unknown";
        const message = typeof event.message === "string" ? event.message : "";
        addTurn({ id: nextId("error"), kind: "error", text: `${code}${message ? `: ${message}` : ""}` });
        break;
      }
      case "response.done": {
        const responseId = typeof event.response_id === "string" ? event.response_id : "unknown";
        const timingsRaw = event.timings;
        if (isRecord(timingsRaw)) {
          const timings: Record<string, number> = {};
          for (const [key, value] of Object.entries(timingsRaw)) {
            if (typeof value === "number") timings[key] = value;
          }
          if (Object.keys(timings).length > 0) {
            const row: LatencyRow = { responseId, timings };
            latencyRows.push(row);
            renderLatencyRow(row);
            if (typeof timings["first_delta"] === "number") {
              firstDeltas.push(timings["first_delta"]);
              updateLatencyMedian();
            }
          }
        }
        break;
      }
      default:
        break;
    }
  }

  function recordToolResult(callId: string, output: string): void {
    updateToolResult(callId, output);
  }

  function clear(): void {
    turns.length = 0;
    latencyRows.length = 0;
    firstDeltas.length = 0;
    listEl!.replaceChildren();
    if (latencyBodyEl) latencyBodyEl.replaceChildren();
    updateEmpty();
    updateLatencyMedian();
  }

  updateEmpty();
  updateLatencyMedian();

  return {
    root,
    pushEvent,
    recordToolResult,
    clear,
    get turns() {
      return turns.slice();
    },
    get latencyRows() {
      return latencyRows.slice();
    },
  };
}
