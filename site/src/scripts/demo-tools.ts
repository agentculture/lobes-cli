/**
 * Browser-safe demo tools — hebrew-realtime, site task.
 *
 * Pure, unit-tested, NO network. These are the "tools" this test harness
 * offers to declare via `session.update` (OpenAI Realtime FLAT shape:
 * `{type: "function", name, description, parameters}` — see
 * `lobes/realtime/_session.py`'s `parse_tools`) and to execute locally when
 * the server calls one back via `response.function_call_arguments.done`.
 * Nothing here talks to the server, the DOM, or the filesystem: every
 * function takes plain values in and returns a plain string out, so the
 * whole module runs the same way in a test and in a real tab.
 *
 * Matching `scripts/realtime-he-accept.py`'s own tool-turn shape: the
 * client parses `arguments` (a JSON STRING, never pre-parsed by the
 * server), runs the tool, and answers with `output` — a string, by
 * convention JSON, but the wire itself never interprets it (see
 * `lobes/realtime/_session.py`'s `FunctionCallOutput`).
 */

/** A tool declaration in OpenAI Realtime's FLAT shape (see `parse_tools`). */
export interface RealtimeToolDeclaration {
  type: "function";
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

/** Hebrew weekday names, Sunday-first — matches `Intl` where available, and
 * stands in where it is not (a headless/old runtime, or a locale bundle
 * that omits `he`). */
const HEBREW_WEEKDAYS = [
  "יום ראשון",
  "יום שני",
  "יום שלישי",
  "יום רביעי",
  "יום חמישי",
  "יום שישי",
  "יום שבת",
];

export const GET_CURRENT_TIME_TOOL: RealtimeToolDeclaration = {
  type: "function",
  name: "get_current_time",
  description:
    "Return the current local date/time (ISO 8601) and the weekday name in Hebrew. Takes no arguments.",
  parameters: { type: "object", properties: {}, additionalProperties: false },
};

export const ROLL_DICE_TOOL: RealtimeToolDeclaration = {
  type: "function",
  name: "roll_dice",
  description: "Roll one die and return the result. Defaults to a six-sided die.",
  parameters: {
    type: "object",
    properties: {
      sides: {
        type: "integer",
        description: "Number of sides on the die (2-1000). Defaults to 6.",
        minimum: 2,
        maximum: 1000,
      },
    },
    additionalProperties: false,
  },
};

export const REMEMBER_NOTE_TOOL: RealtimeToolDeclaration = {
  type: "function",
  name: "remember_note",
  description: "Remember a short text note for the rest of this session.",
  parameters: {
    type: "object",
    properties: {
      text: { type: "string", description: "The note text to remember." },
    },
    required: ["text"],
    additionalProperties: false,
  },
};

export const LIST_NOTES_TOOL: RealtimeToolDeclaration = {
  type: "function",
  name: "list_notes",
  description: "List every note remembered so far this session, oldest first. Takes no arguments.",
  parameters: { type: "object", properties: {}, additionalProperties: false },
};

/** Every demo tool this harness can declare, in the order they are offered. */
export const DEMO_TOOLS: readonly RealtimeToolDeclaration[] = [
  GET_CURRENT_TIME_TOOL,
  ROLL_DICE_TOOL,
  REMEMBER_NOTE_TOOL,
  LIST_NOTES_TOOL,
];

/** An in-memory, session-scoped notes store — cleared by constructing a fresh one. */
export interface NotesStore {
  remember(text: string): void;
  list(): readonly string[];
  clear(): void;
}

export function createNotesStore(): NotesStore {
  const notes: string[] = [];
  return {
    remember(text: string) {
      notes.push(text);
    },
    list() {
      return notes.slice();
    },
    clear() {
      notes.length = 0;
    },
  };
}

/**
 * Build the `get_current_time` result. *now* is injected (default `new
 * Date()`), so the tool is deterministic under test.
 */
export function getCurrentTime(now: Date = new Date()): { iso: string; weekday_he: string } {
  return {
    iso: now.toISOString(),
    weekday_he: HEBREW_WEEKDAYS[now.getDay()]!,
  };
}

export class ToolArgumentValidationError extends Error {}

/**
 * Roll a die. *sides* defaults to 6; anything outside 2..1000, or not an
 * integer, is rejected — the caller (`executeDemoTool`) turns that into an
 * error string rather than letting it throw across the tool boundary.
 * *randomFn* is injected (default `Math.random`) so the result is
 * deterministic under test.
 */
export function rollDice(sides: unknown = 6, randomFn: () => number = Math.random): {
  sides: number;
  result: number;
} {
  const n = sides === undefined || sides === null ? 6 : Number(sides);
  if (!Number.isInteger(n) || n < 2 || n > 1000) {
    throw new ToolArgumentValidationError(
      `sides must be an integer between 2 and 1000, got ${JSON.stringify(sides)}`
    );
  }
  return { sides: n, result: Math.floor(randomFn() * n) + 1 };
}

/**
 * Parse a tool call's `arguments` JSON string. Never throws: malformed JSON
 * or a non-object value is reported as an `{ok: false, error}` result, which
 * `executeDemoTool` turns into a tool OUTPUT string (never a thrown
 * exception) — a malformed call from the model must not crash the harness.
 */
export function parseToolArguments(raw: string): { ok: true; value: Record<string, unknown> } | { ok: false; error: string } {
  let parsed: unknown;
  try {
    parsed = raw.trim() === "" ? {} : JSON.parse(raw);
  } catch (error) {
    return { ok: false, error: `arguments is not valid JSON: ${String(error)}` };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ok: false, error: `arguments must be a JSON object, got ${JSON.stringify(parsed)}` };
  }
  return { ok: true, value: parsed as Record<string, unknown> };
}

/**
 * Run one demo tool by name against already-parsed arguments, returning the
 * tool OUTPUT string (JSON, by convention — see this module's header doc).
 * Never throws: an unknown tool name or a bad argument value comes back as
 * `{"error": "..."}` JSON text, exactly like a real tool failing gracefully
 * — the wire's `output` field is opaque text either way.
 */
export function runDemoTool(
  name: string,
  args: Record<string, unknown>,
  store: NotesStore,
  deps: { now?: Date; randomFn?: () => number } = {}
): string {
  try {
    switch (name) {
      case "get_current_time":
        return JSON.stringify(getCurrentTime(deps.now));
      case "roll_dice":
        return JSON.stringify(rollDice(args["sides"], deps.randomFn));
      case "remember_note": {
        const text = args["text"];
        if (typeof text !== "string" || text.trim() === "") {
          return JSON.stringify({ error: "remember_note requires a non-empty string 'text'" });
        }
        store.remember(text);
        return JSON.stringify({ remembered: text, total_notes: store.list().length });
      }
      case "list_notes":
        return JSON.stringify({ notes: store.list() });
      default:
        return JSON.stringify({ error: `unknown tool ${JSON.stringify(name)}` });
    }
  } catch (error) {
    return JSON.stringify({ error: String(error instanceof Error ? error.message : error) });
  }
}

/**
 * The end-to-end entry point `connection-panel.ts` calls on
 * `response.function_call_arguments.done`: parse the raw `arguments`
 * string, run the named tool, and return the tool OUTPUT string. Never
 * throws — every failure mode (malformed JSON, an unknown tool, a bad
 * argument) becomes a `{"error": "..."}` string the model gets to see and
 * react to, exactly as a real backend tool would report its own failure.
 */
export function executeDemoTool(
  name: string,
  rawArguments: string,
  store: NotesStore,
  deps: { now?: Date; randomFn?: () => number } = {}
): string {
  const parsed = parseToolArguments(rawArguments);
  if (!parsed.ok) {
    return JSON.stringify({ error: parsed.error });
  }
  return runDemoTool(name, parsed.value, store, deps);
}
