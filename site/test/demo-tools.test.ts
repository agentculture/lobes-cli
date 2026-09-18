import { describe, expect, it } from "vitest";

import {
  DEMO_TOOLS,
  createNotesStore,
  executeDemoTool,
  getCurrentTime,
  parseToolArguments,
  rollDice,
  runDemoTool,
  ToolArgumentValidationError,
} from "../src/scripts/demo-tools.ts";

describe("DEMO_TOOLS", () => {
  it("declares every tool in OpenAI Realtime's FLAT shape", () => {
    expect(DEMO_TOOLS.length).toBeGreaterThan(0);
    for (const tool of DEMO_TOOLS) {
      expect(tool.type).toBe("function");
      expect(typeof tool.name).toBe("string");
      expect(typeof tool.description).toBe("string");
      expect(typeof tool.parameters).toBe("object");
    }
    expect(new Set(DEMO_TOOLS.map((t) => t.name)).size).toBe(DEMO_TOOLS.length);
  });
});

describe("getCurrentTime", () => {
  it("returns an ISO timestamp and the Hebrew weekday name", () => {
    // 2026-09-15 is a Tuesday.
    const result = getCurrentTime(new Date("2026-09-15T12:00:00Z"));
    expect(result.iso).toBe("2026-09-15T12:00:00.000Z");
    expect(result.weekday_he).toBe("יום שלישי");
  });

  it("defaults to the current time when none is injected", () => {
    const before = Date.now();
    const result = getCurrentTime();
    const parsed = Date.parse(result.iso);
    expect(parsed).toBeGreaterThanOrEqual(before - 5000);
  });
});

describe("rollDice", () => {
  it("defaults to a six-sided die", () => {
    const result = rollDice(undefined, () => 0);
    expect(result.sides).toBe(6);
    expect(result.result).toBe(1);
  });

  it("respects an injected random function across the full range", () => {
    expect(rollDice(20, () => 0).result).toBe(1);
    expect(rollDice(20, () => 0.999999).result).toBe(20);
  });

  it("rejects a non-integer or out-of-range side count", () => {
    expect(() => rollDice(1)).toThrow(ToolArgumentValidationError);
    expect(() => rollDice(1001)).toThrow(ToolArgumentValidationError);
    expect(() => rollDice(3.5)).toThrow(ToolArgumentValidationError);
    expect(() => rollDice("banana")).toThrow(ToolArgumentValidationError);
  });
});

describe("createNotesStore", () => {
  it("remembers notes in order and lists them back", () => {
    const store = createNotesStore();
    expect(store.list()).toEqual([]);
    store.remember("buy milk");
    store.remember("call mom");
    expect(store.list()).toEqual(["buy milk", "call mom"]);
  });

  it("clear() empties it", () => {
    const store = createNotesStore();
    store.remember("x");
    store.clear();
    expect(store.list()).toEqual([]);
  });

  it("list() returns a defensive copy", () => {
    const store = createNotesStore();
    store.remember("x");
    const copy = store.list() as string[];
    copy.push("y");
    expect(store.list()).toEqual(["x"]);
  });
});

describe("parseToolArguments", () => {
  it("parses a well-formed JSON object", () => {
    const result = parseToolArguments('{"sides": 20}');
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.value).toEqual({ sides: 20 });
  });

  it("treats an empty string as an empty object (get_current_time/list_notes call with no args)", () => {
    const result = parseToolArguments("");
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.value).toEqual({});
  });

  it("never throws on malformed JSON — returns an error result instead", () => {
    expect(() => parseToolArguments("{not json")).not.toThrow();
    const result = parseToolArguments("{not json");
    expect(result.ok).toBe(false);
  });

  it("rejects a JSON array or scalar as not-an-object", () => {
    expect(parseToolArguments("[1,2,3]").ok).toBe(false);
    expect(parseToolArguments("42").ok).toBe(false);
    expect(parseToolArguments("null").ok).toBe(false);
  });
});

describe("runDemoTool", () => {
  it("dispatches get_current_time", () => {
    const store = createNotesStore();
    const out = runDemoTool("get_current_time", {}, store, { now: new Date("2026-09-15T12:00:00Z") });
    expect(JSON.parse(out)).toEqual({ iso: "2026-09-15T12:00:00.000Z", weekday_he: "יום שלישי" });
  });

  it("dispatches roll_dice", () => {
    const store = createNotesStore();
    const out = runDemoTool("roll_dice", { sides: 6 }, store, { randomFn: () => 0 });
    expect(JSON.parse(out)).toEqual({ sides: 6, result: 1 });
  });

  it("dispatches remember_note and list_notes, sharing one store", () => {
    const store = createNotesStore();
    const rememberOut = runDemoTool("remember_note", { text: "hello" }, store);
    expect(JSON.parse(rememberOut)).toEqual({ remembered: "hello", total_notes: 1 });
    const listOut = runDemoTool("list_notes", {}, store);
    expect(JSON.parse(listOut)).toEqual({ notes: ["hello"] });
  });

  it("remember_note with a missing/empty text never throws — returns an error payload", () => {
    const store = createNotesStore();
    expect(JSON.parse(runDemoTool("remember_note", {}, store))).toHaveProperty("error");
    expect(JSON.parse(runDemoTool("remember_note", { text: "  " }, store))).toHaveProperty("error");
    expect(store.list()).toEqual([]);
  });

  it("an unknown tool name never throws — returns an error payload", () => {
    const store = createNotesStore();
    const out = runDemoTool("delete_everything", {}, store);
    expect(JSON.parse(out)).toHaveProperty("error");
  });

  it("a validation error from roll_dice is caught and returned as an error payload, never thrown", () => {
    const store = createNotesStore();
    const out = runDemoTool("roll_dice", { sides: 1 }, store);
    expect(JSON.parse(out)).toHaveProperty("error");
  });
});

describe("executeDemoTool — the end-to-end entry point", () => {
  it("parses arguments, runs the tool, and returns the output string", () => {
    const store = createNotesStore();
    const out = executeDemoTool("roll_dice", '{"sides": 20}', store, { randomFn: () => 0 });
    expect(JSON.parse(out)).toEqual({ sides: 20, result: 1 });
  });

  it("malformed arguments never throw — the model sees a named error string back", () => {
    const store = createNotesStore();
    expect(() => executeDemoTool("roll_dice", "{not json", store)).not.toThrow();
    const out = executeDemoTool("roll_dice", "{not json", store);
    expect(JSON.parse(out)).toHaveProperty("error");
  });

  it("a bare empty string is treated as no-argument call", () => {
    const store = createNotesStore();
    const out = executeDemoTool("list_notes", "", store);
    expect(JSON.parse(out)).toEqual({ notes: [] });
  });
});
