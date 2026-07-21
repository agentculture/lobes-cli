/*
 * no-mic-mute.test.ts — the grep gate, NARROWED per deviation d1 (issue #151 t18).
 *
 * ── Why this file changed shape ─────────────────────────────────────────
 *
 * t11 (issue #151, honesty condition h6) banned every mute mechanism
 * anywhere in shipped site code, full stop: "grep gate: no mic-mute logic
 * anywhere in site code." That was the right rule for a site with no way to
 * own echo cancellation except by muting — `scripts/realtime-voice-loop.py`
 * mutes the mic for its whole synthesize-and-play window for exactly that
 * reason, and that is precisely what barge-in cannot coexist with (see
 * `mic-capture.ts`'s own module doc).
 *
 * Deviation d1 (operator-approved 2026-07-21, recorded against issue #151
 * t18) changed the premise the ban rested on: real hardware now owns AEC at
 * the client edge (Reachy's firmware, this browser's `echoCancellation`
 * constraint), so a mic that stops sending is no longer automatically the
 * AEC-substitute hack — PROVIDED a human put it there, not a playback or
 * response event. The rule therefore NARROWS, it does not disappear:
 *
 *   ✅ NOW ALLOWED   a user muting the mic, or releasing the device, from an
 *                    explicit control (`mic-island.ts`'s "Mute mic" button
 *                    and its existing "Stop mic & playback" control).
 *   ❌ STILL FORBIDDEN  any code path that mutes AUTOMATICALLY, in reaction
 *                    to a playback, response, or connection event — the
 *                    AEC-substitute this file has always existed to catch.
 *
 * ── How the narrowing is enforced ───────────────────────────────────────
 *
 * Still a MECHANISM-level gate, not a word-level one — t11's own rationale
 * for that holds unchanged: the spec, the plan, and this file's own prose
 * say "mute" a great deal while explaining why (most of) it is forbidden, so
 * a word-level gate would fail on its own documentation.
 *
 * Four of the five original mechanisms stay BLANKET-forbidden, everywhere,
 * with no exception — this island's mute never needs any of them (see
 * `mic-island.ts`'s module doc: it works by withholding an already-captured
 * frame from the outbound relay, one layer above the device), so narrowing
 * them would loosen a protection nothing here exercises:
 *
 *   track.enabled = false      the standard MediaStreamTrack mute
 *   node.muted = true          the media-element mute
 *   GainNode / .gain.value = 0 a gain-shaped mute on the capture path
 *   echoCancellation: false    disabling the ONE thing that lets an
 *                              always-open mic coexist with playback at all
 *
 * The fifth — a NAMED mute operation (`mute()`, `setMuted()`, `muteMic()`,
 * …) — is exactly the shape `mic-island.ts`'s legitimate, user-triggered
 * `setMuted` uses, so a blanket ban on it would fail the very feature d1
 * exists to allow. It is narrowed to a ZONE-scoped ban instead, enforced two
 * ways:
 *
 *   1. In-file markers. `mic-island.ts` (a file t18 owns) wraps exactly the
 *      functions that run because the SERVER or the CONNECTION did
 *      something — `handleServerEvent` (every response/session event),
 *      `handlePlaybackStop` (the player finishing, being interrupted, or
 *      torn down), and `notifyDisconnected` (the connection going away) — in
 *      `AUTOMATIC-MUTE-FORBIDDEN-ZONE-START` / `-END` markers. `setMuted`'s
 *      own definition and the click handler that calls it are deliberately
 *      left OUTSIDE any zone, because that is the one call site d1 allows.
 *   2. Whole-file coverage, marker-free. `audio-playback.ts` IS the
 *      playback engine — anything in it is playback-reactive by
 *      definition — but it is a t11 file outside this task's edit scope
 *      (see the task brief's FILE SCOPE), so instead of adding markers to a
 *      file this task does not own, `WHOLE_FILE_FORBIDDEN` below declares
 *      its ENTIRE content off-limits for the zone-scoped mechanism, from
 *      this test file alone.
 *
 * The zone-integrity tests below exist so this narrowing cannot be
 * defeated by quietly deleting a marker or shrinking the whole-file list:
 * they assert the markers are present, balanced, and wrap the specific
 * functions named above, and that the whole-file list still names
 * `audio-playback.ts`.
 *
 * The equivalent by hand, from the repo root (mechanism scan; the
 * zone-scoping below is not expressible as a single grep):
 *
 *   grep -rnE '\.enabled\s*=\s*(false|0|!)|\.muted\s*=\s*true|createGain|GainNode|\.gain\.|echoCancellation\s*:\s*false' \
 *        site/src site/public --include='*.ts' --include='*.js' --include='*.astro' --include='*.css'
 *
 * Scope note: `*.test.ts` files are excluded, because this file necessarily
 * contains every pattern it is looking for. The gate covers everything that
 * ships to a browser plus every non-test module in `src/`.
 */

import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * Walk up from the test runner's cwd to the site root.
 *
 * Not `import.meta.url`: under the jsdom environment that is an http:// URL,
 * not a file:// one, and `fileURLToPath` refuses it.
 */
function findSiteRoot(): string {
  let dir = process.cwd();
  for (let depth = 0; depth < 6; depth += 1) {
    if (existsSync(join(dir, "src", "scripts")) && existsSync(join(dir, "public"))) return dir;
    dir = dirname(dir);
  }
  throw new Error(`could not locate the site root from ${process.cwd()}`);
}

const SITE_ROOT = `${findSiteRoot()}/`;
const SCANNED_ROOTS = ["src", "public"];
const SCANNED_EXTENSIONS = [".ts", ".js", ".mjs", ".astro", ".css"];

/** Mechanisms this island's mute never needs — banned everywhere, no exception. */
const BLANKET_MUTE_MECHANISMS: Array<{ name: string; pattern: RegExp }> = [
  { name: "MediaStreamTrack.enabled = false", pattern: /\.enabled\s*=\s*(?:false|0|!)/ },
  { name: "media element .muted = true", pattern: /\.muted\s*=\s*true/ },
  { name: "GainNode construction", pattern: /createGain\s*\(|new\s+GainNode/ },
  { name: "gain manipulation", pattern: /\.gain\s*\./ },
  { name: "getUserMedia with echoCancellation disabled", pattern: /echoCancellation\s*:\s*false/ },
];

/**
 * The one mechanism a user-triggered mute is now allowed to use — but ONLY
 * outside an AUTOMATIC-MUTE-FORBIDDEN-ZONE. See the module doc above.
 */
const ZONE_SCOPED_MECHANISM = {
  name: "a named mute operation",
  pattern: /\b(?:set)?[Mm]ute(?:d|Mic|Input|Track)?\s*\(/,
};

const ZONE_START = "AUTOMATIC-MUTE-FORBIDDEN-ZONE-START";
const ZONE_END = "AUTOMATIC-MUTE-FORBIDDEN-ZONE-END";

/** Files a forbidden zone MUST exist in, and the functions it must wrap. */
const REQUIRED_ZONE_COVERAGE: Record<string, string[]> = {
  "src/scripts/mic-island.ts": [
    "function handleServerEvent",
    "function handlePlaybackStop",
    "function notifyDisconnected",
  ],
};

/**
 * Files whose ENTIRE content is a forbidden zone for the zone-scoped
 * mechanism, with no in-file markers — see the module doc's "whole-file
 * coverage, marker-free" note. `audio-playback.ts` IS the playback
 * engine (see its own module doc: "There is no gain node anywhere in this
 * island"), so every line in it is playback-reactive by construction; this
 * task does not own that file, so the rule lives here instead of a marker
 * pair inside it.
 */
const WHOLE_FILE_FORBIDDEN = ["src/scripts/audio-playback.ts"];

function walk(dir: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      found.push(...walk(full));
      continue;
    }
    if (!SCANNED_EXTENSIONS.some((ext) => entry.endsWith(ext))) continue;
    if (entry.endsWith(".test.ts") || entry.endsWith(".test.js")) continue;
    found.push(full);
  }
  return found;
}

const FILES = SCANNED_ROOTS.flatMap((root) => walk(join(SITE_ROOT, root)));

/**
 * Every `START..END` span in *source*, START and END markers included (they
 * do not themselves match either mechanism list — see the module doc's
 * marker-text note — so including them changes nothing about what a scan
 * inside the span can find).
 */
function extractForbiddenZones(source: string): string[] {
  const zones: string[] = [];
  let searchFrom = 0;
  for (;;) {
    const startIdx = source.indexOf(ZONE_START, searchFrom);
    if (startIdx === -1) break;
    const endIdx = source.indexOf(ZONE_END, startIdx);
    if (endIdx === -1) {
      throw new Error(
        `unterminated ${ZONE_START} with no matching ${ZONE_END} (searching from index ${startIdx})`,
      );
    }
    zones.push(source.slice(startIdx, endIdx + ZONE_END.length));
    searchFrom = endIdx + ZONE_END.length;
  }
  return zones;
}

describe("the grep gate", () => {
  it("scans the files that actually ship", () => {
    const names = FILES.map((file) => file.slice(SITE_ROOT.length));
    expect(names).toContain("src/scripts/mic-capture.ts");
    expect(names).toContain("src/scripts/mic-island.ts");
    expect(names).toContain("src/scripts/audio-playback.ts");
    expect(names).toContain("public/worklets/pcm-capture-processor.js");
    expect(names).toContain("src/components/MicIsland.astro");
    expect(names.length).toBeGreaterThan(8);
  });

  for (const mechanism of BLANKET_MUTE_MECHANISMS) {
    it(`finds no ${mechanism.name} anywhere in site code (blanket-banned, unchanged by d1)`, () => {
      const hits: string[] = [];
      for (const file of FILES) {
        const lines = readFileSync(file, "utf8").split("\n");
        lines.forEach((line, index) => {
          if (mechanism.pattern.test(line)) {
            hits.push(`${file.slice(SITE_ROOT.length)}:${index + 1}: ${line.trim()}`);
          }
        });
      }
      expect(hits).toEqual([]);
    });
  }

  it(`finds no ${ZONE_SCOPED_MECHANISM.name} inside a forbidden zone — marker-delimited or whole-file (d1)`, () => {
    const hits: string[] = [];
    for (const file of FILES) {
      const relPath = file.slice(SITE_ROOT.length);
      const source = readFileSync(file, "utf8");
      const zones = WHOLE_FILE_FORBIDDEN.includes(relPath)
        ? [source]
        : extractForbiddenZones(source);
      zones.forEach((zone, zoneIndex) => {
        zone.split("\n").forEach((line, lineIndex) => {
          if (ZONE_SCOPED_MECHANISM.pattern.test(line)) {
            hits.push(
              `${relPath} [forbidden zone ${zoneIndex + 1}] line ${lineIndex + 1}: ${line.trim()}`,
            );
          }
        });
      });
    }
    expect(hits).toEqual([]);
  });

  it("keeps echo cancellation on, which is what an always-open mic depends on", () => {
    const capture = readFileSync(join(SITE_ROOT, "src/scripts/mic-capture.ts"), "utf8");
    expect(capture).toContain("echoCancellation: true");
  });

  it("still releases the device on teardown — stopping is not muting", () => {
    const capture = readFileSync(join(SITE_ROOT, "src/scripts/mic-capture.ts"), "utf8");
    expect(capture).toContain("track.stop()");
  });
});

describe("the forbidden-zone markers themselves (gate integrity)", () => {
  it("is balanced and non-empty in every file that declares a zone", () => {
    for (const file of FILES) {
      const source = readFileSync(file, "utf8");
      const starts = (source.match(new RegExp(ZONE_START, "g")) ?? []).length;
      const ends = (source.match(new RegExp(ZONE_END, "g")) ?? []).length;
      const label = file.slice(SITE_ROOT.length);
      expect(starts, `${label}: unbalanced zone markers`).toBe(ends);
      if (starts > 0) {
        // extractForbiddenZones throws on an unterminated pair and every
        // zone it returns spans at least the two marker lines themselves.
        const zones = extractForbiddenZones(source);
        expect(zones.length, `${label}`).toBe(starts);
        for (const zone of zones) {
          expect(zone.length, `${label}: an empty forbidden zone`).toBeGreaterThan(
            ZONE_START.length + ZONE_END.length,
          );
        }
      }
    }
  });

  it("still names every file it scans, so WHOLE_FILE_FORBIDDEN cannot be quietly emptied", () => {
    expect(WHOLE_FILE_FORBIDDEN.length).toBeGreaterThan(0);
    for (const relPath of WHOLE_FILE_FORBIDDEN) {
      const names = FILES.map((file) => file.slice(SITE_ROOT.length));
      expect(names, `${relPath} is not among the files the gate scans`).toContain(relPath);
    }
    expect(WHOLE_FILE_FORBIDDEN).toContain("src/scripts/audio-playback.ts");
  });

  it("wraps every function a playback/response/connection reaction could reach", () => {
    for (const [relPath, functions] of Object.entries(REQUIRED_ZONE_COVERAGE)) {
      const source = readFileSync(join(SITE_ROOT, relPath), "utf8");
      for (const fn of functions) {
        const fnIdx = source.indexOf(fn);
        expect(fnIdx, `${relPath}: could not find "${fn}"`).toBeGreaterThan(-1);

        const precedingStart = source.lastIndexOf(ZONE_START, fnIdx);
        const precedingEnd = source.lastIndexOf(ZONE_END, fnIdx);
        expect(
          precedingStart,
          `${relPath}: "${fn}" is not preceded by ${ZONE_START}`,
        ).toBeGreaterThan(-1);
        // The nearest END before this point (if any) must be older than the
        // nearest START — i.e. we are currently INSIDE a zone, not just
        // somewhere after one that already closed.
        expect(
          precedingStart,
          `${relPath}: "${fn}" sits after a zone closed, not inside one`,
        ).toBeGreaterThan(precedingEnd);

        const followingEnd = source.indexOf(ZONE_END, fnIdx);
        expect(followingEnd, `${relPath}: "${fn}"'s zone never closes`).toBeGreaterThan(-1);
      }
    }
  });

  it("keeps the mute control itself OUTSIDE any forbidden zone", () => {
    const source = readFileSync(join(SITE_ROOT, "src/scripts/mic-island.ts"), "utf8");
    for (const anchor of ["function setMuted", "const onMuteClick"]) {
      const idx = source.indexOf(anchor);
      expect(idx, `could not find "${anchor}"`).toBeGreaterThan(-1);
      const precedingStart = source.lastIndexOf(ZONE_START, idx);
      const precedingEnd = source.lastIndexOf(ZONE_END, idx);
      // Either no zone opened yet before this point, or the nearest one
      // already closed — never "currently inside one".
      expect(
        precedingEnd,
        `"${anchor}" must not be inside an AUTOMATIC-MUTE-FORBIDDEN-ZONE — it is the user-triggered call site d1 allows`,
      ).toBeGreaterThan(precedingStart);
    }
  });
});
