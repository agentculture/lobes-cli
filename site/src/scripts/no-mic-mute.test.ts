/*
 * no-mic-mute.test.ts — the grep gate (issue #151 t11, acceptance criterion 2).
 *
 * "grep gate: no mic-mute logic anywhere in site code."
 *
 * The word "mute" appears in this repo's prose a great deal — the spec, the
 * plan, and several comments in this island all explain at length that muting
 * is the thing barge-in forecloses. A gate on the *word* would therefore fail
 * on its own documentation, and passing it would mean deleting the
 * explanation. So this gate is on the *mechanisms*: the specific code shapes
 * that silence a live microphone.
 *
 *   track.enabled = false      the standard MediaStreamTrack mute
 *   node.muted = true          the media-element mute
 *   GainNode / .gain.value = 0 a gain-shaped mute on the capture path
 *   mute()/setMuted()/…        anything that names the operation outright
 *
 * `track.stop()` is deliberately NOT on that list: releasing the device when
 * the human stops the session, or when the socket dies, is teardown. Muting is
 * silencing a track that is still live, mid-session, while the machine talks —
 * which is precisely what `scripts/realtime-voice-loop.py` does and precisely
 * what this island exists not to do.
 *
 * The equivalent by hand, from the repo root:
 *
 *   grep -rnE '\.enabled\s*=\s*(false|0)|\.muted\s*=\s*true|createGain|GainNode|\.gain\.|(set)?[Mm]ute[A-Za-z]*\s*\(' \
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

/** Code shapes that silence a live microphone. Not the word — the mechanism. */
const MUTE_MECHANISMS: Array<{ name: string; pattern: RegExp }> = [
  { name: "MediaStreamTrack.enabled = false", pattern: /\.enabled\s*=\s*(?:false|0|!)/ },
  { name: "media element .muted = true", pattern: /\.muted\s*=\s*true/ },
  { name: "GainNode construction", pattern: /createGain\s*\(|new\s+GainNode/ },
  { name: "gain manipulation", pattern: /\.gain\s*\./ },
  { name: "a named mute operation", pattern: /\b(?:set)?[Mm]ute(?:d|Mic|Input|Track)?\s*\(/ },
  { name: "getUserMedia with echoCancellation disabled", pattern: /echoCancellation\s*:\s*false/ },
];

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

  for (const mechanism of MUTE_MECHANISMS) {
    it(`finds no ${mechanism.name} anywhere in site code`, () => {
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

  it("keeps echo cancellation on, which is what an always-open mic depends on", () => {
    const capture = readFileSync(join(SITE_ROOT, "src/scripts/mic-capture.ts"), "utf8");
    expect(capture).toContain("echoCancellation: true");
  });

  it("still releases the device on teardown — stopping is not muting", () => {
    const capture = readFileSync(join(SITE_ROOT, "src/scripts/mic-capture.ts"), "utf8");
    expect(capture).toContain("track.stop()");
  });
});
