import { defineConfig } from "vitest/config";

// Shared by every site test suite. It exists as a single committed file — rather
// than each task creating its own — because issue #151's t11 (mic/audio island)
// and t12 (event-stream UI) are built in PARALLEL and would otherwise both add a
// config here and collide at merge.
//
// jsdom, not a real browser: the fixture tests assert encode paths and rendering
// decisions, never live audio hardware. Anything that genuinely needs a
// microphone or speaker belongs in the live acceptance run (t17), not here.
export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts", "src/**/*.test.js", "test/**/*.test.ts"],
  },
});
