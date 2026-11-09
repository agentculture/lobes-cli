# lobes local site: design reference

Confirmed design for the `lobes` local site (spec
`docs/specs/2026-11-09-gradual-fleet-rollout.md`, decision c38). These are the
design-canvas source files, kept as the reference the implementation matches.
They are not the site itself.

- `Main.dc.html`: Mesh view. Graph tab: full-mesh wheel, the gateway with
  outside access at the hub, a positronic brain per machine, split brains,
  pairs, lobes per mind, and the lobe export panel. Table tab: the same data.
- `Rollout.dc.html`: Rollout view with plan, running and halted states.
- `Residents.dc.html`: the generated residents file, needs-you fields, guests.
- `canvas.json`: the canvas layout.

The files use the canvas runtime (`support.js`, `<x-dc>`, `{{holes}}`) and load
Lato from canvas asset URLs (`/_blob/...`). The real site serves the same four
Lato weights (300, 400, 700, 900) and `lato-OFL.txt` from the package (c36).
Machine data in them is the live mesh as of 2026-11-09. The orin-nano guest and
the running and halted rollout states are examples.
