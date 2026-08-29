# fixture-card — a FIXTURE variation (not a real box)

## What this variation is

A **fixture**, not a deployment. It exists so
`tests/test_variation_catalog.py` can exercise the variation-catalog contract
— directory layout, lock completeness, and the info-file rules — without a
real machine. Its `deployment.lock.toml` describes an invented card
(`fixture-card`) serving invented models, and its compose files and
`Dockerfile.gateway` are trimmed stand-ins for the real scaffold.

No hardware ran this. Nothing here may be cited as a measurement of anything.

## Measured result

This fixture models the CITING half of the contract. It cites an existing
transcript — `docs/evidence/2026-08-26-accept-orin-associate.txt` — purely so
a test can assert that a cited path resolves against the repo root.

**The transcript is real; the citation is not a claim about this fixture.**
That transcript measured the `orin-associate` shape on a physical Jetson AGX
Orin, which has nothing to do with `fixture-card`. A published variation cites
the transcript that measured *it*, and cites rather than restates the numbers,
so the two cannot drift.

## Notes

The sibling fixture, `fixture-card__fixture-shape`, models the other half:
a variation with no acceptance transcript, which must say so in the exact
words the contract fixes rather than leave a blank a reader could mistake for
a measurement.
