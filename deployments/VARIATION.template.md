<!-- Copy to deployments/<variation-id>/VARIATION.md and fill in.
     The contract is enforced by lobes/variation_catalog.py; see README.md. -->
# `<variation-id>` — one-line name for this deployment

## What this variation is

Which machine type or setup this is, which roles it hosts, which deployment
shape (if any) is applied, and anything an adopter needs to know before
running it — peers it expects to reach, opt-in gears it assumes, hardware it
requires.

## Measured result

Do exactly one of the following. A blank section here is a failure, not a
default.

**If an acceptance transcript exists**, cite it by path and say what it
covers — do not restate its numbers, or the two will drift:

> Measured live on `DATE`: `docs/evidence/TRANSCRIPT.txt`. Covers `WHAT WAS
> PROBED`. Not covered: `WHAT THE TRANSCRIPT DOES NOT PROVE`.

**If none exists**, delete the block above and keep exactly this line:

> No measured result.

…followed, if useful, by why: what has not been run, and what would have to
happen for a transcript to land.

## Notes

Anything else an adopter should know: known divergences, open issues, the
`MODEL_GEAR_VERSION` this was captured at and whether it is still
installable.
