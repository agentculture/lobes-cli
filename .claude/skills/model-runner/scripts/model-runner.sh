#!/usr/bin/env bash
# model-runner — thin shim over the `lobes` CLI (lobes-cli).
#
# The model-ops logic lives in the lobes package (first-class verbs:
# serve / stop / switch / status / assess / benchmark / init). This shim
# forwards to `lobes` (or the deprecated `model` alias if `lobes` is not yet
# installed) so a maintainer with the repo checked out has a working entry
# point whether or not lobes-cli is pip-installed.
#
# Examples:
#   scripts/model-runner.sh switch nvidia/Qwen3.6-27B-NVFP4 --apply
#   scripts/model-runner.sh assess
#   scripts/model-runner.sh status
#   scripts/model-runner.sh stop --apply
set -euo pipefail

if command -v lobes >/dev/null 2>&1; then
  exec lobes "$@"
fi

if command -v model >/dev/null 2>&1; then
  exec model "$@"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || (cd "$HERE/../../../.." && pwd))"
exec uv run --project "$ROOT" lobes "$@"
