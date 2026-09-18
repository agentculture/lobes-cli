#!/usr/bin/env bash
# lobes-compose — run `docker compose` against the DEPLOYMENT, never the repo.
#
# The compose files in this repo (lobes/templates/**) are PACKAGED TEMPLATES.
# They are not a deployment. Running `docker compose` inside them starts a
# second compose project (named after the folder, e.g. `fleet`) that fights the
# real one for container names, lands on a different network the gateway
# cannot see, and mounts files that only exist in a scaffolded deployment dir.
# This is exactly what happened on the Spark on 2026-09-18 (see
# docs/operating-a-deployment.md).
#
# This wrapper:
#   1. resolves the deployment dir: --compose-dir DIR, else $LOBES_DIR, else ~/.lobes;
#   2. refuses a dir that is inside a lobes-cli checkout's lobes/templates tree;
#   3. asks `lobes fleet files` for the deployment's own -f chain (the CLI's single
#      composition authority, so the shape and override files are never dropped);
#   4. runs from that dir, so the compose project name is the deployment's own.
#
# Read-only subcommands run straight away. Anything else is DRY-RUN: it prints
# the exact command and exits 0 until you add --apply (the repo's
# mutation-safety rule).
#
# Examples:
#   lobes-compose.sh ps
#   lobes-compose.sh logs --tail 50 comfyui
#   lobes-compose.sh build comfyui                           # prints the plan
#   lobes-compose.sh --apply build comfyui
#   lobes-compose.sh --apply up -d --no-deps comfyui         # ALWAYS --no-deps for one service
#   lobes-compose.sh --apply --profile innereye stop comfyui # profile-gated services need --profile
set -euo pipefail

apply=0
dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) apply=1; shift ;;
    --compose-dir) dir="${2:?--compose-dir needs a path}"; shift 2 ;;
    --compose-dir=*) dir="${1#*=}"; shift ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: lobes-compose.sh [--apply] [--compose-dir DIR] <compose args...>" >&2; exit 2; }

dir="${dir:-${LOBES_DIR:-$HOME/.lobes}}"
dir="$(cd "$dir" 2>/dev/null && pwd -P)" || { echo "lobes-compose: no such deployment dir: ${dir}" >&2; exit 2; }

case "$dir" in
  */lobes/templates|*/lobes/templates/*)
    echo "lobes-compose: REFUSED — $dir is a packaged template folder, not a deployment." >&2
    echo "  Edit the template in the repo (and open a PR), then copy the change into the" >&2
    echo "  deployment dir (~/.lobes by default) and run compose THERE." >&2
    exit 2 ;;
esac
[ -f "$dir/docker-compose.yml" ] || { echo "lobes-compose: $dir has no docker-compose.yml (not scaffolded? see 'lobes init')" >&2; exit 2; }

# The deployment's own -f chain, from the CLI (one line per argv element).
mapfile -t files < <(lobes fleet files --compose-dir "$dir")

# First non-option word decides read-only vs mutating. Compose's global options
# that take a value (`--profile innereye`) must not be mistaken for it.
sub=""
skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  case "$a" in
    --profile|-p|--project-name|-f|--file|--env-file|--project-directory|--ansi|--progress|--parallel) skip=1 ;;
    -*) ;;
    *) sub="$a"; break ;;
  esac
done
case "$sub" in
  ps|config|logs|images|ls|top|version|port|events) readonly=1 ;;
  *) readonly=0 ;;
esac

cmd=(docker compose "${files[@]}" "$@")
if [ "$readonly" = 1 ] || [ "$apply" = 1 ]; then
  cd "$dir"
  exec "${cmd[@]}"
fi
printf 'DRY-RUN (add --apply to run) in %s:\n  ' "$dir"
printf '%q ' "${cmd[@]}"
printf '\n'
