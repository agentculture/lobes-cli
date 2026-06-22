#!/usr/bin/env bash
# lobes log wrapper — make the server's output (and its crash trace) durable.
#
# `lobes init` materialises this next to docker-compose.yml; each vLLM service
# bind-mounts it at /usr/local/bin/mg-logwrap and runs it as the entrypoint, so
# the real `command:` (the vllm arg list) arrives here verbatim as "$@".
#
# Why this exists: when a vLLM container is restarted or recreated (docker
# restart, `docker compose up` after `lobes switch`, a `compose down/up`, …) its
# `docker logs` are gone — which is exactly how the EngineCore crash trace in
# issue #50 vanished before anyone could read it. This wrapper tees stdout+stderr
# to a per-boot file under a host-mounted log dir, so the trace survives any
# restart or recreate.
#
# It tees at the process-I/O level (not through Python logging), so it captures
# BOTH Python tracebacks AND native stderr aborts (CUDA / C++ / OOM) — the latter
# bypass Python logging entirely and are the crashes that most need investigating.
# The final `exec` makes the real server the signal target, so `docker stop` still
# drains it cleanly (graceful SIGTERM). If anything about logging fails (no log
# dir, read-only mount, no bash process substitution, …) it falls back to a plain
# exec — logging can never stop the model from serving.
set -u

name="${MG_LOG_NAME:-server}"
dir="${MG_LOG_DIR:-/logs/model-gear}"
ts="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || echo boot)"
log="${dir}/${name}-${ts}.log"

# Per-boot file (the crash boot is preserved as its own file, never overwritten by
# the restart) plus a stable <service>-latest.log pointer for quick tailing and
# `lobes logs`. The `:` no-op probes that the file is actually writable before we
# commit to teeing into it.
if mkdir -p "$dir" 2>/dev/null && : 2>/dev/null >>"$log"; then
    ln -sf "${name}-${ts}.log" "${dir}/${name}-latest.log" 2>/dev/null || true
    printf '=== lobes %s :: boot %s :: %s ===\n' "$name" "$ts" "$*" >>"$log"
    # exec with redirections only (no command): this does NOT replace the shell — it
    # rewires THIS shell's fd1+fd2 to a tee that writes the durable file *and* passes
    # output through to the original stdout, so `docker logs` keeps working too. The
    # tee child gets EOF when the server exits and flushes its buffer.
    exec > >(tee -a "$log") 2>&1
fi

# exec the real command: this DOES replace the shell, so the server (not bash) is
# the container's main process — it receives SIGTERM directly (graceful shutdown)
# and its exit code becomes the container's (so `restart:` behaves as before).
exec "$@"
