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

# --- Drop explicitly-empty flags -------------------------------------------
# A compose `command:` list is static: it cannot omit an argument conditionally.
# So a templated flag whose variable is unset renders as a bare `--flag=` — one
# argv element carrying an empty value — and an arg parser that expects a real
# value there fails at startup. The live case is the `hand` lane's
# `--lora-modules=${HAND_LORA_MODULES:-}`: v1 ships the lane ARMED
# (`--enable-lora`) with an EMPTY inventory, and vLLM would try to parse "" as a
# `name=path` pair.
#
# The rule is deliberately narrow: drop an argument ONLY when it matches
# `--<flag>=` exactly — a long flag, an `=`, and nothing after it. A flag with a
# value, a bare `--flag`, a lone `--`, a short `-x`, and every non-flag argument
# (the model id, `serve`, …) all pass through untouched. Empty is not a value
# any vLLM flag wants, so this can only ever remove an argument that would have
# errored; it can never change one that would have worked.
#
# Done here rather than per-lane because it is a property of compose's static
# arg lists, not of any one model.
_args=()
for _arg in "$@"; do
    case "$_arg" in
        --*=) continue ;;  # e.g. `--lora-modules=` — an unset templated flag
        *) _args+=("$_arg") ;;
    esac
done
set -- ${_args[@]+"${_args[@]}"}

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

# --- Boot hold + staggered release (lane-safety contract, section 2) --------
# After a host reboot dockerd restarts every lane at once (restart policy ignores
# compose `depends_on`), and the simultaneous heavy loads can thermally reset the
# box. So each lane holds until host uptime reaches
#   LOBES_BOOT_DELAY_S + (position of MG_LOG_NAME in LOBES_BOOT_ORDER) * LOBES_BOOT_STAGGER_S
# (a lane not listed: len(order) * stagger, i.e. last). LOBES_BOOT_DELAY_S=0 turns
# the hold and the stagger off together. Uptime is read with builtins only; the
# host's /proc/uptime is not namespaced by docker. Any bad knob or unreadable
# uptime means no hold: this can never stop the model from serving.
_num() { case "$1" in '' | *[!0-9]*) echo "$2" ;; *) echo "$((10#$1))" ;; esac; }

# --- Stoppable waits (2026-10-17 hardware finding) ---------------------------
# The boot hold below must stop on `docker stop`. Two bash rules defeat a plain
# `sleep N`: bash defers a trap until the FOREGROUND child returns (so TERM
# waits out a 120 s hold), and bash as a container's PID 1 ignores TERM unless
# a handler is set — on Spark, Orin and Thor a held lane ran until docker's
# kill timeout. So the wait is `sleep N & wait $!` (the `wait` builtin returns
# at once on a trapped signal) under a TERM/INT handler that kills the pending
# sleep, removes the gate marker (if any) and exits 143/130. The handlers are
# cleared before the final exec, so the server gets default signal handling.
_sleep_pid=""
_marker=""
_on_signal() {
    [ -n "$_sleep_pid" ] && kill "$_sleep_pid" 2>/dev/null
    [ -n "$_marker" ] && rm -f "$_marker"
    exit "$1"
}
trap '_on_signal 143' TERM
trap '_on_signal 130' INT
_wait() {
    sleep "$1" &
    _sleep_pid=$!
    wait "$_sleep_pid" || true
    _sleep_pid=""
}
_delay="$(_num "${LOBES_BOOT_DELAY_S-}" 120)"
if [ "$_delay" -gt 0 ]; then
    _stagger="$(_num "${LOBES_BOOT_STAGGER_S-}" 20)"
    _order=()
    IFS=, read -r -a _order <<<"${LOBES_BOOT_ORDER-}"
    _pos="${#_order[@]}"
    _i=0
    for _n in ${_order[@]+"${_order[@]}"}; do
        _n="${_n// /}"
        if [ "$_n" = "$name" ]; then _pos="$_i"; break; fi
        _i=$((_i + 1))
    done
    _off=$((_pos * _stagger))
    _up=""
    read -r _up _ <"${LOBES_PROC_ROOT:-/proc}/uptime" 2>/dev/null || true
    _up="${_up%%.*}"
    case "$_up" in '' | *[!0-9]*) _up="" ;; esac
    if [ -n "$_up" ]; then
        _hold=$((_delay + _off - 10#$_up))
        if [ "$_hold" -gt 0 ]; then
            printf 'lobes: host up %ss, holding %ss (boot delay %ss + stagger %ss)\n' \
                "$((10#$_up))" "$_hold" "$_delay" "$_off"
            _wait "$_hold"
        fi
    fi
fi

# The waits are over: drop the TERM/INT handlers so the server starts with
# default signal dispositions (exec resets caught signals anyway; this keeps
# the intent explicit and covers no-wait paths too).
trap - TERM INT

# exec the real command: this DOES replace the shell, so the server (not bash) is
# the container's main process — it receives SIGTERM directly (graceful shutdown)
# and its exit code becomes the container's (so `restart:` behaves as before).
exec "$@"
