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
# Every hold below (boot hold, start counter, memory gate) must stop on `docker
# stop`. Two bash rules defeat a plain `sleep N`: bash defers a trap until the
# FOREGROUND child returns (so TERM waits out a 30 s poll, or a 120 s hold),
# and bash as a container's PID 1 ignores TERM unless a handler is set — on
# Spark, Orin and Thor a held lane ran until docker's kill timeout. So each wait
# is `sleep N & wait $!` (the `wait` builtin returns at once on a trapped
# signal) under a TERM/INT handler that kills the pending sleep, removes the
# gate and hold markers (if any) and exits 143/130; an EXIT trap removes them
# on any other exit. The handlers are cleared before the final exec, so the
# server gets default signal handling.
_sleep_pid=""
_marker="" # lane-state/<name>.gated while the memory gate waits
_held=""   # lane-state/<name>.held while a boot or start-counter hold waits
_cleanup() {
    [ -n "$_marker" ] && rm -f "$_marker"
    [ -n "$_held" ] && rm -f "$_held"
    return 0
}
_on_signal() {
    [ -n "$_sleep_pid" ] && kill "$_sleep_pid" 2>/dev/null
    _cleanup
    exit "$1"
}
trap '_on_signal 143' TERM
trap '_on_signal 130' INT
trap _cleanup EXIT
_wait() {
    sleep "$1" &
    _sleep_pid=$!
    wait "$_sleep_pid" || true
    _sleep_pid=""
}

# --- Lane state dir (lane-safety contract, sections 2 to 4) -----------------
# Holds the start counter, the memory-gate marker and the hold marker. If it
# cannot be created they are all skipped (one log line) — the holds and the
# gate themselves only need /proc and still run.
#
# Ownership self-heal (review2-b #3, finding 9): this wrapper runs as root in the
# container, so a lane-state/ it (or an older wrapper) created is root-owned
# and the operator cannot delete a counter. So every start aligns lane-state/
# and each file the wrapper writes into it to the parent log dir's owner
# (`chown --reference`, else the owner `stat` reports). Best-effort and silent:
# a failed chown never blocks the lane.
_chown_like_dir() {
    chown --reference="$dir" "$@" 2>/dev/null && return 0
    local _o
    _o="$(stat -c %u:%g "$dir" 2>/dev/null)" || return 0
    [ -n "$_o" ] && chown "$_o" "$@" 2>/dev/null
    return 0
}
_state="${dir}/lane-state"
if mkdir -p "$_state" 2>/dev/null && [ -w "$_state" ]; then
    _chown_like_dir "$_state"
    # A .held left by an earlier run of THIS lane that was SIGKILLed (no trap
    # runs) would hide this boot from the clear guard: drop it now.
    rm -f "${_state}/${name}.held" 2>/dev/null
else
    printf 'lobes: %s not writable; start counter and gate marker skipped\n' "$_state"
    _state=""
fi
_now() { printf '%(%s)T' -1; }

# A lane in the boot hold or a start-counter hold is held BEFORE exec (no GPU)
# but docker reports it `health: starting`; lane-state/<name>.held (one JSON
# object) tells the lifecycle clear and the watcher not to count it as booting
# (contract section 5). Present for exactly the hold; removed on release and on
# TERM/INT/EXIT.
_hold_begin() { # $1: boot_hold | start_counter
    [ -n "$_state" ] || return 0
    _held="${_state}/${name}.held"
    if printf '{"lane": "%s", "since": %s, "reason": "%s"}\n' \
        "$name" "$(_now)" "$1" >"$_held" 2>/dev/null; then
        _chown_like_dir "$_held"
    else
        _held=""
    fi
    return 0
}
_hold_end() {
    [ -n "$_held" ] && rm -f "$_held"
    _held=""
    return 0
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
            _hold_begin boot_hold
            _wait "$_hold"
            _hold_end
        fi
    fi
fi

# --- Crash backoff: the start counter (contract section 3) ------------------
# File: lane-state/<name>.starts, one epoch second per line. A start HOLDS when
# the file already holds LOBES_START_MAX or more starts inside
# LOBES_START_WINDOW_S — with the defaults (3 in 600 s) the 4th start inside
# 600 s holds. It re-checks every 30 s until the oldest start ages out or the
# file is deleted (`lobes up <role> --apply` deletes it). The start is recorded
# when the lane proceeds past the hold, and lines outside the window (or not
# integers) are dropped on that rewrite. LOBES_START_MAX=0 turns it off.
_count_starts() {
    # sets _kept (in-window epochs) from the counter file
    _kept=()
    local _e _cut=$(($(_now) - _window))
    [ -f "$_starts" ] || return 0
    while read -r _e _; do
        case "$_e" in '' | *[!0-9]*) continue ;; esac
        [ "$((10#$_e))" -gt "$_cut" ] && _kept+=("$((10#$_e))")
    done 2>/dev/null <"$_starts" || true
}
_max="$(_num "${LOBES_START_MAX-}" 3)"
if [ "$_max" -gt 0 ] && [ -n "$_state" ]; then
    _window="$(_num "${LOBES_START_WINDOW_S-}" 600)"
    [ "$_window" -gt 0 ] || _window=600
    _starts="${_state}/${name}.starts"
    case "$name" in # lane name -> role name, for the clear instruction
        primary) _role=cortex ;;
        multimodal) _role=senses ;;
        embed) _role=embedder ;;
        rerank) _role=reranker ;;
        *) _role="$name" ;;
    esac
    _count_starts
    if [ "${#_kept[@]}" -ge "$_max" ]; then
        printf "lobes: %s starts in %ss — holding; clear with 'lobes up %s --apply' or delete %s\n" \
            "${#_kept[@]}" "$_window" "$_role" "$_starts"
        _hold_begin start_counter
        while :; do
            _wait 30
            [ -f "$_starts" ] || break
            _count_starts
            [ "${#_kept[@]}" -lt "$_max" ] && break
        done
        _hold_end
        _count_starts
    fi
    _kept+=("$(_now)")
    printf '%s\n' "${_kept[@]}" >"$_starts" 2>/dev/null && _chown_like_dir "$_starts"
fi

# --- Crash backoff: the memory gate (contract section 4) --------------------
# Runs after the boot hold and stagger (c42). need = ceil(util x MemTotal),
# util being this lane's own --gpu-memory-utilization (`=X` or `X` form); have
# = MemAvailable. Polled every 5 s with lane-state/<name>.gated (JSON) present
# while waiting; removed on pass or give-up. After LOBES_MEM_GATE_TIMEOUT_S of
# polling (counted in poll sleeps, not wall clock) it exits 75 before exec, and
# Docker's restart counts as a start (section 3). No util flag (llama.cpp,
# sidecars) skips it; an unparseable util or unreadable meminfo skips it with a
# log line. LOBES_MEM_GATE=0 turns it off. Integer math only (util scaled to 9
# decimal places), no awk/bc.
_meminfo() {
    _total=""
    _avail=""
    local _k _v
    while read -r _k _v _; do
        case "$_k" in
            MemTotal:) _total="$_v" ;;
            MemAvailable:) _avail="$_v" ;;
        esac
    done 2>/dev/null <"${LOBES_PROC_ROOT:-/proc}/meminfo" || true
    case "$_total" in '' | *[!0-9]*) return 1 ;; esac
    case "$_avail" in '' | *[!0-9]*) return 1 ;; esac
}
_util=""
_has_util=0
_prev=""
for _arg in "$@"; do
    if [ "$_prev" = "--gpu-memory-utilization" ]; then
        _util="$_arg"
        _has_util=1
    fi
    case "$_arg" in
        --gpu-memory-utilization=*)
            _util="${_arg#*=}"
            _has_util=1
            ;;
    esac
    _prev="$_arg"
done
if [ "$(_num "${LOBES_MEM_GATE-}" 1)" != 0 ] && [ "$_has_util" = 1 ]; then
    _scaled=""
    case "$_util" in
        *[!0-9.]* | *.*.* | . | '') ;;
        *)
            _ip="${_util%%.*}"
            _fp=""
            case "$_util" in *.*) _fp="${_util#*.}" ;; esac
            _fp="${_fp}000000000"
            _fp="${_fp:0:9}"
            _scaled=$((10#${_ip:-0} * 1000000000 + 10#$_fp))
            ;;
    esac
    if [ -z "$_scaled" ]; then
        printf "lobes: memory gate skipped (unparseable --gpu-memory-utilization '%s')\n" "$_util"
    elif ! _meminfo; then
        printf 'lobes: memory gate skipped (cannot read %s/meminfo)\n' "${LOBES_PROC_ROOT:-/proc}"
    else
        _need=$(((_total * _scaled + 999999999) / 1000000000))
        _gtimeout="$(_num "${LOBES_MEM_GATE_TIMEOUT_S-}" 900)"
        _marker=""
        [ -n "$_state" ] && _marker="${_state}/${name}.gated"
        _since="$(_now)"
        _waited=0
        while [ "$_avail" -lt "$_need" ]; do
            if [ -n "$_marker" ]; then
                printf '{"lane": "%s", "since": %s, "need_kib": %s, "avail_kib": %s}\n' \
                    "$name" "$_since" "$_need" "$_avail" >"$_marker" 2>/dev/null &&
                    [ "$_waited" = 0 ] && _chown_like_dir "$_marker"
            fi
            if [ "$_waited" -ge "$_gtimeout" ]; then
                printf "lobes: memory gate timed out (need %s KiB, have %s KiB) — run 'lobes watcher install' or the machine CLI's drop-caches\n" \
                    "$_need" "$_avail"
                [ -n "$_marker" ] && rm -f "$_marker"
                exit 75
            fi
            _wait 5
            _waited=$((_waited + 5))
            _meminfo || break # meminfo vanished: never block serving
        done
        [ -n "$_marker" ] && rm -f "$_marker"
        _marker=""
    fi
fi

# The waits are over: drop the TERM/INT handlers so the server starts with
# default signal dispositions (exec resets caught signals anyway; this keeps
# the intent explicit and covers no-wait paths too).
trap - TERM INT EXIT

# exec the real command: this DOES replace the shell, so the server (not bash) is
# the container's main process — it receives SIGTERM directly (graceful shutdown)
# and its exit code becomes the container's (so `restart:` behaves as before).
exec "$@"
