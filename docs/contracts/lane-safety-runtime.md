# Lane safety runtime — the contract

For whoever reads or writes the lane-safety state on a lobes box: the lane
wrapper (`lobes/templates/mg-logwrap.sh`), the compose templates, the lifecycle
verbs (`lobes up`, `serve`, `switch`, `stop`, `fleet`), the host watcher
(`lobes watcher`), `lobes doctor`, and the gateway's thermal pressure signal.
One place names every knob, file and format, so these parts cannot drift
apart.

**Status: draft, 2026-10-17. Not yet implemented.** Lane-safety plan
(`docs/plans/2026-10-17-lane-safety-and-quality-304-307-308-309.md`), for
issues #304, #307, #308 and #309. Thor behaviour is DECLARED/UNVALIDATED (#108): the
Thor is code-only while its experimental work runs.

## 1. Knobs

Each knob is read from the deployment's `.env`. Its off value restores the
behaviour from before this work.

| knob | default | off value | read by | meaning |
|---|---|---|---|---|
| `LOBES_BOOT_DELAY_S` | `120` | `0` | wrapper | Hold until host uptime reaches this many seconds. |
| `LOBES_BOOT_ORDER` | empty (per-card `[host_env]`) | empty | wrapper | Comma-separated `MG_LOG_NAME` values, in release order. |
| `LOBES_BOOT_STAGGER_S` | `20` | `0` | wrapper | Seconds between consecutive lanes in `LOBES_BOOT_ORDER`. |
| `LOBES_START_MAX` | `3` | `0` | wrapper | Starts allowed inside the window before the lane holds. |
| `LOBES_START_WINDOW_S` | `600` | — | wrapper | The start-counter window. |
| `LOBES_MEM_GATE` | `1` | `0` | wrapper | Wait for enough free host memory before starting. |
| `LOBES_MEM_GATE_TIMEOUT_S` | `900` | — | wrapper | How long the gate waits before giving up. |
| `LOBES_VLLM_CACHE_ROOT` | `/cache/vllm` | `/root/.cache/vllm` | compose | Container path vLLM uses for its compile cache (`VLLM_CACHE_ROOT`). The off value is the writable layer, so the mount goes unused. |
| `LOBES_COMPILE_CACHE_DIR` | `./compile-cache` | — | compose, init, doctor | Host directory holding one subdirectory per lane. |
| `LOBES_THERMAL_ZONES` | empty (per-card `[host_env]`) | empty | gateway, status | Comma-separated thermal zone `type` names to watch. Empty means no thermal signal. |
| `LOBES_THERMAL_SHED_C` | empty (per-card) | empty | gateway | Shed above this temperature. |
| `LOBES_THERMAL_RESUME_C` | empty (per-card) | empty | gateway | Stop shedding below this temperature (hysteresis). |
| `LOBES_WATCHER_MIN_INTERVAL_S` | `300` | — | watcher | Minimum seconds between two cache clears. |

Test seams, not operator knobs: `LOBES_PROC_ROOT` (default `/proc`) and
`LOBES_SYS_ROOT` (default `/sys`) let tests point the wrapper and the thermal
reader at a fake tree.

## 2. Boot hold and staggered release

```text
offset   = index of MG_LOG_NAME in LOBES_BOOT_ORDER × LOBES_BOOT_STAGGER_S
           (a lane not listed: len(LOBES_BOOT_ORDER) × LOBES_BOOT_STAGGER_S)
release  = LOBES_BOOT_DELAY_S + offset
hold     = release − host uptime, if positive
```

- Host uptime is the first field of `/proc/uptime`, which Docker does not
  namespace. Measured on the DGX Spark, 2026-10-17: container 237668.11 s, host
  237668.03 s.
- `LOBES_BOOT_DELAY_S=0` disables the hold and the stagger together.
- The hold is logged once, in the lane log, as
  `lobes: host up <u>s, holding <h>s (boot delay <d>s + stagger <o>s)`.
- After a reboot, dockerd restarts containers by restart policy and does not
  honour compose `depends_on` (assumption c32, unverified on DGX OS and JetPack:
  lapse l2). So the hold lives in each container's own entrypoint.

## 3. Crash backoff: the start counter

- **File:** `${MG_LOG_DIR}/lane-state/<MG_LOG_NAME>.starts`. On the host this is
  `<deployment>/logs/lane-state/<name>.starts`, because `MG_LOG_DIR` is the
  mount of `${MODEL_GEAR_LOG_DIR:-./logs}`.
- **Format:** one integer epoch second per line, appended on every start. The
  wrapper drops lines older than `LOBES_START_WINDOW_S` when it rewrites the file.
- **Hold:** when the file already holds `LOBES_START_MAX` or more starts inside
  the window, the wrapper logs
  `lobes: <n> starts in <w>s — holding; clear with 'lobes up <role> --apply' or delete <file>`,
  and re-checks every 30 s until the oldest start ages out or the file is gone.
  With the defaults, the 4th start inside 600 s holds. The start itself is
  recorded when the lane gets past the hold. `<role>` is the role name for the
  lane (`primary` → `cortex`, `multimodal` → `senses`, `embed` → `embedder`,
  `rerank` → `reranker`, any other lane is its own name).
- **Unwritable log dir:** the counter and the gate marker are skipped, with one
  log line. The memory gate still runs.
- **Reset:** a lifecycle verb that starts a lane on purpose deletes that lane's
  file first, so deliberate restarts never count.

## 4. Crash backoff: the memory gate

- **Need:** `ceil(util × MemTotal)`. `util` is the value of the lane's own
  `--gpu-memory-utilization` argument (either `--gpu-memory-utilization=X` or
  `--gpu-memory-utilization X`). A lane without that argument (llama.cpp,
  sidecars, the gateway) skips the gate.
- **Have:** `MemAvailable` from `${LOBES_PROC_ROOT}/meminfo`, polled every 5 s.
- **Marker:** while waiting, the wrapper keeps
  `${MG_LOG_DIR}/lane-state/<MG_LOG_NAME>.gated` holding one JSON object, and
  removes it when the gate passes or gives up:

  ```json
  {"lane": "worker", "since": 1760688000, "need_kib": 57671680, "avail_kib": 1048576}
  ```

- **Timeout:** after `LOBES_MEM_GATE_TIMEOUT_S` the wrapper logs
  `lobes: memory gate timed out (need <n> KiB, have <m> KiB) — run 'lobes watcher install' or the machine CLI's drop-caches`
  and exits `75`. Docker restarts the container, which counts as a start (§3),
  so a lane that never gets its memory ends up held.
  The timeout counts poll time (5 s per poll), not wall-clock time.
- **Skips:** a `--gpu-memory-utilization` value that is not a plain decimal, or
  an unreadable `meminfo`, skips the gate with one log line.
- **Known limit:** on Jetson, memory freed by a crash can stay out of
  `MemAvailable` until the page cache is dropped (mjolnir-agent incident Q7),
  and vLLM checks CUDA free memory rather than `MemAvailable`. The gate can
  therefore wait when vLLM would fit, or pass when vLLM still refuses. The start
  counter is the backstop (plan risk r4).

## 5. The host watcher

- Polls `<deployment>/logs/lane-state/*.gated`.
- On a marker, it runs the machine CLI's `drop-caches` (see
  `machine-cli-window.md`), at most once per `LOBES_WATCHER_MIN_INTERVAL_S`.
  With no grant it logs and does nothing else.
- It never runs a `docker` command.

## 6. Compile cache

- Each vLLM lane mounts `${LOBES_COMPILE_CACHE_DIR}/<lane>` at
  `${LOBES_VLLM_CACHE_ROOT}` and sets `VLLM_CACHE_ROOT` to it.
- `lobes init` and `lobes doctor --fix --apply` create the host directory, owned
  by the invoking user, before any container mounts it.
- Wipe a lane's cache by deleting its directory with no lane running. No sudo is
  needed.

## 7. Thermal signal

- The reader takes zones under `${LOBES_SYS_ROOT}/class/thermal/thermal_zone*`
  whose `type` is in `LOBES_THERMAL_ZONES`, and uses the hottest one. It skips
  a zone whose `temp` read fails (Orin `cv0..cv2-thermal` return ENODATA, read
  2026-10-17) or is outside −40 °C to 150 °C.
- No matching zone means `thermal_c: null`, which is never a reason to shed.
- When shedding, the gateway's response carries `shed_signal: "thermal"`.

## 8. What this does NOT do

- No lobes code pauses, `docker update`s or restarts a serving container.
- Nothing clears the page cache inside a container.
- No clock or fan change outside a benchmark window.
