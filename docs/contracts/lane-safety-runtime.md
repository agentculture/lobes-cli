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
| `LOBES_BOOT_ORDER` | empty (per-card `[host_env]`; a shape's `[host_env]` wins) | empty | wrapper | Comma-separated `MG_LOG_NAME` values, in release order. |
| `LOBES_BOOT_STAGGER_S` | `20` | `0` | wrapper | Seconds between consecutive lanes in `LOBES_BOOT_ORDER`. |
| `LOBES_START_MAX` | `3` | `0` | wrapper | Starts allowed inside the window before the lane holds. |
| `LOBES_START_WINDOW_S` | `600` | — | wrapper | The start-counter window. |
| `LOBES_MEM_GATE` | `1` | `0` | wrapper | Wait for enough free host memory before starting. |
| `LOBES_MEM_GATE_TIMEOUT_S` | `900` | — | wrapper | How long the gate waits before giving up. |
| `LOBES_VLLM_CACHE_ROOT` | `/cache/vllm` | `/root/.cache/vllm` | compose | Container path vLLM uses for its compile cache (`VLLM_CACHE_ROOT`). The off value points vLLM at the writable layer; the host bind (fixed target `/cache/vllm`) goes unused. |
| `LOBES_COMPILE_CACHE_DIR` | `./compile-cache` | — | compose, init, doctor | Host directory holding one subdirectory per lane. |
| `LOBES_THERMAL_ZONES` | empty (per-card `[host_env]`) | empty | gateway, status | Comma-separated thermal zone `type` names to watch. Empty means no thermal signal. |
| `LOBES_THERMAL_SHED_C` | empty (per-card) | empty | gateway | Shed above this temperature. |
| `LOBES_THERMAL_RESUME_C` | empty (per-card) | empty | gateway | Stop shedding below this temperature (hysteresis). |
| `LOBES_WATCHER_MIN_INTERVAL_S` | `300` | — | watcher | Minimum seconds between two cache clears. |

A key that any built-in shape declares in its `[host_env]` (today
`LOBES_BOOT_ORDER` and `LOBES_BOOT_STAGGER_S`, from `orin-associate`) is
written to `.env` by every `lobes init --shape` render: the shape's value,
else the card's, else the default above. Switching shapes therefore
overwrites it, and a value typed by hand is replaced on the next render.

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

- Each vLLM lane mounts `${LOBES_COMPILE_CACHE_DIR}/<lane>` at the fixed path
  `/cache/vllm` and sets `VLLM_CACHE_ROOT=${LOBES_VLLM_CACHE_ROOT:-/cache/vllm}`.
  The mount target is deliberately fixed: if it followed the knob, the off value
  `/root/.cache/vllm` would put the bind mount on the default path and the cache
  would still persist on the host. With the target fixed, the off value points
  vLLM at the writable layer and the bind mount goes unused.
- Lanes are the services whose command is `vllm serve`, plus clef-vllm decider
  sidecars (vLLM runs in-process there). `<lane>` is the lane's `MG_LOG_NAME`
  (the decider service name for deciders). `llamacpp-primary`, `comfyui`, the
  gateway, the audio services and `embed-gemma2-embed` get no mount.
- `lobes init` and `lobes doctor --fix --apply` create the host directory, owned
  by the invoking user, before any container mounts it.
- Wipe a lane's cache by deleting its directory with no lane running. No sudo is
  needed.
- **Verified 2026-10-17** by read-only inspection (`docker run --rm
  --entrypoint bash`, no GPU, no server) of both the pinned nightly
  (`vllm/vllm-openai@sha256:8bd082c2...`, vllm `0.26.1rc1.dev942+g5a4c8d992`) and
  v0.31.0 (`lobes-decider-vllm:odm-local`, built from `vllm/vllm-openai:v0.31.0`,
  vllm `0.31.0`). Paths are under `/usr/local/lib/python3.12/dist-packages/vllm/`;
  line numbers are nightly / v0.31.0.
  - `VLLM_CACHE_ROOT` is read in `envs.py:702` / `:707`; the default is
    `~/.cache/vllm` (`$XDG_CACHE_HOME/vllm` if set), i.e. `/root/.cache/vllm`
    (the image runs as uid 0, and that directory does not exist in either image).
  - The compile cache directory is `$VLLM_CACHE_ROOT/torch_compile_cache/<hash_key>`
    (`compilation/backends.py:1067-1069` / `:1063-1065`), then `rank_<r>_<dp>/<prefix>`
    below it (`:1077` / `:1073`). The same root also holds `inductor_cache` and
    `triton_cache` (`compiler_interface.py:475,478` / `:465`), `startup_plan`
    (`v1/worker/startup_plan.py:83`), `modelinfos` (`model_executor/models/registry.py:923`
    / `:959`), `deep_gemm`, `assets` and the flashinfer autotune cache.
  - Entries are content-hashed, not path-keyed: `hash_key` is the first 10 hex
    chars of sha256 over `[env_hash, config_hash, code_hash, compiler_hash]`
    (`backends.py:1066` / `:1062`). `env_hash` is `envs.compile_factors()`;
    `config_hash` is `vllm_config.compute_hash()` (includes the vLLM version,
    `config/vllm.py:439-441` / `:483-486`); `code_hash` is sha256 of the traced
    model source files' paths and text (`backends.py:1042-1055` / `:1038-1051`); `compiler_hash` comes from the compiler manager.
  - `VLLM_CACHE_ROOT` is in `compile_factors`' `ignored_factors`
    (`envs.py:2224` / `:2305`), so relocating the root does not change `hash_key`.
    Moving the cache to a bind mount is safe: a warm directory is still hit.
  - **Difference between the images: none that affects the mount.** Only line
    numbers differ. A vLLM version change (the #310 upgrade to v0.31.0) changes
    `config_hash` through the version factor, so existing entries are simply not
    matched and new `<hash_key>` subdirectories are written beside the old ones
    in the same per-lane directory. Stale entries are never read; wipe the lane
    directory to reclaim space.

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
