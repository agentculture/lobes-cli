# Durable logs: never lose a crash trace again

When a vLLM container restarts or is recreated, its `docker logs` are gone. That
is exactly how the EngineCore crash trace in
[issue #50](https://github.com/agentculture/model-gear/issues/50) vanished before
anyone could read it: the server 500'd on a tool-calling request, the engine went
down, and by the time the box was looked at, a restart had wiped the logs — so
the root cause could not be investigated for lack of data.

model-gear fixes the **observability gap**, not (yet) the crash itself: it makes
each vLLM service's output durable so the *next* crash leaves a trace you can
read. Pinning the EngineCore root cause (MTP speculative decoding + tools vs FP4)
needs a controlled repro **with that durable trace in hand** — durable logs is
the prerequisite that unblocks it.

## How it works

`model init` scaffolds **`mg-logwrap.sh`** next to `docker-compose.yml`. Each
vLLM service bind-mounts it as the container **entrypoint**, so the real
`command:` (the `vllm serve …` arg list, unchanged) arrives at the wrapper as
`"$@"`. The wrapper:

1. opens a **per-boot** file `<service>-<ISO8601>.log` under the host-mounted log
   dir and points `<service>-latest.log` at it;
2. tees `stdout`+`stderr` to that file **and** passes them through to the console
   (so `docker logs` keeps working too);
3. `exec`s the real command, so **vLLM stays the signal target** — `docker stop`
   still drains it gracefully (SIGTERM reaches vLLM, not a shell), and the
   container's exit code is vLLM's (so `restart: unless-stopped` is unaffected).

Because it tees at the process-I/O level — below Python's logging — it captures
**both** Python tracebacks **and** native `stderr` aborts (CUDA / C++ / OOM),
which is the class of crash that most needs investigating and which Python-level
file logging would miss. If anything about logging fails (no log dir, read-only
mount, no `bash`), the wrapper falls back to a plain `exec "$@"` — logging can
never stop the model from serving.

### Paths

| | Host | In container |
|---|---|---|
| Log dir | `${MODEL_GEAR_LOG_DIR:-<deploy>/logs}` | `/logs/model-gear` |
| Single model | `…/logs/vllm-<boot>.log` | `/logs/model-gear/vllm-<boot>.log` |
| Fleet gears | `…/logs/{primary,embed,rerank}-<boot>.log` | `/logs/model-gear/<svc>-<boot>.log` |

The host dir is created (user-owned) by `model init`, `model serve`, and
`model fleet up` before compose bind-mounts it, so the logs are never
root-owned. Per-boot files mean the **crash boot is preserved as its own file**
and never overwritten by the restart that follows it.

## Reading the logs — `model logs`

`model logs` is read-only and reads the **host** files directly, so it works even
after the crashed container is gone (`docker logs` would not):

```text
model logs                 # list per-boot files (newest first) + the log dir
model logs vllm            # tail the latest boot for a service
model logs vllm --previous # tail the boot BEFORE the latest — i.e. the crashed
                           #   boot, after a restart created a fresh healthy one
model logs primary -n 200  # more lines (fleet service)
model logs --json          # structured listing
```

The `--previous` flag is the #50 investigation path: after a crash+restart, the
latest boot is the healthy one — `--previous` tails the boot that actually
crashed.

### Pruning

Per-boot files accumulate across restarts. They are plain files under the host
log dir; prune old ones with a one-liner, e.g. keep the newest 20 per service:

```bash
ls -1t <deploy>/logs/vllm-*.log | tail -n +21 | xargs -r rm
```

## Why not OTEL?

OpenTelemetry was considered first. vLLM's OTEL support is **traces-only**
(`--otlp-traces-endpoint`, request spans) — it has **no native OTLP log export**,
and a crash traceback is not a span (the engine dies), so OTEL tracing would not
capture the very thing #50 needs. Capturing logs via OTEL would require an OTEL
Collector + `filelog` receiver sidecar reading the same `stderr` plus a backend
to store it — significant new infrastructure for no gain over a host file. So
crash durability is done at the file level; OTEL **traces** remain a future
opt-in for *request* observability (latency/token spans), a separate concern from
crash logs.

## Scope

- Wrapped: the vLLM generate/embed/rerank services (single-model `vllm`, fleet
  `primary` / `embed` / `rerank`).
- Not changed: the model's serving flags/behaviour, the `restart:` policy, or the
  healthcheck. No docker-socket mounts, no new runtime dependencies. Auto-restart
  / autoheal is intentionally **out of scope** here (see #50) — this PR makes the
  crash investigable; recovery is a separate decision.
