# Machine CLIs: benchmark window and page-cache clear — the contract

For whoever implements or calls the per-machine CLIs: **`spark`**
(`dgx-spark-cli`), **`thor`** (`jetson-thor-cli`) and **`orin`**
(`jetson-orin-cli`). lobes is the first caller. It runs a benchmark inside a
locked-clocks window, and it clears the page cache around lane start/stop. The
machine work itself (clocks, fans, sudo) lives in these CLIs, not in lobes
(lane-safety frame, decision c25).

**Status: draft, 2026-10-17. No live exchange exists yet.** The JSON below is
the required *shape*, not a captured payload. Replace each example with a real
one from a box (box + date) when the verbs ship. Tracked by the lane-safety plan,
tasks t2 to t4 (issues #304, #308, #309).

## 1. Verbs

Every verb takes `--json`. With `--json`, stdout carries exactly one JSON object
and diagnostics go to stderr. Without `--json`, the output is human text. The
three CLIs share the verb names and the JSON fields. Card-specific detail goes
under `detail`.

| verb | does | needs root |
|---|---|---|
| `window show` | Reports the current clock / fan / power state and whether a window is active. | no |
| `window store` | Saves the current state so `restore` can return to it. | yes (Jetson: `jetson_clocks --store`) |
| `window lock` | Locks clocks to max and the fan to max. Requires a stored state. | yes |
| `window restore` | Returns to the stored state, then deletes it. | yes |
| `drop-caches` | `sync`, then clears the page cache (`vm.drop_caches=3`). | via the grant only |
| `grant-permissions` | Installs the single passwordless rules the verbs above need. `--check` only reports. | one prompt at install |

## 2. Exit codes

| code | meaning |
|---|---|
| `0` | Done, or nothing to do (e.g. `restore` with no stored state). |
| `1` | Refused by state: `lock` with no stored state; `store` while a window is active; consent declined. |
| `2` | Environment: tool missing, no privilege, card does not support the operation. |

Every JSON object carries `ok` (bool) and `reason` (string, or `null` when
`ok`). `reason` uses the codes in section 5.

## 3. Shapes (draft, not captured)

`window show --json`:

```json
{"ok": true, "reason": null,
 "active": false, "stored": false, "supported": true,
 "clocks_locked": false,
 "power_mode": "MAXN",
 "clocks": {"gpu_mhz": 612, "gpu_min_mhz": 306, "gpu_max_mhz": 1300, "emc_mhz": 3199},
 "fan_profile": "cool",
 "temps_c": {"tj-thermal": 63.8, "gpu-thermal": 63.7},
 "detail": {}}
```

- `active` — clocks are locked by a lobes window (stored state exists and
  lock was applied). It tracks lobes' own window state only.
- `clocks_locked` — `true|false|null`, read from the hardware, not from lobes'
  state: `true` when the GPU devfreq node's `min_freq` equals its `max_freq`
  (on Jetson; the EMC/`bwmgr` node too when it is readable), `false` when they
  differ, `null` when unreadable (the GB10). **A lock can exist without a
  window**: an operator ran `jetson_clocks`, or a crashed run left the box
  pinned. `active` stays `false` in both cases; `clocks_locked` does not.
  *Contract addition, 2026-10-17.* Motivating reading, Jetson AGX Orin
  2026-10-17: `/sys/class/devfreq/17000000.gpu` cur = min = max = 1300500000
  and `/sys/class/devfreq/bwmgr` cur = min = max = 3199000000 (the operator had
  run `jetson_clocks`), yet `window show` reported `"active": false` and
  `lobes doctor` said the clocks were not locked (issue #308). A caller that
  gets no `clocks_locked` key (a CLI from before this addition) falls back to
  `active`.
- `stored` — a saved state exists.
- `supported` — the card accepts a clock lock at all. The GB10 is unverified
  (frame park v2); `spark` returns `supported: false` until a lock is shown to
  work, and then `lock` exits `2` with `reason: "unsupported"`.
- Fields a card cannot read are `null`, never omitted and never guessed.

`window store|lock|restore --json`: the same object as `show`, after the
action, plus `"action": "store"|"lock"|"restore"`. `restore` with no stored
state returns `ok: true`, `"restored": false`.

`drop-caches --json`:

```json
{"ok": true, "reason": null, "cleared": true, "rule": "sysctl"}
```

`grant-permissions --check --json`:

```json
{"ok": true, "reason": null,
 "drop_caches": {"granted": true, "rule": "sysctl", "source": "sudo -n -l"},
 "window": {"granted": false, "rule": null}}
```

## 4. The grant (mirrors `thor-arena/mjolnir` `cachectl.py`)

- **Detection** is `sudo -n -l` (read-only, never prompts). Two rule forms count
  as granted for `drop-caches`:
  - `sysctl` — the listing matches `sysctl.*vm\.drop_caches` (the literal dot).
    The command run is `sudo -n /usr/sbin/sysctl -w vm.drop_caches=3`, passed as
    separate argv elements so sudo matches the rule.
  - `tee` — the listing matches `tee /proc/sys/vm/drop_caches`. The command run
    is `sudo -n tee /proc/sys/vm/drop_caches` with `3` on stdin.
  - `NOPASSWD: ALL` also passes; it is reported as `rule: "all"`. It is never
    installed by these CLIs.
- **An existing equivalent rule is reused**, never duplicated. The Jetson AGX
  Thor already lists `/usr/sbin/sysctl -w vm.drop_caches\=3` (read 2026-10-17);
  `thor grant-permissions` must report it as granted and install nothing.
- **Install** (only when absent, only with consent or `--yes`): write the rule
  to a temp file, validate with `visudo -cf`, install as a `0440 root:root`
  drop-in `/etc/sudoers.d/<cli>-lobes` with one `sudo install`. One rule per
  command. Never full sudo.
- **Window rules** (Jetson): `/usr/bin/jetson_clocks`, `/usr/bin/jetson_clocks
  --store <path>`, `/usr/bin/jetson_clocks --restore <path>`, plus the fan
  control the CLI uses. The CLI owns the exact list and prints it before
  installing.
- **No grant, no prompt.** Every verb uses `sudo -n`. A missing grant is exit
  `2` with `reason: "not_granted"`.

## 5. Reason codes

| reason | when |
|---|---|
| `not_granted` | The passwordless rule a verb needs is absent. |
| `unsupported` | The card cannot do it (e.g. no clock lock on GB10). |
| `tool_missing` | `jetson_clocks`, `nvpmodel`, `nvfancontrol` or `nvidia-smi` not found. |
| `no_stored_state` | `lock` called before `store`. |
| `window_active` | `store` called while a window is active (it would overwrite the stock state with the locked one). |
| `declined` | The user declined the install prompt. |
| `failed` | The privileged command ran and failed; stderr carries its output. |

## 6. What these CLIs do NOT do

- They never change clocks or fans outside an explicit `window` verb, and they
  never make a change persistent across a reboot.
- They never `docker pause`, `docker update` or restart a container.
- They hold no lock between callers. Exclusivity of the window across concurrent
  benchmark runs is the caller's job: lobes holds a host lock around
  store → run → restore. The `window_active` refusal is the backstop.
- `drop-caches` never runs inside a container and never prompts.
