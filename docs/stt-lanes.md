# Speech-to-text lanes: `stt-he` and `stt-en`

**Status: DECLARED/UNVALIDATED (#108).** The gateway routing, advertising and
mesh forwarding are unit-tested only, and the packaged `stt-he` sidecar has not
yet been built or served from these templates. Until a live acceptance
transcript lands under `docs/evidence/`, nothing here may be cited as working.
The numbers at the bottom come from an experiment, not an acceptance.

Operator decision, 2026-10-23: "Zipformer is stt-he, and we can use existing
stt for english for now. so stt-en. I want both."

## What they are

Two **lanes** on `POST /v1/audio/transcriptions`, beside the `stt` role:

| lane | engine | language | where it runs |
|---|---|---|---|
| `stt-he` | `ivrit-ai/zipformer2-66m-streaming-he` @ `ee769a2a13b887f91572eec349458bfb0fe79e7a`, int8 on **sherpa-onnx**, CPU-only | `he` | the packaged sidecar (`docker-compose.stt-he.yml`) |
| `stt-en` | whatever this box's `stt` container serves (`STT_MODEL` / `STT_RUNTIME`; Parakeet by default, ivrit.ai Whisper on a Hebrew overlay) | `en` | the existing `stt` container, asked for English |

They are lanes, not roles. `lobes.roles.ROLES` is unchanged, because adding a
role is effectively irreversible (see
[`colleague-stack.md`](colleague-stack.md#specialist-lanes-not-roles)). The
registry is `lobes/stt_lanes.py`. It is a sibling of the embed-lane registry
with the same shape, so the mesh machinery (announce, verify, place, forward)
is shared. The embed-only paths (`/v1/embeddings` routing, `lobes up` /
`assess`, shapes, the deployment lock) never see an stt lane.

`stt-*` would normally clash with the `{role}-{member}` member-lane spelling.
`validate_lane_name(name, audio_role="stt")` is a narrow exception: it exists
only because `stt` has no member lanes (`MEMBER_LANE_EXCLUDED_ROLES`). So
`stt-he` can never mean "the stt role on a member named `he`", and every other
role prefix is still refused.

A naming note: the issue #329 commits used "stt-he" informally for the Hebrew
**Whisper** overlay (`docker-compose.audio-he.yml`, service `stt`). That
overlay is unchanged. It is what `stt-en` rides on, asked for English, on a box
running it. The lane named `stt-he` is the Zipformer sidecar.

## Selecting a lane

The gateway reads the multipart form field `model`:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "file=@clip.m4a" -F "model=stt-he"
```

The header `X-Lobes-Model: stt-he` is an alternative, for clients that cannot
choose the form value (the decider's header convention).

- **Precedence.** If the form field names a lane, it wins, even when the
  header names a different lane. The form field is the OpenAI-native selector
  and travels inside the body that every hop forwards verbatim. The header is
  used only when the form field names no lane.
- **Unchanged requests.** Any other request keeps the pre-lane path byte for
  byte: the path-keyed `stt` role (proxied, infeasible, or local `AUDIO_URL`).
  That covers no `model` field, `model=stt`, a Whisper id, and an unknown
  `stt-xx`.

### Language injection

Each lane declares a language. When the form has **no** `language` field, the
box that serves the lane appends one multipart part, `language=<lane
language>`. The part is inserted just before the close delimiter, so every
other byte (the audio included) is untouched. A caller's own `language` is
kept as sent. A body that is not well-formed multipart is forwarded unchanged.
Forwarding boxes never inject: the serving box does.

`stt-en` relies on this. The `stt` servers let the form `language` override
`STT_LANGUAGE` (`listen_server_whisper.py`, `resolve_language`). The Zipformer
model is Hebrew-only and ignores `language`.

## Hosting and routing

A lane is hosted on a box when `<LANE>_BASE_URL` is set (`STT_HE_BASE_URL`,
`STT_EN_BASE_URL`) and `<LANE>_FEASIBLE` is not false. Per request:

1. **Hosted here**: forwarded to `<base>/v1/audio/transcriptions`, with
   `X-Lobes-Lane: <lane>` on the answer. A refused dial is **503**
   `backend_unavailable` with `Retry-After`, never another engine. There is no
   cross-lane fallback, and no fallback to the `stt` role.
2. **Reached through the mesh**: the request goes ONE hop to the first
   verified member announcing the lane, with the body verbatim. The forward
   carries the join key, `X-Lobes-Proxied`, `X-Lobes-Mesh-Member` and
   `X-Lobes-Proxied-By`, and a second hop is refused **508**. `stt-he` is
   matched on its pinned checkpoint. `stt-en` has no pinned checkpoint and is
   matched by name. A member still in its boot window answers **503**
   `role_unverified`.
3. **Otherwise**: **404** `role_infeasible` naming the lane (the embed lanes'
   body).

`stt-he`/`stt-en` sent as a JSON `model` on any other route is not a lane
request. The transcribe task is path-routed, so the lanes own no model-field id
and no alias.

## Advertising

`GET /capabilities` carries each lane as a top-level key, but only when it is
wired here or reached through the mesh:

```json
"stt-he": {"lane": true, "task": "transcribe",
  "model": "ivrit-ai/zipformer2-66m-streaming-he", "runtime": "sherpa-onnx",
  "language": "he", "revision": "ee769a2a13b887f91572eec349458bfb0fe79e7a",
  "licence": "CC-BY-4.0", "path": "/v1/audio/transcriptions",
  "feasible": true, "loaded": true, "ready": true, "fingerprint": {}}
```

The `fingerprint` value is elided here. A mesh-reached lane adds `proxied: true`
and `hosted_by`, and carries no fingerprint, so it is never re-announced.

`GET /v1/models` lists a lane id (`stt-he`) only while a request for it would
be answered: hosted, feasible and ready here, or ready on the verified member
that serves it. The gateway probes a hosted lane on its sidecar's
`/v1/health/ready`.

## The `stt-he` sidecar (packaged, not deployed)

`lobes init --stt-he` scaffolds these files. The flag is independent of
`--audio`.

| file | contents |
|---|---|
| `docker-compose.stt-he.yml` | service `stt-he`, container `model-gear-stt-he`, profile `stt-he`, no GPU, no host port, `mem_limit` 1.5g, `cpus` 2 |
| `Dockerfile.stt-zipformer` | `python:3.12-slim` + ffmpeg, `uv pip install --system`, sherpa-onnx 2.0.0 (the version measured live) |
| `listen_server_zipformer.py` | the server (see below) |
| `_readiness.py` | the shared readiness decision |

lobes chains the overlay whenever it is present.

At start, the server fetches only the `sherpa-onnx/` export of the pinned
revision, with `HF_HUB_DISABLE_XET=1`, into the mounted HF cache. It then runs
a warm-up decode, and only after that does `/v1/health/ready` answer 200. It
transcodes any container ffmpeg reads to 16 kHz mono and decodes with two
workers. Endpointing is off, and 1.5 s of tail padding flushes the last word.
It answers `json`, `text` and `verbose_json`. Its experimental WebSocket
(`/v1/audio/stream`) is not fronted by the gateway.

## Licence and attribution

The `stt-he` weights, `ivrit-ai/zipformer2-66m-streaming-he`, are licensed
**CC-BY-4.0** by ivrit.ai. Attribution is required: credit ivrit.ai and link the
model card (<https://huggingface.co/ivrit-ai/zipformer2-66m-streaming-he>)
wherever transcripts or derived work are published. The weights are downloaded
at run time and never committed or baked into an image.

## Measured in an experiment (NOT an acceptance)

The experiment server ran on a Jetson AGX Thor's CPU on 2026-10-23:

- int8, real-time factor ~0.05-0.07, RSS ~280 MB.
- 10 s of silence produced empty text.
- On 12 Hebrew clips it produced text on all 12, while the gated Whisper lane
  dropped 5.
- 0 of 11 non-speech clips produced text.
- **Known limitation:** English speech comes out as phonetic Hebrew-script
  garble. Use `stt-en` for English.

None of this was measured through the packaged image or the gateway.

## Out of scope / follow-ups

- The realtime bridge (`GET /v1/realtime`) choosing a lane by session language.
  Today it keeps using the `stt` role's `STT_URL`.
- `lobes up stt-he`, `lobes status` and `lobes assess` for stt lanes. Start the
  sidecar with the compose chain (`lobes fleet files`) and
  `--profile stt-he up -d --no-deps stt-he`.
- A live acceptance: Thor hosting `stt-he`, the Spark hosting `stt-en` and
  reaching `stt-he` through the mesh.
