# Tool calling over `/v1/realtime` — the client contract

For someone building an agent **on top of** the voice session: a harness that
owns real tools (files, shell, a robot) and lets a person drive them by voice.

**lobes runs no tools.** The server relays the model's tool call to your client
and relays your result back to the model. What the tools are, what they are
allowed to touch, and whether to run them at all is entirely the client's
business. Nothing under `lobes/` imports or knows any tool; the two demo tools
in this repo (`scripts/realtime-he-accept.py`'s `list_directory`, the web
harness's `site/src/lib/demo-tools.ts`) are examples of a client, not part of
the server.

Event names follow the OpenAI Realtime API. Every example below is copied from
a real session log (DGX Spark, 2026-09-18), shortened only where marked `…`.

## 1. Connect

```text
GET ws://<gateway>/v1/realtime?language=he&input_sample_rate=24000&turn_detection=server_vad
Authorization: Bearer <GATEWAY_API_KEY>        (when the gateway's gate is armed)
```

- `language` — `he` for the Hebrew stack; it selects STT language, the Hebrew
  default system prompt and TTS language.
- `input_sample_rate` — `24000` (default) or `16000`. Audio OUT is always 24 kHz.
- A browser cannot send the header; the web harness's dev-server proxy adds it
  (`site/README.md`).

The server answers `session.created` (its `config` shows the effective
language, sample rate and system prompt).

## 2. Declare your tools, then arm the conversation

```json
{"type": "session.update", "session": {
  "tool_choice": "auto",
  "tools": [{
    "type": "function",
    "name": "list_directory",
    "description": "List the sorted names of files and subdirectories at a path…",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string", "description": "…"}},
                   "required": []}
  }]
}}
```

Tools use the **flat** Realtime shape (`name`/`description`/`parameters` at the
top level of each tool), not the nested chat-completions one — the server
translates. The server replies `session.updated` echoing what it accepted; a
malformed declaration is a named `error` and the session stays open. You may
send `session.update` again at any time to change the tool set.

Then arm it — **a session answers nothing until you do**:

```json
{"type": "response.create"}
```

Without this the session is ears-only: you get transcripts and no replies.

## 3. Stream the microphone for the WHOLE session

```json
{"type": "input_audio_buffer.append", "audio": "<base64 PCM16 mono little-endian>"}
```

Keep sending — silence included — from connect to disconnect, on its own
thread or task. Turn detection runs on the server (`server_vad`); if the audio
stops arriving, a turn never ends. Never mute the microphone automatically
while the reply plays: that is what makes interrupting impossible. Echo
cancellation belongs at the client edge (device AEC, or the browser's
`echoCancellation`).

## 4. What comes back for a turn

```text
input_audio_buffer.speech_started        {at_ms, item_id}
input_audio_buffer.speech_stopped        {at_ms, item_id, reason: "silence" | "max_turn"}
conversation.item.input_audio_transcription.completed   {item_id, text}
response.created                         {response_id, item_id}
  … then EITHER a spoken reply …
response.audio.delta                     {response_id, delta: <base64 PCM16 24 kHz>}   (many)
response.text.done                       {response_id, text}     (once; may arrive AFTER the first audio)
response.done                            {response_id, timings: {stt, generate, tts, first_delta, …}}
  … OR a tool call (section 5) …
```

Play `response.audio.delta` as it arrives. On `response.interrupted`
(`{response_id, truncated: true}`) **stop playback immediately and drop what
you have buffered** — the person spoke over the reply.

## 5. The tool round trip

The model decides to call a tool. **Nothing is spoken**, and the response is
not finished:

```json
{"type": "response.function_call_arguments.done",
 "response_id": "resp_aa7b…", "item_id": "item_7e1a…", "output_index": 0,
 "call_id": "chatcmpl-tool-a0789651a059f25d",
 "name": "list_directory",
 "arguments": "{\"path\": \"מסמכים\"}"}
```

`arguments` is a JSON **string** — parse it yourself, and treat it as
untrusted input: it came from a language model that heard a human through a
speech recogniser. Validate paths, refuse what your tool should not do.

Run the tool, then send the result and release the follow-up:

```json
{"type": "conversation.item.create",
 "item": {"type": "function_call_output",
          "call_id": "chatcmpl-tool-a0789651a059f25d",
          "output": "{\"entries\": [\"a.md\", \"b.md\"]}"}}
{"type": "response.create"}
```

`output` is a string (JSON is conventional; an empty string is allowed). An
**error is just an output** — `{"error": "path does not exist: …"}` — and the
model will say so aloud; do not drop the call. The server then generates again
with your result in context and you get either speech (section 4) or another
tool call. One tool call per model step.

### Rules the server enforces

| situation | what you get |
|---|---|
| you answer within `TOOL_WAIT_TIMEOUT_MS` (default 120 s) | the follow-up reply |
| you never answer | the turn fails with a named `error` event whose message says the tool wait timed out; the session continues |
| the person speaks while you are running the tool | `response.interrupted`; the call is **closed**, the model is told it was cancelled, and their new turn proceeds |
| your result arrives after the call was closed | `error` … `call_closed: the tool call '<id>' was closed before this result arrived` — harmless, discard |
| a result for an id the server never issued | `error` … `unknown_call_id` |
| a second result for the same call | `error` … `duplicate_output` |

A cancelled call is not rolled back for you: if your tool has side effects and
the person interrupted it, **your harness** decides whether to finish, abort or
undo. The server only guarantees the model is told the call was cancelled.

## 6. Latency features and what they mean for a tool-owning client

All three are server-side and need no client change. Streaming is on by default; the other two are off until the operator sets them.

- **Sentence streaming** (`GENERATE_STREAM`, default on) — audio starts on the
  first sentence. `response.text.done` can arrive after audio has begun.
- **Hidden speculation** (`VAD_EAGER_MS`) — the server may start the model
  during a pause and throw the work away if the person keeps talking. **You
  never see a discarded speculation**: no event, no tool call. A tool call only
  ever reaches you for a turn that really ended.
- **Continuation merge** (`CONTINUATION_WINDOW_MS`) — with a short end-of-turn
  silence the server may commit a turn, then learn the person was only pausing.
  You will then see `response.interrupted` followed by a second
  `transcription.completed` whose `text` is the WHOLE sentence; treat the
  earlier, shorter transcript as superseded. Because a tool call cannot be
  taken back, the server **holds a finished tool call** for
  `CONTINUATION_TOOL_HOLD_MS` (default 500) after such a commit; if the person
  resumes in that time the call is never sent.

## 7. A minimal client loop

```text
connect → wait session.created
send session.update{tools} → wait session.updated
send response.create
start mic thread: input_audio_buffer.append forever
loop on events:
  response.audio.delta                  → enqueue for playback
  response.interrupted                  → flush playback
  response.function_call_arguments.done → result = run_tool(name, json.loads(arguments))
                                          send conversation.item.create{function_call_output}
                                          send response.create
  error                                 → log it; the session is still open unless it closed
```

`scripts/realtime-he-accept.py` is a complete, dependency-free implementation
of this loop (stdlib WebSocket client, pipewire/ALSA audio, one sandboxed
tool) — read it as a worked example, then replace its tool with yours.

## Known limits (2026-09-18)

- English words inside Hebrew speech — file names especially — are misheard
  and hard to follow when spoken (issue #277). Make path-like arguments
  forgiving on the tool side: case-insensitive, fuzzy, and return near matches
  in the error so the model can recover in one step.
- A noise burst with no words in it still counts as an interruption.
- The operator may set an input-level gate (`VAD_MIN_LEVEL_PCT`): audio whose
  peak stays below it produces **no events at all**. If a quiet speaker gets no
  `speech_started`, that is why — raise the capture gain or ask for a lower
  threshold.
- One session per connection; no resume — a reconnect is a new session with
  empty history.
- The WebSocket is never proxied across mesh boxes: connect to the box that
  hosts the voice stack.
