# Chintu — architecture

Chintu is an autonomous, general-purpose software and automation agent. It executes
code, builds applications, runs background tasks, processes data and researches the
web; video/motion rendering is one capability it reaches for only when the user asks
for it. The process that answers Telegram is deliberately small, and the hardware
that renders video is deliberately ephemeral and disposable.

## 1. Brain vs. engine

| | Brain (`main.py`) | Render engine (E2B microVM) |
| --- | --- | --- |
| Hardware | ~512 MB service | isolated disposable sandbox (FFmpeg + Chromium-capable) |
| Owns | FastAPI, Telegram lifecycle, status card, task routing, LLM orchestration, **local command/script execution** | layout evaluation, headless Chromium, GSAP scrubbing, frame buffers, FFmpeg, voice-over muxing |
| Must never | import Pillow/NumPy/Playwright or run a local FFmpeg render loop | answer Telegram or hold chat state |

`brain_boundary_violations()` parses `main.py` at startup and logs an error if any
media module appears in its import graph. It is asserted empty in the suite, and the
auditor itself is tested against a planted `import PIL`. `subprocess` is intentionally
allowed: ordinary shell/Python work runs on the brain (see §1a), while the media
import guard is what keeps local rendering impossible.

### 1a. Task routing and local execution

Every Telegram turn is classified before any tool runs, and `TASK_CONTEXT` is the
single source of truth for the rest of the turn:

| Mode | Chosen when | Effect |
| --- | --- | --- |
| `video` | the request contains explicit video vocabulary (`render`, `reel`, `mp4`, `footage`, `9:16`, …) or carries an uploaded clip | `render_html_video` / `edit_gameplay_footage` may run, and the video milestone track is used |
| `general` | everything else — code, shell, data, research, web builds | the video tools return a `[ROUTED]` notice without touching E2B, and the general milestone track is used |

`video_dispatch_block()` is the hard guard inside the video tools. It is what stops an
arbitrary prompt, script or command from being pushed into the render pipeline (the
ghost-progress bug): a non-video turn can never dispatch a render, however the model
phrases its tool call.

General work is executed locally with `run_local_command` (`bash -lc`, hard timeout,
default 120 s, maximum 600 s) or `run_local_python` (child interpreter, same budget).
Anything long-running must use `run_local_command(..., background=True)`, which starts
the process in its own session with stdin closed and stdout/stderr redirected to
`local_tasks/<id>.log`; the call returns immediately with the PID and log path, so the
Telegram polling listener never blocks. `execute_in_cloud_microvm` stays available as
an isolated throwaway sandbox but is no longer the default executor.

When an external dispatch raises or times out, `report_dispatch_failure()` pushes the
real error log to the chat immediately, freezes the progress card on a `Failed — …`
stage, flags `PROGRESS_STATE["turn_failed"]`, and returns the same `[FAILED]` text to
the model, so a stalled percentage can never be mistaken for progress.

Transport: the composition is written into the sandbox over the file API and
rendered synchronously with the real `hyperframes` CLI; the finished MP4 is read
back and posted straight to Telegram.

```
brain --write index.html + hyperframes.json--> E2B sandbox
brain --npx hyperframes render (disk-prepped, TMPDIR=/home/user/tmp)--> frames -> MP4
brain --read MP4 bytes--> Telegram /sendVideo to CURRENT_CHAT_ID
```

**Hugging Face was retired as a render path.** Compute Spaces (Docker/Gradio) now
require a paid PRO plan and free accounts can only create Static Spaces, which
cannot run backend media rendering — so the account could not host a worker. The
archived `space/` worker remains in the repository for reference only; it is not in
the dispatch chain, and `hf_render_status` / `hf_space_health` are gone. Requests
with `backend="space"` / `"huggingface"` / `"jobs"` return an explicit retirement
notice instead of a render.

Disk resilience (the production failure was **"Low disk space: 1.0 GB free at /tmp"**):
every sandbox command that produces intermediates runs behind a pre-render prep —
`rm -rf /tmp/* ~/.cache/* /var/tmp/* 2>/dev/null || true`, `mkdir -p /home/user/tmp`
and `export TMPDIR=/home/user/tmp` — so temp writes land on the larger volume, and
frames are piped into FFmpeg over stdin rather than written per-frame.

**Automatic fallback renderer:** if the HyperFrames/Chromium path fails (for example
the tmpfs fills anyway), `render_html_video` engages `_run_pillow_fallback`: a
Pillow + FFmpeg frame compiler written into the sandbox — it parses the
composition's timed scene elements, draws 1080x1920 frames with safe-zone
typography, fade/slide motion and eased timing, and streams raw frames into
FFmpeg (optional edge-tts narration muxed as AAC). The turn still ends with an
observable outcome instead of a `CommandExitException`.

**Hardened audio mux (no silent deliveries):** on BOTH render paths, when narration
is requested it is synthesized in-sandbox with edge-tts (`python3 -m edge_tts
--file narration.txt` — the CLI only accepts `--text`/`--file`, not `--text-file`)
and muxed with an explicit stream map (`-map 0:v:0 -map 1:a:0`, AAC 192k @ 44.1 kHz,
`-shortest`, `-movflags +faststart`) into `renders/final_output.mp4`;
`ffprobe -show_streams -select_streams a` must confirm an active audio stream
before anything is uploaded. Delivery rules:

1. `final_output.mp4` ALWAYS wins in `_deliver_rendered_video` — the raw
   `video.mp4` capture can never shadow it, and no upload is ever *named*
   `video.mp4` (Telegram's player never displays that name).
2. If narration was requested but no verified `final_output.mp4` exists after a
   one-shot re-mux, delivery fails LOUDLY (`[FAILED-NO-AUDIO]`) and the silent
   capture is **not** uploaded.
3. `/sendVideo` posts the explicit `video/mp4` MIME type with
   `supports_streaming=True` so Telegram recognizes the AAC track and shows the
   unmute button.

## 2. Model ladder

1. Gemini cascade over the pinned pool (`gemini-2.0-flash-exp`, `gemini-1.5-flash`,
   `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemini-3.1-flash-lite`) across
   every configured key slot, with rotation on `429`/`500`/`503`.
2. On exhaustion, OpenRouter with the same tool schemas, serialized from
   `ALL_TOOLS_LIST` by signature introspection:
   `openrouter/free` → `nvidia/nemotron-3-super-120b-a12b:free` →
   `cohere/north-mini-code:free` (all verified against the live OpenRouter catalog
   so a rate-limited cascade never hits a 404), 600 s per request, up to 12 tool rounds.
   Every external session carries `FALLBACK_TOOL_ROUND_DIRECTIVE` as a second system
   message, which forbids dumping docs and tells the model to do the requested work
   with the tools — running code locally, reading pages, or calling a video tool only
   when the user explicitly asked for a video.
3. Freebuff (OpenAI-compatible) as the last resort.

If a provider still exhausts the 12-round ceiling, it returns the
`stopped after N tool rounds` notice, logs the ceiling as an **error** with the last
tool output, and records `PROGRESS_STATE["turn_failed"]` for the turn.

`PROVIDER_TELEMETRY` records the provider/model that actually answered and the number
of failovers; `/health` exposes it.

## 3. Live progress

`update_progress_card(percent, stage)` edits the bound Telegram message and is
monotonic: out-of-order tool callbacks can never rewind the bar. Milestones are
emitted by the phase that reaches them, on the track selected by the turn's route.

General track (`GENERAL_PROGRESS_MILESTONES`, the default):

| % | Stage |
| --- | --- |
| 15 | Initializing reasoning & tool pipeline... |
| 30 | Planning the task & loading references... |
| 55 | Executing locally (code, shell, files)... |
| 80 | Verifying results & running checks... |
| 95 | Finalizing the answer... |
| 100 | Task complete. |

Video track (`VIDEO_PROGRESS_MILESTONES`, only when the turn was routed to video):

| % | Stage |
| --- | --- |
| 15 | Initializing reasoning & tool pipeline... |
| 30 | Loading video skills & synthesizing voiceover... |
| 55 | Compiling layouts & dispatching to the E2B render engine... |
| 80 | Headless Chromium / FFmpeg rendering frames... |
| 95 | Finalizing MP4 muxing & binary upload... |
| 100 | Complete! Video delivered. |

`progress_heartbeat()` advances the bar inside the current phase every 15 s, so a long
reasoning or dispatch call never leaves the card frozen. Every milestone is also
appended to `PROGRESS_STATE["history"]` synchronously, so a phase cannot be lost
between loop ticks.

The forced 100% "Complete! Video delivered." card is reserved for turns that actually
produced a video. A turn that ends with `PROGRESS_STATE["turn_failed"]` set (tool-round
ceiling, dispatch error, reasoning timeout) logs the error and finishes the card at its
held percentage with `Stopped — no video produced (...)` on a video turn and
`Stopped — task failed (...)` on a general turn. The whole reasoning phase is bounded by
`REASONING_TIMEOUT_SECONDS` (default 900 s) so a hung provider call cannot park the card
mid-percentage.

## 4. Environment

| Variable | Where | Purpose |
| --- | --- | --- |
| `E2B_API_KEY` | Render | the only render engine (E2B microVM) |
| `E2B_TMP_DIR` | Render | sandbox TMPDIR override (default `/home/user/tmp`) |
| `E2B_SANDBOX_TIMEOUT` | Render | sandbox lifetime for long renders |
| `GEMINI_API_KEY[_2/_3]` | Render | Gemini pool |
| `OPENROUTER_API_KEY` | Render | frontier fallback |
| `TELEGRAM_BOT_TOKEN` | Render | chat transport |
| `REASONING_TIMEOUT_SECONDS` | Render | hard ceiling for one turn's reasoning phase (default 900) |
| `LOCAL_EXEC_TIMEOUT` | Render | default timeout for a foreground `run_local_command` (default 120) |

`HF_TOKEN` remains only for `huggingface_task_runner` (model inference, not rendering).

`/health` reports the compute role, the active render engine (E2B, with the HF
retirement notice), boundary violations, provider telemetry, tool/skill counts and
the failover cascade.

## 5. Verification

`.tools/regression_suite.py` covers, with no cloud credentials required:

- skill registry and live skill loading (6 engines, including Remotion),
- GitHub CDN-first reads, caching and rate-limit messaging,
- composition validation and the E2B render path (fake sandbox, always killed),
- the E2B engine contract: disk prep + TMPDIR before every execution path, the
  Pillow fallback template AST-parses and is written/executed only in the sandbox,
  retirement notices for `backend='space'/'huggingface'/'jobs'` with no HTTP call,
- OpenRouter cascade order (the verified free models), JSON tool schemas, tool
  execution and provider telemetry,
- task routing: general prompts never dispatch video tools, video prompts do,
- local execution: foreground timeout, detached background log, Python snippets,
- dispatch failures surface the real log and freeze the card on a failure stage,
- progress card monotonicity, heartbeat, and full simulated Telegram turns on both
  the general and the video track,
- the archived Space worker still parses and runs standalone (reference code).

```bash
python3 .tools/regression_suite.py     # expects the stubs in .tools/stubs
```
