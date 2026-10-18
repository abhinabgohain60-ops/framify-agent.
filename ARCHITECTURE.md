# Chintu — architecture

Chintu is a distributed agent, not a monolith. The process that answers Telegram is
deliberately too small to render anything; the hardware that renders is deliberately
ephemeral and disposable.

## 1. Brain vs. engine

| | Render brain (`main.py`) | Render engine (E2B microVM) |
| --- | --- | --- |
| Hardware | ~512 MB service | isolated disposable sandbox (FFmpeg + Chromium-capable) |
| Owns | FastAPI, Telegram lifecycle, status card, tool routing, LLM orchestration | layout evaluation, headless Chromium, GSAP scrubbing, frame buffers, FFmpeg, voice-over muxing |
| Must never | import Pillow/NumPy/Playwright/spawn FFmpeg | answer Telegram or hold chat state |

`brain_boundary_violations()` parses `main.py` at startup and logs an error if any
heavy-media module appears in its import graph. It is asserted empty in the suite,
and the auditor itself is tested against a planted `import PIL` / `import subprocess`.

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
FFmpeg (optional edge-tts narration muxed as AAC). The turn still ends with a
delivered MP4 instead of a `CommandExitException`.

**Hardened audio mux (no silent deliveries):** on BOTH render paths, when narration
is requested it is synthesized in-sandbox with edge-tts and muxed with an explicit
stream map (`-map 0:v:0 -map 1:a:0`, AAC 192k @ 44.1 kHz, `-shortest`,
`-movflags +faststart`) into `renders/final_output.mp4`. `ffprobe -show_streams
-select_streams a` must confirm an active audio stream — only then is the muxed
final delivered; a failed mux never aborts the turn, the silent render ships
instead. `/sendVideo` posts with the explicit `video/mp4` MIME type and
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
   so a rate-limited cascade never hits a 404), 600 s per request, up to 6 tool rounds.
3. Freebuff (OpenAI-compatible) as the last resort.

`PROVIDER_TELEMETRY` records the provider/model that actually answered and the number
of failovers; `/health` exposes it.

## 3. Live progress

`update_progress_card(percent, stage)` edits the bound Telegram message and is
monotonic: out-of-order tool callbacks can never rewind the bar. Milestones are
emitted by the phase that reaches them:

| % | Stage |
| --- | --- |
| 15 | Initializing reasoning & tool pipeline... |
| 30 | Loading motion skills & synthesizing voiceover... |
| 55 | Compiling layouts & dispatching to 16 GB worker... |
| 80 | Headless Chromium / FFmpeg rendering frames... |
| 95 | Finalizing MP4 muxing & binary upload... |
| 100 | Complete! Video delivered. |

`progress_heartbeat()` advances the bar inside the current phase every 15 s, so a long
reasoning or render call never leaves the card frozen. Every milestone is also appended
to `PROGRESS_STATE["history"]` synchronously, so a phase cannot be lost between loop ticks.

## 4. Environment

| Variable | Where | Purpose |
| --- | --- | --- |
| `E2B_API_KEY` | Render | the only render engine (E2B microVM) |
| `E2B_TMP_DIR` | Render | sandbox TMPDIR override (default `/home/user/tmp`) |
| `E2B_SANDBOX_TIMEOUT` | Render | sandbox lifetime for long renders |
| `GEMINI_API_KEY[_2/_3]` | Render | Gemini pool |
| `OPENROUTER_API_KEY` | Render | frontier fallback |
| `TELEGRAM_BOT_TOKEN` | Render | chat transport |

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
- progress card monotonicity, heartbeat, and a full simulated Telegram turn,
- the archived Space worker still parses and runs standalone (reference code).

```bash
python3 .tools/regression_suite.py     # expects the stubs in .tools/stubs
```
