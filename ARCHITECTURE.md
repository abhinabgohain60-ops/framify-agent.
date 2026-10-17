# Chintu — architecture

Chintu is a distributed agent, not a monolith. The process that answers Telegram is
deliberately too small to render anything; the hardware that renders is deliberately
too expensive to idle.

## 1. Brain vs. engine

| | Render brain (`main.py`) | 16 GB engine (`space/app.py`) |
| --- | --- | --- |
| Hardware | ~512 MB service | `cpu-basic`: 2 vCPU / 16 GB RAM / 50 GB |
| Owns | FastAPI, Telegram lifecycle, status card, tool routing, LLM orchestration | layout evaluation, headless Chromium, GSAP scrubbing, frame buffers, FFmpeg, voice-over muxing |
| Must never | import Pillow/NumPy/Playwright/spawn FFmpeg | answer Telegram or hold chat state |

`brain_boundary_violations()` parses `main.py` at startup and logs an error if any
heavy-media module appears in its import graph. It is asserted empty in the suite,
and the auditor itself is tested against a planted `import PIL` / `import subprocess`.

Transport (no shared disk, no queue):

```
brain  --POST /render {html, project_slug, fps, quality, narration}-->  Space
brain  --GET  /status/<job_id>  (percent, stage, probe, log_tail)-->    Space
brain  --GET  /artifact/<job_id> (video/mp4)-->                         Space
brain  --POST /sendVideo to CURRENT_CHAT_ID-->                          Telegram
```

The Space renders asynchronously (a request-scoped proxy timeout cannot hold a
multi-minute encode) and keeps at most `MAX_JOBS_RETAINED` artifacts in memory.

Fallbacks, in order: Space → Hugging Face Job (`cpu-basic`, same RAM, artifact
committed to a private dataset and downloaded by the brain) → E2B microVM
(`backend="e2b"`, always `sbx.kill()` in `finally`).

## 2. Model ladder

1. Gemini cascade over the pinned pool (`gemini-2.0-flash-exp`, `gemini-1.5-flash`,
   `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemini-3.1-flash-lite`) across
   every configured key slot, with rotation on `429`/`500`/`503`.
2. On exhaustion, OpenRouter with the same tool schemas, serialized from
   `ALL_TOOLS_LIST` by signature introspection:
   `anthropic/claude-3.5-sonnet` → `deepseek/deepseek-chat` →
   `qwen/qwen-2.5-coder-32b-instruct`, 600 s per request, up to 6 tool rounds.
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
| `HF_SPACE_URL` | Render | Space origin, e.g. `https://<ns>-chintu-render.hf.space` |
| `HF_SPACE_SECRET` | Render | must equal the Space's `HF_RENDER_SECRET` |
| `HF_RENDER_SECRET` | Space | auth for `/render`, `/status`, `/artifact` |
| `HF_TOKEN` | Render | Job fallback + private artifact dataset |
| `HF_ARTIFACT_REPO` | Render | override `<namespace>/chintu-renders` |
| `E2B_API_KEY` | Render | microVM fallback |
| `GEMINI_API_KEY[_2/_3]` | Render | Gemini pool |
| `OPENROUTER_API_KEY` | Render | frontier fallback |
| `TELEGRAM_BOT_TOKEN` | Render | chat transport |

`/health` reports the compute role, the active render engine, boundary violations,
provider telemetry, tool/skill counts and the failover cascade.

## 5. Verification

`.tools/regression_suite.py` covers, with no cloud credentials required:

- skill registry and live skill loading (6 engines, including Remotion),
- GitHub CDN-first reads, caching and rate-limit messaging,
- composition validation and the E2B render path (fake sandbox, always killed),
- the Job script ↔ caller contract (base64, ffprobe proof, Hub commit, secret handling),
- the Space contract end to end: payload, auth, polling, probe facts, artifact download,
  Telegram delivery, non-MP4 refusal, failed-job surfacing,
- OpenRouter cascade order, JSON tool schemas, tool execution and provider telemetry,
- progress card monotonicity, heartbeat, and a full simulated Telegram turn,
- the worker itself: validation, render pipeline, failure capture, mux path handling,
  and the real HTTP surface on a real socket.

```bash
python3 .tools/regression_suite.py     # expects the stubs in .tools/stubs
```
