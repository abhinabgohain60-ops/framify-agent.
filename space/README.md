---
title: Chintu Render Engine
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Chintu Render Engine (16 GB)

The high-memory half of Chintu. The Render brain is small (FastAPI + Telegram +
LLM routing only); every media workload — HTML/CSS layout evaluation, headless
Chromium capture, GSAP timeline scrubbing, FFmpeg encoding, voice-over muxing —
runs here.

Hardware: `cpu-basic` → **2 vCPU / 16 GB RAM / 50 GB ephemeral disk**.
The free CPU tier is the smallest Space size and still has 16 GB of RAM, which
is what HyperFrames' multi-worker capture profile needs.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | readiness, CPU/RAM, toolchain presence, job counts |
| `POST` | `/render` | start a render, returns `202 {"job_id": ...}` |
| `GET` | `/status/<job_id>` | `status` (`running`/`completed`/`failed`), `percent`, `stage`, `probe`, `log_tail` |
| `GET` | `/artifact/<job_id>` | the finished `video/mp4` |

`POST /render` body:

```json
{
  "html": "<!doctype html> … data-composition-id … window.__timelines[\\"main\\"] …",
  "project_slug": "chintu-motion",
  "fps": 30,
  "quality": "looks",
  "narration": "optional voice-over script spoken with edge-tts"
}
```

Rendering is asynchronous by design: a Space proxies one request with a short
gateway timeout, so a multi-minute render cannot be one blocking POST. The
caller polls `/status`, then downloads `/artifact`.

## Auth

Set a Space secret named `HF_RENDER_SECRET`. Every endpoint except `/health`
then requires `Authorization: Bearer <HF_RENDER_SECRET>` on the Space, and the
brain sends the same value from its own `HF_SPACE_SECRET`.

## Deploy

```bash
pip install -U huggingface_hub
hf auth login                                     # token with write scope
hf repo create chintu-render --repo-type space --space-sdk docker
hf upload <your-namespace>/chintu-render space --repo-type space
```

Then, in the Space settings, add `HF_RENDER_SECRET` (and optionally
`RENDER_TIMEOUT_SECONDS`, default 1500).

Point the Render service at it:

```
HF_SPACE_URL=https://<your-namespace>-chintu-render.hf.space
HF_SPACE_SECRET=<same secret>
```

`/health` on the Space is public and harmless; it exposes no composition data.

## Local smoke test

```bash
cd space
PORT=7860 HF_RENDER_SECRET=dev python3 app.py
curl -s localhost:7860/health | python3 -m json.tool
```

Rendering locally additionally needs `node`, `npx`, `ffmpeg` and `ffprobe` on
`PATH`; the Docker image already has all four.
