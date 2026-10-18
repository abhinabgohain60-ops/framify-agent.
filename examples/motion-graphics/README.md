# Motion-graphics render — verified pipeline

A known-good HyperFrames composition plus the exact toolchain and commands needed to
render it. Verified end-to-end on Linux x64 (Ubuntu 24.04, Node 24, no root).

- **Compositions**: `index.html` (root + three sub-compositions) and `single-file.html` (the same
  piece as one HTML string — the shape `render_html_video` can actually deliver)
- **Artifacts**: `renders/video.mp4` — H.264, 1920x1080, 30 fps, 240 frames, 8.000 s;
  `renders/single-file.mp4` — H.264, 1920x1080, 30 fps, 270 frames, 9.000 s (1,965,046 bytes)
  (`ffprobe -count_frames` independently decodes the picture count of both)
- **Capture mode**: `screenshot` + `software gpu` (no GPU in a headless VM)

## Requirements

| Tool | Why |
| --- | --- |
| Node.js >= 22 | `hyperframes` CLI (the package itself declares the floor) |
| `ffmpeg` + `ffprobe` on `PATH` | muxing and post-render verification |
| `chrome-headless-shell` | the capture browser, pinned by the CLI for reproducible pixels |
| Chrome's shared libraries | without them Chrome dies instantly: `error while loading shared libraries: libnspr4.so` |

Chrome's runtime libraries (Debian/Ubuntu):
`libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0
libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0
libcairo2 libatspi2.0-0 fonts-liberation`

`fonts-liberation` (or any real font package) matters: with no fonts installed the
browser still screenshots successfully, it just renders nothing but the background.

## Commands

```bash
npx hyperframes init videos/<project> --non-interactive --example=blank --skill=motion-graphics
npx hyperframes browser ensure                  # fetch the pinned Chrome
npx hyperframes lint                            # after the first HTML pass
npx hyperframes check .                         # final gate: lint + runtime + layout + motion + contrast
npx hyperframes snapshot --at 1.2,4.2,6.9       # proof frames + contact sheet
npx hyperframes render . --skill=motion-graphics -q delivery -o ./renders/video.mp4
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames out.mp4
```

`-q draft` while iterating, `-q looks` (default) for the first real encode, `-q delivery`
for final output.

## Composition contract (the parts that bite)

- **Standalone vs sub-composition.** A top-level `index.html` puts
  `<div data-composition-id="...">` directly in `<body>`. Anything mounted through
  `data-composition-src` must wrap its content in `<template>` — and the runtime clones
  *only* the template, so a `<style>` or `<script>` left in `<head>` is silently dropped
  and every element renders as unstyled default text.
- **Host wiring.** The host clip, the inner `data-composition-id` and the
  `window.__timelines["<id>"]` key must all be the same id, and the host needs its own
  `data-start` / `data-duration` / `data-track-index` / `data-width` / `data-height`.
- **One paused timeline** per composition, registered under the root
  `data-composition-id`. Render length is the root `data-duration`, not the timeline's.
- **Never animate a `.clip` element** with `autoAlpha`/`visibility` (lint:
  `gsap_animates_clip_element`) — animate a child.
- **Never pair a CSS `transform` with a GSAP tween on the same property** (lint:
  `gsap_css_transform_conflict`). Set the initial state inside `fromTo`.
- **A named `font-family` needs an in-file `@font-face`** (`font_family_without_font_face`);
  generic families sidestep it.
- **The root is built from sub-compositions** — nesting markup directly inside a timed
  clip warns `nested_structure_needs_subcomposition`.
- A lint *error* disables the layout and contrast audits, so `check` then reports
  `0 sample(s)` / `0/0 text checks`. A clean-looking report at that point means nothing
  ran; clear lint errors first.

## The single-file variant (`single-file.html`)

`render_html_video` takes **one HTML string**, so a composition that mounts its scenes through
`data-composition-src` cannot be sent through it. `single-file.html` is the same piece built to
that constraint: every scene inline, one registered timeline driving all of them.

The rules that actually bit while building it:

- **Drop the `.clip` wrapper when the markup is inline.** A timeline element containing nested
  markup lints `nested_structure_needs_subcomposition` (the tooling expects clips to *mount*
  sub-compositions). Leaving `#stage` as plain markup keeps the root `data-duration` as the length
  source and the report clean.
- **Never animate `letterSpacing`** — `gsap_non_transform_motion`, an *error*. It reflows text and
  snaps glyphs to integer device pixels, which stutters under seek-by-frame capture. Split the
  wordmark into per-glyph spans and animate each glyph's `x` (the spread) instead.
- **An exit must finish before the next entrance starts.** Overlapping a fading-out scene with an
  entering one is a `content_overlap` layout finding rather than a style choice, because the
  audit samples that instant too.
- A lint **error** silently disables the layout and contrast audits, so a clean-looking report at
  that moment proves nothing.

`check .` on this file: **0 lint, 0 runtime, 0 motion, 0 layout issues across 9 samples, 24/24
WCAG AA text checks**.

## Rendering on the E2B engine

Render itself never renders a frame. `render_html_video` is **synchronous on the E2B microVM**
and posts the finished MP4 directly to Telegram with `/sendVideo`: there is no job id to poll.
(All Hugging Face render backends were retired — compute Spaces now require a paid PRO plan —
so `backend='space'` / `'huggingface'` / `'jobs'` return a retirement notice. The old Space
worker stays in `space/` as reference only.)

Inside the sandbox, one call does:

1. **Disk prep** — `rm -rf /tmp/* ~/.cache/* /var/tmp/* 2>/dev/null || true`, then
   `mkdir -p /home/user/tmp` and `export TMPDIR=/home/user/tmp`. The sandbox tmpfs is ~1 GB
   and previous runs failed with "Low disk space: 1.0 GB free at /tmp"; temp output now lands
   on the larger volume, and the fallback streams frames into FFmpeg over stdin so no frame
   files touch disk at all.
2. **Provisioning** — ffmpeg + the Pillow/NumPy frame stack (and, for the browser path,
   Node 22 + Chrome's runtime libs + `hyperframes browser ensure`).
3. **lint → `render . --skill=motion-graphics -f <fps> -q <quality>`** with the real HyperFrames
   CLI, then the MP4 is read back and delivered to Telegram.
4. **Audio mux + verify** — `narration="…"` is synthesized in-sandbox with edge-tts
   (`--file narration.txt`; the CLI rejects unknown flags like `--text-file`) and muxed with an
   explicit stream map (`-map 0:v:0 -map 1:a:0`, AAC 192k @ 44.1 kHz, `-movflags +faststart`)
   into `renders/final_output.mp4`; `ffprobe -show_streams -select_streams a` must confirm an
   active audio stream before delivery. `final_output.mp4` always wins in the upload step — no
   delivery is ever named `video.mp4` — and if narration was requested but the verified final
   is missing after one re-mux attempt, the render fails loudly with `[FAILED-NO-AUDIO]`
   instead of silently shipping the un-narrated capture. Telegram receives `video/mp4` with
   `supports_streaming=True` so the unmute button appears.
5. **Automatic fallback** — if the browser path fails (e.g. disk exhaustion), a Pillow + FFmpeg
   frame compiler (`_PILLOW_FALLBACK_TEMPLATE`) is written into the sandbox and executed there:
   it parses the composition's timed scenes and draws 1080x1920 frames with safe-zone
   typography, fade/slide motion and eased timing, with edge-tts narration muxed as AAC.

`narration="…"` travels with the composition request and is synthesized in the sandbox.

## Measured on this machine (2 cores, 3.8 GB, software GL)

| Stage | Time |
| --- | --- |
| `check` | ~20 s |
| 3 snapshots | ~15 s |
| `render` 240 frames @1080p, `-q delivery` | 50.2 s (49.2 s capture + encode) |
| `render` 270 frames @1080p, `-q looks` | 67.4 s (compile 0.6 s + capture 66.1 s + assemble 0.1 s) |

Static-frame dedup reused 22/188 predicted frames (~12%) on the plain-wordmark beats. The 270-frame
figure was measured running the shipped job script verbatim (`HF_JOB_INSTALL=0`, this VM's
toolchain, uploads pointed at a local mock of the Hub endpoint).

## Installing the browser without root

`apt` needs root, but `apt-get download` and `dpkg-deb -x` do not — enough to build a
private sysroot for Chrome:

```bash
mkdir -p sysroot && cd debs
for p in libnspr4 libnss3 libgbm1 libasound2 libxkbcommon0 fonts-liberation; do
  apt-get download "$p"          # writes the .deb into the cwd, no root needed
done
for f in *.deb; do dpkg-deb -x "$f" ../sysroot; done
export LD_LIBRARY_PATH="$(find ../sysroot -name '*.so*' -printf '%h\n' | sort -u | paste -sd:)"
```

Point fontconfig at a fonts.conf listing `sysroot/usr/share/fonts` (plus a writable
`cachedir`) via `FONTCONFIG_FILE`. With those two variables set, `npx hyperframes
browser ensure` and `render` work unmodified.
