# Motion-graphics render — verified pipeline

A known-good HyperFrames composition plus the exact toolchain and commands needed to
render it. Verified end-to-end on Linux x64 (Ubuntu 24.04, Node 24, no root).

- **Composition**: kinetic type, three scenes (wordmark → capability rows → lockup)
- **Artifact**: `renders/video.mp4` — H.264, 1920x1080, 30 fps, 240 frames, 8.000 s
  (`ffprobe -count_frames` independently decodes 240 pictures)
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

## Measured on this machine (2 cores, 3.8 GB, software GL)

| Stage | Time |
| --- | --- |
| `check` | ~20 s |
| 3 snapshots | ~15 s |
| `render` 240 frames @1080p, `-q delivery` | 50.2 s (49.2 s capture + encode) |

Static-frame dedup reused 22/188 predicted frames (~12%) on the plain-wordmark beats.

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
