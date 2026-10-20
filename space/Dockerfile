# Chintu 16 GB render engine — Hugging Face Docker Space (cpu-basic: 2 vCPU / 16 GB RAM).
#
# Node 22 satisfies HyperFrames' engine floor; FFmpeg does the encoding; the
# lib* packages below are exactly what chrome-headless-shell links against, so
# Chromium starts without `error while loading shared libraries`.
FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HYPERFRAMES_VERSION=0.8.160 \
    RENDER_WORK_ROOT=/tmp/chintu-render

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg python3 python3-pip curl ca-certificates \
      libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
      libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
      libgbm1 libasound2 libpango-1.0-0 libcairo2 libatspi2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Fonts are optional: a missing name must not fail the image build.
RUN for pkg in fonts-liberation fonts-dejavu-core fonts-inter fonts-noto-color-emoji; do \
      apt-get update -qq >/dev/null 2>&1; \
      apt-get install -y -qq --no-install-recommends "$pkg" >/dev/null 2>&1 \
        || echo "optional font $pkg unavailable"; \
    done; rm -rf /var/lib/apt/lists/*

# edge-tts powers the voice-over track; a failure here must not block rendering.
RUN python3 -m pip install --break-system-packages --no-cache-dir edge-tts \
      || pip3 install --no-cache-dir edge-tts || echo "edge-tts unavailable; narration disabled"

# Warm both the Chromium download and the npx cache during the build so the
# first user request does not pay for either.
RUN npx --yes hyperframes@${HYPERFRAMES_VERSION} browser ensure \
      || echo "WARN: Chrome pre-fetch failed; the worker retries at render time"
RUN npx --yes hyperframes@${HYPERFRAMES_VERSION} doctor || true

# agent-reach installs the web/shell reachability tooling (web read backend,
# GitHub CLI, search router); yt-dlp covers media-site metadata and transcripts.
# Both are build-tolerant: the E2B microVM remains the primary media engine.
RUN python3 -m pip install --break-system-packages --no-cache-dir yt-dlp \
      "agent-reach @ https://github.com/Panniantong/agent-reach/archive/main.zip" \
      || echo "WARN: agent-reach/yt-dlp install failed; web tools degrade"
RUN agent-reach install --env=auto --system \
      || agent-reach install --env=auto \
      || echo "WARN: agent-reach env probe only"

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 7860
CMD ["python3", "/app/app.py"]
