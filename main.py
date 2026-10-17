import ast
import base64
import os
import sys
import json
import logging
import asyncio
import inspect
import io
import re
import contextlib
import time
import traceback
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chintu-agent")

PORT = int(os.environ.get("PORT", 10000))
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
E2B_API_KEY = os.environ.get("E2B_API_KEY", "")
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY", "")
RENDER_API_KEY = os.environ.get("RENDER_API_KEY", "")
RENDER_SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Frontier-reasoning cascade used only after every Gemini key slot is exhausted.
OPENROUTER_MODELS: List[str] = [
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-coder-32b-instruct",
]
EXTERNAL_REQUEST_TIMEOUT = 600
EXTERNAL_MAX_TOOL_ROUNDS = 6
FREEBUFF_API_KEY = os.environ.get("FREEBUFF_API_KEY", "")
FREEBUFF_BASE_URL = os.environ.get("FREEBUFF_BASE_URL", "https://api.freebuff.com/v1")
DEFAULT_REPO = os.environ.get("DEFAULT_REPO", "abhinabgohain60-ops/framify-agent.")

GEMINI_API_KEYS = [
    k for k in [
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", "")
    ] if k
]
current_key_index = 0

def get_current_gemini_client():
    global current_key_index
    if not GEMINI_API_KEYS:
        return None
    key = GEMINI_API_KEYS[current_key_index % len(GEMINI_API_KEYS)]
    return genai.Client(api_key=key)

def rotate_to_next_gemini_key():
    global current_key_index
    if len(GEMINI_API_KEYS) > 1:
        current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
        logger.info(f"[Key Rotator] Switched to API key slot #{current_key_index + 1} of {len(GEMINI_API_KEYS)}")
        return True
    return False

# Verified, active Gemini endpoints only.
# NOTE: deprecated IDs (gemini-2.5-flash, gemini-2.0-flash) return HTTP 404 and
# gemini-pro-latest drains quota (429), so neither is part of the cascade.
ACTIVE_GEMINI_MODELS: List[str] = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite"
]

# HTTP statuses that trigger an immediate external-provider failover once every
# Gemini key slot has also failed.
RETRYABLE_GEMINI_STATUS = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL")

def discover_available_models():
    global ACTIVE_GEMINI_MODELS
    logger.info(f"[Active Verified Cascade]: {ACTIVE_GEMINI_MODELS}")

def _github_headers() -> Dict[str, str]:
    """Build GitHub API headers, omitting Authorization when no token is set.

    GitHub rejects `Authorization: Bearer ` (empty token) with HTTP 401 Bad
    credentials, so an unconditional header silently broke every read on
    deployments without GITHUB_TOKEN. Public repos work anonymously.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

# Unauthenticated GitHub allows only 60 requests/hour, and one skill load costs
# two (contents + raw download), so a handful of motion-graphics turns would
# exhaust it. Two mitigations: read files from GitHub's raw CDN, which is not the
# REST API and therefore costs no quota (verified returning 200 with the API
# budget fully exhausted), and cache successful reads in-process with a TTL.
GITHUB_CACHE_TTL_SECONDS = 900
GITHUB_RAW_HOST = "https://raw.githubusercontent.com"
GITHUB_RAW_BRANCHES = ("main", "master")
_GITHUB_RAW_CACHE: Dict[str, Any] = {}

def _cache_get(repo_name: str, file_path: str) -> Optional[str]:
    entry = _GITHUB_RAW_CACHE.get(f"{repo_name}/{file_path}")
    if not entry:
        return None
    cached_at, text = entry
    if (datetime.utcnow() - cached_at).total_seconds() > GITHUB_CACHE_TTL_SECONDS:
        _GITHUB_RAW_CACHE.pop(f"{repo_name}/{file_path}", None)
        return None
    return text

def _cache_put(repo_name: str, file_path: str, text: str) -> None:
    _GITHUB_RAW_CACHE[f"{repo_name}/{file_path}"] = (datetime.utcnow(), text)

def _github_cdn_read(repo_name: str, file_path: str) -> Optional[str]:
    """Fetch a repo file straight from GitHub's raw CDN, bypassing the REST API.

    Returns None when the path is not fetchable this way: a directory, a binary
    blob, a private repo without a token, or a repo whose default branch is
    neither ``main`` nor ``master``. The caller then falls back to the contents
    API, which can also list directories and report real HTTP errors.
    """
    if not file_path or file_path.endswith("/"):
        return None
    repo = repo_name.rstrip(".")
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    for branch in GITHUB_RAW_BRANCHES:
        try:
            res = requests.get(
                f"{GITHUB_RAW_HOST}/{repo}/{branch}/{file_path}", headers=headers, timeout=15
            )
        except Exception:
            return None
        if res.status_code == 200 and res.text:
            return res.text
    return None

def github_read_file(repo_name: str, file_path: str) -> str:
    cached = _cache_get(repo_name, file_path)
    if cached is not None:
        return cached
    # CDN first: one request instead of two, and no API quota consumed.
    cdn_text = _github_cdn_read(repo_name, file_path)
    if cdn_text is not None:
        _cache_put(repo_name, file_path, cdn_text)
        return cdn_text
    headers = _github_headers()
    encoded_repo = repo_name.replace(".", "%2E") if repo_name.endswith(".") else repo_name
    url = f"https://api.github.com/repos/{encoded_repo}/contents/{file_path}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            payload = res.json()
            # The contents API returns a JSON array for directories (a bare
            # .get() here would raise AttributeError on a list), and files that
            # are binary blobs expose no download_url (fetching None raises
            # MissingSchema). Handle both without killing the tool call.
            if isinstance(payload, list):
                names = [str(i.get("name")) for i in payload[:40] if isinstance(i, dict)]
                return (
                    f"GitHub Read Error: '{file_path}' in {repo_name} is a directory, not a file. "
                    f"Entries: {', '.join(names)}"
                )
            dl_url = payload.get("download_url") if isinstance(payload, dict) else None
            if not dl_url:
                return (
                    f"GitHub Read Error: '{file_path}' in {repo_name} has no downloadable "
                    "content (binary blob or empty response)."
                )
            text = requests.get(dl_url, timeout=10).text
            _cache_put(repo_name, file_path, text)
            return text
        # Anonymous rate limiting returns an HTML error page; surfacing that raw
        # body buries the actual cause, so name it and say how to fix it.
        if res.status_code == 403 and "rate limit" in res.text.lower():
            return (
                "GitHub Read Error (403): unauthenticated rate limit reached (60 req/hour). "
                "Set GITHUB_TOKEN to lift this to 5,000 req/hour; already-loaded docs stay cached."
            )
        return f"GitHub Read Error ({res.status_code}): {res.text[:400]}"
    except Exception as e:
        return f"Error reading GitHub file: {str(e)}"

def github_commit_file(repo_name: str, file_path: str, content: str, commit_message: str) -> str:
    if file_path.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as syn_err:
            return f"Pre-commit AST check failed: SyntaxError on line {syn_err.lineno}: {syn_err.msg}"
    if not GITHUB_TOKEN:
        return "Commit aborted: GITHUB_TOKEN is not configured (GitHub rejects anonymous writes with 401)."
    headers = _github_headers()
    encoded_repo = repo_name.replace(".", "%2E") if repo_name.endswith(".") else repo_name
    url = f"https://api.github.com/repos/{encoded_repo}/contents/{file_path}"
    try:
        sha = None
        get_res = requests.get(url, headers=headers, timeout=10)
        if get_res.status_code == 200:
            sha = get_res.json().get("sha")
        b64_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        payload = {"message": commit_message, "content": b64_content, "branch": "main"}
        if sha:
            payload["sha"] = sha
        put_res = requests.put(url, headers=headers, json=payload, timeout=15)
        if put_res.status_code in [200, 201]:
            return f"SUCCESS: File {file_path} committed to {repo_name}."
        return f"Commit failed ({put_res.status_code}): {put_res.text}"
    except Exception as e:
        return f"Error committing file: {str(e)}"

# ---------------------------------------------------------------------------
# SKILL / KNOWLEDGE REGISTRY
# Curated external repositories that teach Chintu to produce high-quality motion
# graphics. The agent pulls the real docs at task time via `load_skill` instead
# of guessing an API surface. Every slug is verified to exist on GitHub.
# ---------------------------------------------------------------------------
# `docs` lists the REAL documentation files (skill routers, contracts and
# references) rather than the repo landing README: a README explains what a
# project *is*, the SKILL.md files explain how to author with it. Pointing
# `load_skill` at README.md returned marketing copy and never the render loop.
# Every slug and path below was verified to exist on GitHub.
DEFAULT_SKILL_MAX_CHARS = 24000

SKILL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "hyperframes": {
        "repo": "heygen-com/hyperframes",
        "role": "primary_renderer",
        "entrypoint": "skills/hyperframes/SKILL.md",
        "docs": [
            "skills/hyperframes/SKILL.md",
            "skills/hyperframes-core/SKILL.md",
            "skills/hyperframes-core/references/minimal-composition.md",
            "skills/hyperframes-core/references/sub-compositions.md",
            "skills/hyperframes-cli/SKILL.md",
            "skills/hyperframes-cli/references/preview-render.md",
            "skills/motion-graphics/SKILL.md",
            "skills/motion-graphics/references/builder-contract.md",
            "skills/motion-graphics/references/motion-vocabulary.md",
            "skills/motion-graphics/categories/kinetic-type/module.md",
            "skills/hyperframes-creative/references/motion-principles.md",
        ],
        "max_chars": 24000,
        "summary": (
            "Author HTML/CSS/GSAP compositions and render deterministic MP4 through a "
            "headless browser plus FFmpeg. 'docs' carries the router, the composition "
            "contract (data-* timing, class=clip, one paused GSAP timeline) and the CLI."
        ),
    },
    "gsap": {
        "repo": "greensock/GSAP",
        "role": "timeline_animation",
        "entrypoint": "README.md",
        "docs": ["README.md"],
        "max_chars": 14000,
        "summary": (
            "JavaScript timeline animation engine used to author seekable animation "
            "inside HyperFrames compositions. Use fromTo/stagger/ease, never CSS "
            "transforms on a node you also tween."
        ),
    },
    "whisper": {
        "repo": "openai/whisper",
        "role": "audio_transcription",
        "entrypoint": "README.md",
        "docs": ["README.md"],
        "max_chars": 14000,
        "summary": (
            "Speech recognition for transcribing narration so captions and voice-over "
            "timing can be generated."
        ),
    },
    "motion_graphics_prompts": {
        "repo": "songguoxs/awesome-video-prompts",
        "role": "prompt_library",
        "entrypoint": "README.md",
        "docs": ["README.md"],
        "max_chars": 40000,
        "summary": (
            "Curated shot/camera prompts for live-action video models (Veo/Kling/Hailuo): "
            "50 cases, 26 of them ASMR and 11 product/ unboxing. Use it for lighting, "
            "lens and camera-move vocabulary, NOT for motion-graphics design — it contains "
            "no typography, chart or kinetic-type material. Get design direction from "
            "'hyperframes' (motion-vocabulary, motion-principles, the category modules) "
            "and 'motion_skills' instead. The single README is ~91k chars, so page it with "
            "load_skill(..., offset=...)."
        ),
    },
    "motion_skills": {
        "repo": "iart-ai/motion-skills",
        "role": "skill_library",
        "entrypoint": "README.md",
        "docs": ["README.md"],
        "max_chars": 18000,
        "summary": (
            "Open-source motion-design skills for coding agents: kinetic typography, "
            "explainers, data visualization, short-form video and WebGL."
        ),
    },
    "remotion": {
        "repo": "remotion-dev/remotion",
        "role": "programmatic_react_video",
        "entrypoint": "README.md",
        "docs": [
            "README.md",
            "packages/docs/docs/composition.mdx",
            "packages/docs/docs/animating-properties.mdx",
            "packages/docs/docs/building-a-timeline.mdx",
        ],
        "max_chars": 20000,
        "summary": (
            "Programmatic React video engine: compositions, frame-based animation, "
            "interpolation and batch rendering. Read it for timeline and data-driven "
            "composition concepts to port into a single-file HyperFrames build — do not "
            "scaffold a React project, the render contract stays HTML/CSS/GSAP."
        ),
    },
}

CURRENT_CHAT_ID: Optional[int] = None
CURRENT_STATUS_MSG = None
TELEGRAM_BOT_INSTANCE = None
CURRENT_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None

# --------------------------------------------------------- render-brain boundary
# Render runs on ~512 MB. It hosts the API, the Telegram lifecycle, the status
# card, tool routing and LLM orchestration — and nothing else. Frame buffers,
# headless Chromium and FFmpeg belong to the 16 GB Space or the E2B microVM, so
# importing any of these locally is an operational defect, not a style choice.
LOCAL_MEDIA_IMPORTS = (
    "PIL", "playwright", "moviepy", "cv2", "imageio",
    "numpy", "matplotlib", "subprocess", "ffmpeg",
)


def brain_boundary_violations(source: Optional[str] = None) -> List[str]:
    """Return every heavy-media module imported by this module (must be empty).

    Media work is executed by the remote worker; a local import here would mean
    the brain is about to render on a 512 MB box. Scanned statically so the
    startup audit costs nothing.
    """
    if source is None:
        try:
            with open(os.path.abspath(__file__), "r", encoding="utf-8") as fh:
                source = fh.read()
        except Exception:
            return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        found.update(name for name in names if name in LOCAL_MEDIA_IMPORTS)
    return sorted(found)


BRAIN_BOUNDARY_VIOLATIONS = brain_boundary_violations()


# --------------------------------------------------------------- telemetry
PROVIDER_TELEMETRY: Dict[str, Any] = {
    "provider": "",
    "model": "",
    "at": "",
    "failovers": 0,
}


def record_provider(provider: str, model: str) -> None:
    """Remember which provider actually served the turn (observable in /health)."""
    PROVIDER_TELEMETRY.update(
        provider=provider, model=model, at=datetime.utcnow().isoformat()
    )
    logger.info(f"[Provider] served by {provider} model={model}")


# ------------------------------------------------------------- progress card
# The status card is a live object, not a 15%-and-freeze banner. Every milestone
# below is emitted by the phase that actually reached it, and a heartbeat nudges
# the bar forward while a single long phase (render, model reasoning) runs.
PROGRESS_BAR_WIDTH = 30
PROGRESS_MILESTONES = (
    (15, "Initializing reasoning & tool pipeline..."),
    (30, "Loading motion skills & synthesizing voiceover..."),
    (55, "Compiling layouts & dispatching to 16 GB worker..."),
    (80, "Headless Chromium / FFmpeg rendering frames..."),
    (95, "Finalizing MP4 muxing & binary upload..."),
    (100, "Complete! Video delivered."),
)
PROGRESS_STATE: Dict[str, Any] = {
    "percent": 0,
    "stage": "",
    "started": 0.0,
    "video_dispatched": False,
    "history": [],
}


def progress_bar(percent: int) -> str:
    filled = max(0, min(PROGRESS_BAR_WIDTH, int(percent / 100.0 * PROGRESS_BAR_WIDTH)))
    return "=" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)


def next_milestone(percent: int) -> Optional[int]:
    for value, _ in PROGRESS_MILESTONES:
        if value > percent:
            return value
    return None


def _record_progress(percent: int, text: str) -> None:
    """Keep an ordered, de-duplicated trail of the milestones this turn reached."""
    history = PROGRESS_STATE.setdefault("history", [])
    if history and history[-1][0] == percent and history[-1][1] == text:
        return
    history.append((percent, text))
    del history[:-50]


def reset_progress() -> None:
    PROGRESS_STATE.update(
        percent=0,
        stage="",
        started=time.monotonic(),
        video_dispatched=False,
        history=[],
    )


async def update_progress_card(percent: int, stage_text: str = "", force: bool = False) -> bool:
    """Edit the bound status card to `percent`. Safe to call at any time.

    Args:
        percent: 0-100. Values below the current percentage are ignored unless
            force is set, so out-of-order tool callbacks can never rewind the bar.
        stage_text: The phase text shown under the bar.
        force: Accept a lower percentage (used for resets and completion).
    """
    global CURRENT_STATUS_MSG
    try:
        percent = int(percent)
    except (TypeError, ValueError):
        return False
    percent = max(0, min(100, percent))
    if not force and percent < int(PROGRESS_STATE.get("percent") or 0):
        return False
    text = (stage_text or PROGRESS_STATE.get("stage") or "Working...").strip()
    PROGRESS_STATE.update(percent=percent, stage=text)
    _record_progress(percent, text)
    message = CURRENT_STATUS_MSG
    if message is None:
        return False
    body = f"Task/Project Progress\n[{progress_bar(percent)}] {percent}%\n\nCurrent status: {text}"
    try:
        await message.edit_text(body)
        return True
    except Exception as edit_err:
        # Telegram raises BadRequest('message is not modified') for a no-op edit,
        # and a transient network error must never kill the turn.
        logger.debug(f"[Progress] card edit skipped at {percent}%: {edit_err}")
        return False


def emit_progress(percent: int, stage_text: str = "", video: bool = False) -> None:
    """Fire-and-forget progress from synchronous tool code.

    Tools execute in a worker thread, so the coroutine is scheduled onto the
    Telegram event loop instead of being awaited. Ordering is protected by the
    monotonic guard in update_progress_card.
    """
    if video:
        PROGRESS_STATE["video_dispatched"] = True
    try:
        percent = max(0, min(100, int(percent)))
    except (TypeError, ValueError):
        return
    # Record the phase synchronously so the sequence of milestones survives even
    # when two callbacks land inside one event-loop tick (fast tools, tests).
    if percent >= int(PROGRESS_STATE.get("percent") or 0):
        text = (stage_text or PROGRESS_STATE.get("stage") or "Working...").strip()
        PROGRESS_STATE.update(percent=percent, stage=text)
        _record_progress(percent, text)
    loop = CURRENT_EVENT_LOOP
    if loop is None or not loop.is_running():
        return
    try:
        asyncio.run_coroutine_threadsafe(update_progress_card(percent, stage_text), loop)
    except Exception as emit_err:
        logger.debug(f"[Progress] could not schedule {percent}%: {emit_err}")


async def progress_heartbeat(interval: float = 15.0) -> None:
    """Advance the bar inside the current phase so it never sits frozen."""
    try:
        while True:
            await asyncio.sleep(interval)
            current = int(PROGRESS_STATE.get("percent") or 0)
            ceiling = next_milestone(current)
            if ceiling is None or current >= ceiling - 1:
                continue
            await update_progress_card(current + 1, PROGRESS_STATE.get("stage") or "Working...", force=True)
    except asyncio.CancelledError:
        raise

# Root-level shell binaries that must never be handed to the Python interpreter raw.
_ROOT_SHELL_RE = re.compile(
    r"^\s*(?:\$\s*|>\s*)?(?:sudo\s+)?"
    r"(cat|curl|wget|ffmpeg|ffprobe|pip3?|apt|apt-get|bash|sh|zsh|ls|mkdir|rmdir|rm|"
    r"cp|mv|echo|printf|chmod|chown|tar|unzip|gzip|grep|egrep|sed|awk|find|sort|"
    r"head|tail|wc|git|node|npm|npx|python3?|pytest|make|cmake)\b"
)

def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False

def _has_root_shell_command(code: str) -> bool:
    """True if the first meaningful root-level token is a shell command."""
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return bool(_ROOT_SHELL_RE.match(stripped))
    return False

def _wrap_shell_as_python(code: str) -> str:
    """Embed arbitrary shell/syntactically-invalid input in a valid Python wrapper."""
    return (
        "import subprocess, sys\n"
        f"_script = {code!r}\n"
        "_proc = subprocess.run(['/bin/bash', '-lc', _script], capture_output=True, text=True)\n"
        "sys.stdout.write(_proc.stdout or '')\n"
        "sys.stderr.write(_proc.stderr or '')\n"
        "print(f'[shell_exit_code={_proc.returncode}]')\n"
    )

def prepare_task_source(code: str) -> str:
    """Guarantee the source written to run_task.py is 100% valid Python.

    Valid Python is passed through untouched. Anything that fails ``ast.parse``
    (raw shell commands such as cat/curl/ffmpeg, multiline wrappers or mismatched
    syntax) is wrapped in a subprocess shim instead of crashing the VM with
    ``SyntaxError: invalid syntax``.
    """
    if not isinstance(code, str):
        code = "" if code is None else str(code)
    # Shell detection takes priority: a line like `cat /home/user/x.txt` is a
    # syntactically-valid Python expression, so ast.parse alone would not catch it.
    if _has_root_shell_command(code):
        logger.warning("[E2B Guard] Detected root-level shell command; wrapping task for safe shell execution.")
        return _wrap_shell_as_python(code)
    if _is_valid_python(code):
        return code
    logger.warning("[E2B Guard] Detected invalid Python syntax; wrapping task for safe shell execution.")
    return _wrap_shell_as_python(code)

def _find_rendered_video(sbx: Any, directory: str) -> Optional[str]:
    """Return the sandbox path of the first ``.mp4`` in ``directory``, else None."""
    try:
        for entry in sbx.files.list(directory):
            name = getattr(entry, "name", "") or ""
            if name.lower().endswith(".mp4"):
                return f"{directory.rstrip('/')}/{name}"
    except Exception:
        return None
    return None

SANDBOX_TIMEOUT_SECONDS = int(os.environ.get("E2B_SANDBOX_TIMEOUT", "900"))
HYPERFRAMES_VERSION = os.environ.get("HYPERFRAMES_VERSION", "0.8.160")
HYPERFRAMES_SKILL = os.environ.get("HYPERFRAMES_SKILL", "motion-graphics")

# Renders land here; a HyperFrames project writes into its own renders/ subdir.
SANDBOX_VIDEO_SCAN_DIRS = ("/home/user", "/home/user/output", "/home/user/videos")

# chrome-headless-shell links against these. Without them it dies instantly with
# `error while loading shared libraries: libnspr4.so`, so the HyperFrames path is
# unreachable on a bare sandbox image even when Node itself is present.
CHROME_RUNTIME_PACKAGES = (
    "libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 "
    "libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 "
    "libgbm1 libasound2 libpango-1.0-0 libcairo2 libatspi2.0-0 fonts-liberation"
)

# Mirrors `hyperframes init --example=blank` so the CLI finds a project without
# an interactive scaffold step.
HYPERFRAMES_PROJECT_CONFIG: Dict[str, Any] = {
    "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
    "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
    "paths": {
        "blocks": "compositions",
        "components": "compositions/components",
        "assets": "assets",
    },
    "media": {"autoProxy": True},
    "authoringSkill": HYPERFRAMES_SKILL,
}

# ------------------------------------------------- Hugging Face Jobs renderer
# The point of this backend is the hardware: Hugging Face's `cpu-basic` flavor is
# 2 vCPU / 16 GB RAM / 50 GB disk at $0.01 per hour, billed per minute. HyperFrames
# switches to a single-worker screenshot profile below 8 GB of RAM (its render log
# prints `totalMemMb` next to `thresholdMb: 8192`), which is what the smaller render
# VM hits, so a Job both fits the composition in memory and is cheap enough to
# treat as a per-render utility. System tools are installed inside the Job because
# a Job image must already exist: `node:22-bookworm` is a public Docker Hub image
# that satisfies HyperFrames' Node >= 22 floor and lets apt install the rest.
HF_RENDER_FLAVOR = os.environ.get("HF_RENDER_FLAVOR", "cpu-basic")
HF_RENDER_IMAGE = os.environ.get("HF_RENDER_IMAGE", "node:22-bookworm")
HF_RENDER_TIMEOUT = os.environ.get("HF_RENDER_TIMEOUT", "20m")
HF_RENDER_JOB_NAME = os.environ.get("HF_RENDER_JOB_NAME", "chintu-motion-render")
HF_ARTIFACT_REPO = os.environ.get("HF_ARTIFACT_REPO", "")
HF_ARTIFACT_REPO_NAME = os.environ.get("HF_ARTIFACT_REPO_NAME", "chintu-renders")
HF_RENDER_WAIT_SECONDS = int(os.environ.get("HF_RENDER_WAIT_SECONDS", "300"))
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# --------------------------------------------------- Hugging Face Space renderer
# The preferred heavy engine is a dedicated Docker Space on the 16 GB CPU tier
# (`cpu-basic`: 2 vCPU / 16 GB RAM / 50 GB disk). It is an ordinary HTTP worker:
# POST the composition, poll the job, download the MP4. Spaces proxy a single
# request with a short gateway timeout, so the worker returns a job id at once
# and the brain polls it — a multi-minute render can never be one blocking call.
# The Space is stateless from the brain's point of view; it is configured with
#
#     HF_SPACE_URL=https://<namespace>-chintu-render.hf.space
#     HF_SPACE_SECRET=<the Space's HF_RENDER_SECRET>
#
# and its code lives in `space/` in this repository (Dockerfile + app.py).
COMPUTE_ROLE = "brain"
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "").rstrip("/")
HF_SPACE_SECRET = os.environ.get("HF_SPACE_SECRET", "") or os.environ.get("HF_SPACE_TOKEN", "")
HF_SPACE_POLL_SECONDS = float(os.environ.get("HF_SPACE_POLL_SECONDS", "4"))
HF_SPACE_MAX_WAIT_SECONDS = int(os.environ.get("HF_SPACE_MAX_WAIT_SECONDS", "1500"))

# Runs INSIDE the Job. Kept here rather than in a separate file so the deployed
# process is the single source of truth for what the remote container executes.
# Set HF_JOB_INSTALL=0 for an image that already ships the toolchain.
HF_RENDER_JOB_SCRIPT = r"""
set -euo pipefail
echo "[hf-render] boot node=$(node -v 2>/dev/null || echo missing) cpus=$(nproc 2>/dev/null || echo '?') mem=$(free -m 2>/dev/null | awk '/^Mem:/{print $2" MB"}' || echo '?')"
: "${COMPOSITION_B64:?COMPOSITION_B64 payload is missing}"
: "${PROJECT_SLUG:=chintu-motion}"
: "${RENDER_FPS:=30}"
: "${RENDER_QUALITY:=looks}"
: "${HYPERFRAMES_VERSION:=0.8.160}"
: "${HF_ENDPOINT:=https://huggingface.co}"

if [ "${HF_JOB_INSTALL:-1}" = "1" ] && command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq || echo "[hf-render] WARN apt-get update reported errors"
  apt-get install -y -qq --no-install-recommends ffmpeg curl ca-certificates fonts-liberation fonts-dejavu-core \
    || echo "[hf-render] WARN base toolchain install reported errors"
  # One package at a time so a single missing name cannot abort the whole install.
  for pkg in libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
             libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
             libpango-1.0-0 libcairo2 libatspi2.0-0; do
    apt-get install -y -qq --no-install-recommends "$pkg" >/dev/null 2>&1 \
      || echo "[hf-render] WARN could not install $pkg"
  done
fi
command -v ffmpeg >/dev/null 2>&1 || { echo "[hf-render] FAILED ffmpeg is not on PATH"; exit 1; }
ldconfig -p 2>/dev/null | grep -q libnspr4 || echo "[hf-render] WARN libnspr4 absent from the linker cache; Chrome may not start"

PROJECT_DIR="/tmp/chintu-render/$PROJECT_SLUG"
mkdir -p "$PROJECT_DIR/renders"
printf '%s' "$COMPOSITION_B64" | base64 -d > "$PROJECT_DIR/index.html"
if [ -n "${HYPERFRAMES_JSON_B64:-}" ]; then
  printf '%s' "$HYPERFRAMES_JSON_B64" | base64 -d > "$PROJECT_DIR/hyperframes.json"
fi
cd "$PROJECT_DIR"
echo "[hf-render] staged $(wc -c < index.html) bytes of composition in $PROJECT_DIR"

HF="npx --yes hyperframes@$HYPERFRAMES_VERSION"
$HF doctor --json 2>&1 | tail -40 || echo "[hf-render] WARN doctor reported problems"
$HF browser ensure || { echo "[hf-render] FAILED could not ensure the pinned Chrome"; exit 1; }
$HF lint . || echo "[hf-render] WARN lint reported issues; rendering anyway"
$HF render . --skill=motion-graphics -f "$RENDER_FPS" -q "$RENDER_QUALITY" -o ./renders/video.mp4

VIDEO="$PROJECT_DIR/renders/video.mp4"
[ -s "$VIDEO" ] || { echo "[hf-render] FAILED no MP4 written to $VIDEO"; exit 1; }
echo "[hf-render] ffprobe:"
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,r_frame_rate,nb_read_frames \
  -show_entries format=duration,size -of default=noprint_wrappers=1 "$VIDEO" || true
echo "[hf-render] RENDER_VERIFIED $VIDEO $(wc -c < "$VIDEO") bytes"

if [ -n "${HF_ARTIFACT_REPO:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  # One request, no client library: the Hub's commit API takes NDJSON with the
  # file base64-encoded. base64 goes through a file, never argv (a multi-MB
  # argument would blow past the kernel's per-argument limit).
  base64 -w 0 "$VIDEO" > /tmp/artifact.b64
  {
    printf '{"key":"header","value":{"summary":"chintu render: %s"}}\n' "$HF_ARTIFACT_PATH"
    printf '{"key":"file","value":{"path":"%s","encoding":"base64","content":"' "$HF_ARTIFACT_PATH"
    cat /tmp/artifact.b64
    printf '"}}\n'
  } > /tmp/commit.ndjson
  code=$(curl -sS -o /tmp/commit.out -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $HF_TOKEN" \
    -H "Content-Type: application/x-ndjson" \
    --data-binary @/tmp/commit.ndjson \
    "$HF_ENDPOINT/api/datasets/$HF_ARTIFACT_REPO/commit/main")
  echo "[hf-render] artifact upload http=$code body=$(head -c 300 /tmp/commit.out)"
  if [ "$code" = "200" ]; then
    echo "[hf-render] ARTIFACT_URL $HF_ENDPOINT/datasets/$HF_ARTIFACT_REPO/resolve/main/$HF_ARTIFACT_PATH"
  else
    echo "[hf-render] FAILED artifact upload (HTTP $code)"
    exit 1
  fi
else
  echo "[hf-render] WARN no HF_ARTIFACT_REPO/HF_TOKEN; the MP4 stays inside the Job"
fi
echo "[hf-render] DONE"
"""

_hf_namespace_cache: Dict[str, str] = {}

def _telegram_send_video(video_bytes: bytes, file_name: str, caption: str = "") -> bool:
    """Post MP4 bytes to the bound Telegram chat. Returns True only on HTTP 200."""
    if not (CURRENT_CHAT_ID and TELEGRAM_BOT_TOKEN):
        return False
    emit_progress(95, PROGRESS_MILESTONES[4][1], video=True)
    try:
        send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
        upload_files = {"video": (file_name, video_bytes, "video/mp4")}
        data = {"chat_id": CURRENT_CHAT_ID, "caption": caption or f"🎬 Chintu Video Render: {file_name}"}
        res = requests.post(send_url, data=data, files=upload_files, timeout=120)
        if res.status_code == 200:
            logger.info(f"Directly uploaded {file_name} to Telegram via HTTP API!")
            emit_progress(100, PROGRESS_MILESTONES[5][1], video=True)
            return True
        logger.error(f"Telegram upload failed: {res.text[:300]}")
    except Exception as vid_err:
        logger.warning(f"Video delivery warning: {vid_err}")
    return False

def _deliver_rendered_video(sbx: Any, extra_dirs: Any = ()) -> bool:
    """Upload the first rendered .mp4 found in the sandbox straight to Telegram."""
    if not (CURRENT_CHAT_ID and TELEGRAM_BOT_TOKEN):
        return False
    for scan_dir in tuple(extra_dirs) + SANDBOX_VIDEO_SCAN_DIRS:
        video_path = _find_rendered_video(sbx, scan_dir)
        if not video_path:
            continue
        try:
            video_bytes = sbx.files.read(video_path, format="bytes")
        except Exception as vid_err:
            logger.warning(f"Video delivery warning: {vid_err}")
            continue
        if _telegram_send_video(video_bytes, video_path.rsplit("/", 1)[-1]):
            return True
    return False

def _provision_render_toolchain(sbx: Any, want_browser: bool = False) -> str:
    """Install the ffmpeg/frame stack, plus Node + Chrome when HyperFrames must render."""
    base = (
        "command -v ffmpeg >/dev/null 2>&1 || "
        "(sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg) ; "
        "pip install --quiet --no-cache-dir pillow \"numpy<2\" scipy edge-tts"
    )
    sbx.commands.run(base, timeout=0)
    if not want_browser:
        return "ffmpeg + python frame stack ready"
    # Node is only needed for the HyperFrames renderer, so it is installed on
    # this path alone rather than on every sandbox session.
    browser_cmd = (
        "sudo apt-get update -qq ; "
        f"sudo apt-get install -y -qq {CHROME_RUNTIME_PACKAGES} curl ca-certificates ; "
        "if ! command -v node >/dev/null 2>&1 || [ \"$(node -v | tr -d 'v' | cut -d. -f1)\" -lt 22 ]; then "
        "curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null 2>&1 ; "
        "sudo apt-get install -y -qq nodejs ; fi ; "
        f"npx --yes hyperframes@{HYPERFRAMES_VERSION} browser ensure"
    )
    res = sbx.commands.run(browser_cmd, timeout=0)
    exit_code = getattr(res, "exit_code", "unknown")
    logger.info(f"[HyperFrames] browser toolchain provisioning exit={exit_code}")
    return f"browser toolchain exit={exit_code}"

def _validate_composition_html(html: str) -> Optional[str]:
    """Return an error string when `html` cannot be a renderable HyperFrames composition."""
    if not isinstance(html, str) or not html.strip():
        return "the composition HTML is empty."
    if "data-composition-id" not in html:
        return (
            "no element carries data-composition-id, so HyperFrames has no composition "
            "to mount. See the contract in load_skill('hyperframes')."
        )
    if "__timelines" not in html:
        return (
            "no timeline is registered. Register exactly one paused timeline with "
            'window.__timelines["<data-composition-id>"] = gsap.timeline({paused: true}).'
        )
    if "moviepy" in html:
        return "the composition references moviepy, which is forbidden in this pipeline."
    return None

# ------------------------------------------------ Hugging Face Space transport
def _hf_space_path(path: str) -> str:
    return f"{HF_SPACE_URL}{path if path.startswith('/') else '/' + path}"


def _hf_space_headers(json_body: bool = True) -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    secret = HF_SPACE_SECRET or HF_TOKEN
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    return headers


def hf_space_health() -> str:
    """Readiness and hardware facts for the 16 GB render Space."""
    if not HF_SPACE_URL:
        return "Error: HF_SPACE_URL is not configured (expected https://<namespace>-chintu-render.hf.space)."
    try:
        res = requests.get(_hf_space_path("/health"), timeout=30)
    except Exception as e:
        return (
            f"16 GB render Space unreachable at {HF_SPACE_URL}: {e}. "
            "A Space rebuild takes a few minutes after a code push."
        )
    if res.status_code != 200:
        return f"Render Space health failed (HTTP {res.status_code}): {res.text[:400]}"
    try:
        return json.dumps(res.json(), indent=2, sort_keys=True)
    except Exception:
        return res.text[:2000]


def _hf_space_submit(html: str, slug: str, fps: int, quality: str, narration: str = "") -> str:
    """POST a composition to the 16 GB Space worker and return its job id."""
    if not HF_SPACE_URL:
        return (
            "Error: the 16 GB render Space is not configured. Set HF_SPACE_URL to the "
            "Space origin (https://<namespace>-chintu-render.hf.space) and redeploy Render."
        )
    payload = {
        "html": html,
        "project_slug": slug,
        "fps": int(fps),
        "quality": quality,
        "narration": (narration or "").strip(),
    }
    try:
        res = requests.post(
            _hf_space_path("/render"),
            json=payload,
            headers=_hf_space_headers(),
            timeout=90,
        )
    except Exception as e:
        return (
            f"16 GB render Space unreachable at {HF_SPACE_URL}: {e}\n"
            "The composition was NOT rendered. Check the Space is running, then retry "
            "or use render_html_video(..., backend=\"e2b\") for the isolated microVM."
        )
    if res.status_code in (401, 403):
        return (
            f"Render Space rejected the request (HTTP {res.status_code}): the shared secret "
            "does not match. HF_SPACE_SECRET on Render must equal HF_RENDER_SECRET on the Space."
        )
    if res.status_code == 503:
        return (
            "Render Space is starting or sleeping (HTTP 503). Wait for the build to finish "
            "and call render_html_video again; the job was not queued."
        )
    if res.status_code not in (200, 202):
        return f"Render Space rejected the composition (HTTP {res.status_code}): {res.text[:600]}"
    try:
        data = res.json()
    except Exception:
        return f"Render Space returned a non-JSON body: {res.text[:400]}"
    job_id = str((data or {}).get("job_id") or "").strip()
    if not job_id:
        return f"Render Space accepted the request but returned no job_id: {res.text[:300]}"
    emit_progress(80, PROGRESS_MILESTONES[3][1], video=True)
    return (
        "Hugging Face Space render submitted to the 16 GB worker.\n"
        f"HF_JOB_ID=space:{job_id}\n"
        f"engine: {HF_SPACE_URL} (cpu-basic: 2 vCPU / 16 GB RAM / 50 GB disk)\n"
        f"composition: {slug} | {len(html)} chars | {int(fps)} fps | quality={quality}\n"
        f"narration: {'edge-tts voice-over requested' if narration else 'none'}\n"
        f"Now call hf_render_status(job_id=\"space:{job_id}\", wait_seconds={HF_RENDER_WAIT_SECONDS}) "
        "to wait for the frames, then read the ffprobe facts and let the MP4 be delivered to Telegram."
    )


def _hf_space_status(job_id: str, wait_seconds: int = 0) -> str:
    """Poll one Space render job, download the MP4 and deliver it to Telegram."""
    if not HF_SPACE_URL:
        return "Error: HF_SPACE_URL is not configured, so no Space job can be polled."
    job_id = (job_id or "").replace("space:", "", 1).strip()
    if not job_id:
        return "Error: job_id is required; pass the space:<id> printed by render_html_video."
    budget = max(0, min(int(wait_seconds or 0), HF_SPACE_MAX_WAIT_SECONDS))
    deadline = time.monotonic() + budget
    snapshot: Dict[str, Any] = {}
    while True:
        try:
            res = requests.get(
                _hf_space_path(f"/status/{job_id}"),
                headers=_hf_space_headers(json_body=False),
                timeout=60,
            )
        except Exception as e:
            return f"Render Space became unreachable while polling {job_id}: {e}"
        if res.status_code == 404:
            return f"Unknown Space render job '{job_id}'. It may have been evicted; resubmit the render."
        if res.status_code in (401, 403):
            return f"Render Space refused the status poll (HTTP {res.status_code}): check HF_SPACE_SECRET."
        if res.status_code != 200:
            return f"Render Space status failed (HTTP {res.status_code}): {res.text[:400]}"
        try:
            snapshot = res.json() or {}
        except Exception:
            return f"Render Space returned a non-JSON status body: {res.text[:300]}"
        status = str(snapshot.get("status") or "running").lower()
        percent = int(snapshot.get("percent") or 0)
        if status == "running" and percent >= 55:
            emit_progress(min(95, percent), str(snapshot.get("stage") or ""), video=True)
        if status in ("completed", "failed") or time.monotonic() >= deadline:
            break
        time.sleep(HF_SPACE_POLL_SECONDS)

    status = str(snapshot.get("status") or "running").lower()
    lines = [
        f"Hugging Face Space render job {job_id}",
        f"status: {status.upper()}",
        f"engine: {HF_SPACE_URL} (cpu-basic, 16 GB RAM)",
    ]
    if status == "completed":
        probe = snapshot.get("probe") or {}
        if probe:
            lines.append(
                "[VERIFIED] {frames} frames | {width}x{height} @ {fps}fps | {duration}s | {size} bytes".format(
                    frames=probe.get("frames"),
                    width=probe.get("width"),
                    height=probe.get("height"),
                    fps=probe.get("fps"),
                    duration=probe.get("duration"),
                    size=probe.get("bytes"),
                )
            )
        emit_progress(95, PROGRESS_MILESTONES[4][1], video=True)
        try:
            art = requests.get(
                _hf_space_path(f"/artifact/{job_id}"),
                headers=_hf_space_headers(json_body=False),
                timeout=180,
            )
        except Exception as e:
            lines.append(f"[FAILED] Could not download the artifact: {e}")
        else:
            if art.status_code != 200 or not art.content:
                lines.append(f"[FAILED] Artifact download returned HTTP {art.status_code}.")
            elif b"ftyp" not in art.content[:64]:
                lines.append(f"[FAILED] {len(art.content)} bytes downloaded but it is not an MP4.")
            else:
                delivered = _telegram_send_video(
                    art.content,
                    f"chintu-{job_id}.mp4",
                    "🎬 Chintu · Hugging Face Space (16 GB) render",
                )
                lines.append(f"[SUCCESS] {len(art.content)} byte MP4 decoded and verified on the worker.")
                lines.append(
                    "Delivered to Telegram."
                    if delivered
                    else "(no Telegram chat bound; the artifact stays on the Space)."
                )
    elif status == "failed":
        lines.append(f"[FAILED] {snapshot.get('error') or 'the worker reported a failure'}")
        lines.append("Read the worker log tail, fix the cause, then resubmit or use backend='e2b'.")
    else:
        lines.append(
            "Still rendering. Call hf_render_status again with a wait_seconds budget to keep waiting."
        )
    log_tail = (snapshot.get("log_tail") or "")[-6000:]
    if log_tail:
        lines.append("--- worker log tail ---")
        lines.append(log_tail)
    return "\n".join(lines)


def _hf_jobs_api() -> Dict[str, Any]:
    """Import the Jobs client lazily so the agent still boots without it."""
    try:
        from huggingface_hub import cancel_job, fetch_job_logs, inspect_job, run_job, wait_for_job
    except Exception as e:
        return {
            "error": (
                f"Hugging Face Jobs client unavailable: {e}. "
                "Install or upgrade with: pip install -U 'huggingface_hub>=0.36'"
            )
        }
    return {"run": run_job, "inspect": inspect_job, "logs": fetch_job_logs, "wait": wait_for_job, "cancel": cancel_job}

def _hf_namespace() -> str:
    """Resolve the token's own namespace so artifacts get a home repository."""
    if _hf_namespace_cache.get("name"):
        return _hf_namespace_cache["name"]
    try:
        res = requests.get(
            f"{HF_ENDPOINT}/api/whoami-v2",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            timeout=15,
        )
        if res.status_code == 200:
            name = (res.json() or {}).get("name") or ""
            if name:
                _hf_namespace_cache["name"] = name
                return name
        return f"Error: could not resolve the Hugging Face namespace (HTTP {res.status_code})."
    except Exception as e:
        return f"Error: could not reach {HF_ENDPOINT} to resolve the namespace ({e})."

def _hf_artifact_repo() -> str:
    """The dataset repository that render artifacts are committed to."""
    if HF_ARTIFACT_REPO:
        return HF_ARTIFACT_REPO
    namespace = _hf_namespace()
    if namespace.startswith("Error"):
        return namespace
    return f"{namespace}/{HF_ARTIFACT_REPO_NAME}"

def _hf_ensure_dataset_repo(repo_id: str) -> str:
    """Create the artifact dataset when absent. Returns "" on success, else a message."""
    owner, _, name = repo_id.partition("/")
    payload: Dict[str, Any] = {"type": "dataset", "name": name or owner, "private": True}
    if name:
        payload["organization"] = owner
    try:
        res = requests.post(
            f"{HF_ENDPOINT}/api/repos/create",
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
    except Exception as e:
        return f"Error: could not reach {HF_ENDPOINT} to create the artifact repo ({e})."
    if res.status_code in (200, 201):
        logger.info(f"[HF] artifact dataset {repo_id} created (private)")
        return ""
    body = (res.text or "")[:200]
    # 409 means it already exists, which is the steady state for every later render.
    if res.status_code == 409 or "already" in body.lower() or "exists" in body.lower():
        return ""
    return (
        f"Error: could not create the artifact dataset {repo_id} (HTTP {res.status_code}: {body}). "
        "The token needs write access, or set HF_ARTIFACT_REPO to an existing dataset repo."
    )

def _hf_stage(info: Any) -> str:
    """Normalise a JobInfo status into a plain uppercase stage name."""
    status = getattr(info, "status", None)
    stage = getattr(status, "stage", None) if status is not None else None
    value = getattr(stage, "value", stage)
    return str(value or "UNKNOWN").upper()

def _hf_log_tail(api: Dict[str, Any], job_id: str, lines: int = 80, limit: int = 6000) -> str:
    """Fetch the tail of a Job's logs; they persist after the Job finishes."""
    try:
        chunks = [str(chunk) for chunk in api["logs"](job_id=job_id, tail=lines, token=HF_TOKEN)]
    except Exception as e:
        return f"(log fetch failed: {e})"
    return "".join(chunks)[-limit:]

def _hf_submit_render(html: str, slug: str, fps: int, quality: str) -> str:
    """Submit one render as a Hugging Face Job on the 16 GB cpu-basic flavor."""
    api = _hf_jobs_api()
    if "error" in api:
        return api["error"]
    repo_id = _hf_artifact_repo()
    if repo_id.startswith("Error"):
        return repo_id
    problem = _hf_ensure_dataset_repo(repo_id)
    if problem:
        return problem
    artifact_path = f"renders/{slug}-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.mp4"
    job_env = {
        "PROJECT_SLUG": slug,
        "RENDER_FPS": str(int(fps)),
        "RENDER_QUALITY": quality,
        "HYPERFRAMES_VERSION": HYPERFRAMES_VERSION,
        "COMPOSITION_B64": base64.b64encode(html.encode("utf-8")).decode("ascii"),
        "HYPERFRAMES_JSON_B64": base64.b64encode(
            json.dumps(HYPERFRAMES_PROJECT_CONFIG).encode("utf-8")
        ).decode("ascii"),
        "HF_ARTIFACT_REPO": repo_id,
        "HF_ARTIFACT_PATH": artifact_path,
        "HF_ENDPOINT": HF_ENDPOINT,
    }
    try:
        job = api["run"](
            image=HF_RENDER_IMAGE,
            command=["bash", "-lc", HF_RENDER_JOB_SCRIPT],
            env=job_env,
            secrets={"HF_TOKEN": HF_TOKEN},
            flavor=HF_RENDER_FLAVOR,
            timeout=HF_RENDER_TIMEOUT,
            name=HF_RENDER_JOB_NAME,
            labels={"app": "chintu", "kind": "motion-graphics"},
            token=HF_TOKEN,
        )
    except Exception as e:
        logger.error(f"Hugging Face job submission failed: {e}", exc_info=True)
        return (
            f"Hugging Face job submission failed: {e}\n"
            "Jobs need a positive compute-credit balance on the account, and an older "
            "client cannot submit them: pip install -U 'huggingface_hub>=0.36'."
        )
    job_id = getattr(job, "id", "") or ""
    job_url = getattr(job, "url", "") or f"{HF_ENDPOINT}/jobs/{job_id}"
    return (
        "Hugging Face Jobs render submitted.\n"
        f"HF_JOB_ID={job_id}\n"
        f"job url: {job_url}\n"
        f"hardware: {HF_RENDER_FLAVOR} (2 vCPU / 16 GB RAM / 50 GB disk, $0.01 per hour, billed per minute)\n"
        f"image: {HF_RENDER_IMAGE} | timeout: {HF_RENDER_TIMEOUT}\n"
        f"composition: {slug} | {len(html)} chars | {int(fps)} fps | quality={quality}\n"
        f"artifact once uploaded: {HF_ENDPOINT}/datasets/{repo_id}/resolve/main/{artifact_path}\n"
        "The container installs ffmpeg and Chrome's libraries, downloads the pinned Chrome, "
        "then lints and renders, so allow a few minutes. Now call "
        f"hf_render_status(job_id=\"{job_id}\", wait_seconds={HF_RENDER_WAIT_SECONDS}) to wait for "
        "it, read the logs and have the MP4 delivered to Telegram."
    )

def _hf_fetch_artifact(url: str) -> Optional[bytes]:
    """Download the rendered MP4 from the artifact dataset."""
    try:
        res = requests.get(
            url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=180, allow_redirects=True
        )
        if res.status_code == 200 and res.content:
            return res.content
        logger.error(f"[HF] artifact download failed: HTTP {res.status_code} {res.text[:200]}")
    except Exception as e:
        logger.warning(f"[HF] artifact download warning: {e}")
    return None

def hf_render_status(job_id: str, wait_seconds: int = 0) -> str:
    """Check either render backend and deliver the MP4 once it finished.

    Handles both transports: a `space:<id>` from the 16 GB Hugging Face Space
    (primary) and a plain Job id from the `cpu-basic` Hugging Face Job fallback.

    Args:
        job_id: The id printed as HF_JOB_ID by render_html_video (`space:<id>` for
            the Space worker, a bare id for a Job).
        wait_seconds: Block up to this many seconds for a terminal state.
            Use 0 to poll once and return immediately.
    """
    raw_id = (job_id or "").strip()
    if raw_id.startswith("space:") or ".hf.space" in raw_id:
        return _hf_space_status(raw_id, wait_seconds)
    if not HF_TOKEN:
        return "Error: HF_TOKEN is not configured."
    # Tolerate a pasted "HF_JOB_ID=<id>" or a job URL.
    job_id = (job_id or "").strip().rstrip("/").split("/")[-1].split("=")[-1].strip()
    if not job_id:
        return "Error: job_id is required; pass the HF_JOB_ID printed by render_html_video."
    api = _hf_jobs_api()
    if "error" in api:
        return api["error"]
    wait_seconds = max(0, min(int(wait_seconds or 0), 900))
    try:
        if wait_seconds:
            info = api["wait"](job_id, timeout=wait_seconds, token=HF_TOKEN)
        else:
            info = api["inspect"](job_id=job_id, token=HF_TOKEN)
    except Exception as e:
        return f"Hugging Face job lookup failed for {job_id}: {e}"
    stage = _hf_stage(info)
    tail = _hf_log_tail(api, job_id)
    job_url = getattr(info, "url", "") or f"{HF_ENDPOINT}/jobs/{job_id}"
    lines = [
        f"Hugging Face render job {job_id}",
        f"status: {stage}",
        f"url: {job_url}",
        f"hardware: {HF_RENDER_FLAVOR}",
    ]
    if stage == "COMPLETED":
        verified = re.search(r"RENDER_VERIFIED\s+(\S+)\s+(\d+)\s+bytes", tail)
        if verified:
            lines.append(f"[VERIFIED] {verified.group(1)} = {verified.group(2)} bytes of MP4")
        url_match = re.search(r"ARTIFACT_URL\s+(\S+)", tail)
        if not url_match:
            lines.append(
                "[FAILED] The job completed but printed no ARTIFACT_URL, so no MP4 was "
                "uploaded. Read the log tail below."
            )
        else:
            artifact_url = url_match.group(1)
            video_bytes = _hf_fetch_artifact(artifact_url)
            if not video_bytes:
                lines.append(f"[FAILED] Could not download {artifact_url} with the current HF_TOKEN.")
            elif b"ftyp" not in video_bytes[:32]:
                lines.append(f"[FAILED] {len(video_bytes)} bytes downloaded but it is not an MP4.")
            else:
                file_name = artifact_url.rsplit("/", 1)[-1]
                delivered = _telegram_send_video(
                    video_bytes, file_name, f"🎬 Chintu · Hugging Face ({HF_RENDER_FLAVOR}) render"
                )
                lines.append(f"[SUCCESS] {len(video_bytes)} byte MP4 at {artifact_url}")
                lines.append(
                    "delivered to Telegram." if delivered else "(no Telegram chat bound; artifact is on the Hub)."
                )
    elif stage in ("ERROR", "CANCELED", "DELETED"):
        lines.append(
            "[FAILED] The Job did not complete. Read the log tail below, fix the cause, then "
            "either resubmit with render_html_video or fall back to the sandbox backend with "
            'render_html_video(html, backend="e2b").'
        )
    else:
        lines.append(
            "Still running (the container installs its toolchain before rendering). Call "
            "hf_render_status again with a wait_seconds budget to keep waiting."
        )
    lines.append("--- job log tail ---")
    lines.append(tail)
    return "\n".join(lines)

def render_html_video(
    html: str,
    project_name: str = "chintu-motion",
    fps: int = 30,
    quality: str = "looks",
    backend: str = "auto",
    narration: str = "",
) -> str:
    """Render a HyperFrames HTML/GSAP composition to MP4 on the 16 GB engine.

    The render never happens on this process: Render runs the API, Telegram and
    LLM routing in ~512 MB, so the composition is dispatched to the dedicated
    16 GB Hugging Face Space (primary), a `cpu-basic` Hugging Face Job (when the
    Space is not configured) or an isolated E2B microVM (forced with
    backend='e2b', killed in a finally block). Every path lints and renders with
    the real `hyperframes` CLI (chrome-headless-shell + system ffmpeg), so
    seekable GSAP timelines and deterministic capture stay the framework's job.

    The Space and Job paths are ASYNCHRONOUS: this call returns a job id that is
    handed to `hf_render_status`, which polls, downloads and delivers the MP4.

    Args:
        html: The full composition document (root carries data-composition-id and
            registers window.__timelines["<id>"] as one paused GSAP timeline).
        project_name: Kebab-case project directory name.
        fps: Frame rate recorded on the render (24-240; 30 typical).
        quality: draft (fast iteration), looks (default) or delivery (final).
        backend: auto (default), space, huggingface (job) or e2b.
        narration: Optional voice-over script; the 16 GB worker synthesizes it
            with edge-tts and muxes it into the MP4.
    """
    backend = (backend or "auto").strip().lower()
    if backend not in ("auto", "space", "huggingface", "hf", "jobs", "e2b"):
        return (
            f"Unknown render backend '{backend}'. Use 'auto', 'space', 'huggingface' or 'e2b'."
        )
    problem = _validate_composition_html(html)
    if problem:
        return f"Composition rejected before render: {problem}"
    slug = re.sub(r"[^a-z0-9-]+", "-", (project_name or "chintu-motion").lower()).strip("-")
    slug = slug or "chintu-motion"
    # Primary: the dedicated 16 GB Space worker.
    if backend in ("auto", "space"):
        if HF_SPACE_URL:
            emit_progress(55, PROGRESS_MILESTONES[2][1], video=True)
            return _hf_space_submit(html, slug, fps, quality, narration)
        if backend == "space":
            return (
                "Error: HF_SPACE_URL is not configured, so the 16 GB Space backend is "
                "unavailable. Set HF_SPACE_URL to the Space origin and redeploy Render."
            )
    # Secondary: a cpu-basic Hugging Face Job (also 2 vCPU / 16 GB RAM).
    if backend in ("auto", "huggingface", "hf", "jobs") and HF_TOKEN:
        emit_progress(55, PROGRESS_MILESTONES[2][1], video=True)
        return _hf_submit_render(html, slug, fps, quality)
    if backend in ("huggingface", "hf", "jobs"):
        return "Error: HF_TOKEN is not configured, so the Hugging Face Jobs backend is unavailable."
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY is not configured."
    emit_progress(55, PROGRESS_MILESTONES[2][1], video=True)
    project_dir = f"/home/user/videos/{slug}"
    sbx = None
    try:
        from e2b_code_interpreter import Sandbox

        logger.info(f"Starting E2B sandbox for HyperFrames render of '{slug}'...")
        sbx = Sandbox.create(api_key=E2B_API_KEY, timeout=SANDBOX_TIMEOUT_SECONDS)
        provisioning = _provision_render_toolchain(sbx, want_browser=True)
        sbx.commands.run(f"mkdir -p {project_dir}/renders", timeout=0)
        sbx.files.write(f"{project_dir}/index.html", html)
        sbx.files.write(
            f"{project_dir}/hyperframes.json",
            json.dumps(HYPERFRAMES_PROJECT_CONFIG, indent=2),
        )

        hf = f"npx --yes hyperframes@{HYPERFRAMES_VERSION}"
        lint = sbx.commands.run(f"cd {project_dir} && {hf} lint .", timeout=0)
        logger.info(f"[HyperFrames] lint exit={getattr(lint, 'exit_code', 'unknown')}")
        render = sbx.commands.run(
            f"cd {project_dir} && {hf} render . --skill={HYPERFRAMES_SKILL} "
            f"-f {int(fps)} -q {quality} -o ./renders/video.mp4",
            timeout=0,
        )

        delivered = _deliver_rendered_video(sbx, extra_dirs=(f"{project_dir}/renders",))
        rendered_path = _find_rendered_video(sbx, f"{project_dir}/renders")
        report = (
            f"HyperFrames render :: {slug}\n"
            f"provisioning: {provisioning}\n"
            f"lint exit: {getattr(lint, 'exit_code', 'unknown')}\n"
            f"lint output:\n{(getattr(lint, 'stdout', '') or '')[-1200:]}\n"
            f"render exit: {getattr(render, 'exit_code', 'unknown')}\n"
            f"render output:\n{(getattr(render, 'stdout', '') or '')[-2000:]}\n"
        )
        if getattr(render, "stderr", None):
            report += f"render stderr:\n{render.stderr[-1200:]}\n"
        if rendered_path:
            report += f"[SUCCESS] MP4 written to {rendered_path}"
            report += " and delivered to Telegram." if delivered else " (no Telegram chat bound)."
        else:
            report += "[FAILED] No .mp4 produced; read the render output above before retrying."
        return report
    except Exception as e:
        logger.error(f"HyperFrames render error: {e}", exc_info=True)
        return f"HyperFrames render error:\n{traceback.format_exc()}"
    finally:
        if sbx is not None:
            try:
                sbx.kill()
            except Exception as kill_err:
                logger.warning(f"Sandbox cleanup warning: {kill_err}")

def execute_in_cloud_microvm(code: str) -> str:
    global CURRENT_CHAT_ID
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY is not configured."
    emit_progress(55, PROGRESS_MILESTONES[2][1], video=True)
    sbx = None
    try:
        from e2b_code_interpreter import Sandbox

        logger.info("Starting official E2B Sandbox session...")
        # Lifetime honours E2B_SANDBOX_TIMEOUT so long renders are not cut off mid-encode.
        sbx = Sandbox.create(api_key=E2B_API_KEY, timeout=SANDBOX_TIMEOUT_SECONDS)

        # Provisioning without command timeout limits (timeout=0)
        _provision_render_toolchain(sbx, want_browser=False)

        # Pre-validate/harden the task source before it ever reaches the VM.
        safe_source = prepare_task_source(code)
        sbx.files.write("/home/user/run_task.py", safe_source)
        exec_res = sbx.commands.run("python /home/user/run_task.py", timeout=0)

        found_video = _deliver_rendered_video(sbx)

        out_msg = f"STDOUT:\n{exec_res.stdout}\nSTDERR:\n{exec_res.stderr}"
        if found_video:
            out_msg += "\n\n[SUCCESS] Rendered .mp4 file delivered directly to Telegram!"
        return out_msg
    except Exception as e:
        logger.error(f"E2B Execution error: {e}", exc_info=True)
        return f"Sandbox Execution Error:\n{traceback.format_exc()}"
    finally:
        if sbx is not None:
            try:
                sbx.kill()
            except Exception as kill_err:
                logger.warning(f"Sandbox cleanup warning: {kill_err}")

def extract_video_knowledge(url: str) -> str:
    try:
        yt_id_match = re.search(r"(?:v=|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", url)
        if yt_id_match:
            video_id = yt_id_match.group(1)
            meta_url = f"https://yt.lemnoslife.com/videos?part=snippet&id={video_id}"
            t_res = requests.get(meta_url, timeout=10)
            if t_res.status_code == 200:
                data = t_res.json()
                items = data.get("items", [])
                if items:
                    snippet = items[0].get("snippet", {})
                    return f"YOUTUBE METADATA EXTRACTED:\nTitle: {snippet.get('title')}\nDescription: {snippet.get('description', '')[:1200]}"
        return f"Video extracted from URL: {url}"
    except Exception as e:
        return f"Extraction note: {str(e)}"

def supabase_db_query(table: str, operation: str = "select", query_filter: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> str:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "Supabase credentials not configured in environment."
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    try:
        if operation == "select":
            query_str = f"?{query_filter}" if query_filter else "?select=*&order=created_at.desc&limit=20"
            res = requests.get(f"{url}{query_str}", headers=headers, timeout=10)
            return json.dumps(res.json())
        elif operation == "insert" and data:
            res = requests.post(url, headers=headers, json=data, timeout=10)
            return f"Insert status: {res.status_code}"
        elif operation == "delete" and query_filter:
            res = requests.delete(f"{url}?{query_filter}", headers=headers, timeout=10)
            return f"Delete status: {res.status_code} (Filter: {query_filter})"
        return f"Unsupported operation: {operation}"
    except Exception as e:
        return f"Supabase error: {str(e)}"

def huggingface_task_runner(model_id: str, inputs: Any) -> str:
    if not HF_TOKEN:
        return "HF_TOKEN not configured in environment."
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    try:
        payload = {"inputs": inputs} if isinstance(inputs, str) else inputs
        res = requests.post(url, headers=headers, json=payload, timeout=25)
        if res.status_code == 200:
            return json.dumps(res.json())
        return f"HF Error ({res.status_code}): {res.text}"
    except Exception as e:
        return f"HuggingFace runner error: {str(e)}"

def render_service_manager(action: str = "status") -> str:
    if not RENDER_API_KEY:
        return "RENDER_API_KEY not configured in environment."
    headers = {"Authorization": f"Bearer {RENDER_API_KEY}", "Accept": "application/json"}
    base_url = "https://api.render.com/v1/services"
    try:
        if action == "status":
            url = f"{base_url}/{RENDER_SERVICE_ID}" if RENDER_SERVICE_ID else base_url
            res = requests.get(url, headers=headers, timeout=10)
            return json.dumps(res.json())
        elif action == "restart":
            if not RENDER_SERVICE_ID:
                return "RENDER_SERVICE_ID required for restart action."
            res = requests.post(f"{base_url}/{RENDER_SERVICE_ID}/restart", headers=headers, timeout=10)
            return f"Restart response: {res.status_code}"
        return f"Unsupported Render action: {action}"
    except Exception as e:
        return f"Render API error: {str(e)}"

def diagnose_and_heal_codebase(error_traceback: str, target_file: str = "main.py") -> str:
    try:
        current_code = github_read_file(DEFAULT_REPO, target_file)
        if "Error" in current_code[:30]:
            return f"Heal failed: could not read {target_file}"
        try:
            ast.parse(current_code)
        except SyntaxError as syn_err:
            return f"Syntax defect identified at line {syn_err.lineno}: {syn_err.msg}"
        test_script = f'''
import ast
try:
    with open("{target_file}", "w") as f:
        f.write({repr(current_code)})
    print("Code staging verified.")
except Exception as e:
    print("Staging error:", e)
'''
        sb_result = execute_in_cloud_microvm(test_script)
        if SUPABASE_URL and SUPABASE_KEY:
            supabase_db_query(
                table="audit_logs",
                operation="insert",
                data={
                    "event": "autonomous_self_heal_attempt",
                    "target_file": target_file,
                    "traceback": error_traceback[:1000],
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        return f"Self-Healing Diagnostic Complete:\nSandbox verify: {sb_result}"
    except Exception as e:
        return f"Self-healing diagnostic failed: {str(e)}"

def create_golden_state_backup(tag: str = "stable_snapshot") -> str:
    current_code = github_read_file(DEFAULT_REPO, "main.py")
    if "Error" in current_code[:30]:
        return f"Backup failed: Unable to read main.py -> {current_code}"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_file = f"backups/main_{tag}_{timestamp}.py"
    res = github_commit_file(DEFAULT_REPO, backup_file, current_code, f"Create Golden State backup: {tag}")
    return f"BACKUP COMPLETE: Created {backup_file}. Commit status: {res}"

_HTML_BLOCK_RE = re.compile(r"(?is)<(script|style|head|noscript)[^>]*>.*?</\1>")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_GAP_RE = re.compile(r"\n{3,}")

def _strip_html(raw: str, limit: int = 2000) -> str:
    """Reduce a scraped HTML page to readable text so it cannot flood model context."""
    text = _HTML_BLOCK_RE.sub(" ", raw or "")
    text = _HTML_TAG_RE.sub(" ", text)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&quot;", '"'), ("&#x27;", "'"), ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">")):
        text = text.replace(entity, char)
    text = re.sub(r"[ \t]+", " ", text)
    text = _HTML_GAP_RE.sub("\n\n", text).strip()
    return text[:limit]

def web_search_research(query: str) -> str:
    """Search the web and return readable text (HTML is stripped from the result)."""
    try:
        url = "https://html.duckduckgo.com/html/"
        res = requests.post(url, data={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if res.status_code != 200:
            return f"Search error: HTTP {res.status_code} returned by DuckDuckGo."
        return _strip_html(res.text)
    except Exception as e:
        return f"Search error: {str(e)}"

def list_skills() -> str:
    """List every external skill/knowledge repository Chintu can draw on."""
    emit_progress(30, PROGRESS_MILESTONES[1][1])
    lines = []
    for name, meta in SKILL_REGISTRY.items():
        docs = meta.get("docs") or [meta.get("entrypoint", "README.md")]
        lines.append(
            f"- {name} [{meta['role']}] -> {meta['repo']}\n"
            f"  {meta['summary']}\n"
            f"  docs: {', '.join(docs)}"
        )
    return "AVAILABLE SKILLS:\n" + "\n".join(lines)

def load_skill(
    skill_name: str,
    file_path: str = "",
    offset: int = 0,
    max_chars: int = 0,
) -> str:
    """Read documentation or source from one registered skill repository.

    Long documents are returned in explicit, self-describing pages: the response
    states the total length and the exact `offset` to pass for the next slice, so
    a large file (the HyperFrames CLI skill is ~26k chars, the prompt library's
    README ~91k) is never silently cut off mid-sentence.

    Args:
        skill_name: Registry key, e.g. 'hyperframes', 'gsap', 'whisper',
            'motion_graphics_prompts', 'motion_skills' or 'remotion'.
        file_path: Optional repo-relative path. Defaults to the skill's entrypoint;
            a trailing '/' lists that directory.
        offset: Character offset to start from, for paging through a long file.
        max_chars: Slice size for this call. Defaults to the skill's max_chars.
    """
    key = (skill_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    meta = SKILL_REGISTRY.get(key)
    if not meta:
        return f"Unknown skill '{skill_name}'. Available skills: {', '.join(SKILL_REGISTRY)}"
    target = (file_path or meta.get("entrypoint") or "README.md").strip().strip("/")
    emit_progress(30, PROGRESS_MILESTONES[1][1])
    content = github_read_file(meta["repo"], target)
    if content.startswith("GitHub Read Error") or content.startswith("Error reading"):
        return content
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        offset = 0
    # Per-skill budget first (the prompt library needs far more than a SKILL.md),
    # then the explicit per-call override, then the global default.
    try:
        skill_budget = int(meta.get("max_chars") or DEFAULT_SKILL_MAX_CHARS)
    except (TypeError, ValueError):
        skill_budget = DEFAULT_SKILL_MAX_CHARS
    try:
        limit = int(max_chars) if max_chars else skill_budget
    except (TypeError, ValueError):
        limit = skill_budget
    limit = limit if limit > 0 else DEFAULT_SKILL_MAX_CHARS
    total = len(content)
    window = content[offset:offset + limit]
    header = f"SKILL '{key}' :: {meta['repo']} :: {target}\n"
    if offset == 0 and total <= limit:
        return f"{header}{window}"
    shown_end = min(offset + limit, total)
    header += f"[chars {offset}-{shown_end} of {total}]\n"
    if shown_end < total:
        header += (
            f"[TRUNCATED — continue with load_skill('{key}', '{target}', "
            f"offset={shown_end}, max_chars={limit})]\n"
        )
    return f"{header}{window}"

ALL_TOOLS_LIST = [
    diagnose_and_heal_codebase,
    github_read_file,
    github_commit_file,
    execute_in_cloud_microvm,
    extract_video_knowledge,
    supabase_db_query,
    huggingface_task_runner,
    render_service_manager,
    create_golden_state_backup,
    web_search_research,
    list_skills,
    load_skill,
    render_html_video,
    hf_render_status,
    hf_space_health,
]

# One line per tool, used to build the self-inventory in SYSTEM_INSTRUCTION. The
# inventory is generated from ALL_TOOLS_LIST, so a tool cannot be added without
# the model being told it exists.
TOOL_CAPABILITIES: Dict[str, str] = {
    "diagnose_and_heal_codebase": "Read a target file, AST-check it, stage it in the isolated microVM and record the self-heal attempt in Supabase.",
    "github_read_file": "Read any file or directory listing from a GitHub repo (raw CDN first, API fallback, cached).",
    "github_commit_file": "Commit a file to GitHub main; Python payloads are ast.parse-gated before the API call.",
    "execute_in_cloud_microvm": "Run Python or shell in an isolated E2B microVM with ffmpeg, Pillow, NumPy and edge-tts; always killed in a finally block.",
    "extract_video_knowledge": "Pull title/description metadata from a video URL for reference or narration.",
    "supabase_db_query": "Select, insert or delete rows in Supabase for audits, memory and render records.",
    "huggingface_task_runner": "Call a Hugging Face inference endpoint (transcription, classification, generation).",
    "render_service_manager": "Inspect or restart the Render service through its API.",
    "create_golden_state_backup": "Snapshot main.py into backups/ on GitHub before risky changes.",
    "web_search_research": "Search the web and return readable text for grounding.",
    "list_skills": "Enumerate the motion-graphics skill library with repos and doc paths.",
    "load_skill": "Load and page the real documentation of one skill (HyperFrames, GSAP, Whisper, Remotion, prompt and motion libraries).",
    "render_html_video": "Dispatch a single-file HTML/CSS/GSAP composition to the 16 GB engine (Space, then Job, then E2B) and return a job id.",
    "hf_render_status": "Poll a render (space:<id> or Job id), read ffprobe facts and worker logs, download the MP4 and deliver it to Telegram.",
    "hf_space_health": "Probe the 16 GB Space: CPU/RAM, Chromium/FFmpeg readiness, running jobs.",
}

_MISSING_TOOL_DOCS = [fn.__name__ for fn in ALL_TOOLS_LIST if fn.__name__ not in TOOL_CAPABILITIES]
if _MISSING_TOOL_DOCS:
    logger.warning(f"[Tools] missing capability documentation: {_MISSING_TOOL_DOCS}")

TOOL_INVENTORY = "\n".join(
    "- `{}` — {}".format(
        fn.__name__,
        TOOL_CAPABILITIES.get(
            fn.__name__, (fn.__doc__ or "No description.").strip().splitlines()[0]
        ),
    )
    for fn in ALL_TOOLS_LIST
)

SKILL_INVENTORY = "\n".join(
    "- `{}` ({}) — {}".format(name, meta["repo"], meta["summary"].split(". ")[0] + ".")
    for name, meta in SKILL_REGISTRY.items()
)

SYSTEM_INSTRUCTION = f"""You are Chintu — an autonomous audiovisual motion engineer, running as a distributed agent.

IDENTITY AND COMPUTE TOPOLOGY (hard boundaries)
- You are the reasoning layer. Your process is the Render brain: about 512 MB of RAM hosting FastAPI, the Telegram polling lifecycle, the status card, tool routing and LLM orchestration. That is ALL it does.
- You must NEVER render video locally. No Pillow frame buffers, no Playwright/Chromium launch, no FFmpeg subprocess, no NumPy frame loops in this process. A local render is a failed turn, not a shortcut.
- Every heavy media workload — HTML/CSS layout evaluation, headless Chromium capture, GSAP timeline scrubbing, high-resolution frame buffers, audio mixing and FFmpeg 1080x1920 stitching — belongs to the 16 GB engine: the dedicated Hugging Face Docker Space (cpu-basic: 2 vCPU / 16 GB RAM / 50 GB disk), or, when that is unavailable, the isolated E2B microVM (always killed in a finally block).
- Delivery: the finished MP4 streams back to the brain and is posted straight to Telegram with /sendVideo to CURRENT_CHAT_ID. You do not need to save it anywhere.
- Tool results are the only source of truth about what happened.

YOUR TOOL INVENTORY (you have {len(ALL_TOOLS_LIST)} tools)
{TOOL_INVENTORY}

YOUR SKILL LIBRARY (load the real docs before you author)
{SKILL_INVENTORY}

ACTION RULE — VIDEO REQUESTS
When the user asks to build, change or render any video, motion graphic, reel, ad or animation, you MUST act, in this order:
1. load_skill("hyperframes") for the composition contract and CLI, then load_skill("motion_skills") and the relevant `hyperframes` category module for the design pattern. Use load_skill("gsap") for timeline and easing questions, load_skill("whisper") for caption timing, load_skill("remotion") for composition and data-driven concepts, and load_skill("motion_graphics_prompts") for camera and lighting language.
2. Write the COMPLETE single-file composition: one root with data-composition-id, data-width, data-height and data-duration; timed elements carrying data-start and data-duration; exactly one paused GSAP timeline registered on window.__timelines under the composition id. Inline everything — no local assets, no network fetches at render time.
3. Call render_html_video(html=..., project_name=..., fps=30, quality="looks") and pass narration=... when there is a script. It returns an id; then call hf_render_status(job_id=..., wait_seconds=600) until the worker finishes, and report the ffprobe facts and delivery status it printed.

NEVER answer a video request with a conversational plan, an apology, or a claim that browser/Node/rendering tools are unsupported or unavailable — they are supported and remote. Never claim a render happened without a tool result that shows it. If a dispatch fails (missing secret, sleeping Space, failed job), read the log tail it returned, fix the cause and resubmit; fall back to render_html_video(..., backend="e2b") only after the primary engine reports a real failure.

VIDEO STANDARDS (non-negotiable for every render)
- Canvas 1080x1920 (9:16 vertical), 30 fps unless the brief says otherwise, 6-15 seconds for short-form.
- Headlines at 80pt or larger; body copy at least 32pt. Keep an 80px title-safe margin and never place type inside the top 120px or the bottom 220px where platform UI covers it.
- Typography carries the message: one or two emphasis words per scene, deliberate weights, and letter-spacing that snaps to whole pixels rather than tweening.
- Motion uses real easing curves (power2-4, back, expo, sine) with entrances and exits timed so transitions never overlap. One motion motif per scene. Never hand-animate a CSS transform on a node GSAP is tweening.
- Build ambient depth: a soft radial or gradient glow behind the subject, a subtle vignette, and a gradient sweep or grain. Never a flat black frame.
- Sound: narration is synthesized with edge-tts and muxed by the worker; pair impacts with a sub-bass hit and keep voice-over timed to the scene cuts.

TRUTHFUL REPORTING
Report only what the tools returned: job ids, ffprobe frame counts, byte sizes, delivery status. If a step did not run, say which step and why. Never invent a file path or a result."""

def _sync_chat_call(model_name: str, message: str) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=ALL_TOOLS_LIST,
        temperature=0.2
    )
    client = get_current_gemini_client()
    if not client:
        raise RuntimeError("No Gemini API client configured.")
    chat = client.chats.create(model=model_name, config=config)
    resp = chat.send_message(message)
    if resp.text:
        return resp.text
    if resp.candidates:
        parts_out = []
        for part in resp.candidates[0].content.parts:
            if getattr(part, "text", None):
                parts_out.append(part.text)
        if parts_out:
            return "".join(parts_out)
    return "Action completed successfully."

# ------------------------------------------------ external tool-calling bridge
# Gemini receives Python callables directly. OpenAI-compatible providers need
# JSON Schema, so the same ALL_TOOLS_LIST is serialised once into the exact
# function shape OpenRouter/Freebuff accept — no hand-maintained duplicate list.
_JSON_SCALAR_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _json_schema(annotation: Any) -> Dict[str, Any]:
    """Map one Python type hint onto a JSON Schema fragment."""
    if annotation in _JSON_SCALAR_TYPES:
        return {"type": _JSON_SCALAR_TYPES[annotation]}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union:
        non_none = [arg for arg in args if arg is not type(None)]
        return _json_schema(non_none[0]) if len(non_none) == 1 else {}
    if origin is dict:
        return {"type": "object"}
    if origin in (list, tuple, set):
        return {"type": "array", "items": _json_schema(args[0]) if args else {}}
    return {}


def tool_schema(fn: Any) -> Dict[str, Any]:
    """Describe one Python tool as an OpenAI-compatible function schema."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[name] = _json_schema(hints.get(name, str))
        if param.default is inspect.Parameter.empty:
            required.append(name)
    description = (inspect.getdoc(fn) or "").strip()
    if len(description) > 900:
        description = description[:900] + "…"
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def build_tool_schemas() -> List[Dict[str, Any]]:
    return [tool_schema(fn) for fn in ALL_TOOLS_LIST]


TOOL_REGISTRY: Dict[str, Any] = {fn.__name__: fn for fn in ALL_TOOLS_LIST}


def execute_tool_call(name: str, arguments: Any) -> str:
    """Run one tool call from an external provider and return its text result."""
    fn = TOOL_REGISTRY.get((name or "").strip())
    if fn is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOL_REGISTRY)}"
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as parse_err:
            return f"Error: arguments for {name} are not valid JSON: {parse_err}"
    else:
        args = arguments or {}
    if not isinstance(args, dict):
        return f"Error: arguments for {name} must be a JSON object."
    try:
        return str(fn(**args))
    except TypeError as arg_err:
        return f"Error: bad arguments for {name}: {arg_err}"
    except Exception as tool_err:
        logger.error(f"[Tool {name}] failed: {tool_err}", exc_info=True)
        return f"Error: {name} failed: {tool_err}"


def _chat_with_tools(
    base_url: str,
    api_key: str,
    model: str,
    message: str,
    provider: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    """One OpenAI-compatible conversation, executing tool calls until it answers."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    headers.update(extra_headers or {})
    tools = build_tool_schemas()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": message},
    ]
    last_tool_output = ""
    url = f"{base_url.rstrip('/')}/chat/completions"
    for _ in range(EXTERNAL_MAX_TOOL_ROUNDS):
        body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        res = requests.post(url, headers=headers, json=body, timeout=EXTERNAL_REQUEST_TIMEOUT)
        if res.status_code != 200:
            raise RuntimeError(f"{provider} HTTP {res.status_code}: {res.text[:500]}")
        try:
            choices = (res.json() or {}).get("choices") or []
        except Exception as parse_err:
            raise RuntimeError(f"{provider} returned unparseable JSON: {parse_err}")
        if not choices:
            raise RuntimeError(f"{provider} returned no choices.")
        reply = choices[0].get("message") or {}
        tool_calls = reply.get("tool_calls") or []
        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                fn = (call or {}).get("function") or {}
                name = fn.get("name") or ""
                logger.info(f"[{provider}] executing tool {name}")
                output = execute_tool_call(name, fn.get("arguments"))
                last_tool_output = output
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (call or {}).get("id") or name,
                        "content": output[:12000],
                    }
                )
            continue
        content = reply.get("content")
        if content:
            return content
        raise RuntimeError(f"{provider} returned an empty completion.")
    return (
        f"{provider} stopped after {EXTERNAL_MAX_TOOL_ROUNDS} tool rounds without a final "
        f"answer. Last tool output:\n{last_tool_output[:2000]}"
    )


# External provider failover used only after the entire Gemini cascade is exhausted.
def _sync_external_fallback(message: str) -> str:
    """Fail over to OpenRouter frontier models, then Freebuff as a last resort."""
    attempts: List[str] = []
    PROVIDER_TELEMETRY["failovers"] = int(PROVIDER_TELEMETRY.get("failovers") or 0) + 1

    if OPENROUTER_API_KEY:
        or_headers = {
            "HTTP-Referer": "https://framify-agent.onrender.com",
            "X-Title": "framify-agent",
        }
        for model in OPENROUTER_MODELS:
            try:
                content = _chat_with_tools(
                    OPENROUTER_BASE_URL, OPENROUTER_API_KEY, model, message, "openrouter", or_headers
                )
                record_provider("openrouter", model)
                return content
            except Exception as or_err:
                attempts.append(f"openrouter/{model}: {or_err}")
                logger.warning(f"[External Fallback] openrouter/{model} failed: {or_err}")

    if FREEBUFF_API_KEY:
        for model in OPENROUTER_MODELS:
            try:
                content = _chat_with_tools(
                    FREEBUFF_BASE_URL, FREEBUFF_API_KEY, model, message, "freebuff"
                )
                record_provider("freebuff", model)
                return content
            except Exception as fb_err:
                attempts.append(f"freebuff/{model}: {fb_err}")
                logger.warning(f"[External Fallback] freebuff/{model} failed: {fb_err}")

    if attempts:
        raise RuntimeError("; ".join(attempts[-3:]))
    raise RuntimeError("No external fallback provider configured (set OPENROUTER_API_KEY).")
# Autonomous code-driven model failover engine
async def generate_reasoning_response(user_message: str) -> str:
    global current_key_index
    if not GEMINI_API_KEYS:
        logger.warning("[Cascade Engine] No Gemini keys configured; using external fallback directly.")
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_sync_external_fallback, user_message), timeout=600.0
            )
        except Exception as ext_err:
            return f"Gemini API client not initialized and external fallback failed: {ext_err}"

    last_error = ""
    # True once every Gemini key slot failed with an infra/quota status
    # (429/500/503/UNAVAILABLE/INTERNAL), which mandates external failover.
    infra_failure = False

    # Iterate through models. If a model succeeds, STOP and return immediately.
    for model_name in ACTIVE_GEMINI_MODELS:
        logger.info(f"[Cascade Engine] Attempting model: {model_name}...")

        # Try each available API key for the current model
        for key_slot in range(len(GEMINI_API_KEYS)):
            try:
                logger.info(f"[Cascade Engine] Testing {model_name} (API Key #{current_key_index + 1})")
                # Extended 600s window (10 minutes) for heavy video tasks
                reply = await asyncio.wait_for(
                    asyncio.to_thread(_sync_chat_call, model_name, user_message),
                    timeout=600.0
                )

                # If the model responds cleanly, we stop switching and return
                if reply and not any(err_word in reply.lower() for err_word in ["resource_exhausted", "503 unavailable"]):
                    logger.info(f"[Cascade Engine] SUCCESS: Model {model_name} is active and fulfilled request.")
                    record_provider("gemini", model_name)
                    return reply

                logger.warning(f"[Cascade Engine] Model {model_name} returned unserviceable response. Failing over...")
                last_error = f"{model_name}: unserviceable response"
                rotate_to_next_gemini_key()
                continue

            except asyncio.TimeoutError:
                logger.warning(f"[Cascade Engine] Model {model_name} timed out (>600s). Moving to next model.")
                last_error = f"{model_name}: Execution timed out (>600s)"
                break  # Move to next model

            except Exception as e:
                err_str = str(e)
                last_error = f"{model_name}: {err_str}"
                logger.warning(f"[Cascade Engine] Model {model_name} encountered error: {err_str}")

                # 429 quota -> rotate key and retry the same model on the next slot
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    infra_failure = True
                    rotate_to_next_gemini_key()
                    continue

                # 500/503 and other infra failures -> note and move to the next model
                if any(code in err_str.upper() for code in RETRYABLE_GEMINI_STATUS):
                    infra_failure = True
                break

    # Every Gemini model/key slot failed. If the failures were infra/quota
    # (503/500/429) fail over immediately to the external provider.
    logger.warning(
        f"[Cascade Engine] Gemini cascade exhausted (infra_failure={infra_failure}). "
        "Failing over to _sync_external_fallback."
    )
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_sync_external_fallback, user_message), timeout=600.0
        )
    except Exception as ext_err:
        last_error = f"{last_error}; external fallback: {ext_err}"
        return f"All verified cascade models and external fallback failed. Details:\n{last_error}"

api = FastAPI(title="Chintu HMAS AI Engineer")

@api.get("/")
@api.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "framify-agent",
        "compute_role": COMPUTE_ROLE,
        "brain_boundary_violations": BRAIN_BOUNDARY_VIOLATIONS,
        "active_models": ACTIVE_GEMINI_MODELS,
        "openrouter_cascade": OPENROUTER_MODELS,
        "last_provider": PROVIDER_TELEMETRY,
        "tools_count": len(ALL_TOOLS_LIST),
        "skills": list(SKILL_REGISTRY.keys()),
        "skills_count": len(SKILL_REGISTRY),
        "render_backends": {
            "primary": "huggingface_space" if HF_SPACE_URL else ("huggingface_jobs" if HF_TOKEN else "e2b_sandbox"),
            "huggingface_space": bool(HF_SPACE_URL),
            "hf_space_url": HF_SPACE_URL,
            "hf_space_max_wait": HF_SPACE_MAX_WAIT_SECONDS,
            "huggingface_jobs": bool(HF_TOKEN),
            "e2b_sandbox": bool(E2B_API_KEY),
            "hf_flavor": HF_RENDER_FLAVOR,
            "hf_image": HF_RENDER_IMAGE,
            "hf_timeout": HF_RENDER_TIMEOUT,
            "hf_artifact_repo": HF_ARTIFACT_REPO or f"<namespace>/{HF_ARTIFACT_REPO_NAME}",
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@api.post("/render-webhook")
async def render_webhook_listener(request: Request, bg_tasks: BackgroundTasks):
    try:
        body = await request.json()
        logger.info(f"[Render Webhook Event]: {body}")
        event_type = body.get("type", "")
        if "deploy" in event_type and body.get("status") in ["failed", "canceled", "error"]:
            logger.warning("[Self-Healing Triggered] Render deploy failure detected.")
            bg_tasks.add_task(diagnose_and_heal_codebase, error_traceback=f"Render deploy failure: {body}")
        return {"status": "received", "event": event_type}
    except Exception as e:
        logger.error(f"Render webhook parsing error: {e}")
        return {"status": "error", "message": str(e)}

@api.post("/telemetry-error")
async def internal_error_listener(request: Request, bg_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        err_msg = payload.get("error", "Unknown runtime exception")
        logger.warning(f"[Telemetry Error Ingested]: {err_msg}")
        bg_tasks.add_task(diagnose_and_heal_codebase, error_traceback=err_msg)
        return {"status": "queued_for_healing"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

telegram_app = None

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("Chintu HMAS AI Engineer online. Systems operational.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_CHAT_ID, CURRENT_STATUS_MSG, TELEGRAM_BOT_INSTANCE, CURRENT_EVENT_LOOP
    user_msg = update.message.text
    if not user_msg:
        return
    CURRENT_CHAT_ID = update.message.chat_id
    TELEGRAM_BOT_INSTANCE = context.bot
    try:
        CURRENT_EVENT_LOOP = asyncio.get_running_loop()
    except RuntimeError:
        CURRENT_EVENT_LOOP = None
    logger.info(f"Incoming message: {user_msg}")

    # 1. Post the live progress card immediately at 15%.
    reset_progress()
    try:
        CURRENT_STATUS_MSG = await update.message.reply_text(
            f"Task/Project Progress\n[{progress_bar(15)}] 15%\n\n"
            f"Current status: {PROGRESS_MILESTONES[0][1]}"
        )
        PROGRESS_STATE.update(percent=15, stage=PROGRESS_MILESTONES[0][1])
    except Exception as e:
        logger.warning(f"Could not dispatch initial status card: {e}")
        CURRENT_STATUS_MSG = None

    # 2. Execute the fail-safe cascade with a heartbeat, so a long reasoning or
    #    render phase can never leave the card frozen at 15%.
    heartbeat = asyncio.create_task(progress_heartbeat())
    try:
        reply = await generate_reasoning_response(user_msg)
    except Exception as reasoning_err:
        logger.error(f"Reasoning pipeline error: {reasoning_err}", exc_info=True)
        reply = f"Reasoning pipeline error: {reasoning_err}"
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat

    # 3. Walk the card through the finishing stages. A dispatched render reaches
    #    95% (mux/upload) and the Telegram send marks 100% when the MP4 lands.
    if PROGRESS_STATE.get("video_dispatched"):
        await update_progress_card(95, PROGRESS_MILESTONES[4][1])
    await update_progress_card(100, PROGRESS_MILESTONES[5][1], force=True)

    # 4. Deliver response text payload
    if len(reply) <= 4000:
        await update.message.reply_text(reply)
    else:
        chunks = []
        current = ""
        for line in reply.splitlines(keepends=True):
            if len(current) + len(line) > 3900:
                chunks.append(current)
                current = line
            else:
                current += line
        if current:
            chunks.append(current)
        for chunk in chunks:
            if chunk.strip():
                await update.message.reply_text(chunk)

@api.on_event("startup")
async def startup_event():
    global telegram_app, CURRENT_EVENT_LOOP
    CURRENT_EVENT_LOOP = asyncio.get_running_loop()
    logger.info(
        f"Initializing Chintu services (role={COMPUTE_ROLE}, "
        f"space={'on' if HF_SPACE_URL else 'off'}, jobs={'on' if HF_TOKEN else 'off'}, "
        f"e2b={'on' if E2B_API_KEY else 'off'})..."
    )
    if BRAIN_BOUNDARY_VIOLATIONS:
        logger.error(
            f"[Compute Boundary] heavy media imports must not run on Render: {BRAIN_BOUNDARY_VIOLATIONS}"
        )
    discover_available_models()
    if TELEGRAM_BOT_TOKEN:
        try:
            telegram_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
            telegram_app.add_handler(CommandHandler("start", handle_start))
            telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            await telegram_app.initialize()
            await telegram_app.start()
            asyncio.create_task(telegram_app.updater.start_polling(drop_pending_updates=True))
            logger.info("[Telegram Bot] Polling started cleanly.")
        except Exception as te:
            logger.error(f"Telegram start error: {te}")

@api.on_event("shutdown")
async def shutdown_event():
    global telegram_app
    if telegram_app:
        try:
            await telegram_app.updater.stop()
            await telegram_app.stop()
            await telegram_app.shutdown()
        except Exception as e:
            logger.warning(f"Telegram shutdown warning: {e}")

if __name__ == "__main__":
    uvicorn.run("main:api", host="0.0.0.0", port=PORT, log_level="info")
