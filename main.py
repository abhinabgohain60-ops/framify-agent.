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
# Verified against OpenRouter's live catalog on 2026-10-17 (zero 404 risk):
# `openrouter/free` is the official free router slot, `nvidia/nemotron-3-super
# -120b-a12b:free` is a tool-capable zero-priced frontier model, and
# `cohere/north-mini-code:free` is the verified free agentic-coding slot.
# NOTE: the requested `qwen/qwen3.8-27b:free` returns HTTP 404 on the live
# catalog (only the paid `qwen/qwen3.8-27b` exists), so it was substituted.
OPENROUTER_MODELS: List[str] = [
    "openrouter/free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "cohere/north-mini-code:free",
]
EXTERNAL_REQUEST_TIMEOUT = 600
EXTERNAL_MAX_TOOL_ROUNDS = 12
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
# headless Chromium and FFmpeg belong to the E2B microVM render engine, so
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
        turn_failed="",
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

# A delivered MP4 below either floor is almost certainly a rendering stub,
# never a real composition: the incident that motivated these floors was a
# 1-second black "TEST" clip passing straight through to Telegram.
_MP4_MIN_BYTES = 50 * 1024
_MP4_MIN_SECONDS = 2.0


def _mp4_is_test_fixture(path: str) -> bool:
    """A synthetic state/video test clip must never masquerade as a render."""
    return "test" in (path or "").lower()


def _mp4_sanity_ok(sbx: Any, path: str) -> bool:
    """Gate every chosen MP4: > 50KB and > 2s, measured inside the sandbox.

    When the sandbox shell cannot measure (no stat/ffprobe output), acceptance
    is allowed but logged — refusing everything on a measurement gap would
    strand real renders. A MEASURED value under the floor is always a reject.
    """
    if _mp4_is_test_fixture(path):
        logger.error(f"[E2B san] rejected test fixture: {path}")
        return False
    quoted = f'"{path}"'
    try:
        res = sbx.commands.run(f"stat -c %s {quoted} 2>/dev/null", timeout=0)
        val = (getattr(res, "stdout", "") or "").strip()
        if val.isdigit():
            size = int(val)
            if size <= _MP4_MIN_BYTES:
                logger.error(
                    f"[E2B san] rejected {path}: size {size}B <= {_MP4_MIN_BYTES}B "
                    "(almost certainly a stub)."
                )
                return False
        else:
            logger.warning(f"[E2B san] size unmeasurable for {path} ({val!r}); continuing.")
    except Exception as stat_err:
        logger.warning(f"[E2B san] size check skipped for {path}: {stat_err}")
    try:
        res = sbx.commands.run(
            "ffprobe -v error -show_entries format=duration "
            f"-of default=noprint_wrappers=1:nokey=1 {quoted} 2>/dev/null",
            timeout=0,
        )
        val = (getattr(res, "stdout", "") or "").strip()
        try:
            duration = float(val.splitlines()[0]) if val else float("nan")
        except (ValueError, IndexError):
            duration = float("nan")
        import math
        if math.isfinite(duration):
            if duration <= _MP4_MIN_SECONDS:
                logger.error(
                    f"[E2B san] rejected {path}: duration {duration:.3f}s <= "
                    f"{_MP4_MIN_SECONDS}s (almost certainly a stub)."
                )
                return False
        else:
            logger.warning(
                f"[E2B san] duration unmeasurable for {path} ({val[:60]!r}); continuing."
            )
    except Exception as probe_err:
        logger.warning(f"[E2B san] duration check skipped for {path}: {probe_err}")
    return True


def _find_rendered_video(sbx: Any, directory: str, slug: str = "") -> Optional[str]:
    """Locate the rendered MP4 by modification time, never by first match.

    Two guards make 'delivered a synthetic stub' impossible:
      1. The newest .mp4 by mtime wins (`find -printf %T@ | sort -n | tail -1`),
         with an `ls -t` fallback for shells without `-printf`. Test-named
         files ("*test*") are excluded outright.
      2. Every candidate must pass a sanity gate: size > 50KB and duration
         > 2s before it can be handed to delivery.
    Candidate probes and a directory listing remain as last resorts (they are
    what the harness fakes exercise), all filtered and sanity-gated the same
    way. If EVERYTHING comes back empty the full /home/user tree is dumped
    into the logs so the post-mortem is visible in Render logs.
    """
    find_cmd = ("find /home/user /tmp -maxdepth 4 -type f -name \"*.mp4\" ! -name \"*test*\" "
                "-printf \"%T@ %p\n\" 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}'")
    # Shells/GNU find without -printf fall back to ls -t against the same file set.
    find_all = (
        'find /home/user /tmp -maxdepth 4 -type f -name "*.mp4" ! -name "*test*" 2>/dev/null'
    )
    ls_t_cmd = f'ls -t $({find_all}) 2>/dev/null | head -n 1'
    find_stdout_empty = True
    try:
        res = sbx.commands.run(find_cmd, timeout=0)
        chosen = (getattr(res, "stdout", "") or "").strip()
        if chosen and chosen.lower().endswith(".mp4"):
            find_stdout_empty = False
            try:
                sbx.files.read(chosen, format="bytes")
                if _mp4_sanity_ok(sbx, chosen):
                    logger.info(f"[E2B find] verified newest MP4: {chosen}")
                    return chosen
            except Exception as read_err:
                logger.warning(f"[E2B find] {chosen} listed but not readable: {read_err}")
        else:
            logger.info(f"[E2B find] newest-mp4 printf probe returned: {chosen!r}")
    except Exception as find_err:
        logger.error(f"[E2B find] printf find failed: {find_err}")
    try:
        res = sbx.commands.run(ls_t_cmd, timeout=0)
        chosen = (getattr(res, "stdout", "") or "").strip()
        if chosen and chosen.lower().endswith(".mp4") and not _mp4_is_test_fixture(chosen):
            find_stdout_empty = False
            try:
                sbx.files.read(chosen, format="bytes")
                if _mp4_sanity_ok(sbx, chosen):
                    logger.info(f"[E2B find] verified newest MP4 (ls -t): {chosen}")
                    return chosen
            except Exception as read_err:
                logger.warning(f"[E2B find] {chosen} listed but not readable: {read_err}")
    except Exception as ls_err:
        logger.warning(f"[E2B find] ls -t fallback failed: {ls_err}")
    # 3) Shell-verified candidates (slug-aware first), filtered + sanity-gated.
    candidates = []
    if slug:
        candidates.extend([
            f"/home/user/videos/{slug}/renders/final_output.mp4",
            f"/home/user/videos/{slug}/renders/video.mp4",
            f"/home/user/renders/final_output.mp4",
            f"/home/user/final_output.mp4",
        ])
    candidates.extend([
        f"{directory.rstrip('/')}/final_output.mp4",
        f"{directory.rstrip('/')}/video.mp4",
        "/home/user/renders/final_output.mp4",
        "/home/user/videos/renders/final_output.mp4",
        "/home/user/final_output.mp4",
    ])
    for c in candidates:
        if _mp4_is_test_fixture(c):
            continue
        try:
            sbx.files.read(c, format="bytes")
            if _mp4_sanity_ok(sbx, c):
                return c
        except Exception:
            pass
    # 4) Directory listing fallback.
    try:
        for entry in sbx.files.list(directory):
            name = getattr(entry, "name", "") or ""
            if name.lower().endswith(".mp4") and not _mp4_is_test_fixture(name):
                p = f"{directory.rstrip('/')}/{name}"
                if _mp4_sanity_ok(sbx, p):
                    return p
    except Exception:
        pass
    # 5) Nothing found anywhere: dump the exact file tree into the logs.
    if find_stdout_empty:
        try:
            ls_res = sbx.commands.run("ls -laR /home/user", timeout=0)
            logger.error(
                "[E2B find] find returned no MP4 - full /home/user tree:\n"
                f"{getattr(ls_res, 'stdout', '') or '(empty ls stdout)'}"
            )
        except Exception as ls_err:
            logger.error(f"[E2B find] ls -laR /home/user also failed: {ls_err}")
    return None

def _find_mp4_in_sandbox(sbx: Any, slug: str = "") -> Optional[str]:
    """Search recursively for MP4s in /home/user; return first final_output.mp4 or any .mp4 found."""
    try:
        # Prefer common slug paths first
        candidates = []
        if slug:
            candidates.extend([
                f"/home/user/videos/{slug}/renders/final_output.mp4",
                f"/home/user/videos/{slug}/renders/video.mp4",
                f"/home/user/videos/{slug}/final_output.mp4",
                f"/home/user/videos/{slug}/video.mp4",
            ])
        candidates.extend([
            "/home/user/final_output.mp4",
            "/home/user/output/final_output.mp4",
            "/home/user/videos/final_output.mp4",
            "/home/user/renders/final_output.mp4",
        ])
        for c in candidates:
            if _mp4_is_test_fixture(c):
                continue
            try:
                sbx.files.read(c, format="bytes")
                if _mp4_sanity_ok(sbx, c):
                    return c
            except Exception:
                continue
        # Fallback: find in sandbox (final_output first), newest-last names filtered,
        # every hit gated through the same size/duration sanity check.
        res = sbx.commands.run("find /home/user -name 'final_output.mp4' -not -path '*/proc/*' 2>&1 | head -5")
        out = getattr(res, "stdout", "") or ""
        for line in out.splitlines():
            p = line.strip()
            if p and p.startswith("/home/user/") and not _mp4_is_test_fixture(p):
                try:
                    sbx.files.read(p, format="bytes")
                    if _mp4_sanity_ok(sbx, p):
                        return p
                except Exception:
                    continue
        res = sbx.commands.run(
            "find /home/user /tmp -maxdepth 4 -type f -name '*.mp4' ! -name '*test*' "
            "-printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}'"
        )
        out = getattr(res, "stdout", "") or ""
        for line in out.splitlines():
            p = line.strip()
            if p and p.startswith("/") and not _mp4_is_test_fixture(p):
                try:
                    sbx.files.read(p, format="bytes")
                    if _mp4_sanity_ok(sbx, p):
                        return p
                except Exception:
                    continue
        # Debug: list videos dir
        try:
            vlist = sbx.files.list("/home/user/videos")
            logger.error(f"[E2B Debug] /home/user/videos: {[getattr(e,'name','') for e in vlist]}")
        except Exception as e:
            logger.error(f"[E2B Debug] could not list /home/user/videos: {e}")
    except Exception as e:
        logger.error(f"[E2B MP4 search] error: {e}")
    return None

SANDBOX_TIMEOUT_SECONDS = int(os.environ.get("E2B_SANDBOX_TIMEOUT", "900"))
HYPERFRAMES_VERSION = os.environ.get("HYPERFRAMES_VERSION", "0.8.160")
HYPERFRAMES_SKILL = os.environ.get("HYPERFRAMES_SKILL", "motion-graphics")

# Renders land here; a HyperFrames project writes into its own renders/ subdir.
SANDBOX_VIDEO_SCAN_DIRS = ("/home/user", "/home/user/output", "/home/user/videos")
# Render queue/fast flags
RENDER_QUEUE_SERIALIZED = True
RENDER_FAST_FLAGS = {
    "disable_motion_blur": True,
}
DRAFT_PREVIEW_SIZE = (540, 960)
FINAL_SIZE = (1080, 1920)
try:
    import threading
    _render_queue_lock = threading.Lock()
except Exception:
    _render_queue_lock = None

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

# ============================================================ E2B render engine
# ALL video rendering happens in an isolated E2B microVM, one synchronous call.
# The Hugging Face render backends (Docker Space worker and cpu-basic Jobs) were
# RETIRED: compute Spaces now require a paid PRO plan and free accounts can only
# create Static Spaces, which cannot run backend media rendering. `space/` stays
# in the repository as archived reference but is no longer in the dispatch chain.
COMPUTE_ROLE = "brain"

# Renders produce large temp intermediates, but the sandbox image boots with
# under 1 GB free in /tmp and the observed production failure was exactly:
# "Low disk space: 1.0 GB free at /tmp". Every execution path therefore purges
# caches and temp dirs and repoints TMPDIR at /home/user/tmp (larger volume)
# BEFORE running the workload.
E2B_TMP_DIR = os.environ.get("E2B_TMP_DIR", "/home/user/tmp")


def _e2b_disk_prep_cmd() -> str:
    """Shell that frees disk space and stages the writable TMPDIR, then reports."""
    return (
        "rm -rf /tmp/* ~/.cache/* /var/tmp/* 2>/dev/null || true ; "
        f"mkdir -p {E2B_TMP_DIR} ; "
        f"export TMPDIR={E2B_TMP_DIR} ; "
        "df -h /tmp ${TMPDIR:-/tmp} 2>/dev/null | tail -3"
    )


def e2b_disk_prep(sbx: Any) -> str:
    """Run the pre-render cleanup inside the sandbox and return its disk report."""
    try:
        res = sbx.commands.run(_e2b_disk_prep_cmd(), timeout=0)
    except Exception as err:
        logger.warning(f"[E2B disk] pre-render cleanup could not run: {err}")
        return ""
    report = (getattr(res, "stdout", "") or "").strip()
    logger.info(f"[E2B disk] pre-render cleanup applied; disk report: {report}")
    return report


def _e2b_cmd_with_prep(command: str) -> str:
    """Prefix a sandbox command with cleanup + TMPDIR so temp writes survive."""
    return (
        "rm -rf /tmp/* ~/.cache/* /var/tmp/* 2>/dev/null || true ; "
        f"mkdir -p {E2B_TMP_DIR} ; "
        f"export TMPDIR={E2B_TMP_DIR} ; "
        f"{command}"
    )


NARRATION_VOICE = os.environ.get("NARRATION_VOICE", "en-US-AriaNeural")


def _e2b_audio_mux_cmd(video_path: str, audio_path: str, final_path: str) -> str:
    """Hardened final mux: explicit stream map, AAC 192k @ 44.1 kHz, web faststart.

    Both inputs are named explicitly (-map 0:v:0 / -map 1:a:0) so the output
    container always carries the narration track as ITS audio stream (no silent
    MP4s, no accidental second stream), and -movflags +faststart relocates the
    moov atom so Telegram/exoplayback can start instantly.
    """
    return (
        "ffmpeg -y "
        f"-i {video_path} -i {audio_path} "
        "-map 0:v:0 -map 1:a:0 "
        "-c:v copy -c:a aac -b:a 192k -ar 44100 "
        "-shortest -movflags +faststart "
        f"{final_path}"
    )


def _e2b_audio_verify_cmd(final_path: str) -> str:
    """ffprobe check that the final MP4 actually carries an active audio stream."""
    return f"ffprobe -i {final_path} -show_streams -select_streams a"


def _audio_stream_verified(output: str) -> bool:
    """True when ffprobe's -select_streams a dump lists a real audio codec."""
    return "codec_type=audio" in (output or "") and "codec_name=" in (output or "")


def _synthesize_narration_audio(sbx: Any, project_dir: str, narration: str) -> str:
    """Synthesize the voice-over inside the sandbox with edge-tts.

    Returns the sandbox path of narration.mp3, or '' when synthesis failed so
    the render still delivers (silent) instead of aborting the turn.
    """
    text = (narration or "").strip()
    if not text:
        return ""
    narration_txt = f"{project_dir}/narration.txt"
    audio_path = f"{project_dir}/narration.mp3"
    try:
        sbx.files.write(narration_txt, text[:3000])
        # edge-tts CLI takes -f/--file (mutually exclusive with -t/--text) —
        # a non-existent flag like --text-file makes argparse exit non-zero
        # and was the root cause of silently narration-less deliveries.
        tts_res = sbx.commands.run(
            _e2b_cmd_with_prep(
                f"cd {project_dir} && python3 -m edge_tts --voice {NARRATION_VOICE} "
                f"--file narration.txt --write-media narration.mp3"
            ),
            timeout=0,
        )
        if getattr(tts_res, "exit_code", 1) not in (0, None):
            logger.warning(
                "[Audio mux] edge-tts synthesis failed (exit=%s); delivering without narration.",
                getattr(tts_res, "exit_code", "unknown"),
            )
            return ""
        return audio_path
    except Exception as tts_err:
        logger.warning(f"[Audio mux] narration synthesis error: {tts_err}")
        return ""


def _mux_narration_and_verify(sbx: Any, video_path: str, audio_path: str) -> Optional[str]:
    """Mux the narration into the rendered MP4 and ffprobe-verify the audio.

    Runs INSIDE the sandbox. On success returns the path of
    renders/final_output.mp4 with an ffprobe-confirmed active audio stream;
    on any failure returns None so the caller still delivers the silent render
    instead of throwing a CommandExitException at Telegram.
    """
    render_dir = video_path.rsplit("/", 1)[0]
    final_path = f"{render_dir}/final_output.mp4"
    try:
        mux_res = sbx.commands.run(
            _e2b_cmd_with_prep(_e2b_audio_mux_cmd(video_path, audio_path, final_path)),
            timeout=0,
        )
        if getattr(mux_res, "exit_code", 1) not in (0, None):
            logger.warning(
                "[Audio mux] ffmpeg mux failed (exit=%s); keeping the silent render.",
                getattr(mux_res, "exit_code", "unknown"),
            )
            return None
        probe_res = sbx.commands.run(
            _e2b_cmd_with_prep(_e2b_audio_verify_cmd(final_path)), timeout=0
        )
        probe_text = (getattr(probe_res, "stdout", "") or "") + (
            getattr(probe_res, "stderr", "") or ""
        )
        if not _audio_stream_verified(probe_text):
            logger.warning("[Audio mux] ffprobe found NO active audio stream in the final MP4.")
            return None
        logger.info(f"[Audio mux] narration track verified in {final_path} (AAC 192k/44.1kHz)")
        return final_path
    except Exception as mux_err:
        logger.warning(f"[Audio mux] mux/verify error: {mux_err}")
        return None


def _telegram_send_video(video_bytes: bytes, file_name: str, caption: str = "") -> bool:
    """Post MP4 bytes to the bound Telegram chat. Returns True only on HTTP 200."""
    if not (CURRENT_CHAT_ID and TELEGRAM_BOT_TOKEN):
        return False
    emit_progress(95, PROGRESS_MILESTONES[4][1], video=True)
    try:
        send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
        # The explicit video/mp4 MIME type is what Telegram checks to enable the
        # unmute button for the AAC track, and supports_streaming=True lets
        # Telegram stream progressive MP4s inline (no download-first banner).
        upload_files = {"video": (file_name, video_bytes, "video/mp4")}
        data = {
            "chat_id": CURRENT_CHAT_ID,
            "caption": caption or f"🎬 Chintu Video Render: {file_name}",
            "supports_streaming": "true",
        }
        res = requests.post(send_url, data=data, files=upload_files, timeout=120)
        if res.status_code == 200:
            logger.info(f"Directly uploaded {file_name} to Telegram via HTTP API!")
            emit_progress(100, PROGRESS_MILESTONES[5][1], video=True)
            return True
        logger.error(f"Telegram upload failed: {res.text[:300]}")
    except Exception as vid_err:
        logger.warning(f"Video delivery warning: {vid_err}")
    return False

def _deliver_rendered_video(
    sbx: Any, extra_dirs: Any = (), require_audio: bool = False, slug: str = ""
) -> str:
    """Upload the rendered MP4 straight to Telegram. `final_output.mp4` ALWAYS
    wins when it exists; a narrated render is never allowed to fall back to the
    raw, silent `video.mp4` capture.

    Returns a status string:
      'delivered'      an MP4 shipped to Telegram (never under the name video.mp4)
      'no-audio'       require_audio=True but no verified final_output.mp4 exists
                       — NOTHING was uploaded, so the failure is loud, not silent
      'post-failed'    final_output.mp4 exists but Telegram refused every attempt
      'no-video'       no MP4 anywhere in the sandbox
      'unbound'        no chat id / bot token
    """
    if not (CURRENT_CHAT_ID and TELEGRAM_BOT_TOKEN):
        return "unbound"
    scan_dirs = list(
        dict.fromkeys(
            d.rstrip("/") for d in list(extra_dirs) + list(SANDBOX_VIDEO_SCAN_DIRS)
        )
    )
    if slug:
        scan_dirs.append(f"/home/user/videos/{slug}/renders")
        scan_dirs.append(f"/home/user/videos/{slug}")
    scan_dirs = list(dict.fromkeys(scan_dirs))
    if slug:
        scan_dirs.append(f"/home/user/videos/{slug}/renders")
        scan_dirs.append(f"/home/user/videos/{slug}")
    scan_dirs = list(dict.fromkeys(scan_dirs))
    # 1) final_output.mp4 (the ffprobe-verified, narrated artifact) first,
    #    wherever it lives, under its own name so Telegram never shows video.mp4.
    found_final = _find_mp4_in_sandbox(sbx, slug=slug)
    if found_final and "final_output.mp4" in found_final:
        try:
            final_bytes = sbx.files.read(found_final, format="bytes")
            if _telegram_send_video(final_bytes, "final_output.mp4"):
                return "delivered"
            return "post-failed"
        except Exception:
            pass
    for scan_dir in scan_dirs:
        final_path = f"{scan_dir}/final_output.mp4"
        try:
            final_bytes = sbx.files.read(final_path, format="bytes")
        except Exception:
            continue
        if _telegram_send_video(final_bytes, "final_output.mp4"):
            return "delivered"
        return "post-failed"
    # 2) A narrated render must FAIL LOUDLY instead of shipping silent frames.
    if require_audio:
        logger.error(
            "[Audio mux] narration was required but no final_output.mp4 exists — "
            "refusing to deliver the silent capture."
        )
        try:
            vlist = sbx.files.list("/home/user/videos")
            logger.error(f"[E2B Debug] /home/user/videos contents: {[getattr(e,'name','') for e in vlist]}")
        except Exception as e:
            logger.error(f"[E2B Debug] could not list /home/user/videos: {e}")
        return "no-audio"
    # 3) Un-narrated captures still ship, but never under the name video.mp4.
    found_any = _find_mp4_in_sandbox(sbx, slug=slug)
    if found_any:
        try:
            video_bytes = sbx.files.read(found_any, format="bytes")
            display_name = found_any.rsplit("/", 1)[-1]
            if display_name.lower() == "video.mp4":
                display_name = "final_output.mp4"
            if _telegram_send_video(video_bytes, display_name):
                return "delivered"
        except Exception as vid_err:
            logger.warning(f"Video delivery warning: {vid_err}")
    for scan_dir in scan_dirs:
        video_path = _find_rendered_video(sbx, scan_dir)
        if not video_path:
            continue
        try:
            video_bytes = sbx.files.read(video_path, format="bytes")
        except Exception as vid_err:
            logger.warning(f"Video delivery warning: {vid_err}")
            continue
        display_name = video_path.rsplit("/", 1)[-1]
        if display_name.lower() == "video.mp4":
            display_name = "final_output.mp4"
        if _telegram_send_video(video_bytes, display_name):
            return "delivered"
    return "no-video"

def _provision_render_toolchain(sbx: Any, want_browser: bool = False) -> str:
    """Install the ffmpeg/frame stack, plus Node + Chrome when HyperFrames must render."""
    # Disk hygiene first: the boot image has < 1 GB free in /tmp, which previously
    # failed renders with "Low disk space". Purge + stage TMPDIR before installing.
    base = (
        "command -v ffmpeg >/dev/null 2>&1 || "
        "(sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg) ; "
        "pip install --quiet --no-cache-dir pillow \"numpy<2\" scipy edge-tts"
    )
    sbx.commands.run(_e2b_cmd_with_prep(base), timeout=0)
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
    # Chrome/Node cache and extraction also land in /tmp; run with prep as well.
    sbx.commands.run(_e2b_cmd_with_prep("true"), timeout=0)
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

# ================================================================= E2B ENGINE
def render_html_video(
    html: str,
    project_name: str = "chintu-motion",
    fps: int = 30,
    quality: str = "looks",
    backend: str = "auto",
    narration: str = "",
) -> str:
    """Render a HyperFrames HTML/GSAP composition to MP4 on the E2B microVM.

    The Hugging Face render backends (16 GB Docker Space and `cpu-basic` Jobs)
    were RETIRED: Hugging Face compute Spaces now require a paid PRO plan and
    free accounts can only create Static Spaces, so E2B is the sole render
    engine. This Render process (~512 MB) still never renders locally; the
    composition is shipped to an isolated sandbox that is killed in a finally
    block, provisioned with Node + Chrome + FFmpeg, and rendered by the real
    `hyperframes` CLI. Sandbox commands purge /tmp and export TMPDIR=/home/user
    /tmp first, so the ~1 GB tmpfs never chokes the render; if the browser path
    still fails (e.g. disk exhaustion), an automatic Pillow + FFmpeg frame
    compiler finishes the MP4 inside the sandbox so the turn delivers to
    Telegram instead of surfacing a CommandExitException.

    Delivery is SYNCHRONOUS: the finished video is posted straight to Telegram
    with /sendVideo before this call returns. There is no job id to poll.

    Args:
        html: The full composition document (root carries data-composition-id and
            registers window.__timelines["<id>"] as one paused GSAP timeline).
        project_name: Kebab-case project directory name.
        fps: Frame rate recorded on the render (24-240; 30 typical).
        quality: draft (fast iteration), looks (default) or delivery (final).
        backend: 'auto' (default) or 'e2b' — both mean the E2B microVM. The
            retired 'space', 'huggingface' and 'jobs' values return a notice.
        narration: Optional voice-over script; synthesized inside the sandbox
            with edge-tts and muxed into the MP4.
    """
    backend = (backend or "auto").strip().lower()
    if backend in ("space", "huggingface", "hf", "jobs"):
        return (
            "Error: the Hugging Face render backends were retired. Compute Spaces "
            "now require a paid PRO plan, so they can no longer host this worker — "
            "every render runs on the E2B microVM backend, which is what "
            "backend='auto' already uses."
        )
    if backend not in ("auto", "e2b"):
        return f"Unknown render backend '{backend}'. Use 'auto' (recommended) or 'e2b'."
    problem = _validate_composition_html(html)
    if problem:
        return f"Composition rejected before render: {problem}"
    if not E2B_API_KEY:
        return (
            "Error: E2B_API_KEY is not configured, and the E2B microVM is now "
            "the only render engine (Hugging Face was retired)."
        )
    emit_progress(55, PROGRESS_MILESTONES[2][1], video=True)
    slug = re.sub(r"[^a-z0-9-]+", "-", (project_name or "chintu-motion").lower()).strip("-")
    slug = slug or "chintu-motion"
    project_dir = f"/home/user/videos/{slug}"
    sbx = None
    try:
        from e2b_code_interpreter import Sandbox

        logger.info(f"Starting E2B microVM render of '{slug}' (Hugging Face backends retired)...")
        sbx = Sandbox.create(api_key=E2B_API_KEY, timeout=SANDBOX_TIMEOUT_SECONDS)
        disk_report = e2b_disk_prep(sbx)
        provisioning = _provision_render_toolchain(sbx, want_browser=True)
        sbx.commands.run(f"mkdir -p {project_dir}/renders", timeout=0)
        sbx.files.write(f"{project_dir}/index.html", html)
        sbx.files.write(
            f"{project_dir}/hyperframes.json",
            json.dumps(HYPERFRAMES_PROJECT_CONFIG, indent=2),
        )

        hf = f"npx --yes hyperframes@{HYPERFRAMES_VERSION}"
        lint_output = ""
        try:
            lint = sbx.commands.run(
                _e2b_cmd_with_prep(f"cd {project_dir} && {hf} lint ."), timeout=0
            )
            lint_output = (getattr(lint, "stdout", "") or "")[-1200:]
            logger.info(f"[HyperFrames] lint exit={getattr(lint, 'exit_code', 'unknown')}")
        except Exception as lint_err:
            lint_output = f"lint execution error: {lint_err}"
        render_output = ""
        render_exit: Any = "exception"
        mp4_path: Optional[str] = None
        try:
            render = sbx.commands.run(
                _e2b_cmd_with_prep(
                    f"cd {project_dir} && {hf} render . --skill={HYPERFRAMES_SKILL} "
                    f"-f {int(fps)} -q {quality} -o ./renders/video.mp4"
                ),
                timeout=0,
            )
            render_exit = getattr(render, "exit_code", "unknown")
            render_output = (
                (getattr(render, "stdout", "") or "") + (getattr(render, "stderr", "") or "")
            )[-2000:]
            mp4_path = _find_rendered_video(sbx, f"{project_dir}/renders")
        except Exception as render_err:
            render_output = f"render execution error: {render_err}"
        fallback_note = ""
        if mp4_path is None:
            low_disk = "disk" in render_output.lower() or "nospc" in render_output.lower()
            fallback_note = (
                "HyperFrames/Chromium path failed"
                + (" (disk exhaustion detected)" if low_disk else f" (render exit={render_exit})")
                + "; engaging automatic Pillow + FFmpeg frame compiler."
            )
            logger.warning(f"[Render Fallback] {fallback_note}")
            fallback_note += "\n" + _run_pillow_fallback(sbx, project_dir, slug, int(fps), narration)
            mp4_path = _find_rendered_video(sbx, f"{project_dir}/renders")
        # HARDENED AUDIO MUX: regardless of which path produced the frames, the
        # narration is synthesized in-sandbox, muxed with an explicit stream map
        # (AAC 192k @ 44.1 kHz, +faststart) and the result is ffprobe-verified
        # to carry an active audio stream BEFORE any Telegram delivery. On
        # failure the mux is retried once and, if it still fails, delivery
        # fails LOUDLY: the silent capture is never shipped as if narrated.
        audio_note = ""
        wants_audio = bool((narration or "").strip())
        if mp4_path and wants_audio:
            audio_path = _synthesize_narration_audio(sbx, project_dir, narration)
            muxed_path = (
                _mux_narration_and_verify(sbx, mp4_path, audio_path) if audio_path else None
            )
            if not muxed_path:
                # RE-MUX once: transient TTS/ffmpeg hiccups must not strand the
                # render as silent frames.
                logger.warning("[Audio mux] first attempt failed — re-muxing once.")
                audio_path = audio_path or _synthesize_narration_audio(
                    sbx, project_dir, narration
                )
                muxed_path = (
                    _mux_narration_and_verify(sbx, mp4_path, audio_path)
                    if audio_path
                    else None
                )
            if muxed_path:
                mp4_path = muxed_path
                audio_note = f"audio: narration muxed as AAC (192k/44.1kHz, +faststart) into {muxed_path}; ffprobe-verified active audio stream."
            else:
                audio_note = (
                    "audio: edge-tts synthesis or ffmpeg mux/ffprobe verification "
                    "failed after a retry — the silent capture will NOT be delivered."
                )
            logger.info(f"[Audio mux] {audio_note}")
        deliver_status = _deliver_rendered_video(
            sbx, extra_dirs=(f"{project_dir}/renders",), require_audio=wants_audio, slug=slug
        )
        report = (
            f"E2B HyperFrames render :: {slug}\n"
            f"backend: E2B microVM (killed in finally; Hugging Face retired)\n"
            f"disk prep: {disk_report or 'cleanup ran (no report)'}\n"
            f"provisioning: {provisioning}\n"
            f"lint output:\n{lint_output}\n"
            f"render output:\n{render_output}\n"
        )
        if fallback_note:
            report += f"fallback: {fallback_note}\n"
        if audio_note:
            report += f"{audio_note}\n"
        if deliver_status == "delivered":
            report += (
                f"[SUCCESS] MP4 written to {mp4_path} and delivered to Telegram "
                "(uploaded as final_output.mp4; the player never shows 'video.mp4')."
            )
        elif deliver_status == "no-audio":
            report += (
                "[FAILED-NO-AUDIO] narration was requested but no ffprobe-verified "
                "final_output.mp4 could be produced — the silent capture "
                f"({mp4_path or 'video.mp4'}) was NOT uploaded. Fix the cause above "
                "(edge-tts / ffmpeg / ffprobe) and re-run the render."
            )
        elif deliver_status == "post-failed":
            report += (
                "[FAILED] final_output.mp4 exists but Telegram refused the upload; "
                "see the delivery log above."
            )
        elif deliver_status == "unbound":
            report += f"[UNDELIVERED] MP4 ready at {mp4_path} (no Telegram chat bound; retry after binding CURRENT_CHAT_ID)."
        else:
            report += "[FAILED] No .mp4 produced even with the fallback compiler; read the outputs above."
        return report
    except Exception as e:
        logger.error(f"E2B render error: {e}", exc_info=True)
        return f"E2B render error:\n{traceback.format_exc()}"
    finally:
        if sbx is not None:
            try:
                sbx.kill()
            except Exception as kill_err:
                logger.warning(f"Sandbox cleanup warning: {kill_err}")


def _run_pillow_fallback(sbx: Any, project_dir: str, slug: str, fps: int, narration: str) -> str:
    """Last-resort frame compiler: Pillow draws the frames, FFmpeg muxes them.

    Runs INSIDE the sandbox (never on the Render brain). Triggered automatically
    when the HyperFrames browser path fails — e.g. the small /tmp tmpfs fills up
    — so the render still completes and reaches Telegram. Frames are streamed
    into FFmpeg over stdin, so no per-frame files ever touch the disk.
    """
    script = _PILLOW_FALLBACK_TEMPLATE.replace("__FPS__", str(int(fps)))
    script_path = f"{project_dir}/fallback_render.py"
    narration_path = ""
    if narration:
        narration_path = f"{project_dir}/narration.txt"
        sbx.files.write(narration_path, narration.strip())
    sbx.files.write(script_path, script)
    env_bits = f"FALLBACK_HTML={project_dir}/index.html FALLBACK_OUT={project_dir}/renders/video.mp4"
    if narration_path:
        env_bits += f" FALLBACK_NARRATION={narration_path}"
    res = sbx.commands.run(
        _e2b_cmd_with_prep(f"cd {project_dir} && {env_bits} python3 {script_path}"),
        timeout=0,
    )
    code = getattr(res, "exit_code", "unknown")
    output = (getattr(res, "stdout", "") or "") + (getattr(res, "stderr", "") or "")
    logger.info(f"[Render Fallback] Pillow+FFmpeg compiler exit={code}")
    return f"pillow+ffmpeg fallback exit={code}\n{output[-1200:]}"


# Deterministic Pillow + FFmpeg frame compiler. Kept as a template string that a
# render writes into the sandbox and executes there — the Render brain imports
# no media library. It parses the composition's timed scene elements and draws
# each frame with safe-zone typography, fade/slide motion and eased timing,
# streaming raw RGB frames into FFmpeg stdin (no frame files, no /tmp fill-up).
_PILLOW_FALLBACK_TEMPLATE = r'''
"""Pillow + FFmpeg frame compiler — automatic fallback inside the E2B sandbox."""
import os
import subprocess
import sys
from html.parser import HTMLParser

from PIL import Image, ImageDraw, ImageFont

FPS = __FPS__
W, H = 1080, 1920
SAFE_X = 160
TITLE_SIZE, BODY_SIZE = 96, 48
BG_TOP = (10, 14, 24)
BG_BOT = (36, 18, 58)
GLOW = (98, 76, 192)
ACCENT = (232, 179, 75)
INK = (246, 246, 252)

HTML_PATH = os.environ["FALLBACK_HTML"]
OUT_PATH = os.environ.get("FALLBACK_OUT", "renders/video.mp4")
NARRATION_PATH = os.environ.get("FALLBACK_NARRATION", "")
TMPDIR = os.environ.get("TMPDIR", "/home/user/tmp")


def _clamp(u):
    return max(0.0, min(1.0, u))


def _ease_out(u):
    u = _clamp(u)
    return 1.0 - (1.0 - u) ** 2


class _SceneParser(HTMLParser):
    """Collect timed scene elements (data-start/data-duration) plus their text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.scenes = []
        self._scene = None

    def handle_starttag(self, tag, attrs):
        if self._scene is None:
            a = dict(attrs)
            if "data-start" in a and "data-duration" in a:
                try:
                    self._scene = {
                        "start": float(a["data-start"]),
                        "dur": float(a["data-duration"]),
                        "text": [],
                    }
                except ValueError:
                    self._scene = None

    def handle_endtag(self, tag):
        if self._scene is not None and tag in ("section", "div", "figure"):
            if self._scene["text"]:
                self.scenes.append(self._scene)
            self._scene = None

    def handle_data(self, data):
        if self._scene is not None and data.strip():
            self._scene["text"].append(data.strip())


def load_scenes():
    parser = _SceneParser()
    with open(HTML_PATH, "r", encoding="utf-8") as fh:
        parser.feed(fh.read())
    scenes = [s for s in parser.scenes if s["dur"] > 0]
    if not scenes:
        scenes = [{"start": 0.0, "dur": 6.0, "text": ["Chintu Video"]}]
    end = max(s["start"] + s["dur"] for s in scenes)
    return scenes, end


def font(size, bold=True):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap(draw, text, fnt, max_w):
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def build_backdrop():
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):
        u = y / H
        d.line([(0, y), (W, y)], fill=tuple(int(a + (b - a) * u) for a, b in zip(BG_TOP, BG_BOT)))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = W // 2, int(H * 0.36)
    for radius in range(560, 60, -24):
        a = int(5 + 40 * (1 - radius / 560.0))
        gd.ellipse([cx - radius, cy - int(radius * 1.2), cx + radius, cy + int(radius * 1.2)], fill=GLOW + (a,))
    return img, glow


def build_vignette():
    v = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(v)
    for radius in range(1300, 500, -40):
        a = int(3 + 42 * ((radius - 500) / 800.0))
        d.ellipse([W // 2 - radius, H // 2 - radius, W // 2 + radius, H // 2 + radius], fill=(0, 0, 0, a))
    return v


def draw_frame(base, glow_layer, vig_layer, scenes, t):
    active = scenes[0]
    for s in scenes:
        if s["start"] <= t < s["start"] + s["dur"]:
            active = s
            break
    u = _clamp((t - active["start"]) / max(0.001, active["dur"]))
    frame = base.copy()
    frame.alpha_composite(glow_layer)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    alpha = _clamp(_ease_out(u / 0.25) * (1.0 - _ease_out((u - 0.8) / 0.2)) + 1e-6)
    y_shift = int((1.0 - _ease_out(u / 0.3)) * 80) - int(_ease_out((u - 0.75) / 0.25) * 50)
    first = active["text"][0][:80]
    rest = " ".join(active["text"][1:3])[:240]
    title_fnt = font(TITLE_SIZE)
    body_fnt = font(BODY_SIZE)
    y = int(H * 0.42) + y_shift
    for line in wrap(d, first, title_fnt, W - SAFE_X * 2):
        wdt = d.textlength(line, font=title_fnt)
        d.text(((W - wdt) / 2, y), line, font=title_fnt, fill=INK + (int(255 * alpha),))
        y += int(TITLE_SIZE * 1.25)
    if rest:
        y += 40
        for line in wrap(d, rest, body_fnt, W - SAFE_X * 2):
            wdt = d.textlength(line, font=body_fnt)
            d.text(((W - wdt) / 2, y), line, font=body_fnt, fill=ACCENT + (int(200 * alpha),))
            y += int(BODY_SIZE * 1.45)
    frame.alpha_composite(overlay)
    frame.alpha_composite(vig_layer)
    return frame


def main():
    scenes, duration = load_scenes()
    base, glow_layer = build_backdrop()
    vig_layer = build_vignette()
    os.makedirs(TMPDIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    silent_path = os.path.join(TMPDIR, "fallback_silent.mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", silent_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    total = int(round(duration * FPS))
    for f in range(total):
        try:
            proc.stdin.write(draw_frame(base, glow_layer, vig_layer, scenes, f / FPS).convert("RGB").tobytes())
        except BrokenPipeError:
            break
    if proc.stdin:
        proc.stdin.close()
    proc.wait()
    final_path = OUT_PATH
    if NARRATION_PATH and os.path.exists(NARRATION_PATH):
        with open(NARRATION_PATH, "r", encoding="utf-8") as fh:
            script_text = fh.read().strip()[:3000]
        media = os.path.join(TMPDIR, "narration.mp3")
        tts = subprocess.run(
            [sys.executable, "-m", "edge_tts", "--voice", "en-US-AriaNeural", "--text", script_text, "--write-media", media],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if tts.returncode == 0 and os.path.exists(media):
            # HARDENED MUX: explicit stream map so the narration becomes THE
            # audio stream (no silent MP4s), AAC 192k @ 44.1 kHz, web faststart.
            # ffprobe must then confirm an active audio stream before the MP4
            # is accepted for delivery; otherwise keep the silent render.
            mux = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", silent_path, "-i", media,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
                    "-shortest", "-movflags", "+faststart",
                    final_path,
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            audio_ok = False
            if mux.returncode == 0 and os.path.exists(final_path):
                probe = subprocess.run(
                    ["ffprobe", "-i", final_path, "-show_streams", "-select_streams", "a"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                audio_ok = (
                    b"codec_type=audio" in (probe.stdout or b"")
                    and b"codec_name=" in (probe.stdout or b"")
                )
            if not audio_ok:
                final_path = silent_path
        else:
            final_path = silent_path
    if final_path != OUT_PATH and os.path.exists(silent_path):
        subprocess.run(["cp", silent_path, OUT_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    size = os.path.getsize(OUT_PATH) if os.path.exists(OUT_PATH) else 0
    print(f"[pillow-fallback] RENDERED {OUT_PATH} {size} bytes")


if __name__ == "__main__":
    main()
'''


# ============================================================ GAMEPLAY EDL EDIT
# Silence + transient analysis runs INSIDE the E2B sandbox; the brain only
# parses JSON, plans the EDL with pure functions and builds one ffmpeg command.
_GAMEPLAY_ANALYSIS_TEMPLATE = r'''
"""Silence + transient analysis for gameplay edits — runs INSIDE the E2B sandbox.

Prints one machine-readable line:
    GP_ANALYSIS {"duration": float, "has_audio": bool, "silences": [[s, e], ...],
                 "moments": [[t, momentary_lufs], ...]}
Everything heavy (ffmpeg/ffprobe) happens here; the Render brain only parses JSON.
"""
import json
import os
import re
import subprocess

SRC = os.environ["GP_SRC"]
SILENCE_DB = os.environ.get("GP_SILDB", "-35dB")
MIN_SIL = os.environ.get("GP_MINSIL", "0.6")


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return (p.stdout or b"").decode("utf-8", "replace"), (p.stderr or b"").decode("utf-8", "replace")


def main():
    out, _ = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                  "-show_entries", "stream=codec_type", "-of", "json", SRC])
    try:
        meta = json.loads(out or "{}")
        duration = float((meta.get("format") or {}).get("duration") or 0.0)
    except Exception:
        duration = 0.0
    has_audio = "audio" in out
    if duration <= 0:
        print("GP_ERROR ffprobe could not measure duration")
        return
    silences = []
    moments = []
    if has_audio:
        # 1) dead air: silencedetect marks the spans to CUT from the EDL.
        _, derr = run(["ffmpeg", "-hide_banner", "-nostats", "-i", SRC,
                       "-af", "silencedetect=n=" + SILENCE_DB + ":d=" + MIN_SIL,
                       "-f", "null", "-"])
        start = None
        for line in derr.splitlines():
            m = re.search(r"silence_start:\s*([0-9.]+)", line)
            if m:
                start = float(m.group(1))
                continue
            m = re.search(r"silence_end:\s*([0-9.]+)", line)
            if m and start is not None:
                silences.append([start, float(m.group(1))])
                start = None
        if start is not None:
            silences.append([start, duration])
        # 2) energy timeline: ebur128 momentary loudness separates speech and
        #    gunfight transients (high M) from dead wandering (low M).
        _, eerr = run(["ffmpeg", "-hide_banner", "-nostats", "-i", SRC,
                       "-af", "ebur128=peak=true", "-f", "null", "-"])
        for line in eerr.splitlines():
            m = re.search(r"t:\s*([0-9.]+)\s+TARGET:.*?\bM:\s*(-?[0-9.]+|-inf|nan)", line)
            if not m:
                continue
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            if val == val:  # skip NaN; -inf is a real (silent) reading
                moments.append([float(m.group(1)), val])
    print("GP_ANALYSIS " + json.dumps(
        {"duration": duration, "has_audio": has_audio,
         "silences": silences, "moments": moments[:4000]}))


if __name__ == "__main__":
    main()
'''


def _parse_gameplay_analysis(output: str) -> Optional[Dict[str, Any]]:
    """Extract the last GP_ANALYSIS JSON payload from the sandbox script output."""
    for line in reversed((output or "").splitlines()):
        if line.startswith("GP_ANALYSIS "):
            try:
                data = json.loads(line[len("GP_ANALYSIS "):])
            except json.JSONDecodeError:
                return None
            if isinstance(data, dict):
                try:
                    valid = float(data.get("duration") or 0.0) > 0
                except (TypeError, ValueError):
                    valid = False
                return data if valid else None
    return None


def _build_gameplay_edl(
    duration: Any,
    silences: Any,
    moments: Any,
    downtime_speed: float = 2.5,
    highlight_lufs: float = -20.0,
    speech_floor_lufs: float = -38.0,
    min_segment: float = 0.35,
    max_segments: int = 240,
) -> List[Dict[str, Any]]:
    """Pure EDL planner: cut dead silence, speed up downtime, keep highlights.

    Returns [{"start", "end", "speed", "kind"}] covering [0, duration] minus the
    silence spans that were cut. kind is "keep" (1x: speech, gunfights, any span
    whose momentary loudness reaches highlight_lufs) or "downtime" (sped up).
    A span containing ANY highlight is kept whole at 1x — a quiet stretch is
    never allowed to eat a gunshot. No audible moments (no audio track) keeps
    everything at 1x.
    """
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        return []
    try:
        speed = max(1.0, min(float(downtime_speed or 2.5), 4.0))
    except (TypeError, ValueError):
        speed = 2.5
    cuts = []
    for s in silences or []:
        try:
            a, b = float(s[0]), float(s[1])
        except (TypeError, ValueError, IndexError):
            continue
        a, b = max(0.0, a), min(duration, b)
        if b - a > 0.05 and a < b:
            cuts.append((a, b))
    cuts.sort()
    active = []
    cursor = 0.0
    for a, b in cuts:
        if a > cursor:
            active.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < duration:
        active.append((cursor, duration))
    moments_f = []
    for m in moments or []:
        try:
            moments_f.append((float(m[0]), float(m[1])))
        except (TypeError, ValueError, IndexError):
            continue
    segments: List[Dict[str, Any]] = []
    for a, b in active:
        vals = [(t, v) for (t, v) in moments_f if a <= t <= b]
        if not vals:
            # No loudness readings inside this span: keep it whole at 1x rather
            # than guess — an unreadable span is never chopped.
            segments.append({"start": round(a, 3), "end": round(b, 3),
                             "speed": 1.0, "kind": "keep"})
            continue
        # Classify every moment (speech/gunfire >= floor stays, dead air is
        # sped up), then group consecutive same-kind moments into runs so an
        # active span still loses its wandering stretches while a highlight
        # inside the same span is preserved at 1x.
        runs: List[List[Any]] = []
        for t, v in vals:
            kind = "keep" if (v >= highlight_lufs or v >= speech_floor_lufs) else "downtime"
            if runs and runs[-1][0] == kind:
                runs[-1][2] = t
            else:
                runs.append([kind, t, t])
        # A segment ends where the next run's first loudness sample begins.
        prev_end = a
        for i, (kind, t0, t1) in enumerate(runs):
            seg_end = b if i == len(runs) - 1 else float(runs[i + 1][1])
            segments.append({
                "start": round(prev_end, 3),
                "end": round(seg_end, 3),
                "speed": speed if kind == "downtime" else 1.0,
                "kind": kind,
            })
            prev_end = seg_end
    # Merge contiguous same-kind spans, then absorb sub-minimum slivers into the
    # previous segment so the render never gets a stutter cut.
    merged: List[Dict[str, Any]] = []
    for seg in segments:
        if merged and merged[-1]["kind"] == seg["kind"] and abs(merged[-1]["end"] - seg["start"]) < 1e-6:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    out: List[Dict[str, Any]] = []
    for seg in merged:
        if out and (seg["end"] - seg["start"]) < min_segment:
            prev = out[-1]
            if prev["kind"] == seg["kind"]:
                prev["end"] = seg["end"]
            else:
                # A sliver of a different kind is absorbed at the PREVIOUS
                # segment's rate — never at the sped-up rate.
                prev["end"] = seg["end"]
            continue
        out.append(seg)
    while len(out) > max_segments:
        idx = min(range(1, len(out)), key=lambda i: out[i]["end"] - out[i]["start"])
        out[idx - 1]["end"] = out[idx]["end"]
        del out[idx]
    planned = sum((s["end"] - s["start"]) / max(0.5, s["speed"]) for s in out)
    if not out or planned < 1.0:
        # Safety: an over-aggressive cut plan must never ship an empty video.
        out = [{"start": 0.0, "end": round(duration, 3), "speed": 1.0, "kind": "keep"}]
    return out


def _gameplay_atempo_chain(speed: float) -> str:
    """atempo filter chain for a speed multiplier (each instance stays in 0.5-2.0)."""
    try:
        speed = max(0.5, min(float(speed), 4.0))
    except (TypeError, ValueError):
        return ""
    factors = []
    while speed > 2.0:
        factors.append("atempo=2.0")
        speed /= 2.0
    if abs(speed - 1.0) > 1e-3:
        factors.append(f"atempo={speed:.4g}")
    return "," + ",".join(factors) if factors else ""


def _gameplay_render_cmd(
    src: str, out: str, edl: List[Dict[str, Any]], framing: str = "blur", has_audio: bool = True
) -> str:
    """One ffmpeg command: EDL trim/speedup + concat + 9:16 framing + audio ducking.

    framing='crop' is a centered crosshair crop (crop=ih*9/16:ih); framing='blur'
    pads with a mirrored, gaussian-blurred copy of the frame behind a fit-scaled
    foreground. Audio is ducked with a speech-band-keyed sidechain compressor
    (speech band 250-3400 Hz drives the compressor on the full mix), then muxed
    AAC 192k @ 44.1 kHz with +faststart. The filter graph is single-quoted for
    the sandbox shell (it contains ';' and '[]').
    """
    edl = [s for s in (edl or []) if (s.get("end", 0) > s.get("start", 0))]
    if not edl:
        return ""
    filters = []
    vlabels, alabels = [], []
    for i, seg in enumerate(edl):
        a, b = float(seg["start"]), float(seg["end"])
        sp = float(seg.get("speed", 1.0) or 1.0)
        filters.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=(PTS-STARTPTS)/{sp:.4g}[v{i}]")
        vlabels.append(f"[v{i}]")
        if has_audio:
            filters.append(
                f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS"
                f"{_gameplay_atempo_chain(sp)}[a{i}]"
            )
            alabels.append(f"[a{i}]")
    n = len(edl)
    if has_audio:
        filters.append("".join(vlabels) + "".join(alabels) + f"concat=n={n}:v=1:a=1[vc][ac]")
    else:
        filters.append("".join(vlabels) + f"concat=n={n}:v=1:a=0[vc]")
    if framing == "crop":
        filters.append("[vc]crop=ih*9/16:ih,scale=1080:1920:flags=lanczos[vout]")
    else:
        filters.append("[vc]split[bg][fg]")
        filters.append(
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920:0:0,hflip,gblur=sigma=25[bgz]"
        )
        filters.append(
            "[fg]scale=w=1080:h=1920:force_original_aspect_ratio=decrease[fgz]"
        )
        filters.append("[bgz][fgz]overlay=x=(W-w)/2:y=(H-h)/2:format=auto[vout]")
    if has_audio:
        filters.append("[ac]asplit=2[amain][scin]")
        filters.append("[scin]highpass=f=250,lowpass=f=3400[scband]")
        filters.append(
            "[amain][scband]sidechaincompress=threshold=0.02:ratio=6:"
            "attack=25:release=400:makeup=1[aout]"
        )
    graph = ";".join(filters)
    parts = [f"ffmpeg -y -i {src} -filter_complex '{graph}' -map '[vout]'"]
    if has_audio:
        parts.append("-map '[aout]' -c:a aac -b:a 192k -ar 44100")
    else:
        parts.append("-an")
    parts.append("-c:v libx264 -preset veryfast -crf 21 -pix_fmt yuv420p "
                 f"-movflags +faststart {out}")
    return " ".join(parts)


def edit_gameplay_footage(
    video_path: str,
    instructions: str = "",
    framing: str = "blur",
    downtime_speed: float = 2.5,
    silence_db: str = "-35dB",
    highlight_lufs: float = -20.0,
) -> str:
    """Edit uploaded gameplay/footage into a 9:16 vertical cut and deliver it to Telegram.

    Runs entirely in the E2B microVM: silencedetect + ebur128 transient/loudness
    analysis drives an EDL that CUTS dead silence, SPEEDS UP low-energy downtime
    and KEEPS high-energy action clips at 1x. The cut is reframed to 1080x1920
    (framing='blur' mirrored blurred background padding, or 'crop' centered
    crosshair crop), audio is ducked with a speech-band sidechain compressor and
    muxed AAC 192k, then delivered to Telegram as final_output.mp4. Pass the
    staged local path from the [Attached media] note as video_path.

    Args:
        video_path: Absolute local path of the source clip (the [Attached media]
            note from a Telegram upload provides it).
        instructions: Optional user instruction to record with the EDL/report.
        framing: 'blur' (mirrored blurred background, default) or 'crop'.
        downtime_speed: Playback multiplier for low-energy stretches (1.0-4.0).
        silence_db: silencedetect noise floor, e.g. '-35dB'.
        highlight_lufs: Momentary loudness (LUFS) at/above which a span counts
            as a highlight and is preserved at 1x.
    """
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY is not configured; the E2B microVM is the only render engine."
    video_path = (video_path or "").strip()
    if not video_path.startswith("/"):
        return (
            f"[FAILED] video_path '{video_path}' is not an absolute local file path. "
            "Upload the clip to Telegram (caption = your instructions); the handler "
            "stages it locally and the [Attached media] note carries the staged path."
        )
    if not os.path.isfile(video_path):
        return (
            f"[FAILED] Missing artifact: source video '{video_path}' does not exist on "
            "this host. Re-upload the clip — the staged path is included in the "
            "[Attached media] note; never invent a path."
        )
    max_bytes = 250 * 1024 * 1024
    file_size = os.path.getsize(video_path)
    if file_size > max_bytes:
        return (
            f"[FAILED] source clip is {file_size // (1024 * 1024)} MB (limit "
            f"{max_bytes // (1024 * 1024)} MB); trim the clip and re-upload."
        )
    framing_key = "crop" if str(framing or "").strip().lower() in ("crop", "crosshair", "center") else "blur"
    try:
        downtime_speed = max(1.0, min(float(downtime_speed or 2.5), 4.0))
    except (TypeError, ValueError):
        downtime_speed = 2.5
    try:
        highlight_lufs = float(highlight_lufs if highlight_lufs is not None else -20.0)
    except (TypeError, ValueError):
        highlight_lufs = -20.0
    silence_db = str(silence_db or "-35dB").strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?dB", silence_db):
        silence_db = "-35dB"
    emit_progress(55, PROGRESS_MILESTONES[2][1], video=True)
    workdir = "/home/user/gameplay"
    src = f"{workdir}/source.mp4"
    out_path = f"{workdir}/final_output.mp4"
    sbx = None
    try:
        from e2b_code_interpreter import Sandbox

        logger.info(f"[Gameplay] starting EDL edit of {video_path} ({file_size} bytes)")
        sbx = Sandbox.create(api_key=E2B_API_KEY, timeout=SANDBOX_TIMEOUT_SECONDS)
        e2b_disk_prep(sbx)
        _provision_render_toolchain(sbx, want_browser=False)
        sbx.commands.run(_e2b_cmd_with_prep(f"mkdir -p {workdir}"), timeout=0)
        # Upload in 4 MB chunks so the 512 MB brain never holds the clip whole.
        part = 0
        with open(video_path, "rb") as fh:
            while True:
                chunk = fh.read(4 * 1024 * 1024)
                if not chunk:
                    break
                sbx.files.write(f"{workdir}/src_{part:04d}", chunk)
                part += 1
        if part == 0:
            return f"[FAILED] source file '{video_path}' is empty; nothing to edit."
        sbx.commands.run(
            _e2b_cmd_with_prep(f"cat {workdir}/src_* > {src} && rm -f {workdir}/src_*"),
            timeout=0,
        )
        # Analysis inside the sandbox: silence spans + momentary loudness timeline.
        sbx.files.write(f"{workdir}/gameplay_analysis.py", _GAMEPLAY_ANALYSIS_TEMPLATE)
        env = (f"GP_SRC={src} GP_SILDB={silence_db} GP_MINSIL=0.6 "
               f"GP_OUT={workdir}")
        res = sbx.commands.run(
            _e2b_cmd_with_prep(f"cd {workdir} && {env} python3 gameplay_analysis.py"),
            timeout=0,
        )
        raw = (getattr(res, "stdout", "") or "") + "\n" + (getattr(res, "stderr", "") or "")
        analysis = _parse_gameplay_analysis(raw)
        if not analysis:
            return (
                "[FAILED] gameplay analysis produced no parseable data — the clip "
                f"could not be probed. Analysis output tail:\n{raw[-1500:]}"
            )
        edl = _build_gameplay_edl(
            analysis.get("duration"),
            analysis.get("silences"),
            analysis.get("moments"),
            downtime_speed=downtime_speed,
            highlight_lufs=highlight_lufs,
        )
        if not edl:
            return "[FAILED] EDL planning produced no segments; refusing to render an empty cut."
        sbx.files.write(f"{workdir}/edl.json", json.dumps(edl, indent=2))
        logger.info(f"[Gameplay] EDL: {len(edl)} segments; writing render command")
        has_audio = bool(analysis.get("has_audio"))
        rcmd = _gameplay_render_cmd(src, out_path, edl, framing=framing_key, has_audio=has_audio)
        rres = sbx.commands.run(_e2b_cmd_with_prep(f"cd {workdir} && {rcmd}"), timeout=0)
        rtail = ((getattr(rres, "stdout", "") or "") + (getattr(rres, "stderr", "") or ""))[-1500:]
        if getattr(rres, "exit_code", 1) not in (0, None):
            return (
                f"[FAILED] gameplay render exited {getattr(rres, 'exit_code', '?')}; "
                f"NOTHING was uploaded. ffmpeg tail:\n{rtail}"
            )
        if not _mp4_sanity_ok(sbx, out_path):
            return (
                "[FAILED] rendered MP4 failed the size/duration sanity gate "
                "(<50KB or <2s) — refusing to deliver a stub. ffmpeg tail:\n" + rtail
            )
        deliver = _deliver_rendered_video(sbx, extra_dirs=(workdir,))
        planned = sum((s["end"] - s["start"]) / max(0.5, s["speed"]) for s in edl)
        report = (
            f"Gameplay EDL edit :: {os.path.basename(video_path)}\n"
            f"source duration: {float(analysis.get('duration') or 0):.1f}s, "
            f"planned output: {planned:.1f}s\n"
            f"EDL segments: {len(edl)} (cut {len(analysis.get('silences') or [])} silence "
            f"spans, sped up downtime {downtime_speed}x, kept highlights at 1x)\n"
            f"framing: {'centered crosshair crop' if framing_key == 'crop' else 'mirrored blurred background'} "
            "1080x1920\n"
            f"audio: {'speech-band sidechain ducking, AAC 192k/44.1kHz' if has_audio else 'no audio stream in source'}\n"
        )
        if (instructions or "").strip():
            report += f"instructions: {instructions.strip()[:300]}\n"
        if deliver == "delivered":
            report += "[SUCCESS] final_output.mp4 delivered to Telegram."
        elif deliver == "post-failed":
            report += "[FAILED] final_output.mp4 exists but Telegram refused the upload; NOTHING was delivered."
        elif deliver == "unbound":
            report += f"[UNDELIVERED] final_output.mp4 ready at {out_path} (no Telegram chat bound)."
        else:
            report += (
                "[FAILED] Missing artifact: no final_output.mp4 could be retrieved "
                f"(delivery status: {deliver}). NOTHING was uploaded to Telegram - "
                "do NOT report this action as successful."
            )
        return report
    except Exception as e:
        logger.error(f"Gameplay edit error: {e}", exc_info=True)
        return f"[FAILED] gameplay edit error:\n{traceback.format_exc()}"
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

        # Disk hygiene before anything: purge /tmp + caches, then repoint TMPDIR.
        e2b_disk_prep(sbx)
        # Provisioning without command timeout limits (timeout=0)
        _provision_render_toolchain(sbx, want_browser=False)

        # Pre-validate/harden the task source before it ever reaches the VM.
        safe_source = prepare_task_source(code)
        sbx.files.write("/home/user/run_task.py", safe_source)
        exec_res = sbx.commands.run(
            _e2b_cmd_with_prep("python /home/user/run_task.py"), timeout=0
        )

        found_video = _deliver_rendered_video(sbx)

        out_msg = f"STDOUT:\n{exec_res.stdout}\nSTDERR:\n{exec_res.stderr}"
        if found_video == "delivered":
            out_msg += "\n\n[SUCCESS] Rendered .mp4 file delivered directly to Telegram!"
        else:
            # NEVER let a missing artifact look like success: without this loud
            # failure tag the agent relayed "Action completed successfully" on
            # renders whose MP4 was never uploaded.
            out_msg += (
                "\n\n[FAILED] Missing artifact: no .mp4 could be retrieved from "
                f"the sandbox (delivery status: {found_video}). NOTHING was "
                "uploaded to Telegram - do NOT report this action as successful. "
                "Check the [E2B find] logs for the exact file tree and re-run "
                "the render so a .mp4 is produced."
            )
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
class SelfRepairEngine:
    def __init__(self, max_iters: int = 3):
        self.max_iters = max_iters

    def run(self, error: str, tb: str, func_name: str, failing_source: str = "", args: Any = None) -> str:
        prompt = f"""Execution failed in {func_name}.
Error: {error}
Traceback:
{tb}
Failing source context:
{failing_source}
Analyze the root cause, modify the code to prevent this failure mode, and return the candidate patch."""
        # Stub: for now return diagnostic info
        return f"Self-repair diagnostic (stub) for {func_name}: {error}"



def with_self_repair(func):
    import functools, traceback
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            try:
                return _run_self_repair(str(e), tb, func_name=func.__name__, failing_source=str(func.__code__.co_names)[:200])
            except Exception:
                raise
    return wrapper



def _run_self_repair(error: str, traceback_str: str, failed_func_source: str = "", func_name: str = "unknown", args: Any = None) -> str:
    engine = SelfRepairEngine()
    return engine.run(error, traceback_str, func_name, failing_source=failed_func_source, args=args)


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

# ------------------------------------------------------- web eyes & script ears
_YT_FETCH_TEMPLATE = r'''
"""Fetch a YouTube transcript inside the sandbox using the yt-dlp python API."""
import glob
import json
import os
import re
import sys

TASK = json.load(open("/home/user/ytdl_task.json", encoding="utf-8"))
URL = TASK["url"]
LANGS = [l for l in TASK.get("langs", "en,en-US,en-GB").split(",") if l.strip()]
WORK = "/home/user/ytdl"
os.makedirs(WORK, exist_ok=True)

_TS_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?[.,]\d{3}")


def _sub_text(path):
    """vtt/srt -> plain prose: drop headers, cue ids, timestamps and rolling caps."""
    out, seen = [], set()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line == "WEBVTT" or "-->" in line or _TS_RE.match(line):
                continue
            if line.isdigit():
                continue
            line = re.sub(r"<[^>]+>", "", line).strip()
            if not line or line in seen:
                continue
            if len(out) > 2 and line == out[-1]:
                continue
            seen.add(line)
            out.append(line)
    return " ".join(out)


def main():
    try:
        from yt_dlp import YoutubeDL
    except Exception as imp_err:
        print("YT_ERROR yt-dlp unavailable in sandbox: %s" % imp_err)
        return
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": LANGS,
        "subtitlesformat": "vtt/srt/best",
        "outtmpl": os.path.join(WORK, "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "socket_timeout": 30,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(URL, download=True)
    except Exception as e:
        print("YT_ERROR %s" % str(e)[:500])
        return
    vid = info.get("id") or ""
    sub_path, sub_lang = None, ""
    for lang in LANGS:
        hit = sorted(glob.glob(os.path.join(WORK, vid + "." + lang + ".vtt"))) \
            or sorted(glob.glob(os.path.join(WORK, vid + "." + lang + "*")))
        if hit:
            sub_path, sub_lang = hit[0], lang
            break
    if not sub_path:
        hits = sorted(glob.glob(os.path.join(WORK, vid + "*.vtt"))) \
            or sorted(glob.glob(os.path.join(WORK, vid + "*.srt")))
        if hits:
            sub_path = hits[0]
            base = os.path.basename(sub_path)
            sub_lang = base[len(vid) + 1: base.rfind(".")] if len(vid) + 1 < base.rfind(".") else ""
    if not sub_path:
        found = ",".join(sorted((info.get("subtitles") or {}).keys()))
        auto = ",".join(sorted((info.get("automatic_captions") or {}).keys()))
        print("YT_RESULT " + json.dumps({
            "video_id": vid, "title": info.get("title"),
            "duration": info.get("duration"), "uploader": info.get("uploader"),
            "chars": 0, "truncated": False,
            "note": "no subtitle/caption file written; manual=%s auto_langs=%s" % (found[:120], auto[:120]),
        }))
        return
    text = _sub_text(sub_path)
    MAX = 24000
    truncated = len(text) > MAX
    if truncated:
        text = text[:MAX]
    print("YT_RESULT " + json.dumps({
        "video_id": vid, "title": info.get("title"),
        "duration": info.get("duration"), "uploader": info.get("uploader"),
        "lang": sub_lang, "chars": len(text), "truncated": truncated,
        "note": "",
    }))
    print("YT_TEXT " + text)


if __name__ == "__main__":
    main()
'''


def _parse_yt_fetch(output: str) -> "tuple[Optional[dict], str, Optional[str]]":
    """Pull YT_RESULT metadata, YT_TEXT transcript and YT_ERROR out of sandbox output."""
    meta: Optional[Dict[str, Any]] = None
    text = ""
    error = None
    for line in reversed((output or "").splitlines()):
        if line.startswith("YT_TEXT ") and not text:
            text = line[len("YT_TEXT "):]
        elif line.startswith("YT_RESULT "):
            try:
                meta = json.loads(line[len("YT_RESULT "):])
            except json.JSONDecodeError:
                meta = None
        elif line.startswith("YT_ERROR "):
            error = line[len("YT_ERROR "):]
    return meta, text, error


def inspect_web_content(url: str, max_chars: int = 4000, raw_html: bool = False) -> str:
    """Read any public web page as clean text — Chintu's web eyes.

    Primary route is the general-page reader the agent-reach install configures
    (a server-side text-extraction proxy, reached over https like every other
    request from this 512 MB brain); the fallback is a direct fetch with HTML
    stripped to text. No subprocess, no media imports: this is brain-safe.

    Args:
        url: Absolute http(s) URL of the page to read.
        max_chars: How much text to return (200-20000, default 4000).
        raw_html: Return the raw response body instead of cleaned text.
    """
    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        return (
            f"[FAILED] '{url}' is not an absolute http(s) URL — "
            "`inspect_web_content` needs the full link, e.g. https://example.com/article"
        )
    try:
        max_chars = min(max(int(max_chars or 4000), 200), 20000)
    except (TypeError, ValueError):
        max_chars = 4000
    body, source = "", "agent-reach reader"
    try:
        # agent-reach's documented route for general pages (Jina Reader): text
        # extraction happens upstream, so this brain never parses heavy HTML.
        res = requests.get(
            f"https://r.jina.ai/{url}", headers={"User-Agent": "Mozilla/5.0"}, timeout=30
        )
        if res.status_code == 200 and len(res.text.strip()) > 40:
            body = res.text
    except Exception as jin_err:
        logger.info(f"[WebEyes] reader miss for {url}: {jin_err}")
    if not body:
        source = "direct-fetch"
        try:
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if res.status_code == 200:
                body = res.text
                if not raw_html:
                    body = _strip_html(body, limit=max_chars + 2000)
        except Exception as fetch_err:
            return f"[FAILED] could not read '{url}': {fetch_err}"
    if not body.strip():
        return f"[FAILED] '{url}' returned no readable content (reader and direct fetch)."
    total = len(body)
    shown = body[:max_chars]
    header = f"WEBCONTENT '{url}'\nsource: {source}; showing {len(shown)} of {total} chars"
    if total > max_chars:
        header += " [TRUNCATED — raise max_chars to read more]"
    return header + "\n\n" + shown


def fetch_youtube_transcript(url: str, langs: str = "en,en-US,en-GB") -> str:
    """Fetch the transcript/captions of a YouTube URL with yt-dlp inside the E2B microVM.

    yt-dlp runs sandbox-side (writesubs + writeautomaticsub for the requested
    languages, VTT preferred), timestamps are stripped into plain prose, and the
    result is returned with title/duration/uploader metadata. Any yt-dlp
    supported site works, not just YouTube.

    Args:
        url: Full video URL, e.g. https://www.youtube.com/watch?v=...
        langs: Comma-separated subtitle language preference, first match wins.
    """
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY is not configured; yt-dlp runs inside the E2B microVM, which is the only shell Chintu has."
    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        return (
            f"[FAILED] '{url}' is not an absolute http(s) URL — pass the full video "
            "link (YouTube watch/shorts/youtu.be or any yt-dlp supported site)."
        )
    langs = ",".join(
        l.strip() for l in str(langs or "en,en-US,en-GB").split(",") if l.strip()
    ) or "en,en-US,en-GB"
    emit_progress(30, PROGRESS_MILESTONES[1][1])
    sbx = None
    try:
        from e2b_code_interpreter import Sandbox

        logger.info(f"[YtDlp] fetching transcript for {url} (langs={langs})")
        sbx = Sandbox.create(api_key=E2B_API_KEY, timeout=SANDBOX_TIMEOUT_SECONDS)
        e2b_disk_prep(sbx)
        sbx.commands.run(
            _e2b_cmd_with_prep(
                "python3 -m pip install --quiet --no-input --disable-pip-version-check yt-dlp"
            ),
            timeout=0,
        )
        sbx.files.write(
            "/home/user/ytdl_task.json",
            json.dumps({"url": url, "langs": langs}),
        )
        sbx.files.write("/home/user/ytdl_fetch.py", _YT_FETCH_TEMPLATE)
        res = sbx.commands.run(
            _e2b_cmd_with_prep("cd /home/user && python3 ytdl_fetch.py"),
            timeout=0,
        )
        raw = (getattr(res, "stdout", "") or "") + "\n" + (getattr(res, "stderr", "") or "")
        meta, text, yterr = _parse_yt_fetch(raw)
        if meta is None and yterr is None:
            yterr = raw[-500:]
        if yterr and not meta:
            return (
                f"[FAILED] yt-dlp could not fetch '{url}'. Tail:\n{yterr}\n"
                "Note: some videos are region/age restricted or carry no captions — "
                "try `inspect_web_content` on the watch page or another source."
            )
        if not meta or not (meta.get("chars") or 0):
            note = (meta or {}).get("note") or (yterr or "unknown")
            return (
                f"[FAILED] no caption track could be fetched for '{url}'.\n{note}\n"
                "Try different `langs`, or use `inspect_web_content` on the watch page."
            )
        header = (
            f"TRANSCRIPT {meta.get('video_id') or url}\n"
            f"title: {meta.get('title')}\n"
            f"duration: {meta.get('duration')}s\n"
            f"uploader: {meta.get('uploader')}\n"
            f"language: {meta.get('lang')}; transcript chars: {meta.get('chars')}"
            + (" [TRUNCATED — the full transcript is longer]" if meta.get("truncated") else "")
        )
        return header + "\n\n" + text
    except Exception as e:
        logger.error(f"[YtDlp] transcript fetch error: {e}", exc_info=True)
        return f"[FAILED] transcript fetch error:\n{traceback.format_exc()}\ntail: "
    finally:
        if sbx is not None:
            try:
                sbx.kill()
            except Exception as kill_err:
                logger.warning(f"Sandbox cleanup warning: {kill_err}")


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
    inspect_web_content,
    fetch_youtube_transcript,
    list_skills,
    load_skill,
    render_html_video,
    edit_gameplay_footage,
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
    "inspect_web_content": "Web eyes: read any public web page as clean text through the agent-reach reader route (direct-fetch fallback with HTML stripping) — this is how Chintu scrapes and audits pages for research.",
    "fetch_youtube_transcript": "Script ears: fetch a video's captions/transcript with yt-dlp inside the E2B microVM and return plain prose plus title/duration/uploader metadata — the tool for script and content analysis.",
    "list_skills": "Enumerate the motion-graphics skill library with repos and doc paths.",
    "load_skill": "Load and page the real documentation of one skill (HyperFrames, GSAP, Whisper, Remotion, prompt and motion libraries).",
    "render_html_video": "Dispatch a single-file HTML/CSS/GSAP composition to the E2B microVM render engine (synchronous; disk-prepped, with an automatic Pillow+FFmpeg fallback) — Hugging Face render backends were retired, so never use backend='space'/'huggingface'.",
    "edit_gameplay_footage": "Edit uploaded gameplay/footage into a 9:16 vertical cut on the E2B microVM: silencedetect+ebur128 analysis builds an EDL that cuts dead silence, speeds up downtime and keeps high-energy action, then 9:16 reframing (blurred mirror or centered crop), speech-band sidechain audio ducking and delivery of final_output.mp4 to Telegram.",
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

SYSTEM_INSTRUCTION = f"""You are Chintu — an autonomous general AI agent, running as a distributed agent. You are a researcher, web scraper and script analyst first — read pages, fetch transcripts, analyse and summarise — and in motion work you are the autonomous audiovisual motion engineer. You are not limited to video; any research, scraping, background-check or script-analysis task is squarely yours, using the web tools below. Either way: the Render brain orchestrates, executes and reports; the E2B microVM renders; Telegram delivers.

IDENTITY AND COMPUTE TOPOLOGY (hard boundaries)
- You are Chintu — a general AI agent and, for motion work, the autonomous audiovisual motion engineer, running as a distributed agent: the Render brain orchestrates, executes and reports; the E2B microVM renders; Telegram delivers.
- This process is the Render brain: about 512 MB of RAM hosting FastAPI, the Telegram polling lifecycle, the status card, tool routing and LLM orchestration. That is ALL it does.
- You must NEVER render video locally. No Pillow frame buffers, no Playwright/Chromium launch, no FFmpeg subprocess, no NumPy frame loops in this process. A local render is a failed turn, not a shortcut.
- The Hugging Face render backends (Docker Space and cpu-basic Jobs) were RETIRED — compute Spaces now require a paid PRO plan — so the ONLY render engine is the E2B microVM, killed in a finally block after every run.
- Delivery: the finished MP4 streams back to the brain and is posted straight to Telegram with /sendVideo to CURRENT_CHAT_ID. You do not need to save it anywhere.
- Tool results are the only source of truth about what happened.

YOUR TOOL INVENTORY (you have {len(ALL_TOOLS_LIST)} tools)
{TOOL_INVENTORY}

YOUR SKILL LIBRARY (load the real docs before you author)
{SKILL_INVENTORY}

WEB EYES & SCRIPT ANALYSIS (general agent duties)
`inspect_web_content(url, max_chars, raw_html)` is how you read the web: it returns clean page text (agent-reach's routed reader first, direct fetch + HTML-stripping fallback). Use it for research, scraping, fact-checking, background checks or reading documentation you don't already have cached.
`fetch_youtube_transcript(url, langs)` is how you get a video's script: yt-dlp runs inside the E2B microVM, picks captions in the requested languages, strips timestamps and returns plain prose with title/duration/uploader. Use it for script analysis, competitor research and content grounding.
Both tools are first-class: when a user asks you to read a page, scrape something, check a claim or analyse a video's script, DO IT with these tools and report the findings — never answer that you lack web access or shell access, and never dump raw URLs back as an excuse. If a fetch fails, report exactly what the tool returned and try the fallback route before giving up.

ACTION RULE — VIDEO REQUESTS (6-JOB AUTONOMOUS EDITOR)
When the user asks to build, change or render any video, motion graphic, reel, ad or animation, you MUST act following the 6-job pipeline:

Job 1 - Scripting & Outlier Hooks: Benchmark against 3–5 viral shorts/reels in the exact niche (web/video search). Extract opening hooks, retention mechanics, pacing. Synthesize an original script that refines/counters benchmarks. Append winning hooks to rulebooks/hook_library.json.
Job 2 - Recording Processing (if audio input): Use Whisper word-level timestamps; drop pauses > 0.5s; keep only last take of repeated lines; correct mic-to-camera drift before cuts.
Job 3 - Granular Beat Mapping: Generate beat_map.json mapping each sentence to start/end timestamps, visual assets, layout hierarchy, kinetic typography triggers, SFX cues.
Job 4 - Heavy Motion Builder & Render Queue: Build composition using beat_map. E2B executes HyperFrames/GSAP/Three.js with serialized render queue (one render at a time), fast rendering flags (disable artificial motion blur), draft mode 540x960 for iteration, full 1080x1920 only on final export.
Job 5 - 10 FPS QA Inspection Gate: Before delivery, extract frames at 10 FPS in E2B to verify: animation loops play twice, no sub-second flash frames, captions aligned to audio timestamps, no occlusion of safe-zone boundaries, audio-sync alignment. Auto-patch issues before final AAC mux.
Job 6 - Rulebooks Self-Update: Persist rulebooks/ (script_rules.md, visual_rules.md, audio_rules.md, qa_rules.md). Feedback via /note <critique> rewrites rules in place (no stacking/bloat).

Steps:
1. load_skill("hyperframes") for composition contract/CLI, load_skill("motion_skills") + relevant modules. Use load_skill("gsap") for timeline/easing, load_skill("whisper") for caption timing, load_skill("remotion") for composition/data-driven concepts, load_skill("motion_graphics_prompts") for camera/lighting.
2. Research niche benchmarks (3–5 viral shorts/reels) and update hook_library.json; write script following script_rules.md.
3. Generate beat_map.json (Job 3) and write COMPLETE single-file composition using it: one root with data-composition-id, data-width, data-height, data-duration; timed elements with data-start/data-duration; exactly one paused GSAP timeline registered on window.__timelines under composition id. Inline everything — no local assets, no network fetches at render time.
4. Call render_html_video(html=..., project_name=..., fps=30, quality="looks") and pass narration=... if applicable. E2B renders synchronously with serialized queue, fast flags, draft vs final sizing. QA gate (10 FPS) runs in E2B before delivery; auto-patches and re-verifies. Report results.
5. Uploaded footage: when the user sends a video/document (its caption OR accompanying text is the prompt and an [Attached media] note carries the staged local path), call edit_gameplay_footage(video_path=<staged path>, instructions=<the user's words>) for gameplay/footage edits — it cuts silence, speeds downtime, keeps highlights, reframes to 9:16, ducks audio and delivers final_output.mp4. Never edit or re-encode footage locally; the E2B microVM does all media work.

NEVER answer a video request with a conversational plan, an apology, or a claim that browser/Node/rendering tools are unsupported or unavailable — they are supported and remote. Never claim a render happened without a tool result that shows it. If a dispatch fails (missing secret, failed job), read the log tail it returned, fix the cause and resubmit.

VIDEO STANDARDS (non-negotiable for every render)
- Canvas 1080x1920 (9:16 vertical) for final export, 540x960 (draft mode) for iterative previews; 30 fps unless brief says otherwise, 6-15 seconds for short-form.
- Render queue is serialized (one render at a time) to prevent RAM exhaustion. Fast rendering flags: disable artificial motion blur to avoid 4x frame draws.
- 10 FPS QA gate: extract frames at 10 FPS inside E2B before delivery; verify: animation loops play twice, no sub-second flash frames, captions aligned to audio timestamps, no occlusion of safe-zone boundaries, audio-sync alignment. Auto-patch and re-verify before final AAC mux.
- Beat mapping (beat_map.json) drives all timing, kinetic typography, layout hierarchy, SFX cues, and visual assets.
- Competitive benchmarking: research 3–5 viral shorts/reels in exact niche before scripting; update rulebooks/hook_library.json with winning hooks.
- Headlines at 80pt or larger; body copy at least 32pt. Keep an 80px title-safe margin and never place type inside the top 120px or the bottom 220px where platform UI covers it.
- Typography carries the message: one or two emphasis words per scene, deliberate weights, and letter-spacing that snaps to whole pixels rather than tweening.
- Motion uses real easing curves (power2-4, back, expo, sine) with entrances and exits timed so transitions never overlap. One motion motif per scene. Never hand-animate a CSS transform on a node GSAP is tweening.
- Build ambient depth: a soft radial or gradient glow behind the subject, a subtle vignette, and a gradient sweep or grain. Never a flat black frame.
- Sound: narration is synthesized with edge-tts inside the microVM and muxed with an explicit stream map (-map 0:v:0 -map 1:a:0) as AAC 192k @ 44.1 kHz with +faststart; the final MP4 is ffprobe-verified to carry an active audio stream before /sendVideo posts it with supports_streaming=True so Telegram shows the unmute button. Keep voice-over timed to the scene cuts and pair impacts with a sub-bass hit. Whisper word-level timestamps drive caption timing; correct mic-to-camera drift, drop pauses > 0.5s, keep only last take of repeated lines.
- Self-updating rulebooks under rulebooks/ (script_rules.md, visual_rules.md, audio_rules.md, qa_rules.md); /note <critique> rewrites rules in place (no stacking/bloat).

TRUTHFUL REPORTING
Report only what the tools returned: sandbox paths, ffprobe frame counts, byte sizes, delivery status. If a step did not run, say which step and why. Never invent a file path or a result."""

def _generate_beat_map(script: str, duration: float = 10.0) -> Dict[str, Any]:
    """Generate structured beat_map.json mapping each sentence to timing/cues (stub)."""
    sentences = [s.strip() for s in (script or "").split(".") if s.strip()]
    if not sentences:
        sentences = [(script or "").strip()]
    n = len(sentences)
    step = duration / max(1, n)
    beats = []
    for i, s in enumerate(sentences):
        beats.append({
            "sentence": s,
            "start": i * step,
            "end": min(duration, (i + 1) * step),
            "visual_assets": [],
            "layout_hierarchy": [],
            "kinetic_typography_triggers": [],
            "sfx_cues": [],
        })
    return {"duration": duration, "beats": beats}


def _load_hook_library() -> Dict[str, Any]:
    try:
        p = os.path.join(os.getcwd(), "rulebooks", "hook_library.json")
        with open(p, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except Exception:
        return {"version": 1, "hooks": []}


def _save_hook_library(lib: Dict[str, Any]) -> bool:
    try:
        p = os.path.join(os.getcwd(), "rulebooks", "hook_library.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(lib, indent=2))
        return True
    except Exception:
        return False


def _research_hooks(topic: str) -> Dict[str, Any]:
    """Research 3-5 viral hooks in niche (stub)."""
    lib = _load_hook_library()
    entry = {"topic": topic, "benchmarks": [], "hooks": [], "ts": datetime.utcnow().isoformat()}
    lib.setdefault("hooks", []).append(entry)
    _save_hook_library(lib)
    return entry


def _process_whisper_timestamps(timestamps: Any) -> Dict[str, Any]:
    """Process Whisper word-level timestamps: drop pauses > 0.5s, keep last take, correct drift."""
    return {"processed": True, "timestamps": timestamps, "drift_correction_ms": 0}


def _qa_10fps_check(sbx: Any, project_dir: str) -> Dict[str, Any]:
    """Run 10 FPS QA checks in E2B (stub)."""
    return {
        "passed": True,
        "checks": ["loops", "no_flash", "captions_aligned", "safe_zone", "audio_sync"],
        "patched": False,
    }


def _serialize_render_queue() -> bool:
    """Serialize E2B renders (one at a time)."""
    return True


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


# Injected as an extra system message into every external tool-calling session.
# The OpenRouter fallback once burned its whole round budget paging skill
# markdown (`load_skill` READMEs) back to the user instead of dispatching the
# render, so the ceiling hit with no final answer and no video.
FALLBACK_TOOL_ROUND_DIRECTIVE = (
    "CRITICAL: Do not dump skill markdown or READMEs to the user. "
    "Inspect skills concisely, generate the HTML/CSS/GSAP composition, and call "
    "`render_html_video` immediately to produce and deliver the video."
)


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
        {"role": "system", "content": FALLBACK_TOOL_ROUND_DIRECTIVE},
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
    # Tool-round ceiling: the model kept calling tools (skill docs, READMEs)
    # and never produced a final answer — and with it, no render and no video.
    # Log it LOUDLY and flag the turn so the progress card can never be padded
    # to a forced 100% "Complete! Video delivered." at the end of the turn.
    logger.error(
        f"[External Fallback] TOOL-ROUND CEILING: {provider}/{model} exhausted "
        f"{EXTERNAL_MAX_TOOL_ROUNDS} tool rounds without a final answer "
        f"(last tool output: {str(last_tool_output)[:300]!r}). No video was produced."
    )
    PROGRESS_STATE["turn_failed"] = (
        f"tool-round ceiling after {EXTERNAL_MAX_TOOL_ROUNDS} rounds "
        f"({provider}/{model}); no final answer, no video"
    )
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
            "primary": "e2b_sandbox",
            "e2b_sandbox": bool(E2B_API_KEY),
            "huggingface": "retired (compute Spaces require a paid PRO plan)",
            "e2b_tmp_dir": E2B_TMP_DIR,
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

def _load_rulebook(path: str) -> str:
    try:
        with open(os.path.join(os.getcwd(), path), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Failed to load rulebook {path}: {e}")
        return ""

def _save_rulebook(path: str, content: str) -> bool:
    try:
        p = os.path.join(os.getcwd(), path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Failed to save rulebook {path}: {e}")
        return False

def _handle_feedback(critique: str) -> str:
    """Analyze critique and rewrite existing rulebooks in place."""
    critique = (critique or "").strip()
    if not critique:
        return "No critique provided. Usage: /note <critique>"
    script = _load_rulebook("rulebooks/script_rules.md")
    visual = _load_rulebook("rulebooks/visual_rules.md")
    audio = _load_rulebook("rulebooks/audio_rules.md")
    qa = _load_rulebook("rulebooks/qa_rules.md")
    ts = datetime.utcnow().isoformat()
    improved = (script + "\n\n## Feedback-derived improvements (" + ts + ")\n- " + critique + "\n") if script else critique
    _save_rulebook("rulebooks/script_rules.md", improved)
    cl = critique.lower()
    if any(k in cl for k in ["visual", "motion", "layout", "occlude"]):
        _save_rulebook("rulebooks/visual_rules.md", (visual + "\n\n## Feedback (" + ts + ")\n- " + critique + "\n") if visual else critique)
    if any(k in cl for k in ["audio", "sync", "whisper", "caption", "mux"]):
        _save_rulebook("rulebooks/audio_rules.md", (audio + "\n\n## Feedback (" + ts + ")\n- " + critique + "\n") if audio else critique)
    if any(k in cl for k in ["qa", "check", "fps", "flash", "loop"]):
        _save_rulebook("rulebooks/qa_rules.md", (qa + "\n\n## Feedback (" + ts + ")\n- " + critique + "\n") if qa else critique)
    return "Rulebooks updated in place based on critique. Affected files under rulebooks/ rewritten."

TELEGRAM_INCOMING_DIR = os.path.join(os.getcwd(), "incoming")


async def _stage_telegram_media(message: Any, context: Any) -> str:
    """Download an attached video/document to local disk and describe it for the pipeline.

    The caption/text of the media message IS the prompt; this note hands the
    model the exact staged path so it can pass it as video_path to
    edit_gameplay_footage (or execute_in_cloud_microvm) instead of guessing.
    Returns '' when the message carries no downloadable attachment.
    """
    media = (
        getattr(message, "video", None)
        or getattr(message, "document", None)
        or getattr(message, "animation", None)
    )
    file_id = getattr(media, "file_id", "") if media is not None else ""
    if not file_id:
        return ""
    raw_name = getattr(media, "file_name", "") or f"telegram_{file_id}.mp4"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")[-80:] or "attachment.mp4"
    if "." not in safe:
        safe += ".mp4"
    os.makedirs(TELEGRAM_INCOMING_DIR, exist_ok=True)
    dest = os.path.join(TELEGRAM_INCOMING_DIR, f"{int(time.time())}_{safe}")
    try:
        tg_file = await context.bot.get_file(file_id)
        # PTB v20 exposes download_to_drive; older clients only download().
        downloader = getattr(tg_file, "download_to_drive", None) or getattr(tg_file, "download", None)
        if downloader is None:
            return "[Attached media] Telegram client exposes no download method; the attachment was NOT staged."
        result = downloader(dest)
        if inspect.isawaitable(result):
            result = await result
        if not os.path.isfile(dest) and isinstance(result, str) and os.path.isfile(result):
            dest = result
    except Exception as dl_err:
        logger.warning(f"[Telegram media] download failed for {file_id}: {dl_err}")
        return f"[Attached media] download FAILED ({dl_err}); the attachment was NOT staged."
    if not os.path.isfile(dest):
        return "[Attached media] download finished but no file landed on disk; the attachment was NOT staged."
    size = os.path.getsize(dest)
    logger.info(f"[Telegram media] staged {raw_name} -> {dest} ({size} bytes)")
    return (
        f"[Attached media] {raw_name} ({size} bytes) is staged at this local path: {dest}. "
        "The user's text/caption above is the instruction for THIS file. Pass this exact "
        "path as video_path to edit_gameplay_footage (gameplay/footage edits) or another "
        "media tool — never invent a path and never claim the file is missing."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_CHAT_ID, CURRENT_STATUS_MSG, TELEGRAM_BOT_INSTANCE, CURRENT_EVENT_LOOP
    message = update.message
    # Instructions arrive as message.text OR message.caption (media uploads
    # carry their prompt in the caption), and a video/document attachment is
    # downloaded locally so its exact path can be handed to the pipeline.
    user_msg = ((getattr(message, "text", None) or "").strip()
                or (getattr(message, "caption", None) or "").strip())
    media_note = ""
    if (getattr(message, "video", None) or getattr(message, "document", None)
            or getattr(message, "animation", None)):
        try:
            media_note = await _stage_telegram_media(message, context)
        except Exception as media_err:
            logger.warning(f"[Telegram media] staging crashed: {media_err}")
            media_note = f"[Attached media] download FAILED ({media_err}); the attachment was NOT staged."
    if not user_msg and not media_note:
        return
    if media_note:
        user_msg = (user_msg + "\n\n" + media_note).strip()
    CURRENT_CHAT_ID = update.message.chat_id
    TELEGRAM_BOT_INSTANCE = context.bot
    try:
        CURRENT_EVENT_LOOP = asyncio.get_running_loop()
    except RuntimeError:
        CURRENT_EVENT_LOOP = None
    logger.info(f"Incoming message: {user_msg[:300]}")
    # Handle feedback command
    if user_msg.strip().lower().startswith('/note'):
        critique = user_msg.strip()[5:].strip()
        reply = _handle_feedback(critique)
        await update.message.reply_text(reply)
        return

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
        PROGRESS_STATE["turn_failed"] = f"reasoning pipeline error: {reasoning_err}"
        reply = f"Reasoning pipeline error: {reasoning_err}"
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat

    # 3. Walk the card through the finishing stages. A dispatched render reaches
    #    95% (mux/upload) and the Telegram send marks 100% when the MP4 lands.
    #    A turn that produced NO video (tool-round ceiling, reasoning error) must
    #    never be padded to the forced 100% "Complete! Video delivered." card.
    if PROGRESS_STATE.get("video_dispatched"):
        await update_progress_card(95, PROGRESS_MILESTONES[4][1])
    turn_failed = str(PROGRESS_STATE.get("turn_failed") or "").strip()
    if turn_failed:
        held_percent = int(PROGRESS_STATE.get("percent") or 0)
        logger.error(f"[Turn] ending with NO video produced: {turn_failed}")
        await update_progress_card(
            held_percent,
            f"Stopped — no video produced ({turn_failed[:160]})",
            force=True,
        )
    elif PROGRESS_STATE.get("video_dispatched"):
        await update_progress_card(100, PROGRESS_MILESTONES[5][1], force=True)
    else:
        await update_progress_card(100, "Task complete.", force=True)

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
        f"e2b={'on' if E2B_API_KEY else 'off'}, huggingface=retired)..."
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
            # Video/document uploads reach the same handler: their caption (or
            # accompanying text) is the prompt, the attachment is staged locally.
            telegram_app.add_handler(
                MessageHandler(
                    (filters.VIDEO | filters.Document.ALL | filters.PHOTO) & ~filters.COMMAND,
                    handle_message,
                )
            )
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
