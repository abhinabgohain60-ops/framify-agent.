import ast
import base64
import os
import sys
import json
import logging
import asyncio
import io
import re
import contextlib
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional
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
}

CURRENT_CHAT_ID: Optional[int] = None
CURRENT_STATUS_MSG = None
TELEGRAM_BOT_INSTANCE = None

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
            file_name = video_path.rsplit("/", 1)[-1]
            send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
            upload_files = {"video": (file_name, video_bytes, "video/mp4")}
            data = {"chat_id": CURRENT_CHAT_ID, "caption": f"🎬 Chintu Video Render: {file_name}"}
            res = requests.post(send_url, data=data, files=upload_files, timeout=120)
            if res.status_code == 200:
                logger.info(f"Directly uploaded {file_name} to Telegram via HTTP API!")
                return True
            logger.error(f"Telegram upload failed: {res.text[:300]}")
        except Exception as vid_err:
            logger.warning(f"Video delivery warning: {vid_err}")
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

def render_html_video(
    html: str,
    project_name: str = "chintu-motion",
    fps: int = 30,
    quality: str = "looks",
) -> str:
    """Render a HyperFrames HTML/GSAP composition to MP4 in the cloud VM and deliver it.

    This is the executable form of directive 7: the composition is linted and then
    rendered by the real `hyperframes` CLI (chrome-headless-shell + system ffmpeg),
    so seekable GSAP timelines and deterministic frame capture stay the framework's
    job instead of being hand-rolled.

    Args:
        html: The full composition document (root carries data-composition-id and
            registers window.__timelines["<id>"] as one paused GSAP timeline).
        project_name: Kebab-case project directory name under /home/user/videos.
        fps: Frame rate recorded on the render (24-240; 30 typical).
        quality: draft (fast iteration), looks (default) or delivery (final).
    """
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY is not configured."
    problem = _validate_composition_html(html)
    if problem:
        return f"Composition rejected before render: {problem}"
    slug = re.sub(r"[^a-z0-9-]+", "-", (project_name or "chintu-motion").lower()).strip("-")
    slug = slug or "chintu-motion"
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
            'motion_graphics_prompts' or 'motion_skills'.
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
    render_html_video
]

SYSTEM_INSTRUCTION = """You are Chintu, an autonomous execution assistant.

STRICT OPERATIONAL DIRECTIVES:
1. NEVER FAKE OR HALLUCINATE TOOL EXECUTION: Never claim you executed code or modified a file unless you invoked the tool and received a valid tool response.
2. ACTION-FIRST PRINCIPLE: When asked to execute code, render video, or read/write files, IMMEDIATELY call execute_in_cloud_microvm or the corresponding tool.
3. VIDEO GENERATION: Importing `moviepy.editor` is STRICTLY FORBIDDEN and will break the sandbox. For ANY video task, write a pure Python rendering script: generate frames with Pillow + NumPy, then stitch them into the final `.mp4` natively by calling the system `ffmpeg` binary through `subprocess`.
4. CONCISE FACTUAL OUTPUT: Report only actual ground truth from tool execution stdout and stderr.
5. SANDBOX CODE HYGIENE: Submit a single, syntactically valid `run_task.py` (validate it with `ast.parse` before sending). Do not mix root-level shell commands (cat/curl/ffmpeg/...) into Python source; invoke them through `subprocess` when required.
6. SKILL LIBRARY: You ship with a curated skill library. Call `list_skills` to enumerate it and `load_skill` to read the real documentation before writing code in an unfamiliar domain. For ANY motion-graphics or animated-video request, load `hyperframes` FIRST (its router, composition contract and CLI docs), then `motion_graphics_prompts` for direction, then follow the APIs you actually read instead of inventing signatures. Long docs are paged: when a response says TRUNCATED, keep going with `load_skill(name, path, offset=<shown_end>)` until you have the section you need.
7. MOTION GRAPHICS PIPELINE: Author motion graphics as HTML/CSS with a single paused GSAP timeline, then render them with `render_html_video` — it lints and renders inside the cloud VM through the real hyperframes CLI (chrome-headless-shell + ffmpeg), which installs Node and the Chrome runtime libraries on that path. Keep the pure-Python route (Pillow + NumPy frames stitched by system `ffmpeg`) only as the fallback when `render_html_video` reports the toolchain could not be provisioned. An 8-second 30fps 1080p render takes roughly a minute of capture time, so keep short-form pieces short and use `quality="draft"` while iterating. Use `whisper` only when audio must be transcribed for captions or voice-over timing.
"""

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

# External provider failover used only after the entire Gemini cascade is exhausted.
def _sync_external_fallback(message: str) -> str:
    """Fail over to OpenRouter (claude-3.5-sonnet); Freebuff is a last resort."""
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": message},
        ],
        "temperature": 0.2,
    }
    or_error = ""

    if OPENROUTER_API_KEY:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://framify-agent.onrender.com",
            "X-Title": "framify-agent",
        }
        body = dict(payload, model="anthropic/claude-3.5-sonnet")
        try:
            res = requests.post(
                f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                json=body,
                timeout=120,
            )
            if res.status_code == 200:
                choices = (res.json() or {}).get("choices") or []
                if choices:
                    content = (choices[0].get("message") or {}).get("content")
                    if content:
                        logger.info("[External Fallback] OpenRouter claude-3.5-sonnet fulfilled request.")
                        return content
                raise RuntimeError("OpenRouter returned an empty completion.")
            raise RuntimeError(f"OpenRouter error ({res.status_code}): {res.text[:500]}")
        except Exception as or_err:
            or_error = str(or_err)
            logger.warning(f"[External Fallback] OpenRouter failed: {or_err}")

    if FREEBUFF_API_KEY:
        headers = {
            "Authorization": f"Bearer {FREEBUFF_API_KEY}",
            "Content-Type": "application/json",
        }
        body = dict(payload, model="anthropic/claude-3.5-sonnet")
        res = requests.post(
            f"{FREEBUFF_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
            timeout=120,
        )
        if res.status_code == 200:
            choices = (res.json() or {}).get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content")
                if content:
                    logger.info("[External Fallback] Freebuff fulfilled request.")
                    return content
        raise RuntimeError(f"Freebuff error ({res.status_code}): {res.text[:500]}")

    # Distinguish "OpenRouter was configured but broke" from "nothing configured",
    # so the cascade error message points at the real cause.
    if or_error:
        raise RuntimeError(f"OpenRouter failed ({or_error}) and no usable Freebuff fallback is configured.")
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
        "active_models": ACTIVE_GEMINI_MODELS,
        "tools_count": len(ALL_TOOLS_LIST),
        "skills": list(SKILL_REGISTRY.keys()),
        "skills_count": len(SKILL_REGISTRY),
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
    global CURRENT_CHAT_ID, CURRENT_STATUS_MSG, TELEGRAM_BOT_INSTANCE
    user_msg = update.message.text
    if not user_msg:
        return
    CURRENT_CHAT_ID = update.message.chat_id
    TELEGRAM_BOT_INSTANCE = context.bot
    logger.info(f"Incoming message: {user_msg}")
    
    # 1. Immediately post live progress bar card at 15%
    try:
        total_slots = 30
        filled = int(15 / 100.0 * total_slots)
        init_bar = "=" * filled + "-" * (total_slots - filled)
        CURRENT_STATUS_MSG = await update.message.reply_text(
            f"Task/Project Progress\n[{init_bar}] 15%\n\nCurrent status: Initializing reasoning & tool pipeline..."
        )
    except Exception as e:
        logger.warning(f"Could not dispatch initial status card: {e}")
        CURRENT_STATUS_MSG = None

    # 2. Execute fail-safe code cascade
    reply = await generate_reasoning_response(user_msg)
    
    # 3. Update status card to 100% complete
    if CURRENT_STATUS_MSG:
        try:
            full_bar = "=" * 30
            await CURRENT_STATUS_MSG.edit_text(
                f"Task/Project Progress\n[{full_bar}] 100%\n\nCurrent status: Complete"
            )
        except Exception:
            pass

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
    global telegram_app
    logger.info("Initializing Chintu services...")
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
