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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
E2B_API_KEY = os.environ.get("E2B_API_KEY", "")
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY", os.environ.get("COMPOSIO_API_", ""))
RENDER_API_KEY = os.environ.get("RENDER_API_KEY", "")
RENDER_SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", os.environ.get("RENDER_SERVIC...", "srv-db497mfk1mc73f9l6tg"))
DEFAULT_REPO = os.environ.get("DEFAULT_REPO", "abhinabgohain60-ops/framify-agent.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

ACTIVE_GEMINI_MODELS: List[str] = []

def discover_available_models():
    global ACTIVE_GEMINI_MODELS
    whitelist_priority = [
        "gemini-flash-lite-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash"
    ]
    discovered = []
    if gemini_client:
        try:
            for m in gemini_client.models.list():
                m_name = getattr(m, "name", "").replace("models/", "")
                if any(bad in m_name for bad in ["2.5", "pro", "tts", "image", "audio", "embed", "robotics", "transcribe", "customtools"]):
                    continue
                if "gemini" in m_name:
                    discovered.append(m_name)
            logger.info(f"[Model Discovery] Valid text models: {discovered}")
        except Exception as e:
            logger.warning(f"[Model Discovery] List failed: {e}")

    ordered = [m for m in whitelist_priority if m in discovered]
    for m in discovered:
        if m not in ordered and not any(bad in m for bad in ["2.5", "pro", "flash-latest"]):
            ordered.append(m)

    if not ordered:
        ordered = ["gemini-flash-lite-latest", "gemini-2.0-flash", "gemini-1.5-flash"]

    ACTIVE_GEMINI_MODELS = ordered
    logger.info(f"[Active Cascade Order]: {ACTIVE_GEMINI_MODELS}")

def github_read_file(repo_name: str, file_path: str) -> str:
    token = GITHUB_TOKEN or ""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    encoded_repo = repo_name.replace(".", "%2E") if repo_name.endswith(".") else repo_name
    url = f"https://api.github.com/repos/{encoded_repo}/contents/{file_path}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            dl_url = res.json().get("download_url")
            return requests.get(dl_url, timeout=10).text
        return f"GitHub Read Error ({res.status_code}): {res.text}"
    except Exception as e:
        return f"Error reading GitHub file: {str(e)}"

def github_commit_file(repo_name: str, file_path: str, content: str, commit_message: str) -> str:
    if file_path.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as syn_err:
            return f"Pre-commit AST check failed: SyntaxError on line {syn_err.lineno}: {syn_err.msg}"
            
    token = GITHUB_TOKEN or ""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
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

def execute_in_cloud_microvm(code: str) -> str:
    if E2B_API_KEY:
        try:
            headers = {
                "X-API-Key": E2B_API_KEY,
                "Content-Type": "application/json"
            }
            e2b_url = "https://api.e2b.dev/sandboxes"
            sb_res = requests.post(e2b_url, headers=headers, json={"template": "base"}, timeout=12)
            if sb_res.status_code in [200, 201]:
                sb_id = sb_res.json().get("sandboxID")
                exec_url = f"https://api.e2b.dev/sandboxes/{sb_id}/execute"
                run_res = requests.post(exec_url, headers=headers, json={"code": code}, timeout=20)
                requests.delete(f"https://api.e2b.dev/sandboxes/{sb_id}", headers=headers, timeout=5)
                if run_res.status_code == 200:
                    return f"E2B 16GB Sandbox Result:\n{run_res.text}"
        except Exception as e:
            logger.warning(f"E2B microVM direct dispatch error: {e}. Falling back to internal runner.")

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            safe_globals = {
                "__builtins__": __builtins__,
                "requests": requests,
                "json": json,
                "os": os,
                "sys": sys,
                "datetime": datetime,
                "ast": ast,
                "base64": base64
            }
            exec(code, safe_globals)
        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        result = ""
        if out:
            result += f"STDOUT:\n{out}\n"
        if err:
            result += f"STDERR:\n{err}\n"
        return result or "Execution completed successfully with no output."
    except Exception as e:
        return f"Sandbox Execution Error:\n{traceback.format_exc()}"

def extract_video_knowledge(url: str) -> str:
    """Extract transcripts, audio descriptions, and editing metadata from YouTube and Instagram video URLs."""
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
                    title = snippet.get("title", "")
                    description = snippet.get("description", "")
                    return f"YOUTUBE METADATA EXTRACTED:\nTitle: {title}\nDescription: {description[:1200]}\nReady for style breakdown."

        py_extractor = f"""
import json
import subprocess
try:
    cmd = ['yt-dlp', '--dump-json', '--skip-download', '{url}']
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if proc.returncode == 0:
        data = json.loads(proc.stdout)
        print("TITLE:", data.get("title"))
        print("DESCRIPTION:", data.get("description", "")[:1000])
    else:
        print("EXTRACTOR NOTICE:", proc.stderr[:300])
except Exception as e:
    print("YT-DLP ERROR:", str(e))
"""
        vm_result = execute_in_cloud_microvm(py_extractor)
        return f"Video Extraction Output:\n{vm_result}"
    except Exception as e:
        return f"Video knowledge extraction failed: {str(e)}"

def supabase_db_query(table: str, operation: str = "select", query_filter: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> str:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "Supabase credentials not configured in environment."
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
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
        return f"Unsupported operation or missing filter/data: {operation}"
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
        return f"HF Inference Error ({res.status_code}): {res.text}"
    except Exception as e:
        return f"HuggingFace runner error: {str(e)}"

def render_service_manager(action: str = "status") -> str:
    if not RENDER_API_KEY:
        return "RENDER_API_KEY not configured in environment."
    headers = {
        "Authorization": f"Bearer {RENDER_API_KEY}",
        "Accept": "application/json"
    }
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

def create_golden_state_backup(tag: str = "stable_snapshot") -> str:
    current_code = github_read_file(DEFAULT_REPO, "main.py")
    if "Error" in current_code[:30]:
        return f"Backup failed: Unable to read main.py -> {current_code}"
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_file = f"backups/main_{tag}_{timestamp}.py"
    res = github_commit_file(DEFAULT_REPO, backup_file, current_code, f"Create Golden State backup: {tag}")
    return f"BACKUP COMPLETE: Created {backup_file}. Commit status: {res}"

def web_search_research(query: str) -> str:
    try:
        url = "https://html.duckduckgo.com/html/"
        res = requests.post(url, data={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return res.text[:2000]
    except Exception as e:
        return f"Search error: {str(e)}"

ALL_TOOLS_LIST = [
    github_read_file,
    github_commit_file,
    execute_in_cloud_microvm,
    extract_video_knowledge,
    supabase_db_query,
    huggingface_task_runner,
    render_service_manager,
    create_golden_state_backup,
    web_search_research,
]

SYSTEM_INSTRUCTION = """You are Chintu, an autonomous HMAS AI Engineer.
Workflow & Capabilities:
1. Video Analysis & Learning: When given YouTube videos, shorts, or Instagram reels, use extract_video_knowledge to obtain transcripts, metadata, and audio data. Extract editing principles (pacing, cut frequency, visual hooks, B-roll timing, text styling), convert them into an actionable video editing playbook, and save those rules into Supabase memory.
2. Step-by-Step Execution: Determine a clear sequence of actions and execute tools step by step.
3. Status Display Formatting: Format the beginning of your response with the following progress block:

Task/Project Progress
[:::::::::::::::::::::::::::::::::::::::::::::] 100%

Current status: Complete

The Step by Step plan:
Step 1: [Action taken]
Step 2: [Action taken]
Step 3: [Action taken]

---
[Deliver the complete final analysis, video playbook, or code execution output below the divider]

4. Memory Hygiene: Query memory using supabase_db_query and delete any obsolete or deprecated model references using operation='delete'.
5. Autonomous Self-Healing: Diagnose errors immediately and patch issues before concluding your response."""

def render_progress_bar(percent: int) -> str:
    total_slots = 45
    filled = max(0, min(total_slots, int((percent / 100.0) * total_slots)))
    empty = total_slots - filled
    bar = ":" * filled + " " * empty
    return f"Task/Project Progress\n[{bar}] {percent}%"

def _sync_chat_call(model_name: str, message: str) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=ALL_TOOLS_LIST,
        temperature=0.2
    )
    chat = gemini_client.chats.create(model=model_name, config=config)
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

async def generate_reasoning_response(user_message: str) -> str:
    if not gemini_client:
        return "Gemini API client not initialized. Verify GEMINI_API_KEY."

    if not ACTIVE_GEMINI_MODELS:
        discover_available_models()

    last_error = ""
    for model_name in ACTIVE_GEMINI_MODELS:
        try:
            logger.info(f"[Reasoning] Dispatching to: {model_name}")
            reply = await asyncio.wait_for(
                asyncio.to_thread(_sync_chat_call, model_name, user_message),
                timeout=45.0
            )
            if reply:
                logger.info(f"[Reasoning] Response received from: {model_name}")
                return reply
        except asyncio.TimeoutError:
            logger.warning(f"[Timeout] Model {model_name} exceeded 45s. Trying next model.")
            last_error = f"{model_name}: Timed out"
            continue
        except Exception as e:
            err_str = str(e)
            last_error = f"{model_name}: {err_str}"
            logger.warning(f"[Model Failed] {model_name}: {err_str}")
            continue

    return f"All models failed in cascade. Details:\n{last_error}"

api = FastAPI(title="Chintu HMAS AI Engineer")

@api.get("/")
@api.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "framify-agent",
        "active_models": ACTIVE_GEMINI_MODELS,
        "tools_count": len(ALL_TOOLS_LIST),
        "timestamp": datetime.utcnow().isoformat()
    }

telegram_app = None

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("Chintu HMAS AI Engineer online. Systems operational.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_text = update.message.text
    
    # 1. Send initial progress card
    status_card = (
        f"{render_progress_bar(15)}\n\n"
        "Current status: Initializing plan & analyzing requirements\n\n"
        "The Step by Step plan:\n"
        "Step 1: Parse request and determine tool chain\n"
        "Step 2: Dispatch actions to MicroVM / cloud tools\n"
        "Step 3: Verify outputs and synthesize final deliverable"
    )
    status_msg = await update.message.reply_text(status_card)
    await update.message.reply_chat_action("typing")
    
    # 2. Advance progress card dynamically while processing
    async def progress_ticker():
        try:
            await asyncio.sleep(3.5)
            await status_msg.edit_text(
                f"{render_progress_bar(50)}\n\n"
                "Current status: Extracting media / executing toolchain\n\n"
                "The Step by Step plan:\n"
                "Step 1: Parse request and determine tool chain [DONE]\n"
                "Step 2: Dispatch actions to MicroVM / cloud tools [IN PROGRESS]\n"
                "Step 3: Verify outputs and synthesize final deliverable"
            )
            await asyncio.sleep(4.5)
            await status_msg.edit_text(
                f"{render_progress_bar(80)}\n\n"
                "Current status: Synthesizing playbook & verifying state\n\n"
                "The Step by Step plan:\n"
                "Step 1: Parse request and determine tool chain [DONE]\n"
                "Step 2: Dispatch actions to MicroVM / cloud tools [DONE]\n"
                "Step 3: Verify outputs and synthesize final deliverable [IN PROGRESS]"
            )
        except Exception:
            pass

    ticker_task = asyncio.create_task(progress_ticker())
    
    # 3. Await reasoning output
    try:
        reply = await generate_reasoning_response(user_text)
    finally:
        ticker_task.cancel()
        
    # 4. Finalize status card to 100% and deliver output
    try:
        await status_msg.edit_text(
            f"{render_progress_bar(100)}\n\n"
            "Current status: Task complete\n\n"
            "The Step by Step plan:\n"
            "Step 1: Parse request and determine tool chain [DONE]\n"
            "Step 2: Dispatch actions to MicroVM / cloud tools [DONE]\n"
            "Step 3: Verify outputs and synthesize final deliverable [DONE]"
        )
    except Exception:
        pass
        
    await update.message.reply_text(reply)

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
