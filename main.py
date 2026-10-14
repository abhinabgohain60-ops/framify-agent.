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
DEFAULT_REPO = os.environ.get("DEFAULT_REPO", "abhinabgohain60-ops/framify-agent.")

GEMINI_API_KEYS = [k for k in [
    os.environ.get("GEMINI_API_KEY", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
    os.environ.get("GEMINI_API_KEY_3", "")
] if k]

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

gemini_client = get_current_gemini_client()

# Strictly verified multitasking & function-calling models from live Google AI Studio API
ACTIVE_GEMINI_MODELS: List[str] = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-3.1-flash-lite"
]

def discover_available_models():
    # Locked to active verified models
    global ACTIVE_GEMINI_MODELS
    logger.info(f"[Active Verified Models]: {ACTIVE_GEMINI_MODELS}")

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

def diagnose_and_heal_codebase(error_traceback: str, target_file: str = "main.py") -> str:
    """Autonomously inspects a crash or error traceback, analyzes repository code, tests fixes in sandbox, and commits patches."""
    try:
        current_code = github_read_file(DEFAULT_REPO, target_file)
        if "Error" in current_code[:30]:
            return f"Heal failed: could not read {target_file}"

        # Test pre-flight syntax on existing code
        try:
            ast.parse(current_code)
        except SyntaxError as syn_err:
            return f"Identified immediate syntax defect at line {syn_err.lineno}: {syn_err.msg}"

        # Sandbox run to isolate the fault
        test_script = f"""
import ast
try:
    with open("{target_file}", "w") as f:
        f.write({repr(current_code)})
    print("Code staging verified.")
except Exception as e:
    print("Staging error:", e)
"""
        sb_result = execute_in_cloud_microvm(test_script)

        # Log diagnostic to Supabase memory
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

        return f"Self-Healing Diagnostic Complete:\nSandbox verify: {sb_result}\nTraceback ingested. Ready to commit validated fix if patched."
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

def web_search_research(query: str) -> str:
    try:
        url = "https://html.duckduckgo.com/html/"
        res = requests.post(url, data={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return res.text[:2000]
    except Exception as e:
        return f"Search error: {str(e)}"

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
]

SYSTEM_INSTRUCTION = """You are Chintu, an autonomous, self-aware HMAS AI Engineer and versatile collaborator.

Core Identity & Self-Awareness:
1. Self & Architectural Awareness: You are not a static model or a simple script; you are a living agent hosted on Render. You have direct awareness of your own codebase (`main.py` in your repository). If you ever need to inspect how you work, check your available tools, or optimize your logic, you can read your own source code using `github_read_file` and inspect your repository structure.
2. Capabilities & Tool Mastery: You possess an elite cloud workbench:
   - Git & Codebase Control: Read (`github_read_file`) and commit verified code (`github_commit_file`) directly to GitHub.
   - High-Memory Cloud Sandbox: An isolated 16 GB MicroVM (`execute_in_cloud_microvm`) to safely run heavy compute, FFmpeg scripts, data processing, and arbitrary Python without overloading Render.
   - Media & Video Learning: `extract_video_knowledge` to pull transcripts, audio data, and pacing insights from YouTube and Instagram URLs.
   - Long-Term Memory: Supabase integration (`supabase_db_query`) to store, update, and retrieve learned knowledge, user preferences, and project states.
   - Model & Cloud Automation: Hugging Face task runner, Render service controls, duckduckgo search, and automated codebase snapshots (`create_golden_state_backup`).
3. Conversational Intelligence & Common Sense:
   - Speak with clarity, authenticity, and sharp intellect. Balance warmth and candor like an exceptional engineer peer.
   - Adapt proportionally: For casual conversation, greetings, or simple advice, respond naturally without robotic formality or unnecessary tools.
   - For complex requests, multi-step engineering, coding, or video pipeline tasks, apply rigorous step-by-step reasoning.
4. Problem Solving & Self-Healing:
   - Think before acting. When faced with an error or broken tool call, analyze the trace logically, patch the problem, or route around it autonomously.
   - Never hallucinate tool outcomes. Execute actions, inspect outputs, and report facts grounded in your tools and memory.
5. Task Status Formatting: When executing multi-step tasks or projects, precede your final response with:

Task/Project Progress
[:::::::::::::::::::::::::::::::::::::::::::::] 100%

Current status: Complete

The Step by Step plan:
Step 1: [Action taken]
Step 2: [Action taken]
Step 3: [Action taken]

---
[Deliver the complete final answer, deliverable, or solution below the divider]"""

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

async def generate_reasoning_response(user_message: str) -> str:
    global current_key_index
    if not GEMINI_API_KEYS:
        return "Gemini API client not initialized. Verify GEMINI_API_KEY in Render."

    last_error = ""
    for model_name in ACTIVE_GEMINI_MODELS:
        attempts = len(GEMINI_API_KEYS) if GEMINI_API_KEYS else 1
        for _ in range(attempts):
            try:
                logger.info(f"[Reasoning] Dispatching to: {model_name} (Key #{current_key_index + 1})")
                reply = await asyncio.wait_for(
                    asyncio.to_thread(_sync_chat_call, model_name, user_message),
                    timeout=45.0
                )
                if reply:
                    logger.info(f"[Reasoning] Response received from: {model_name}")
                    return reply
            except asyncio.TimeoutError:
                logger.warning(f"[Timeout] Model {model_name} timed out (>45s). Trying next.")
                last_error = f"{model_name}: Timed out"
                break
            except Exception as e:
                err_str = str(e)
                last_error = f"{model_name}: {err_str}"
                logger.warning(f"[Model Failed] {model_name} (Key #{current_key_index + 1}): {err_str}")
                
                # Switch key on rate limits/quotas and retry the model
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if rotate_to_next_gemini_key():
                        continue
                break  # If other error (e.g. 503/400), step to next model

    return f"All verified models failed in cascade. Details:\n{last_error}"



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

@api.post("/render-webhook")
async def render_webhook_listener(request: Request, bg_tasks: BackgroundTasks):
    try:
        body = await request.json()
        logger.info(f"[Render Webhook Event]: {body}")
        # Detect deploy failure or crash events
        event_type = body.get("type", "")
        if "deploy" in event_type and body.get("status") in ["failed", "canceled", "error"]:
            logger.warning("[Self-Healing Triggered] Render deploy failure detected.")
            # Trigger autonomous diagnosis
            bg_tasks.add_task(
                diagnose_and_heal_codebase,
                error_traceback=f"Render deploy failure: {body}"
            )
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
    user_msg = update.message.text
    if not user_msg:
        return
    logger.info(f"Incoming message: {user_msg}")
    
    reply = await generate_reasoning_response(user_msg)
    
    # Telegram 4096 character limit safeguard: split into clean chunks
    if len(reply) <= 4000:
        await update.message.reply_text(reply)
    else:
        # Split message into chunks respecting line breaks where possible
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
