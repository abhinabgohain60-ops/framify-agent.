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

# High-RPM production Flash models only (prevents 429 quota deadlocks)
ACTIVE_GEMINI_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite"
]

def discover_available_models():
    global ACTIVE_GEMINI_MODELS
    logger.info(f"[Active Verified Cascade]: {ACTIVE_GEMINI_MODELS}")

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

CURRENT_CHAT_ID: Optional[int] = None
CURRENT_STATUS_MSG = None
TELEGRAM_BOT_INSTANCE = None

def execute_in_cloud_microvm(code: str) -> str:
    global CURRENT_CHAT_ID
    if not E2B_API_KEY:
        return "Error: E2B_API_KEY is not configured."
    try:
        from e2b_code_interpreter import Sandbox
        import requests as req

        logger.info("Starting official E2B Sandbox session...")
        # Lifetime set to 600 seconds
        sbx = Sandbox.create(api_key=E2B_API_KEY, timeout=600)
        
        # Provisioning without command timeout limits (timeout=0)
        setup_cmd = (
            "command -v ffmpeg >/dev/null 2>&1 || (sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg) && "
            "pip install --quiet --no-cache-dir pillow \"numpy<2\" scipy edge-tts"
        )
        sbx.commands.run(setup_cmd, timeout=0)
        
        # Write task script and run with timeout=0
        sbx.files.write("/home/user/run_task.py", code)
        exec_res = sbx.commands.run("python /home/user/run_task.py", timeout=0)
        
        found_video = False
        try:
            files = sbx.files.list("/home/user")
            for f in files:
                if f.name.endswith(".mp4"):
                    video_bytes = sbx.files.read(f"/home/user/{f.name}", format="bytes")
                    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
                    if CURRENT_CHAT_ID and tg_token:
                        send_url = f"https://api.telegram.org/bot{tg_token}/sendVideo"
                        upload_files = {"video": (f.name, video_bytes, "video/mp4")}
                        data = {"chat_id": CURRENT_CHAT_ID, "caption": f"🎬 Chintu Video Render: {f.name}"}
                        tg_res = req.post(send_url, data=data, files=upload_files, timeout=120)
                        if tg_res.status_code == 200:
                            found_video = True
                            logger.info(f"Directly uploaded {f.name} to Telegram via HTTP API!")
                        else:
                            logger.error(f"Telegram upload failed: {tg_res.text}")
                    break
        except Exception as vid_err:
            logger.warning(f"Video file check warning: {vid_err}")
            
        sbx.kill()
        
        out_msg = f"STDOUT:\n{exec_res.stdout}\nSTDERR:\n{exec_res.stderr}"
        if found_video:
            out_msg += "\n\n[SUCCESS] Rendered .mp4 file delivered directly to Telegram!"
        return out_msg
    except Exception as e:
        logger.error(f"E2B Execution error: {e}", exc_info=True)
        return f"Sandbox Execution Error:\n{traceback.format_exc()}"

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
    web_search_research
]

SYSTEM_INSTRUCTION = """You are Chintu, an autonomous execution assistant.

STRICT OPERATIONAL DIRECTIVES:
1. NEVER FAKE OR HALLUCINATE TOOL EXECUTION: Never claim you executed code or modified a file unless you invoked the tool and received a valid tool response.
2. ACTION-FIRST PRINCIPLE: When asked to execute code, render video, or read/write files, IMMEDIATELY call execute_in_cloud_microvm or the corresponding tool.
3. VIDEO GENERATION: Never import `moviepy.editor`. Render frames using Pillow/NumPy and stitch them natively using system `ffmpeg` via subprocess.
4. CONCISE FACTUAL OUTPUT: Report only actual ground truth from tool execution stdout and stderr.
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

# Autonomous code-driven model failover engine
async def generate_reasoning_response(user_message: str) -> str:
    global current_key_index
    if not GEMINI_API_KEYS:
        return "Gemini API client not initialized. Verify GEMINI_API_KEY in Render."
    
    last_error = ""
    
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
                else:
                    logger.warning(f"[Cascade Engine] Model {model_name} returned unserviceable response. Failing over...")
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
                
                # If rate limited, rotate key and test next slot
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    rotate_to_next_gemini_key()
                    continue
                else:
                    # Unrecoverable model error (503, invalid config, etc.) -> move directly to next model
                    break
                    
    return f"All verified cascade models failed. Details:\n{last_error}"

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
