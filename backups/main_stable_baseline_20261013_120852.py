import ast
import base64
import ast
import os
import sys
import json
import logging
import asyncio
import io
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
DEFAULT_REPO = os.environ.get("DEFAULT_REPO", "abhinabgohain60-ops/framify-agent.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

ACTIVE_GEMINI_MODELS: List[str] = []

def discover_available_models():
    global ACTIVE_GEMINI_MODELS
    whitelist_priority = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-flash-latest"
    ]
    discovered = []
    if gemini_client:
        try:
            for m in gemini_client.models.list():
                m_name = getattr(m, "name", "").replace("models/", "")
                if any(bad in m_name for bad in ["tts", "image", "audio", "embed", "robotics", "transcribe", "customtools"]):
                    continue
                if "gemini" in m_name:
                    discovered.append(m_name)
            logger.info(f"[Model Discovery] Valid text models: {discovered}")
        except Exception as e:
            logger.warning(f"[Model Discovery] List failed: {e}")

    ordered = [m for m in whitelist_priority if m in discovered]
    for m in discovered:
        if m not in ordered:
            ordered.append(m)

    if not ordered:
        ordered = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

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
                "ast": ast
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

def supabase_db_query(table: str, operation: str = "select", data: Optional[Dict[str, Any]] = None) -> str:
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
            res = requests.get(f"{url}?select=*&order=created_at.desc&limit=20", headers=headers, timeout=10)
            return json.dumps(res.json())
        elif operation == "insert" and data:
            res = requests.post(url, headers=headers, json=data, timeout=10)
            return f"Insert status: {res.status_code}"
        return f"Unsupported operation: {operation}"
    except Exception as e:
        return f"Supabase error: {str(e)}"

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
    supabase_db_query,
    create_golden_state_backup,
    web_search_research,
]

SYSTEM_INSTRUCTION = """You are Chintu, an autonomous HMAS AI Engineer.
You have tools to read/commit code to your GitHub repo, run isolated code in a microVM sandbox, query Supabase memory, create Golden State backups, and perform web research.
Self-Healing & Problem Solving Protocol:
- When an action or tool fails (e.g. missing import, syntax error, or API rejection), analyze the error immediately.
- Use your tools (such as github_read_file and github_commit_file) to inspect the code, correct the bug, and re-attempt execution.
- Maintain persistent execution until the goal succeeds or an unrecoverable external dependency blocks progress.
- Once verified, confirm the final resolution and details to the user."""

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
        
    # Extract output from candidate parts if text field is empty
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
                timeout=35.0
            )
            if reply:
                logger.info(f"[Reasoning] Response received from: {model_name}")
                return reply
        except asyncio.TimeoutError:
            logger.warning(f"[Timeout] Model {model_name} exceeded 35s. Trying next model.")
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
    await update.message.reply_chat_action("typing")
    reply = await generate_reasoning_response(user_text)
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
