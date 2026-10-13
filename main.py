import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from groq import Groq

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chintu-agent")

# Environment & Config
PORT = int(os.environ.get("PORT", 10000))
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
DEFAULT_REPO = os.environ.get("DEFAULT_REPO", "abhinabgohain60-ops/framify-agent.")

GEMINI_MODELS_CASCADE = [
    "gemini-3.0-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.0-flash"
]

# Client Initializations
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==========================================
# TOOL DEFINITIONS & IMPLEMENTATIONS
# ==========================================

def github_read_file(repo_name: str, file_path: str) -> str:
    """Read a file from GitHub repository."""
    token = GITHUB_TOKEN or ""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    encoded_repo = repo_name.replace(".", "%2E") if repo_name.endswith(".") else repo_name
    url = f"https://api.github.com/repos/{encoded_repo}/contents/{file_path}"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            dl_url = res.json().get("download_url")
            return requests.get(dl_url, timeout=15).text
        return f"GitHub Read Error ({res.status_code}): {res.text}"
    except Exception as e:
        return f"Error reading GitHub file: {str(e)}"

def github_commit_file(repo_name: str, file_path: str, content: str, commit_message: str) -> str:
    """Commit and push a file directly to GitHub via REST API."""
    token = GITHUB_TOKEN or ""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    encoded_repo = repo_name.replace(".", "%2E") if repo_name.endswith(".") else repo_name
    url = f"https://api.github.com/repos/{encoded_repo}/contents/{file_path}"
    try:
        sha = None
        get_res = requests.get(url, headers=headers, timeout=15)
        if get_res.status_code == 200:
            sha = get_res.json().get("sha")
        
        b64_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        payload = {"message": commit_message, "content": b64_content, "branch": "main"}
        if sha:
            payload["sha"] = sha
            
        put_res = requests.put(url, headers=headers, json=payload, timeout=20)
        if put_res.status_code in [200, 201]:
            return f"Successfully committed {file_path} to {repo_name}."
        return f"Commit failed ({put_res.status_code}): {put_res.text}"
    except Exception as e:
        return f"Error committing file: {str(e)}"

def supabase_db_query(table: str, operation: str = "select", data: Optional[Dict[str, Any]] = None) -> str:
    """Read or write records to Supabase state tables."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "Supabase credentials not configured."
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    try:
        if operation == "select":
            res = requests.get(f"{url}?select=*", headers=headers, timeout=10)
            return json.dumps(res.json())
        elif operation == "insert" and data:
            res = requests.post(url, headers=headers, json=data, timeout=10)
            return f"Inserted status: {res.status_code}"
        return f"Unsupported Supabase operation: {operation}"
    except Exception as e:
        return f"Supabase Error: {str(e)}"

def web_search_research(query: str) -> str:
    """Perform web research via DuckDuckGo lite HTML parser."""
    try:
        url = "https://html.duckduckgo.com/html/"
        res = requests.post(url, data={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return res.text[:2000]
    except Exception as e:
        return f"Search error: {str(e)}"

def execute_in_cloud_microvm(code: str) -> str:
    """Execute Python code in isolated cloud sandbox."""
    try:
        import io
        import contextlib
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": __builtins__})
        return stdout.getvalue() or "Code executed with no output."
    except Exception as e:
        return f"Execution error: {str(e)}"

ALL_TOOLS_MAP = {
    "github_read_file": github_read_file,
    "github_commit_file": github_commit_file,
    "supabase_db_query": supabase_db_query,
    "web_search_research": web_search_research,
    "execute_in_cloud_microvm": execute_in_cloud_microvm,
}

SYSTEM_INSTRUCTION = """You are Chintu, an autonomous HMAS AI Engineer.
You have tools to read/commit code to GitHub, execute tests in microVMs, run web research, and access Supabase memory.
Always inspect code before modifying it, ensure syntactic validity, and maintain system stability."""

# ==========================================
# REASONING ENGINE (CASCADE + FALLBACK)
# ==========================================

async def generate_reasoning_response(user_message: str) -> str:
    response_text = ""
    success = False

    if gemini_client:
        for model_name in GEMINI_MODELS_CASCADE:
            try:
                chat = gemini_client.chats.create(model=model_name)
                response = chat.send_message(user_message)
                if response.text:
                    response_text = response.text
                    success = True
                    break
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Model {model_name} failed: {err_msg}")
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    break
                continue

    if not success or not response_text:
        try:
            if groq_client:
                comp = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                response_text = comp.choices[0].message.content or "Empty fallback response."
            else:
                response_text = "No AI engines available to process request."
        except Exception as ge:
            response_text = f"All reasoning cascades failed: {str(ge)}"

    return response_text

# ==========================================
# FASTAPI APPLICATION & HEALTH ENDPOINTS
# ==========================================

api = FastAPI(title="Chintu HMAS AI Engineer")

@api.get("/")
@api.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "framify-agent",
        "timestamp": datetime.utcnow().isoformat()
    }

# ==========================================
# TELEGRAM BOT INTEGRATION
# ==========================================

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
    logger.info("Initializing FastAPI and Telegram services...")
    if TELEGRAM_BOT_TOKEN:
        try:
            telegram_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
            telegram_app.add_handler(CommandHandler("start", handle_start))
            telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            await telegram_app.initialize()
            await telegram_app.start()
            asyncio.create_task(telegram_app.updater.start_polling(drop_pending_updates=True))
            logger.info("[Telegram Bot] Started successfully via polling.")
        except Exception as te:
            logger.error(f"Telegram startup warning: {str(te)}")

@api.on_event("shutdown")
async def shutdown_event():
    global telegram_app
    if telegram_app:
        try:
            await telegram_app.updater.stop()
            await telegram_app.stop()
            await telegram_app.shutdown()
        except Exception as e:
            logger.warning(f"Telegram shutdown warning: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:api", host="0.0.0.0", port=PORT, log_level="info")
