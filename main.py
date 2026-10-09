import os
import asyncio
import subprocess
import time
import urllib.request
import ssl
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "8513926902"))

client = genai.Client(api_key=GEMINI_API_KEY)
# Switched to gemini-2.5-flash for 250-1500 RPD free quota limit
MODEL_NAME = "gemini-2.5-flash"

def run_terminal_command(command: str) -> str:
    """Executes a bash shell command with a safety timeout."""
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        out = res.stdout if res.stdout else res.stderr
        return out[:3000] if out else "Command executed with no output."
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as e:
        return f"Execution error: {str(e)}"

def write_project_file(file_path: str, content: str) -> str:
    """Writes files to disk, creating directories as needed."""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File '{file_path}' written successfully."

def fetch_webpage(url: str) -> str:
    """Fetches web text with a 10s non-blocking timeout and clean browser headers."""
    try:
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
            status_code = response.getcode()
            html = response.read().decode("utf-8", errors="ignore")
            cleaned = re.sub(r"<(script|style).*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", cleaned)
            text = " ".join(text.split())
            preview = text[:2000] if text else "Empty body"
            return f"Status {status_code}: {preview}"
    except urllib.error.HTTPError as e:
        return f"HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Connection failed: {str(e.reason)}"
    except Exception as e:
        return f"Fetch error: {str(e)}"

agent_tools = [run_terminal_command, write_project_file, fetch_webpage]

def run_agent(prompt: str) -> str:
    max_retries = 3
    delay = 2.0
    
    for attempt in range(max_retries):
        try:
            chat = client.chats.create(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are an autonomous engineering agent with live internet access, terminal execution, "
                        "and file writing tools. Complete programming and systems tasks directly and completely."
                    ),
                    tools=agent_tools,
                    temperature=0.2
                )
            )
            response = chat.send_message(prompt)
            return response.text if response.text else "Task completed with no text output."
        except Exception as e:
            err_str = str(e)
            if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return f"Agent error: {err_str}"

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    if ALLOWED_USER_ID != 0 and update.effective_user.id != ALLOWED_USER_ID:
        await update.message.reply_text("Unauthorized access.")
        return

    task = update.message.text
    status = await update.message.reply_text(f"Task received:\n'{task[:60]}...'\nProcessing on {MODEL_NAME}...")
    loop = asyncio.get_running_loop()
    
    try:
        # Safety cutoff to ensure tasks never hang Telegram polling indefinitely
        result = await asyncio.wait_for(
            loop.run_in_executor(None, run_agent, task),
            timeout=90.0
        )
    except asyncio.TimeoutError:
        result = "Error: Agent task timed out after 90 seconds."

    await status.edit_text(result[:4000])

async def run_telegram_worker():
    await asyncio.sleep(4)
    while True:
        try:
            bot_app = (
                ApplicationBuilder()
                .token(TELEGRAM_TOKEN)
                .connect_timeout(30.0)
                .read_timeout(30.0)
                .write_timeout(30.0)
                .build()
            )
            bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_msg))
            
            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling(drop_pending_updates=True)
            
            while True:
                await asyncio.sleep(3600)
        except Exception as e:
            await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_telegram_worker())
    yield
    task.cancel()

api = FastAPI(lifespan=lifespan)

@api.get("/")
def home():
    return {"status": "Agent Online", "model": MODEL_NAME}
