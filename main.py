import os
import asyncio
import subprocess
import time
import urllib.request
import ssl
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "8513926902"))

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.5-flash-lite"

last_progress_time = time.time()

def record_activity():
    global last_progress_time
    last_progress_time = time.time()

def run_terminal_command(command: str) -> str:
    """Executes a bash shell command with output truncation and timeouts."""
    record_activity()
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        record_activity()
        out = res.stdout if res.stdout else res.stderr
        return out[:3000] if out else "Command executed successfully with no stdout/stderr."
    except subprocess.TimeoutExpired:
        record_activity()
        return "ERROR: Command timed out after 60 seconds."
    except Exception as e:
        record_activity()
        return f"ERROR: Execution failed: {str(e)}"

def write_project_file(file_path: str, content: str) -> str:
    """Writes files cleanly to disk, creating parent directories automatically."""
    record_activity()
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        record_activity()
        return f"SUCCESS: File '{file_path}' written ({len(content)} bytes)."
    except Exception as e:
        record_activity()
        return f"ERROR: Could not write file: {str(e)}"

def fetch_webpage(url: str) -> str:
    """Fetches web text cleanly without scripts/styles."""
    record_activity()
    try:
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, timeout=15, context=ctx) as response:
            record_activity()
            html = response.read().decode("utf-8", errors="ignore")
            cleaned = re.sub(r"<(script|style).*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", cleaned)
            text = " ".join(text.split())
            return f"Status {response.getcode()}: {text[:2500]}"
    except Exception as e:
        record_activity()
        return f"Fetch error: {str(e)}"

agent_tools = [run_terminal_command, write_project_file, fetch_webpage]

SYSTEM_PROMPT = (
    "You are an Elite Full-Stack Systems Architect and DevOps Engineer running directly on an Ubuntu server container.\n\n"
    "OPERATIONAL PROTOCOL:\n"
    "1. PLAN BEFORE ACTING: Always formulate a 2-sentence logical plan before calling any tools.\n"
    "2. AUTONOMOUS RECOVERY: If a bash command or tool returns an ERROR, do NOT give up or stop. Analyze the error output, determine the root cause, and attempt up to 2 alternate approaches.\n"
    "3. COMPLETE CODE: Never output placeholders, ellipses ('// TODO'), or partial snippets. Always write clean, production-ready code.\n"
    "4. MANDATORY TEXT SUMMARY: After calling tools, you MUST provide a final concise markdown report detailing what was accomplished and direct next steps."
)

def run_autonomous_agent(prompt: str) -> str:
    """Executes a multi-turn chat session with automatic recovery loops."""
    global last_progress_time
    record_activity()
    
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=agent_tools,
            temperature=0.2  # Low temperature dramatically reduces coding hallucinations
        )
    )
    
    # Send user prompt
    response = chat.send_message(prompt)
    record_activity()
    
    # If the model called tools but didn't output text, prompt it for the final report
    if not (response.text and response.text.strip()):
        record_activity()
        follow_up = chat.send_message(
            "Synthesize your actions: Summarize the changes you made, list created files, and outline the exact results."
        )
        if follow_up.text and follow_up.text.strip():
            return follow_up.text
        return "All tools and tasks executed successfully."
        
    return response.text

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.text:
            return
        if ALLOWED_USER_ID != 0 and update.effective_user.id != ALLOWED_USER_ID:
            await update.message.reply_text("Unauthorized access.")
            return

        task = update.message.text
        status = await update.message.reply_text(f"Task Queued:\n'{task[:60]}...'\nProcessing with peak reasoning...")
        loop = asyncio.get_running_loop()
        
        record_activity()
        agent_future = loop.run_in_executor(None, run_autonomous_agent, task)
        
        # Idle watchdog: 120s max between tool steps
        max_idle_seconds = 120.0
        stuck = False
        while not agent_future.done():
            await asyncio.sleep(2.0)
            if (time.time() - last_progress_time) > max_idle_seconds:
                stuck = True
                break

        if stuck:
            agent_future.cancel()
            await status.edit_text("Halt: Agent exceeded idle timeout (no active progress for > 2 minutes).")
            return

        result = await agent_future
        await status.edit_text(result[:4000] if result else "Execution completed.")
    except Exception as e:
        print(f"Update handler error: {e}", flush=True)

async def run_telegram_worker():
    await asyncio.sleep(3)
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
            await bot_app.updater.start_polling(drop_pending_updates=False)
            
            print("Telegram poller running...", flush=True)
            while True:
                await asyncio.sleep(3600)
        except Exception as e:
            print(f"Poller restarted: {e}", flush=True)
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_telegram_worker())
    yield
    task.cancel()

api = FastAPI(lifespan=lifespan)

# Static file serving & health route
if os.path.exists("public"):
    api.mount("/static", StaticFiles(directory="public"), name="static")

@api.get("/")
def home():
    return {"status": "Agent Online", "model": MODEL_NAME}

@api.get("/watermark")
def serve_watermark():
    if os.path.exists("public/index.html"):
        return FileResponse("public/index.html")
    return {"error": "Frontend UI file not found in public/index.html"}
