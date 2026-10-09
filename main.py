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
from duckduckgo_search import DDGS

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "8513926902"))

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.5-flash-lite"

last_progress_time = time.time()

def record_activity():
    global last_progress_time
    last_progress_time = time.time()

# ----------------- CLAUDE-EQUIVALENT CORE TOOLS -----------------

def web_search(query: str) -> str:
    """Performs live web searches to find documentation, code libraries, and answers."""
    record_activity()
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No web results found."
        
        output = []
        for r in results:
            output.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
        return "\n---\n".join(output)
    except Exception as e:
        record_activity()
        return f"Search error: {str(e)}"

def fetch_webpage(url: str) -> str:
    """Fetches and extracts clean, readable text from any web URL."""
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
            return f"Status {response.getcode()}:\n{text[:3000]}"
    except Exception as e:
        record_activity()
        return f"Fetch error: {str(e)}"

def run_terminal_command(command: str) -> str:
    """Executes a shell command in the workspace directory with output capture."""
    record_activity()
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        record_activity()
        out = res.stdout if res.stdout else res.stderr
        return out[:3000] if out else "Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        record_activity()
        return "ERROR: Command timed out after 60 seconds."
    except Exception as e:
        record_activity()
        return f"ERROR: Execution failed: {str(e)}"

def write_project_file(file_path: str, content: str) -> str:
    """Creates or completely overwrites a file on disk."""
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

def read_file(file_path: str, start_line: int = 1, line_count: int = 100) -> str:
    """Reads specific lines of a file to inspect code without loading huge files."""
    record_activity()
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File '{file_path}' does not exist."
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        start = max(1, start_line) - 1
        end = start + line_count
        chunk = "".join(lines[start:end])
        return f"Lines {start+1}-{min(len(lines), end)} of '{file_path}':\n{chunk}" if chunk else "Empty range."
    except Exception as e:
        record_activity()
        return f"ERROR: Could not read file: {str(e)}"

def patch_file(file_path: str, target_block: str, replacement_block: str) -> str:
    """Surgically replaces a snippet of text inside a file without rewriting the entire file."""
    record_activity()
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File '{file_path}' not found."
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if target_block not in content:
            return f"ERROR: Target block not found in '{file_path}'. Verify lines using read_file first."
        
        updated = content.replace(target_block, replacement_block, 1)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated)
        record_activity()
        return f"SUCCESS: Patched '{file_path}' successfully."
    except Exception as e:
        record_activity()
        return f"ERROR: Patch failed: {str(e)}"

agent_tools = [
    web_search, 
    fetch_webpage, 
    run_terminal_command, 
    write_project_file, 
    read_file, 
    patch_file
]

SYSTEM_PROMPT = (
    "You are an Elite Autonomous Full-Stack AI Engineer running on an Ubuntu container.\n\n"
    "TOOL USAGE PROTOCOLS:\n"
    "1. RESEARCH: Use `web_search` and `fetch_webpage` whenever you need current docs, solutions, or API specifications.\n"
    "2. SAFE CODE EDITS: Use `read_file` to inspect code and `patch_file` for modifications. Do not rewrite large files if you can patch them.\n"
    "3. TERMINAL RESILIENCE: Run terminal commands to test and verify your work. If a command fails, read the stderr, fix the problem, and retry.\n"
    "4. MANDATORY REPORT: Always finish with a clear text summary detailing the actions you took and the outcome."
)

def run_autonomous_agent(prompt: str) -> str:
    global last_progress_time
    record_activity()
    
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=agent_tools,
            temperature=0.2
        )
    )
    
    response = chat.send_message(prompt)
    record_activity()
    
    if not (response.text and response.text.strip()):
        record_activity()
        follow_up = chat.send_message("Synthesize and summarize what you did and the exact results achieved.")
        if follow_up.text and follow_up.text.strip():
            return follow_up.text
        return "All tools executed successfully."
        
    return response.text

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.text:
            return
        if ALLOWED_USER_ID != 0 and update.effective_user.id != ALLOWED_USER_ID:
            await update.message.reply_text("Unauthorized access.")
            return

        task = update.message.text
        status = await update.message.reply_text(f"Task Received:\n'{task[:60]}...'\nExecuting...")
        loop = asyncio.get_running_loop()
        
        record_activity()
        agent_future = loop.run_in_executor(None, run_autonomous_agent, task)
        
        # 120s watchdog between tool operations
        max_idle_seconds = 120.0
        stuck = False
        while not agent_future.done():
            await asyncio.sleep(2.0)
            if (time.time() - last_progress_time) > max_idle_seconds:
                stuck = True
                break

        if stuck:
            agent_future.cancel()
            await status.edit_text("Halt: Agent was idle for > 2 minutes with no progress.")
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

@api.get("/")
def home():
    return {"status": "Agent Online", "model": MODEL_NAME}
