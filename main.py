import os
import asyncio
import subprocess
import time
import urllib.request
import ssl
import re
import json
import contextvars
from contextlib import asynccontextmanager
from fastapi import FastAPI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from groq import Groq
from duckduckgo_search import DDGS
from e2b_code_interpreter import Sandbox
from supabase import create_client, Client
from github import Github, GithubException

# Environment Variables & Auth (Sanitized)
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
ALLOWED_USER_ID = int((os.getenv("TELEGRAM_ADMIN_ID") or "8513926902").strip() or 0)
E2B_API_KEY = (os.getenv("E2B_API_KEY") or "").strip()
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip()
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
DEFAULT_REPO = (os.getenv("GITHUB_REPO") or "abhinabgohain60-ops/framify-agent.").strip()

# Clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None
github_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

GEMINI_MODEL = "gemini-3.5-flash-lite"
LLAMA_MODEL = "llama-3.3-70b-versatile"

# Context tracking for Telegram file dispatch
active_chat_id: contextvars.ContextVar[int] = contextvars.ContextVar("active_chat_id", default=0)
bot_instance = None
main_loop = None

last_progress_time = time.time()

def record_activity():
    global last_progress_time
    last_progress_time = time.time()

# ----------------- GITHUB & SELF-IMPROVEMENT TOOLS -----------------

def github_read_file(file_path: str, repo_name: str = "", branch: str = "main") -> str:
    """Reads raw text content from any file in a GitHub repository."""
    record_activity()
    if not github_client:
        return "ERROR: GITHUB_TOKEN is not configured on Render."
    target_repo = repo_name.strip() if repo_name.strip() else DEFAULT_REPO
    try:
        repo = github_client.get_repo(target_repo)
        file_content = repo.get_contents(file_path, ref=branch)
        record_activity()
        return file_content.decoded_content.decode("utf-8")
    except Exception as e:
        record_activity()
        return f"GitHub read error: {str(e)}"

def github_commit_file(file_path: str, content: str, commit_message: str, repo_name: str = "", branch: str = "main") -> str:
    """Creates or updates a file directly in GitHub and commits it, triggering auto-deploy on Render."""
    record_activity()
    if not github_client:
        return "ERROR: GITHUB_TOKEN is not configured on Render."
    target_repo = repo_name.strip() if repo_name.strip() else DEFAULT_REPO
    try:
        repo = github_client.get_repo(target_repo)
        try:
            existing_file = repo.get_contents(file_path, ref=branch)
            repo.update_file(
                path=file_path,
                message=commit_message,
                content=content,
                sha=existing_file.sha,
                branch=branch
            )
            record_activity()
            return f"SUCCESS: Updated '{file_path}' in '{target_repo}' ({branch})."
        except GithubException as ge:
            if ge.status == 404:
                repo.create_file(
                    path=file_path,
                    message=commit_message,
                    content=content,
                    branch=branch
                )
                record_activity()
                return f"SUCCESS: Created '{file_path}' in '{target_repo}' ({branch})."
            raise ge
    except Exception as e:
        record_activity()
        return f"GitHub commit error: {str(e)}"

def github_create_repository(repo_name: str, description: str = "", private: bool = False) -> str:
    """Creates a new GitHub repository under your authenticated GitHub account."""
    record_activity()
    if not github_client:
        return "ERROR: GITHUB_TOKEN is not configured on Render."
    try:
        user = github_client.get_user()
        new_repo = user.create_repo(name=repo_name, description=description, private=private, auto_init=True)
        record_activity()
        return f"SUCCESS: Created repository '{new_repo.full_name}'."
    except Exception as e:
        record_activity()
        return f"GitHub repo creation error: {str(e)}"

# ----------------- TELEGRAM MEDIA DELIVERY TOOLS -----------------

def send_telegram_photo(file_path: str, caption: str = "") -> str:
    """Sends an image file (PNG, JPG, WEBP) directly to the Telegram user chat."""
    record_activity()
    if not os.path.exists(file_path):
        return f"ERROR: File '{file_path}' does not exist on disk."
    chat_id = active_chat_id.get()
    if not chat_id or not bot_instance or not main_loop:
        return "ERROR: Telegram dispatch bot or chat_id is unavailable."

    try:
        async def _send():
            with open(file_path, "rb") as f:
                await bot_instance.send_photo(chat_id=chat_id, photo=f, caption=caption[:1024])

        asyncio.run_coroutine_threadsafe(_send(), main_loop).result(timeout=30)
        record_activity()
        return f"SUCCESS: Sent photo '{file_path}' to Telegram."
    except Exception as e:
        record_activity()
        return f"Failed to send photo: {str(e)}"

def send_telegram_document(file_path: str, caption: str = "") -> str:
    """Sends any file (PDF, CSV, ZIP, TXT, code file) as a downloadable document to Telegram."""
    record_activity()
    if not os.path.exists(file_path):
        return f"ERROR: File '{file_path}' does not exist on disk."
    chat_id = active_chat_id.get()
    if not chat_id or not bot_instance or not main_loop:
        return "ERROR: Telegram dispatch bot or chat_id is unavailable."

    try:
        async def _send():
            with open(file_path, "rb") as f:
                await bot_instance.send_document(chat_id=chat_id, document=f, caption=caption[:1024])

        asyncio.run_coroutine_threadsafe(_send(), main_loop).result(timeout=30)
        record_activity()
        return f"SUCCESS: Sent document '{file_path}' to Telegram."
    except Exception as e:
        record_activity()
        return f"Failed to send document: {str(e)}"

# ----------------- LONG-TERM MEMORY TOOLS -----------------

def remember_information(key: str, value: str, category: str = "general") -> str:
    """Stores or updates persistent knowledge, project notes, or facts in Supabase."""
    record_activity()
    if not supabase:
        return "ERROR: Supabase is not configured on Render."
    try:
        data = {
            "key": key.strip().lower(),
            "value": value.strip(),
            "category": category.strip().lower()
        }
        supabase.table("chintu_memory").upsert(data, on_conflict="key").execute()
        record_activity()
        return f"SUCCESS: Remembered '{key}' under category '{category}'."
    except Exception as e:
        record_activity()
        return f"Memory save error: {str(e)}"

def recall_information(query_key: str = "") -> str:
    """Searches and retrieves stored persistent knowledge or facts from Supabase."""
    record_activity()
    if not supabase:
        return "ERROR: Supabase is not configured on Render."
    try:
        if query_key.strip():
            res = (
                supabase.table("chintu_memory")
                .select("key, value, category, updated_at")
                .ilike("key", f"%{query_key.strip()}%")
                .limit(5)
                .execute()
            )
        else:
            res = (
                supabase.table("chintu_memory")
                .select("key, value, category, updated_at")
                .order("updated_at", desc=True)
                .limit(10)
                .execute()
            )
        
        record_activity()
        if not res.data:
            return f"No memories found matching '{query_key}'."
            
        memories = [f"[{m.get('category', 'general')}] {m.get('key')}: {m.get('value')}" for m in res.data]
        return "Stored Memories:\n" + "\n---\n".join(memories)
    except Exception as e:
        record_activity()
        return f"Memory recall error: {str(e)}"

# ----------------- BASE TOOLS -----------------

def execute_in_cloud_microvm(code: str) -> str:
    """Spins up an isolated Linux MicroVM sandbox in the cloud and executes Python code safely."""
    record_activity()
    if not E2B_API_KEY:
        return "ERROR: E2B_API_KEY environment variable is missing on Render."
    try:
        with Sandbox.create(api_key=E2B_API_KEY) as sandbox:
            execution = sandbox.run_code(code)
            record_activity()
            output = []
            if execution.text:
                output.append(f"Result:\n{execution.text}")
            if execution.logs.stdout:
                output.append("Stdout:\n" + "".join(execution.logs.stdout))
            if execution.logs.stderr:
                output.append("Stderr:\n" + "".join(execution.logs.stderr))
            if execution.error:
                output.append(f"Execution Error: {execution.error.name}: {execution.error.value}")
            return "\n---\n".join(output) if output else "Executed successfully in MicroVM with no output."
    except Exception as e:
        record_activity()
        return f"MicroVM error: {str(e)}"

def web_search(query: str) -> str:
    """Performs live web searches using DuckDuckGo."""
    record_activity()
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No web results found."
        output = [f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}" for r in results]
        return "\n---\n".join(output)
    except Exception as e:
        record_activity()
        return f"Search error: {str(e)}"

def fetch_webpage(url: str) -> str:
    """Fetches clean text content from a web URL."""
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
            text = " ".join(re.sub(r"<[^>]+>", " ", cleaned).split())
            return f"Status {response.getcode()}:\n{text[:3000]}"
    except Exception as e:
        record_activity()
        return f"Fetch error: {str(e)}"

def run_terminal_command(command: str) -> str:
    """Executes a bash shell command directly on the Render server."""
    record_activity()
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        record_activity()
        out = res.stdout if res.stdout else res.stderr
        return out[:3000] if out else "Success (no output)."
    except Exception as e:
        record_activity()
        return f"Execution failed: {str(e)}"

def write_project_file(file_path: str, content: str) -> str:
    """Writes content cleanly to a workspace file."""
    record_activity()
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        record_activity()
        return f"SUCCESS: File '{file_path}' written."
    except Exception as e:
        record_activity()
        return f"ERROR: {str(e)}"

def read_file(file_path: str, start_line: int = 1, line_count: int = 100) -> str:
    """Reads specific lines from a workspace file."""
    record_activity()
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File '{file_path}' does not exist."
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        start = max(1, start_line) - 1
        end = start + line_count
        return f"Lines {start+1}-{min(len(lines), end)}:\n{''.join(lines[start:end])}"
    except Exception as e:
        record_activity()
        return f"ERROR: {str(e)}"

def patch_file(file_path: str, target_block: str, replacement_block: str) -> str:
    """Surgically replaces a snippet of text inside a file."""
    record_activity()
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File '{file_path}' not found."
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if target_block not in content:
            return f"ERROR: Target block not found in '{file_path}'."
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.replace(target_block, replacement_block, 1))
        record_activity()
        return f"SUCCESS: Patched '{file_path}'."
    except Exception as e:
        record_activity()
        return f"ERROR: {str(e)}"

# ----------------- LLAMA SPECIALIST TOOL -----------------

def consult_llama_specialist(task_description: str, code_or_context: str) -> str:
    """Delegates deep reasoning, complex algorithmic work, architecture design, or difficult debugging to Llama 3.3 70B."""
    record_activity()
    if not groq_client:
        return "ERROR: GROQ_API_KEY is not configured on Render. Unable to consult Llama."

    try:
        system_msg = (
            "You are the Lead Reasoning Specialist (Llama 3.3 70B). "
            "You receive complex sub-tasks, code architecture problems, and deep logic queries from Gemini. "
            "Analyze the problem rigorously, fix bugs, optimize algorithms, and provide clean, production-ready solutions."
        )
        user_prompt = f"TASK:\n{task_description}\n\nCONTEXT/CODE:\n{code_or_context}"
        response = groq_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        record_activity()
        return f"[Llama 3.3 70B Specialist Analysis]:\n{response.choices[0].message.content}"
    except Exception as e:
        record_activity()
        return f"Llama consultation error: {str(e)}"

agent_tools = [
    github_read_file,
    github_commit_file,
    github_create_repository,
    send_telegram_photo,
    send_telegram_document,
    remember_information,
    recall_information,
    consult_llama_specialist,
    execute_in_cloud_microvm,
    web_search,
    fetch_webpage,
    run_terminal_command,
    write_project_file,
    read_file,
    patch_file
]

SYSTEM_PROMPT = (
    "You are Chintu, an Autonomous Full-Stack AI Engineer and Team Coordinator.\n\n"
    "GITHUB & SELF-IMPROVEMENT PROTOCOL:\n"
    "- When requested to commit or update code on GitHub, you MUST execute `github_commit_file` directly. Do not simulate file creation.\n"
    "- You can inspect existing repository code using `github_read_file`.\n"
    "- Commits to the main branch trigger auto-deployments on Render.\n"
    "- Use `github_create_repository` when asked to scaffold fresh repositories.\n\n"
    "MEDIA DELIVERY PROTOCOL:\n"
    "- Use `send_telegram_photo` for generated graphs, diagrams, and plots.\n"
    "- Use `send_telegram_document` for files, reports, logs, and zip archives.\n\n"
    "CO-WORK & MEMORY PROTOCOL:\n"
    "1. LONG-TERM MEMORY: Permanent cloud recall via Supabase. Call `recall_information` to load context; call `remember_information` to save facts.\n"
    "2. ROUTER & SCOUT: Coordinate tasks, inspect files, and run web searches.\n"
    "3. SPECIALIST ESCALATION: Call `consult_llama_specialist` for deep algorithmic or mathematical problems.\n"
    "4. EXECUTION: Use `execute_in_cloud_microvm` for safe code testing or `run_terminal_command` for local server commands.\n"
    "5. Provide crisp, direct summaries."
)

def run_autonomous_agent(prompt: str, chat_id: int) -> str:
    global last_progress_time
    record_activity()
    active_chat_id.set(chat_id)
    
    chat = gemini_client.chats.create(
        model=GEMINI_MODEL,
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
        follow_up = chat.send_message("Synthesize and summarize the work done.")
        if follow_up.text and follow_up.text.strip():
            return follow_up.text
        return "Task completed across agent team."
        
    return response.text

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.text:
            return
        if ALLOWED_USER_ID != 0 and update.effective_user.id != ALLOWED_USER_ID:
            await update.message.reply_text("Unauthorized access.")
            return

        task = update.message.text
        status = await update.message.reply_text(f"Task Received:\n'{task[:60]}...'\nProcessing...")
        loop = asyncio.get_running_loop()
        chat_id = update.effective_chat.id
        
        record_activity()
        agent_future = loop.run_in_executor(None, run_autonomous_agent, task, chat_id)
        
        max_idle_seconds = 120.0
        stuck = False
        while not agent_future.done():
            await asyncio.sleep(2.0)
            if (time.time() - last_progress_time) > max_idle_seconds:
                stuck = True
                break

        if stuck:
            agent_future.cancel()
            await status.edit_text("Halt: Agent was idle for > 2 minutes.")
            return

        result = await agent_future
        await status.edit_text(result[:4000] if result else "Execution completed.")
    except Exception as e:
        print(f"Update handler error: {e}", flush=True)

async def run_telegram_worker():
    global bot_instance, main_loop
    main_loop = asyncio.get_running_loop()
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
            bot_instance = bot_app.bot
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
    return {
        "status": "Agent Team Online",
        "models": [GEMINI_MODEL, LLAMA_MODEL],
        "memory": "Supabase Enabled",
        "media": "Telegram Delivery Enabled",
        "github": "
