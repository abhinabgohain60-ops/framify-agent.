import os
import asyncio
import subprocess
import time
import urllib.request
import ssl
import re
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Environment & Config
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
ALLOWED_USER_ID = int((os.getenv("TELEGRAM_ADMIN_ID") or "8513926902").strip() or 0)
E2B_API_KEY = (os.getenv("E2B_API_KEY") or "").strip()
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip()
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
DEFAULT_REPO = (os.getenv("GITHUB_REPO") or "abhinabgohain60-ops/framify-agent.").strip()

# Email Configuration
GMAIL_ADDRESS = (os.getenv("GMAIL_ADDRESS") or "").strip()
GMAIL_APP_PASSWORD = (os.getenv("GMAIL_APP_PASSWORD") or "").strip().replace(" ", "")

# Clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None
github_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

GEMINI_MODEL = "gemini-3.5-flash-lite"
LLAMA_MODEL = "llama-3.3-70b-versatile"

active_chat_id: contextvars.ContextVar[int] = contextvars.ContextVar("active_chat_id", default=0)
bot_instance = None
main_loop = None
scheduler = AsyncIOScheduler()
last_progress_time = time.time()

def record_activity():
    global last_progress_time
    last_progress_time = time.time()

# ----------------- OUTREACH & EMAIL TOOLS -----------------

def send_client_email(to_email: str, subject: str, message_body: str) -> str:
    """Sends a professional email or project proposal to a client or recipient via Gmail SMTP."""
    record_activity()
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        return "ERROR: GMAIL_ADDRESS or GMAIL_APP_PASSWORD missing in environment."
    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = to_email.strip()
        msg["Subject"] = subject.strip()
        msg.attach(MIMEText(message_body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=25)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        record_activity()
        return f"SUCCESS: Sent email to '{to_email.strip()}' with subject '{subject.strip()}'."
    except Exception as e:
        record_activity()
        return f"Email sending error: {str(e)}"

# ----------------- GITHUB TOOLS -----------------

def github_read_file(file_path: str, repo_name: str = "", branch: str = "main") -> str:
    """Reads raw contents of a file from GitHub."""
    record_activity()
    if not github_client: return "ERROR: GITHUB_TOKEN missing."
    target_repo = repo_name.strip() if repo_name.strip() else DEFAULT_REPO
    try:
        repo = github_client.get_repo(target_repo)
        fc = repo.get_contents(file_path, ref=branch)
        record_activity()
        return fc.decoded_content.decode("utf-8")
    except Exception as e:
        record_activity()
        return f"GitHub read error: {str(e)}"

def github_commit_file(file_path: str, content: str, commit_message: str, repo_name: str = "", branch: str = "main") -> str:
    """Commits or updates a file in GitHub, triggering Render redeployment."""
    record_activity()
    if not github_client: return "ERROR: GITHUB_TOKEN missing."
    target_repo = repo_name.strip() if repo_name.strip() else DEFAULT_REPO
    try:
        repo = github_client.get_repo(target_repo)
        try:
            cur = repo.get_contents(file_path, ref=branch)
            repo.update_file(path=file_path, message=commit_message, content=content, sha=cur.sha, branch=branch)
            record_activity()
            return f"SUCCESS: Updated '{file_path}' in '{target_repo}'."
        except GithubException as ge:
            if ge.status == 404:
                repo.create_file(path=file_path, message=commit_message, content=content, branch=branch)
                record_activity()
                return f"SUCCESS: Created '{file_path}' in '{target_repo}'."
            raise ge
    except Exception as e:
        record_activity()
        return f"GitHub commit error: {str(e)}"

def github_create_repository(repo_name: str, description: str = "", private: bool = False) -> str:
    """Creates a new repository under the GitHub account."""
    record_activity()
    if not github_client: return "ERROR: GITHUB_TOKEN missing."
    try:
        u = github_client.get_user()
        r = u.create_repo(name=repo_name, description=description, private=private, auto_init=True)
        record_activity()
        return f"SUCCESS: Created repo '{r.full_name}'."
    except Exception as e:
        record_activity()
        return f"GitHub create repo error: {str(e)}"

# ----------------- TELEGRAM MEDIA DELIVERY -----------------

def send_telegram_photo(file_path: str, caption: str = "") -> str:
    """Dispatches a local photo or chart directly to Telegram."""
    record_activity()
    if not os.path.exists(file_path): return f"ERROR: File '{file_path}' not found."
    chat_id = active_chat_id.get() or ALLOWED_USER_ID
    if not chat_id or not bot_instance or not main_loop: return "ERROR: Telegram dispatch unavailable."
    try:
        async def _send():
            with open(file_path, "rb") as f:
                await bot_instance.send_photo(chat_id=chat_id, photo=f, caption=caption[:1024])
        asyncio.run_coroutine_threadsafe(_send(), main_loop).result(timeout=30)
        record_activity()
        return f"SUCCESS: Sent photo '{file_path}'."
    except Exception as e:
        record_activity()
        return f"Photo dispatch error: {str(e)}"

def send_telegram_document(file_path: str, caption: str = "") -> str:
    """Dispatches any local file directly to Telegram."""
    record_activity()
    if not os.path.exists(file_path): return f"ERROR: File '{file_path}' not found."
    chat_id = active_chat_id.get() or ALLOWED_USER_ID
    if not chat_id or not bot_instance or not main_loop: return "ERROR: Telegram dispatch unavailable."
    try:
        async def _send():
            with open(file_path, "rb") as f:
                await bot_instance.send_document(chat_id=chat_id, document=f, caption=caption[:1024])
        asyncio.run_coroutine_threadsafe(_send(), main_loop).result(timeout=30)
        record_activity()
        return f"SUCCESS: Sent document '{file_path}'."
    except Exception as e:
        record_activity()
        return f"Doc dispatch error: {str(e)}"

# ----------------- MEMORY TOOLS -----------------

def remember_information(key: str, value: str, category: str = "general") -> str:
    """Persists memories to Supabase."""
    record_activity()
    if not supabase: return "ERROR: Supabase missing."
    try:
        supabase.table("chintu_memory").upsert({"key": key.strip().lower(), "value": value.strip(), "category": category.strip().lower()}, on_conflict="key").execute()
        record_activity()
        return f"SUCCESS: Remembered '{key}'."
    except Exception as e:
        record_activity()
        return f"Memory save error: {str(e)}"

def recall_information(query_key: str = "") -> str:
    """Recalls memories from Supabase."""
    record_activity()
    if not supabase: return "ERROR: Supabase missing."
    try:
        if query_key.strip():
            res = supabase.table("chintu_memory").select("key, value, category").ilike("key", f"%{query_key.strip()}%").limit(5).execute()
        else:
            res = supabase.table("chintu_memory").select("key, value, category").order("updated_at", desc=True).limit(10).execute()
        record_activity()
        if not res.data: return f"No memories found for '{query_key}'."
        return "\n".join([f"[{m.get('category')}] {m.get('key')}: {m.get('value')}" for m in res.data])
    except Exception as e:
        record_activity()
        return f"Recall error: {str(e)}"

# ----------------- CRON SCHEDULER TOOLS -----------------

async def _scheduled_task_runner(task_id: str, prompt: str, target_chat_id: int):
    """Internal runner executed in background by APScheduler."""
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, run_autonomous_agent, prompt, target_chat_id)
        if bot_instance and target_chat_id:
            await bot_instance.send_message(
                chat_id=target_chat_id,
                text=f"⏰ [Cron Task: {task_id}]\n\n{res[:3900]}"
            )
    except Exception as e:
        print(f"Scheduled task error for {task_id}: {e}", flush=True)

def schedule_recurring_task(task_id: str, prompt: str, interval_minutes: int) -> str:
    """Schedules an autonomous task to run repeatedly every interval_minutes and dispatch results to Telegram."""
    record_activity()
    target_chat = active_chat_id.get() or ALLOWED_USER_ID
    clean_id = task_id.strip().lower().replace(" ", "_")
    interval = max(1, int(interval_minutes))
    try:
        scheduler.add_job(
            _scheduled_task_runner,
            "interval",
            minutes=interval,
            id=clean_id,
            replace_existing=True,
            args=[clean_id, prompt, target_chat]
        )
        if supabase:
            job_meta = json.dumps({"prompt": prompt, "interval_minutes": interval, "chat_id": target_chat})
            remember_information(f"cron_{clean_id}", job_meta, category="cron_schedule")
        record_activity()
        return f"SUCCESS: Scheduled recurring task '{clean_id}' every {interval} minute(s)."
    except Exception as e:
        record_activity()
        return f"Scheduling error: {str(e)}"

def list_scheduled_tasks() -> str:
    """Lists all active background recurring cron tasks."""
    record_activity()
    jobs = scheduler.get_jobs()
    if not jobs: return "No active scheduled recurring tasks found."
    lines = [f"- ID: '{j.id}' (Next Run: {j.next_run_time})" for j in jobs]
    return "Active Scheduled Tasks:\n" + "\n".join(lines)

def cancel_scheduled_task(task_id: str) -> str:
    """Cancels and removes a scheduled recurring task by ID."""
    record_activity()
    clean_id = task_id.strip().lower().replace(" ", "_")
    try:
        scheduler.remove_job(clean_id)
        record_activity()
        return f"SUCCESS: Cancelled scheduled task '{clean_id}'."
    except Exception as e:
        record_activity()
        return f"Failed to cancel task '{clean_id}': {str(e)}"

# ----------------- SYSTEM & SPECIALIST -----------------

def execute_in_cloud_microvm(code: str) -> str:
    """Executes Python code in an E2B microVM sandbox."""
    record_activity()
    if not E2B_API_KEY: return "ERROR: E2B_API_KEY missing."
    try:
        with Sandbox.create(api_key=E2B_API_KEY) as s:
            r = s.run_code(code)
            record_activity()
            out = []
            if r.text: out.append(r.text)
            if r.logs.stdout: out.append("".join(r.logs.stdout))
            if r.logs.stderr: out.append("".join(r.logs.stderr))
            if r.error: out.append(f"{r.error.name}: {r.error.value}")
            return "\n".join(out) if out else "Executed without output."
    except Exception as e:
        record_activity()
        return f"MicroVM error: {str(e)}"

def web_search(query: str) -> str:
    """Live web search via DuckDuckGo."""
    record_activity()
    try:
        with DDGS() as d:
            res = list(d.text(query, max_results=4))
        record_activity()
        return "\n\n".join([f"{r.get('title')}: {r.get('body')} ({r.get('href')})" for r in res]) if res else "No results."
    except Exception as e:
        record_activity()
        return f"Search error: {str(e)}"

def fetch_webpage(url: str) -> str:
    """Fetches text content from a web page."""
    record_activity()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
            record_activity()
            t = " ".join(re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style).*?</\1>", "", r.read().decode("utf-8", errors="ignore"), flags=re.DOTALL)).split())
            return t[:3000]
    except Exception as e:
        record_activity()
        return f"Fetch error: {str(e)}"

def run_terminal_command(command: str) -> str:
    """Runs a bash command on the local container."""
    record_activity()
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        record_activity()
        out = r.stdout or r.stderr
        return out[:3000] if out else "Success."
    except Exception as e:
        record_activity()
        return f"Bash error: {str(e)}"

def write_project_file(file_path: str, content: str) -> str:
    """Writes content to disk."""
    record_activity()
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f: f.write(content)
        record_activity()
        return f"SUCCESS: Wrote '{file_path}'."
    except Exception as e:
        record_activity()
        return f"Write error: {str(e)}"

def read_file(file_path: str, start_line: int = 1, line_count: int = 100) -> str:
    """Reads lines from disk."""
    record_activity()
    try:
        if not os.path.exists(file_path): return f"ERROR: File '{file_path}' missing."
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f: lines = f.readlines()
        record_activity()
        s = max(1, start_line) - 1
        return "".join(lines[s:s + line_count])
    except Exception as e:
        record_activity()
        return f"Read error: {str(e)}"

def patch_file(file_path: str, target_block: str, replacement_block: str) -> str:
    """Patches content in a file."""
    record_activity()
    try:
        if not os.path.exists(file_path): return f"ERROR: '{file_path}' missing."
        with open(file_path, "r", encoding="utf-8") as f: c = f.read()
        if target_block not in c: return f"Target block missing in '{file_path}'."
        with open(file_path, "w", encoding="utf-8") as f: f.write(c.replace(target_block, replacement_block, 1))
        record_activity()
        return f"SUCCESS: Patched '{file_path}'."
    except Exception as e:
        record_activity()
        return f"Patch error: {str(e)}"

def consult_llama_specialist(task_description: str, code_or_context: str) -> str:
    """Consults Llama 3.3 70B specialist on Groq for deep reasoning."""
    record_activity()
    if not groq_client: return "ERROR: GROQ_API_KEY missing."
    try:
        resp = groq_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are Llama 3.3 70B Specialist. Solve complex engineering tasks."},
                {"role": "user", "content": f"TASK:\n{task_description}\n\nCONTEXT:\n{code_or_context}"}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        record_activity()
        return f"[Llama Specialist]: {resp.choices[0].message.content}"
    except Exception as e:
        record_activity()
        return f"Llama error: {str(e)}"

agent_tools = [
    send_client_email,
    schedule_recurring_task,
    list_scheduled_tasks,
    cancel_scheduled_task,
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
    "You are Chintu, an Autonomous Full-Stack AI Engineer.\n"
    "- CLIENT OUTREACH: Use `send_client_email` to contact clients, deliver proposals, or send notifications directly.\n"
    "- AUTONOMOUS SCHEDULING: Use `schedule_recurring_task`, `list_scheduled_tasks`, and `cancel_scheduled_task` to manage background jobs.\n"
    "- GITHUB: Use `github_commit_file` to commit changes directly, `github_read_file` to read repo code, and `github_create_repository` for new repos.\n"
    "- MEDIA: Use `send_telegram_photo` for charts and `send_telegram_document` for files.\n"
    "- MEMORY: Use `remember_information` and `recall_information` with Supabase.\n"
    "- REASONING: Use `consult_llama_specialist` for complex logic or coding tasks."
)

def run_autonomous_agent(prompt: str, chat_id: int) -> str:
    global last_progress_time
    record_activity()
    active_chat_id.set(chat_id)
    chat = gemini_client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=agent_tools, temperature=0.2)
    )
    res = chat.send_message(prompt)
    record_activity()
    if not (res.text and res.text.strip()):
        f_up = chat.send_message("Summarize results.")
        return f_up.text if f_up.text else "Task completed."
    return res.text

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.text: return
        if ALLOWED_USER_ID != 0 and update.effective_user.id != ALLOWED_USER_ID:
            await update.message.reply_text("Unauthorized.")
            return
        task = update.message.text
        status = await update.message.reply_text(f"Task: {task[:50]}...\nProcessing...")
        loop = asyncio.get_running_loop()
        chat_id = update.effective_chat.id
        record_activity()
        fut = loop.run_in_executor(None, run_autonomous_agent, task, chat_id)
        while not fut.done():
            await asyncio.sleep(2.0)
            if (time.time() - last_progress_time) > 120.0:
                fut.cancel()
                await status.edit_text("Halt: Agent timed out.")
                return
        out = await fut
        await status.edit_text(out[:4000] if out else "Execution complete.")
    except Exception as e:
        print(f"Handler error: {e}", flush=True)

async def run_telegram_worker():
    global bot_instance, main_loop
    main_loop = asyncio.get_running_loop()
    await asyncio.sleep(3)
    while True:
        try:
            bot_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            bot_instance = bot_app.bot
            bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_msg))
            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling(drop_pending_updates=False)
            print("Telegram poller running...", flush=True)
            while True: await asyncio.sleep(3600)
        except Exception as e:
            print(f"Poller restarted: {e}", flush=True)
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    worker_task = asyncio.create_task(run_telegram_worker())
    yield
    scheduler.shutdown()
    worker_task.cancel()

api = FastAPI(lifespan=lifespan)

@api.get("/")
def home():
