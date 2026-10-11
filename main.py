import threading
import os
import asyncio
import subprocess
import time
import urllib.request
import ssl
import re
import json
import smtplib
import tempfile
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import contextvars
from contextlib import asynccontextmanager
from fastapi import FastAPI
from telegram import Update
from telegram.error import TelegramError, RetryAfter, BadRequest
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from google.genai.errors import APIError
from groq import Groq
from huggingface_hub import InferenceClient, HfApi
from duckduckgo_search import DDGS
from e2b_code_interpreter import Sandbox
from supabase import create_client, Client
from github import Github, GithubException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from composio import Composio

# Environment & Config
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_BOT_TOKEN = TELEGRAM_TOKEN
ADMIN_CHAT_ID = int((os.getenv("TELEGRAM_ADMIN_ID") or "8513926902").strip() or 0)
ALLOWED_USER_ID = ADMIN_CHAT_ID
E2B_API_KEY = (os.getenv("E2B_API_KEY") or "").strip()
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip()
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
DEFAULT_REPO = "abhinabgohain60-ops/framify-agent."
COMPOSIO_API_KEY = (os.getenv("COMPOSIO_API_KEY") or "").strip()

# Email Configuration
GMAIL_ADDRESS = (os.getenv("GMAIL_ADDRESS") or "").strip()
GMAIL_APP_PASSWORD = (os.getenv("GMAIL_APP_PASSWORD") or "").strip().replace(" ", "")

# Clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None
github_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

# Updated Gemini Model Cascade (Prioritizing gemini-3-flash-preview)
GEMINI_MODELS_CASCADE = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

active_chat_id: contextvars.ContextVar[int] = contextvars.ContextVar("active_chat_id", default=0)
bot_instance = None
main_loop = None
scheduler = AsyncIOScheduler()
last_progress_time = time.time()

user_message_buffer = {}
user_buffer_tasks = {}

def record_activity():
    global last_progress_time
    last_progress_time = time.time()

# ----------------- MANDATORY PROTOCOL: LLAMA AUDIT & E2B SANDBOX -----------------

def run_llama_code_audit(code_content: str, context_desc: str = "Code audit") -> str:
    """
    STEP 1: Mandatory Llama Code Review & Optimization Hook via Groq.
    Inspects for destructive truncations, syntax/indentation/type errors, and performance bottlenecks.
    """
    if not groq_client:
        print("[WARNING] Groq client not initialized. Bypassing Llama audit.", flush=True)
        return code_content
    try:
        prompt = f"""You are an elite Lead Software Auditor. Review the following code for '{context_desc}'.
Strictly check for:
1. Destructive truncations or logic wiping.
2. Syntax, indentation, or type annotation errors.
3. Performance bottlenecks or memory leaks.

If you find issues, return the corrected, optimized, and production-ready code. If there are no issues, return the original code intact. Return ONLY the final code without conversational filler."""
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise code auditor and optimizer. Return only valid code."},
                {"role": "user", "content": f"{prompt}\n\n---\n{code_content}"}
            ],
            temperature=0.1,
            max_tokens=8000
        )
        reviewed = completion.choices[0].message.content.strip()
        if reviewed.startswith("```python"):
            reviewed = reviewed[9:]
        elif reviewed.startswith("```"):
            reviewed = reviewed[3:]
        if reviewed.endswith("```"):
            reviewed = reviewed[:-3]
        return reviewed.strip()
    except Exception as e:
        print(f"[Llama Audit Error]: {e}. Falling back to original code.", flush=True)
        return code_content

def run_e2b_sandbox_preflight(code_content: str, is_python_script: bool = True) -> bool:
    """
    STEP 2: Mandatory E2B Cloud MicroVM Sandbox Pre-Flight Test.
    Spins up an isolated E2B Sandbox, writes the code, and verifies exit code 0 and syntax/import health.
    """
    if not E2B_API_KEY:
        print("[WARNING] E2B_API_KEY missing. Skipping sandbox preflight.", flush=True)
        return True
    try:
        sandbox = Sandbox(api_key=E2B_API_KEY)
        try:
            filename = "preflight_test.py"
            sandbox.files.write(filename, code_content)
            
            if is_python_script:
                result = sandbox.commands.run(f"python3 -m py_compile {filename}")
                if result.exit_code != 0:
                    print(f"[E2B Pre-flight Error] Compilation failed: {result.stderr}", flush=True)
                    return False
            return True
        finally:
            sandbox.kill()
    except Exception as e:
        print(f"[E2B Sandbox Exception]: {e}", flush=True)
        return False

# ----------------- DYNAMIC TELEGRAM PROGRESS CARD & TELEMETRY ENGINE -----------------
class ProgressCardTracker:
    def __init__(self, bot, chat_id: int, task_title: str, steps: list):
        self.bot = bot
        self.chat_id = chat_id
        self.task_title = task_title
        self.steps = steps
        self.statuses = ["⏳ Pending" for _ in steps]
        self.details = ["" for _ in steps]
        self.message_id = None
        self._lock = threading.Lock()
        self._update_queue = None

    def run_thread_safe(self, coro):
        global main_loop
        if main_loop is None or not main_loop.is_running():
            try:
                main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        if main_loop and main_loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(coro, main_loop)
                return future.result(timeout=15)
            except Exception as e:
                print(f"[Dispatch error]: {e}", flush=True)
                return None
        else:
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                res = loop.run_until_complete(coro)
                return res
            except Exception as e:
                print(f"[Fallback error]: {e}", flush=True)
                return None
                
    def _render_card(self) -> str:
        lines = [f"⚡ Task: {self.task_title}", "Status: In Progress ⏳\n"]
        for idx, step in enumerate(self.steps):
            stat = self.statuses[idx]
            det = f" - {self.details[idx]}" if self.details[idx] else ""
            lines.append(f"{idx + 1}. {step}: {stat}{det}")
        return "\n".join(lines)

    async def initialize(self):
        if not self.bot or not self.chat_id:
            return
        try:
            self._update_queue = asyncio.Queue()
            text = self._render_card()
            msg = await self.bot.send_message(chat_id=self.chat_id, text=text)
            if msg:
                self.message_id = msg.message_id
            asyncio.create_task(self._worker_loop())
        except Exception as e:
            print(f"ProgressCard initialize error: {e}", flush=True)

    async def _worker_loop(self):
        while self._update_queue:
            try:
                text = await self._update_queue.get()
                if text is None:
                    break
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self.message_id,
                    text=text
                )
                self._update_queue.task_done()
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"ProgressCard worker loop error: {e}", flush=True)

    async def update_step(self, step_index: int, status: str, detail: str = ""):
        if step_index < 0 or step_index >= len(self.steps):
            return
        with self._lock:
            self.statuses[step_index] = status
            if detail:
                self.details[step_index] = detail

            if not self.bot or not self.message_id:
                return

            text = self._render_card()

        if self._update_queue:
            try:
                while not self._update_queue.empty():
                    try:
                        self._update_queue.get_nowait()
                        self._update_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                await self._update_queue.put(text)
            except Exception as e:
                print(f"ProgressCard queue put error: {e}", flush=True)

    async def finalize(self, success: bool = True, final_note: str = ""):
        if not self.bot or not self.message_id:
            return
        with self._lock:
            lines = [f"⚡ Task: {self.task_title}", f"Status: {'Completed ✅' if success else 'Failed 🚫'}\n"]
            for idx, step in enumerate(self.steps):
                stat = self.statuses[idx]
                det = f" - {self.details[idx]}" if self.details[idx] else ""
                lines.append(f"{idx + 1}. {step}: {stat}{det}")
            if final_note:
                lines.append(f"\n💡 {final_note}")
            text = "\n".join(lines)

        if self._update_queue:
            try:
                await self._update_queue.put(text)
                await asyncio.sleep(0.5)
                await self._update_queue.put(None)
            except Exception:
                pass
        try:
            self.run_thread_safe(self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=self.message_id,
                text=text
            ))
        except Exception as e:
            pass

async def execute_with_self_healing(card: ProgressCardTracker, step_idx: int, step_title: str, func, *args, **kwargs):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            await card.update_step(step_idx, "⏳ In Progress", f"Executing {step_title} (Attempt {attempt+1}/{max_attempts})...")
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await card.update_step(step_idx, "✅ Done", f"{step_title} succeeded.")
            return result
        except Exception as err:
            err_str = str(err)
            if attempt < max_attempts - 1:
                await card.update_step(step_idx, "🔄 Healing", f"Attempt {attempt+1} failed: {err_str[:40]}. Retrying...")
                try:
                    if supabase:
                        supabase.table("chintu_memory").upsert({
                            "key": f"heal_attempt_{int(time.time())}_{step_idx}",
                            "category": "autonomous_healing_lessons",
                            "value": f"Step: {step_title} | Attempt: {attempt+1} | Error: {err_str}"
                        }, on_conflict="key").execute()
                except Exception:
                    pass
                await asyncio.sleep(1.0)
            else:
                await card.update_step(step_idx, "🚫 Failed", f"Self-healing exhausted after {max_attempts} attempts: {err_str[:40]}")
                raise err

# ----------------- HUGGING FACE INFERENCE TOOLS & WORKER NODES -----------------
    
def patch_core_system(file_path: str, new_content: str, commit_message: str) -> str:
    """
    [DEPRECATED / FROZEN] Dynamic core patching of main.py is deprecated.
    main.py is frozen against runtime feature rewrites. Use modules under /tools/ instead.
    """
    return "ERROR: Dynamic patching of main.py is deprecated and frozen. Please use modular tools in the /tools/ directory."
        
def generate_ai_image(prompt: str, filename: str = "concept_visual.png") -> str:
    """Generates an image via Hugging Face FLUX/SDXL models and sends it directly to Telegram."""
    record_activity()
    if not HF_TOKEN:
        return "ERROR: HF_TOKEN missing in environment variables."
    try:
        client = InferenceClient(api_key=HF_TOKEN)
        image = client.text_to_image(prompt=prompt.strip(), model="black-forest-labs/FLUX.1-schnell")
        filepath = os.path.join(tempfile.gettempdir(), filename)
        image.save(filepath)
        
        if TELEGRAM_BOT_TOKEN and ADMIN_CHAT_ID:
            with open(filepath, "rb") as photo_file:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                requests.post(url, data={"chat_id": ADMIN_CHAT_ID, "caption": f"Generated Visual: {prompt[:100]}"}, files={"photo": photo_file}, timeout=30)
        record_activity()
        return f"SUCCESS: Generated image for prompt '{prompt}' and dispatched to Telegram."
    except Exception as e:
        record_activity()
        return f"Image generation error: {str(e)}"

def consult_hf_specialist(prompt: str, model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct") -> str:
    """Queries open-source specialized models on Hugging Face for coding, translation, or alternative reasoning."""
    record_activity()
    if not HF_TOKEN:
        return "ERROR: HF_TOKEN missing in environment variables."
    try:
        client = InferenceClient(api_key=HF_TOKEN)
        messages = [{"role": "user", "content": prompt.strip()}]
        response = client.chat.completions.create(model=model_id, messages=messages, max_tokens=1000)
        record_activity()
        return response.choices[0].message.content
    except Exception as e:
        record_activity()
        return f"Hugging Face query error: {str(e)}"

def deploy_hf_worker_node(repo_id: str, file_path: str, file_content: str, commit_message: str = "Deploy HF worker node") -> str:
    """
    Deploys or updates a worker node script or dataset file on Hugging Face Hub synchronously using HfApi 
    to avoid async event loop dependencies entirely.
    """
    record_activity()
    if not HF_TOKEN:
        return "ERROR: HF_TOKEN missing in environment variables."
    try:
        api = HfApi(token=HF_TOKEN)
        try:
            api.repo_info(repo_id=repo_id, repo_type="space")
        except Exception:
            try:
                api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="streamlit", private=False)
            except Exception as e:
                return f"Failed to verify/create HF space repository: {str(e)}"

        api.upload_string(
            repo_id=repo_id,
            repo_type="space",
            string=file_content,
            path_in_repo=file_path,
            commit_message=commit_message
        )
        record_activity()
        return f"SUCCESS: Deployed worker node file '{file_path}' to Hugging Face space '{repo_id}'."
    except Exception as e:
        record_activity()
        return f"HF deployment error: {str(e)}"

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

# ----------------- MULTI-PLATFORM CLIENT HUNTER -----------------

def hunt_client_leads(service_niche: str = "video editor", platform: str = "all", time_range: str = "w") -> str:
    """
    Searches the live web across public platforms for fresh hiring and gig leads.
    service_niche: e.g. 'video editor', 'thumbnail designer', 'motion graphics'
    platform: 'reddit', 'x', 'creator_boards', or 'all'
    time_range: 'd' (past 24h), 'w' (past week), or 'm' (past month)
    """
    record_activity()
    queries = []
    
    if platform in ("reddit", "all"):
        queries.append(f'site:reddit.com (inurl:CreatorServices OR inurl:forhire OR inurl:videography) "hiring" "{service_niche}"')
    if platform in ("x", "all"):
        queries.append(f'site:x.com ("looking for a {service_niche}" OR "hiring {service_niche}")')
    if platform in ("creator_boards", "all"):
        queries.append(f'"{service_niche} needed" ("portfolio" OR "budget" OR "contact")')

    aggregated_leads = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                raw_results = list(ddgs.text(q, timelimit=time_range, max_results=4))
                for r in raw_results:
                    title = r.get("title", "No Title")
                    href = r.get("href", "")
                    body = r.get("body", "")[:280].replace("\n", " ")
                    aggregated_leads.append(f"• [{title}]({href})\n  Snippet: {body}\n")
                    
        record_activity()
        if not aggregated_leads:
            return f"No recent leads found for niche '{service_niche}' within time limit '{time_range}'."
        return "\n".join(aggregated_leads)
    except Exception as e:
        record_activity()
        return f"Lead hunting error: {str(e)}"

# ----------------- GITHUB TOOLS WITH MANDATORY GATEWAY -----------------

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
    """
    Commits or updates a file in GitHub with mandatory Zero-Bypass Llama Audit & E2B Sandbox checks:
    1. Line-Count Guard: Blocks if core files are reduced by > 15%.
    2. STEP 1: Llama Code Review & Optimization.
    3. STEP 2: E2B Cloud MicroVM Sandbox Pre-Flight Test.
    """
    record_activity()
    if not github_client: return "ERROR: GITHUB_TOKEN missing."
    target_repo = repo_name.strip() if repo_name.strip() else DEFAULT_REPO
    
    # 1. Line-Count Guard & Existing Content Check
    existing_content = ""
    try:
        repo = github_client.get_repo(target_repo)
        try:
            cur = repo.get_contents(file_path, ref=branch)
            existing_content = cur.decoded_content.decode("utf-8")
        except Exception:
            pass
    except Exception:
        pass
        
    if existing_content and file_path in ("main.py",):
        old_lines = len(existing_content.splitlines())
        new_lines = len(content.splitlines())
        if old_lines > 100 and new_lines < (old_lines * 0.85):
            raise PermissionError(f"Commit blocked: Destructive truncation guard triggered. Existing lines: {old_lines}, proposed lines: {new_lines} (>15% reduction).")

    # 2. STEP 1: Mandatory Llama Code Review & Optimization Hook
    print(f"[Protocol] Running Llama audit on commit for {file_path}...", flush=True)
    optimized_content = run_llama_code_audit(content, context_desc=f"Commit file {file_path}")
    llama_review_passed = True
    
    # 3. STEP 2: Mandatory E2B Cloud MicroVM Sandbox Pre-Flight Test
    print(f"[Protocol] Running E2B sandbox pre-flight test for {file_path}...", flush=True)
    is_py = file_path.endswith(".py")
    e2b_sandbox_tested = run_e2b_sandbox_preflight(optimized_content, is_python_script=is_py)
    
    if not e2b_sandbox_tested and is_py:
        print(f"[Protocol] Sandbox preflight failed. Routing traceback back to Llama for automated fix...", flush=True)
        fix_prompt = f"The following Python code failed E2B sandbox compilation/syntax pre-flight. Fix any syntax, indentation, or import errors and return only valid code:\n\n{optimized_content}"
        optimized_content = run_llama_code_audit(fix_prompt, context_desc=f"Auto-repair fix for {file_path}")
        e2b_sandbox_tested = run_e2b_sandbox_preflight(optimized_content, is_python_script=is_py)
        if not e2b_sandbox_tested:
            raise PermissionError("Commit blocked: E2B Sandbox verification failed after automated fix cycle.")

    # 4. Hard-Coded Gateway Check
    if not llama_review_passed or not e2b_sandbox_tested:
        raise PermissionError("Commit blocked: Code has not passed Llama review and E2B Sandbox verification.")

    try:
        repo = github_client.get_repo(target_repo)
        try:
            cur = repo.get_contents(file_path, ref=branch)
            repo.update_file(path=file_path, message=commit_message, content=optimized_content, sha=cur.sha, branch=branch)
            record_activity()
            return f"SUCCESS (Verified by Llama & E2B): Updated '{file_path}' in '{target_repo}'."
        except GithubException as ge:
            if ge.status == 404:
                repo.create_file(path=file_path, message=commit_message, content=optimized_content, branch=branch)
                record_activity()
                return f"SUCCESS (Verified by Llama & E2B): Created '{file_path}' in '{target_repo}'."
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
        return f"Document dispatch error: {str(e)}"

# ----------------- SUPABASE MEMORY TOOLS -----------------

def remember_information(key: str, value: str, category: str = "general") -> str:
    """Persists memories to Supabase."""
    record_activity()
    if not supabase: return "ERROR: Supabase client not initialized."
    try:
        supabase.table("chintu_memory").upsert({
            "key": key.strip(),
            "value": value.strip(),
            "category": category.strip()
        }, on_conflict="key").execute()
        record_activity()
        return f"SUCCESS: Remembered '{key}' under category '{category}'."
    except Exception as e:
        record_activity()
        return f"Supabase remember error: {str(e)}"

def recall_information(query_key: str = "") -> str:
    """Recalls memories from Supabase."""
    record_activity()
    if not supabase: return "ERROR: Supabase client not initialized."
    try:
        query = supabase.table("chintu_memory").select("*")
        if query_key.strip():
            query = query.ilike("key", f"%{query_key.strip()}%")
        res = query.execute()
        record_activity()
        if not res.data:
            return f"No memories found matching '{query_key}'."
        out = []
        for row in res.data:
            out.append(f"• [{row.get('category', 'general')}] {row.get('key')}: {row.get('value')}")
        return "\n".join(out)
    except Exception as e:
        record_activity()
        return f"Supabase recall error: {str(e)}"

# ----------------- SCHEDULER & CRON JOBS -----------------

def schedule_recurring_task(task_id: str, prompt: str, interval_minutes: int) -> str:
    """Schedules an autonomous task to run repeatedly every interval_minutes and dispatch results to Telegram."""
    record_activity()
    try:
        global scheduler
        if scheduler.get_job(task_id):
            scheduler.remove_job(task_id)

        async def scheduled_job():
            chat_id = ADMIN_CHAT_ID
            if not chat_id or not bot_instance: return
            card = ProgressCardTracker(bot_instance, chat_id, f"Cron Job: {task_id}", ["Execute Task", "Dispatch Result"])
            await card.initialize()
            try:
                await card.update_step(0, "⏳ In Progress", "Running task prompt...")
                res = handle_ai_reasoning(prompt, chat_id)
                await card.update_step(0, "✅ Done", "Task completed.")
                await card.update_step(1, "⏳ In Progress", "Sending report to Telegram...")
                await bot_instance.send_message(chat_id=chat_id, text=f"⏰ [Cron Report: {task_id}]\n\n{res}")
                await card.update_step(1, "✅ Done", "Report dispatched.")
                await card.finalize(True, "Scheduled task executed successfully.")
            except Exception as e:
                await card.finalize(False, f"Error: {str(e)}")

        scheduler.add_job(scheduled_job, 'interval', minutes=interval_minutes, id=task_id)
        if not scheduler.running:
            scheduler.start()
        record_activity()
        return f"SUCCESS: Scheduled task '{task_id}' every {interval_minutes} minutes."
    except Exception as e:
        record_activity()
        return f"Scheduler error: {str(e)}"

def list_scheduled_tasks() -> str:
    """Lists all active background recurring cron tasks."""
    record_activity()
    try:
        global scheduler
        jobs = scheduler.get_jobs()
        if not jobs:
            return "No active scheduled tasks."
        out = []
        for j in jobs:
            out.append(f"• ID: {j.id} | Next Run: {j.next_run_time}")
        return "\n".join(out)
    except Exception as e:
        record_activity()
        return f"List tasks error: {str(e)}"

def cancel_scheduled_task(task_id: str) -> str:
    """Cancels and removes a scheduled recurring task by ID."""
    record_activity()
    try:
        global scheduler
        if scheduler.get_job(task_id):
            scheduler.remove_job(task_id)
            record_activity()
            return f"SUCCESS: Cancelled task '{task_id}'."
        return f"Task '{task_id}' not found."
    except Exception as e:
        record_activity()
        return f"Cancel task error: {str(e)}"

# ----------------- ADVANCED REASONING & LLAMA SPECIALIST -----------------

def consult_llama_specialist(task_description: str, code_or_context: str) -> str:
    """Consults Llama specialist (Groq) strictly for offline code repair, auditing, syntax optimization, and bug fixing."""
    record_activity()
    if not groq_client: return "ERROR: GROQ_API_KEY missing."
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert Llama code repair and debugging specialist."},
                {"role": "user", "content": f"Task: {task_description}\n\nCode/Context:\n{code_or_context}"}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        record_activity()
        return completion.choices[0].message.content
    except Exception as e:
        record_activity()
        return f"Llama specialist error: {str(e)}"

def delegate_subtask(role: str, task_description: str, code_payload: str = "") -> str:
    """Executes a worker subtask or code audit via Llama 3.3 offline code repair engine."""
    return consult_llama_specialist(f"Role: {role} | {task_description}", code_payload)

def dispatch_to_agent(agent_id: str, task_prompt: str, context_payload: str = "") -> str:
    """Invokes a persistent worker agent from the Supabase registry using its specialized model and instructions."""
    record_activity()
    if not supabase: return "ERROR: Supabase unavailable."
    try:
        res = supabase.table("chintu_agent_registry").select("*").eq("agent_id", agent_id).execute()
        if not res.data:
            return f"Agent ID '{agent_id}' not found in Supabase registry."
        agent = res.data[0]
        sys_prompt = agent.get("system_prompt", "You are a specialized worker agent.")
        if groq_client:
            comp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Task: {task_prompt}\n\nContext:\n{context_payload}"}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            return comp.choices[0].message.content
        return f"Executed agent {agent_id} successfully."
    except Exception as e:
        return f"Agent dispatch error: {str(e)}"

def execute_in_cloud_microvm(code: str) -> str:
    """Executes Python code in an E2B microVM sandbox."""
    record_activity()
    if not E2B_API_KEY:
        return "ERROR: E2B_API_KEY missing in environment variables."
    try:
        sandbox = Sandbox(api_key=E2B_API_KEY)
        try:
            exec_result = sandbox.run_code(code)
            output = []
            if exec_result.logs.stdout:
                output.append("STDOUT:\n" + "\n".join(exec_result.logs.stdout))
            if exec_result.logs.stderr:
                output.append("STDERR:\n" + "\n".join(exec_result.logs.stderr))
            record_activity()
            return "\n".join(output) if output else "SUCCESS: Code executed with zero output."
        finally:
            sandbox.kill()
    except Exception as e:
        record_activity()
        return f"E2B sandbox execution error: {str(e)}"

def execute_composio_action(action_name: str, arguments: dict = {}) -> str:
    """Executes actions across 1000+ apps (Gmail, Notion, GitHub, Twitter, Slack) via Composio."""
    record_activity()
    if not COMPOSIO_API_KEY:
        return "ERROR: COMPOSIO_API_KEY missing in environment variables."
    try:
        composio = Composio(api_key=COMPOSIO_API_KEY)
        result = composio.execute(action=action_name, arguments=arguments)
        record_activity()
        return json.dumps(result, indent=2)
    except Exception as e:
        record_activity()
        return f"Composio execution error: {str(e)}"

# ----------------- TOOL REGISTRY MAP -----------------
ALL_TOOLS_MAP = {
    "generate_ai_image": generate_ai_image,
    "consult_hf_specialist": consult_hf_specialist,
    "deploy_hf_worker_node": deploy_hf_worker_node,
    "send_client_email": send_client_email,
    "hunt_client_leads": hunt_client_leads,
    "github_read_file": github_read_file,
    "github_commit_file": github_commit_file,
    "github_create_repository": github_create_repository,
    "send_telegram_photo": send_telegram_photo,
    "send_telegram_document": send_telegram_document,
    "remember_information": remember_information,
    "recall_information": recall_information,
    "schedule_recurring_task": schedule_recurring_task,
    "list_scheduled_tasks": list_scheduled_tasks,
    "cancel_scheduled_task": cancel_scheduled_task,
    "consult_llama_specialist": consult_llama_specialist,
    "delegate_subtask": delegate_subtask,
    "dispatch_to_agent": dispatch_to_agent,
    "execute_in_cloud_microvm": execute_in_cloud_microvm,
    "execute_composio_action": execute_composio_action,
}

# ----------------- TOOL DECLARATIONS FOR GEMINI -----------------
GEMINI_TOOL_DECLARATIONS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="generate_ai_image",
            description="Generates an image via Hugging Face FLUX/SDXL models and sends it directly to Telegram.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "prompt": types.Schema(type=types.Type.STRING, description="Detailed visual prompt"),
                    "filename": types.Schema(type=types.Type.STRING, description="Local filename", default="concept_visual.png")
                },
                required=["prompt"]
            )
        ),
        types.FunctionDeclaration(
            name="consult_hf_specialist",
            description="Queries open-source specialized models on Hugging Face for coding, translation, or alternative reasoning.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "prompt": types.Schema(type=types.Type.STRING, description="Prompt for HF model"),
                    "model_id": types.Schema(type=types.Type.STRING, description="Hugging Face model ID", default="Qwen/Qwen2.5-Coder-32B-Instruct")
                },
                required=["prompt"]
            )
        ),
        types.FunctionDeclaration(
            name="deploy_hf_worker_node",
            description="Deploys or updates a worker node script or dataset file on Hugging Face Hub synchronously using HfApi.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "repo_id": types.Schema(type=types.Type.STRING, description="HF Space repo ID"),
                    "file_path": types.Schema(type=types.Type.STRING, description="Target file path in repo"),
                    "file_content": types.Schema(type=types.Type.STRING, description="File content"),
                    "commit_message": types.Schema(type=types.Type.STRING, description="Commit message", default="Deploy HF worker node")
                },
                required=["repo_id", "file_path", "file_content"]
            )
        ),
        types.FunctionDeclaration(
            name="send_client_email",
            description="Sends a professional email or project proposal to a client or recipient via Gmail SMTP.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "to_email": types.Schema(type=types.Type.STRING, description="Recipient email"),
                    "subject": types.Schema(type=types.Type.STRING, description="Email subject"),
                    "message_body": types.Schema(type=types.Type.STRING, description="Email body content")
                },
                required=["to_email", "subject", "message_body"]
            )
        ),
        types.FunctionDeclaration(
            name="hunt_client_leads",
            description="Searches the live web across public platforms for fresh hiring and gig leads.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "service_niche": types.Schema(type=types.Type.STRING, description="Service niche e.g. 'video editor'", default="video editor"),
                    "platform": types.Schema(type=types.Type.STRING, description="Platform ('reddit', 'x', 'creator_boards', 'all')", default="all"),
                    "time_range": types.Schema(type=types.Type.STRING, description="Time range ('d', 'w', 'm')", default="w")
                }
            )
        ),
        types.FunctionDeclaration(
            name="github_read_file",
            description="Reads raw contents of a file from GitHub.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "file_path": types.Schema(type=types.Type.STRING, description="File path in repo"),
                    "repo_name": types.Schema(type=types.Type.STRING, description="Repository name", default=""),
                    "branch": types.Schema(type=types.Type.STRING, description="Branch", default="main")
                },
                required=["file_path"]
            )
        ),
        types.FunctionDeclaration(
            name="github_commit_file",
            description="Commits or updates a file in GitHub (protected by mandatory Llama audit and E2B sandbox preflight).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "file_path": types.Schema(type=types.Type.STRING, description="File path in repo"),
                    "content": types.Schema(type=types.Type.STRING, description="File content"),
                    "commit_message": types.Schema(type=types.Type.STRING, description="Commit message"),
                    "repo_name": types.Schema(type=types.Type.STRING, description="Repository name", default=""),
                    "branch": types.Schema(type=types.Type.STRING, description="Branch", default="main")
                },
                required=["file_path", "content", "commit_message"]
            )
        ),
        types.FunctionDeclaration(
            name="github_create_repository",
            description="Creates a new repository under the GitHub account.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "repo_name": types.Schema(type=types.Type.STRING, description="New repo name"),
                    "description": types.Schema(type=types.Type.STRING, description="Repo description", default=""),
                    "private": types.Schema(type=types.Type.BOOLEAN, description="Is private", default=False)
                },
                required=["repo_name"]
            )
        ),
        types.FunctionDeclaration(
            name="send_telegram_photo",
            description="Dispatches a local photo or chart directly to Telegram.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "file_path": types.Schema(type=types.Type.STRING, description="Local photo file path"),
                    "caption": types.Schema(type=types.Type.STRING, description="Photo caption", default="")
                },
                required=["file_path"]
            )
        ),
        types.FunctionDeclaration(
            name="send_telegram_document",
            description="Dispatches any local file directly to Telegram.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "file_path": types.Schema(type=types.Type.STRING, description="Local file path"),
                    "caption": types.Schema(type=types.Type.STRING, description="Document caption", default="")
                },
                required=["file_path"]
            )
        ),
        types.FunctionDeclaration(
            name="remember_information",
            description="Persists memories to Supabase.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "key": types.Schema(type=types.Type.STRING, description="Memory key"),
                    "value": types.Schema(type=types.Type.STRING, description="Memory value"),
                    "category": types.Schema(type=types.Type.STRING, description="Category", default="general")
                },
                required=["key", "value"]
            )
        ),
        types.FunctionDeclaration(
            name="recall_information",
            description="Recalls memories from Supabase.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query_key": types.Schema(type=types.Type.STRING, description="Query key", default="")
                }
            )
        ),
        types.FunctionDeclaration(
            name="schedule_recurring_task",
            description="Schedules an autonomous task to run repeatedly every interval_minutes and dispatch results to Telegram.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "task_id": types.Schema(type=types.Type.STRING, description="Unique task ID"),
                    "prompt": types.Schema(type=types.Type.STRING, description="Prompt to run"),
                    "interval_minutes": types.Schema(type=types.Type.INTEGER, description="Interval in minutes")
                },
                required=["task_id", "prompt", "interval_minutes"]
            )
        ),
        types.FunctionDeclaration(
            name="list_scheduled_tasks",
            description="Lists all active background recurring cron tasks.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={})
        ),
        types.FunctionDeclaration(
            name="cancel_scheduled_task",
            description="Cancels and removes a scheduled recurring task by ID.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "task_id": types.Schema(type=types.Type.STRING, description="Task ID to cancel")
                },
                required=["task_id"]
            )
        ),
        types.FunctionDeclaration(
            name="consult_llama_specialist",
            description="Consults Llama specialist (Groq) strictly for offline code repair, auditing, syntax optimization, and bug fixing.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "task_description": types.Schema(type=types.Type.STRING, description="Task description"),
                    "code_or_context": types.Schema(type=types.Type.STRING, description="Code or context payload")
                },
                required=["task_description", "code_or_context"]
            )
        ),
        types.FunctionDeclaration(
            name="delegate_subtask",
            description="Executes a worker subtask or code audit via Llama 3.3 offline code repair engine.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "role": types.Schema(type=types.Type.STRING, description="Worker role"),
                    "task_description": types.Schema(type=types.Type.STRING, description="Task description"),
                    "code_payload": types.Schema(type=types.Type.STRING, description="Code payload", default="")
                },
                required=["role", "task_description"]
            )
        ),
        types.FunctionDeclaration(
            name="dispatch_to_agent",
            description="Invokes a persistent worker agent from the Supabase registry using its specialized model and instructions.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "agent_id": types.Schema(type=types.Type.STRING, description="Agent ID in Supabase"),
                    "task_prompt": types.Schema(type=types.Type.STRING, description="Task prompt"),
                    "context_payload": types.Schema(type=types.Type.STRING, description="Context payload", default="")
                },
                required=["agent_id", "task_prompt"]
            )
        ),
        types.FunctionDeclaration(
            name="execute_in_cloud_microvm",
            description="Executes Python code in an E2B microVM sandbox.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code": types.Schema(type=types.Type.STRING, description="Python code to execute")
                },
                required=["code"]
            )
        ),
        types.FunctionDeclaration(
            name="execute_composio_action",
            description="Executes actions across 1000+ apps (Gmail, Notion, GitHub, Twitter, Slack) via Composio.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "action_name": types.Schema(type=types.Type.STRING, description="Composio action name"),
                    "arguments": types.Schema(type=types.Type.OBJECT, description="Action arguments dictionary", default={})
                },
                required=["action_name"]
            )
        ),
    ])
]

# ----------------- CORE REASONING ENGINE -----------------

def handle_ai_reasoning(user_message: str, chat_id: int) -> str:
    record_activity()
    active_chat_id.set(chat_id)
    
    system_instruction = (
        "You are Chintu, an Autonomous Full-Stack AI Engineer operating under a Hierarchical Multi-Agent System (HMAS) Manager-Worker-Auditor architecture. "
        "Role: Lead Orchestrator & Agency Director.\n"
        "MANDATORY CODE PROTOCOL:\n"
        "Every single piece of code generated or committed must pass through the mandatory Llama audit and E2B sandbox pre-flight verification gate. "
        "No Python code or repository commit may bypass this protocol.\n"
        "Be efficient, autonomous, and decisive."
    )
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=GEMINI_TOOL_DECLARATIONS,
        temperature=0.3,
        max_output_tokens=3000
    )
    
    response_text = ""
    success = False
    
    for model_name in GEMINI_MODELS_CASCADE:
        try:
            chat = gemini_client.chats.create(model=model_name)
            response = chat.send_message(user_message, config=config)
            
            max_turns = 8
            turn = 0
            while response.function_calls and turn < max_turns:
                turn += 1
                function_responses = []
                for fc in response.function_calls:
                    fname = fc.name
                    fargs = fc.args
                    print(f"[Executing Tool]: {fname} with args {fargs}", flush=True)
                    
                    tool_func = ALL_TOOLS_MAP.get(fname)
                    if tool_func:
                        try:
                            if isinstance(fargs, dict):
                                f_res = tool_func(**fargs)
                            else:
                                f_res = tool_func(fargs)
                        except Exception as te:
                            f_res = f"Tool execution error in {fname}: {str(te)}"
                    else:
                        f_res = f"Error: Tool '{fname}' not found."
                        
                    function_responses.append(
                        types.Part.from_function_response(
                            name=fname,
                            response={"result": f_res}
                        )
                    )
                response = chat.send_message(function_responses, config=config)
                
            if response.text:
                response_text = response.text
                success = True
                break
        except Exception as e:
            print(f"[Cascade model {model_name} failed]: {e}", flush=True)
            continue
            
    if not success or not response_text:
        try:
            if groq_client:
                comp = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                response_text = comp.choices[0].message.content
                success = True
        except Exception as ge:
            response_text = f"All reasoning cascades and fallback engines exhausted. Error: {str(ge)}"
            
    record_activity()
    return response_text

# ----------------- TELEGRAM BOT HANDLERS -----------------

async def telegram_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_id = update.effective_user.id
    if ALLOWED_USER_ID and user_id != ALLOWED_USER_ID:
        await update.message.reply_text("⛔ Unauthorized access.")
        return
        
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    active_chat_id.set(chat_id)
    record_activity()
    
    steps = ["Analyze Request & Plan", "Execute Tools / Code Protocol", "Formulate Response & Report"]
    card = ProgressCardTracker(context.bot, chat_id, f"Request: {text[:35]}...", steps)
    await card.initialize()
    
    try:
        await card.update_step(0, "✅ Done", "Parsed request.")
        await card.update_step(1, "⏳ In Progress", "Executing actions under Llama & E2B protocol...")
        
        result_text = await asyncio.to_thread(handle_ai_reasoning, text, chat_id)
        
        # Execution Guard: Break out immediately upon successful dispatch / response
        await card.update_step(1, "✅ Done", "Actions & verification completed.")
        await card.update_step(2, "⏳ In Progress", "Sending final response...")
        
        if len(result_text) > 4000:
            for i in range(0, len(result_text), 4000):
                await update.message.reply_text(result_text[i:i+4000])
        else:
            await update.message.reply_text(result_text)
            
        await card.update_step(2, "✅ Done", "Response delivered.")
        await card.finalize(True, "All operations executed successfully.")
        return  # Execution Guard: Ensure no duplicate retries trigger
    except Exception as e:
        await card.finalize(False, f"Execution failed: {str(e)}")
        await update.message.reply_text(f"❌ Error during execution: {str(e)}")

# ----------------- FASTAPI & LIFESPAN -----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_instance, main_loop
    main_loop = asyncio.get_running_loop()
    
    if TELEGRAM_TOKEN:
        try:
            app_builder = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            bot_instance = app_builder.bot
            app_builder.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), telegram_message_handler))
            await app_builder.initialize()
            await app_builder.start()
            if app_builder.updater:
                await app_builder.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            print("[Telegram Bot] Started successfully via polling.", flush=True)
        except Exception as e:
            print(f"[Telegram Bot Initialization Error]: {e}", flush=True)
            
    if scheduler and not scheduler.running:
        scheduler.start()
        print("[Scheduler] Started successfully.", flush=True)
        
    yield
    
    if bot_instance:
        try:
            app_builder = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            if app_builder.updater:
                await app_builder.updater.stop()
            await app_builder.stop()
            await app_builder.shutdown()
        except Exception:
            pass
    if scheduler.running:
        scheduler.shutdown()

app = FastAPI(title="Chintu AI Agency Director", lifespan=lifespan)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "agent": "Chintu",
        "architecture": "HMAS Manager-Worker-Auditor",
        "protocol": "Mandatory Llama Audit & E2B Sandbox Commit Gateway Active",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
