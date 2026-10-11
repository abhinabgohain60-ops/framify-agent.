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

# Resilient Speed-to-Power Pure Gemini Free-Tier Cascade
GEMINI_MODELS_CASCADE = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
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

# Autonomous Self-Healing Telemetry Wrapper with True Recursive Retry Loop
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
    CRITICAL INFRASTRUCTURE TOOL: Allows the agent to modify its own source code.
    Use this to patch bugs in main.py, fix the event loop, or add new capabilities.
    This commits directly to the framify-agent repository and triggers a Render restart.
    """
    import urllib.request
    import json
    import base64
    
    repo = "abhinabgohain60-ops/framify-agent."
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Chintu-Auto-Patcher"
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            sha = json.loads(response.read().decode())['sha']
            
        encoded_content = base64.b64encode(new_content.encode('utf-8')).decode('utf-8')
        data = {
            "message": commit_message, 
            "content": encoded_content, 
            "sha": sha, 
            "branch": "main"
        }
        
        put_req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method="PUT")
        with urllib.request.urlopen(put_req) as response:
            return f"SUCCESS: {file_path} patched. Render is now rebooting the server to apply your changes."
            
    except Exception as e:
        return f"FAILED to patch core system: {str(e)}"
        
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
