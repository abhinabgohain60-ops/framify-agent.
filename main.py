import os
import threading
import asyncio
import subprocess
from fastapi import FastAPI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types

# Secure environment variable lookup (Render injects these safely)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "8513926902"))

client = genai.Client(api_key=GEMINI_API_KEY)

def run_terminal_command(command: str) -> str:
    """Executes bash commands, tests scripts, and runs system tools."""
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = res.stdout if res.stdout else res.stderr
        return out[:3000] if out else "Executed successfully with no output."
    except Exception as e:
        return f"Execution error: {str(e)}"

def write_project_file(file_path: str, content: str) -> str:
    """Writes code, creates automation scripts, or builds new tools dynamically."""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File '{file_path}' written successfully."

tools = [run_terminal_command, write_project_file]

def run_agent(prompt: str) -> str:
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an autonomous engineering agent with full bash terminal execution and file writing tools. "
                "You have complete freedom to write custom scripts, install dependencies with pip, run code, "
                "and build your own tools to accomplish user objectives. Always return clear final results."
            ),
            tools=tools,
            temperature=0.2
        )
    )
    return chat.send_message(prompt).text

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_USER_ID != 0 and update.effective_user.id != ALLOWED_USER_ID:
        await update.message.reply_text("Unauthorized access.")
        return
    
    task = update.message.text
    status = await update.message.reply_text(f"Task received:\n'{task[:60]}...'\nExecuting in cloud sandbox...")
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_agent, task)
    await status.edit_text(result[:4000])

def start_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_msg))
    app.run_polling()

if TELEGRAM_TOKEN:
    t = threading.Thread(target=start_bot, daemon=True)
    t.start()

api = FastAPI()

@api.get("/")
def home():
    return {"status": "Agent Online"}
