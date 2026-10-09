import os
import asyncio
import subprocess
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

# Use the exact model requested by Google's API for your account
MODEL_NAME = "gemini-3.8-flash"

def run_terminal_command(command: str) -> str:
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = res.stdout if res.stdout else res.stderr
        return out[:3000] if out else "Command executed with no output."
    except Exception as e:
        return f"Execution error: {str(e)}"

def write_project_file(file_path: str, content: str) -> str:
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File '{file_path}' written successfully."

tools = [run_terminal_command, write_project_file]

def run_agent(prompt: str) -> str:
    try:
        chat = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an autonomous engineering agent with full bash terminal execution and file writing tools. "
                    "You have complete freedom to write custom scripts, install dependencies with pip, run code, "
                    "and build your own tools to accomplish user objectives. Return clean results."
                ),
                tools=tools,
                temperature=0.2
            )
        )
        return chat.send_message(prompt).text
    except Exception as e:
        return f"Agent error: {str(e)}"

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    if ALLOWED_USER_ID != 0 and update.effective_user.id != ALLOWED_USER_ID:
        await update.message.reply_text("Unauthorized access.")
        return

    task = update.message.text
    status = await update.message.reply_text(f"Task received:\n'{task[:60]}...'\nProcessing on {MODEL_NAME}...")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_agent, task)
    await status.edit_text(result[:4000])

async def run_telegram_worker():
    # Delay startup slightly so Render's previous container terminates cleanly
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
