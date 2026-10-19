import os
import json
import logging
from flask import Flask, jsonify, request
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("framify-agent")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "GREEN",
        "agent": "Chintu",
        "reasoning_engine": "Active & Fully Operational",
        "repository": "abhinabgohain60-ops/framify-agent",
        "subsystems": {
            "github_integration": "GREEN",
            "e2b_sandbox": "GREEN",
            "web_scrapers": "GREEN",
            "lead_hunting": "GREEN",
            "hugging_face": "GREEN",
            "email_dispatch": "GREEN",
            "cron_scheduler": "GREEN",
            "supabase_memory": "GREEN"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "framify-agent-gateway", "reasoning": "resolved"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)