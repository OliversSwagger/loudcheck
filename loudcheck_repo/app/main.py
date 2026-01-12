from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import os

from app.audio_analysis import analyze_audio
from app.evaluation import evaluate_analysis
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta


app = FastAPI(title="LoudCheck")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Simple in-memory usage tracker
USAGE = {}

MAX_FREE_PER_DAY = 3

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/templates/index.html") as f:
        return f.read()

@app.post("/analyze")
async def analyze(file: UploadFile):
    try:
        user = "demo"  # For now, anonymous demo user
        today = datetime.now().date()

        # Initialize
        if user not in USAGE:
            USAGE[user] = {"date": today, "count": 0}

        # Reset daily usage
        if USAGE[user]["date"] != today:
            USAGE[user] = {"date": today, "count": 0}

        # Check limit
        if USAGE[user]["count"] >= MAX_FREE_PER_DAY:
            return {"status": "error", "message": "Free limit reached. Upgrade for unlimited use."}

        USAGE[user]["count"] += 1

        # Save file and analyze as before
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        result = analyze_audio(file_path)
        evaluation = evaluate_analysis(result)
        targets = {
            "lufs": "-14 ±1 LUFS",
            "true_peak": "≤ -1.0 dBTP",
            "low_end": "≤ 35%"
        }

        return {
            "status": "ok",
            "analysis": result,
            "evaluation": evaluation,
            "targets": targets,
            "remaining_free": MAX_FREE_PER_DAY - USAGE[user]["count"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


