# main.py (Plan-gated /analyze)
from fastapi import FastAPI, UploadFile, Header, File, Form, HTTPException
import os
import uuid
from datetime import datetime
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.audio_analysis import analyze_audio
from app.evaluation import evaluate_analysis
from app.mobile_warnings import generate_mobile_warnings
from app.consultation.automated import generate_consultation
from app.controllers.plans import PLAN_CAPABILITIES
from app.readiness import assess_readiness
from app.batch_summary import build_batch_summary
from typing import List
from app.controllers.analysis_policy import ANALYSIS_EXPOSURE
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# App configuration
# -------------------------------------------------
app = FastAPI(title="LoudCheck AES Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "https://loudcheck.elty.co.za",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------
# Usage limits (educational free tier only)
# -------------------------------------------------
USAGE = {}
MAX_FREE_PER_DAY = 3000

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/templates/index.html") as f:
        return f.read()


@app.post("/analyze")
async def analyze(file: UploadFile, x_clerk_user_id: str = Header(None), x_user_plan: str = Header(None)):
    """
    Plan-gated LoudCheck analysis endpoint.

    Free, Basic, Pro users see only what their plan allows.
    """
    try:
        # -------------------------------------------------
        # Identity & plan resolution
        # -------------------------------------------------
        #user = "demo"  # TODO: replace with Clerk user ID
        #plan = "pro"   # TODO: derive from Clerk / Paystack


        user = x_clerk_user_id
        plan = x_user_plan or "free"


        caps = PLAN_CAPABILITIES.get(plan, PLAN_CAPABILITIES["free"])
        policy = ANALYSIS_EXPOSURE.get(plan, ANALYSIS_EXPOSURE["free"])
        
        # Safety check: policy must be valid
        assert policy == "ALL" or isinstance(policy, list), "Invalid analysis exposure policy"

        today = datetime.now().date()

        # -------------------------------------------------
        # Usage tracking (free tier only)
        # -------------------------------------------------
        if user not in USAGE or USAGE[user].get("date") != today:
            USAGE[user] = {"date": today, "count": 0}

        if plan == "free" and USAGE[user]["count"] >= MAX_FREE_PER_DAY:
            return {
                "status": "error",
                "message": "Free validation limit reached. Upgrade to continue."
            }

        USAGE[user]["count"] += 1

        # -------------------------------------------------
        # File handling
        # -------------------------------------------------
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # -------------------------------------------------
        # Core deterministic analysis (plan-agnostic)
        # -------------------------------------------------
        analysis = analyze_audio(file_path)
        evaluation = evaluate_analysis(analysis)
        readiness = assess_readiness(analysis, evaluation) if caps.get("readiness_verdict") else None

        # -------------------------------------------------
        # Build plan-gated response
        # -------------------------------------------------
        # Minimal analysis for free users
        # response_analysis = {}
        # if caps["distribution_simulation"]:
        #     response_analysis = analysis
        # else:
        #     response_analysis = {
        #         "integrated_lufs": analysis.get("integrated_lufs"),
        #         "true_peak_db": analysis.get("true_peak_db"),
        #         "plr": analysis.get("plr"),
        #         "content_type": analysis.get("content_type"),
        #     }
        # -------------------------------------------------
        # Build plan-gated analysis response (POLICY-DRIVEN)
        # -------------------------------------------------
        if policy == "ALL":
            response_analysis = analysis
        else:
            response_analysis = {
                key: analysis.get(key)
                for key in policy
                if key in analysis
            }


        # Evaluation/confidence
        response_evaluation = evaluation if caps["compliance_language"] else {}

        # Mobile warnings
        # mobile_warnings = generate_mobile_warnings(analysis) if caps["mobile_warnings"] else None
        # Mobile warnings (ALWAYS return a list)
        mobile_warnings = (
            generate_mobile_warnings(analysis)
            if caps.get("mobile_warnings")
            else []
        )

        # Risk warnings
        warnings = []
        if caps["risk_flags"]:
            if analysis["distribution_simulation"]["music_streaming"]["will_limit"]:
                warnings.append(
                    "Upward normalization is likely to trigger playback limiters, risking transient distortion."
                )
            if analysis["true_peak_db"] > -1.0:
                warnings.append(
                    "True peak exceeds −1 dBTP. Inter-sample peaks may distort on consumer playback systems."
                )

        # Export (Pro only)
        response_export = {
            "json": analysis,
            "evaluation": evaluation,
            "schema": "loudcheck-aes-1.0"
        } if caps["export"] else None

        # Consultation (Pro only)
        response_consultation = {
            "type": "automated",
            "actions": generate_consultation(analysis)
        } if caps["consultation"] else None

        # -------------------------------------------------
        # Compose response
        # -------------------------------------------------
        # response = {
        #     "status": "ok",
        #     "plan": plan,
        #     "analysis": response_analysis,
        #     "evaluation": response_evaluation,
        #     "warnings": warnings,
        #     "mobile_warnings": mobile_warnings,
        #     "export": response_export,
        #     "consultation": response_consultation,
        #     "readiness": readiness,
        #     "remaining_free": (
        #         MAX_FREE_PER_DAY - USAGE[user]["count"]
        #         if plan == "free"
        #         else None
        #     )
        # }

        response = {
            "status": "ok",
            "plan": plan,
            "analysis": response_analysis,
            "evaluation": response_evaluation or {},
            "warnings": warnings or [],
            "mobile_warnings": mobile_warnings or [],
            "export": response_export or {},
            "consultation": response_consultation or {},
            "readiness": readiness,
            "remaining_free": (
                MAX_FREE_PER_DAY - USAGE[user]["count"]
                if plan == "free"
                else None
            )
        }


        if readiness:
            response["readiness"] = readiness

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
"""
from fastapi import Request

@app.post("/analyze/batch")
async def analyze_batch(request: Request):
    print("Headers:", request.headers)
    form = await request.form()
    print("Form keys:", form.keys())
    for key in form.keys():
        print(key, type(form[key]))
    return {"status": "debug"}
"""

@app.post("/analyze/batch")
async def analyze_batch(files: List[UploadFile] = File(...), clerk_user_id: str = Form(...), plan: str = Form(...)):
    # , x_clerk_user_id: str = Header(None), x_user_plan: str = Header(None)
    results = []
    print("FILES RECEIVED COUNT:", len(files))



    user = clerk_user_id
    plan = plan

    if plan != "pro":
        return {
            "status": "error",
            "message": "Batch analysis is only available for Pro users"
        }
    
    print(f"This is the plan bro: {plan}")
    caps = PLAN_CAPABILITIES.get(plan, PLAN_CAPABILITIES["free"])
    policy = ANALYSIS_EXPOSURE.get(plan, ANALYSIS_EXPOSURE["free"])
    
    assert policy == "ALL" or isinstance(policy, list)


    for file in files:
        # Save uploaded file to disk
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f_out:
            f_out.write(await file.read())

        # Run analysis on saved file path
        analysis = analyze_audio(file_path)
        evaluation = evaluate_analysis(analysis)
        #readiness = assess_readiness(analysis, evaluation)
        readiness = (
            assess_readiness(analysis, evaluation)
            if caps.get("readiness_verdict")
            else None
        )

        consultation = (
            {"type": "automated", "actions": generate_consultation(analysis)}
            if caps.get("consultation")
            else {}
        )



        # results.append({
        #     "filename": file.filename,
        #     "analysis": analysis,
        #     "evaluation": evaluation,
        #     "readiness": readiness
        # })

        results.append({
            "filename": file.filename,
            "analysis": analysis if policy == "ALL" else {
                k: analysis.get(k) for k in policy if k in analysis
            },
            "evaluation": evaluation if caps.get("compliance_language") else {},
            "readiness": readiness,
            "consultation": consultation, 
        })


    batch_summary = build_batch_summary(results)
    
    return {
        "tracks": results,
        "batch_summary": batch_summary
    }



"""
@app.post("/analyze/batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
):
    print("FILES RECEIVED:", files, flush=True)
    return {
        "count": len(files),
        "names": [f.filename for f in files]
    }
"""
