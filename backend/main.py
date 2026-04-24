"""
NeuroPath — main.py
====================
FastAPI backend server.
Replaces Gradio's app.launch() with proper REST API endpoints.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import datetime
import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from cnn_engine import run_cnn_analysis, correlate_symptoms, is_valid_brain_mri
from gemini_engine import chat_with_triage, extract_symptoms_from_history, generate_report

# ===============================================================================
# App Setup
# ===============================================================================

app = FastAPI(
    title="NeuroPath AI Backend",
    description="REST API for NeuroPath — Symptom Triage + Brain MRI Analysis",
    version="1.0.0",
)

# Allow React dev server (localhost:5173) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================================================================
# In-memory session state
# Stores the latest analysis results so /report can use them
# (for a group demo with one user at a time, this is sufficient)
# ===============================================================================

_session = {
    "pred_class":     None,
    "confidence":     None,
    "all_probs":      None,
    "symptom_text":   "",
    "triage_summary": "",
    "correlation":    "",
}


# ===============================================================================
# Request / Response Models
# ===============================================================================

class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []

class ChatResponse(BaseModel):
    reply: str
    triage_summary: str
    history: list[ChatMessage]

class ReportResponse(BaseModel):
    report: str
    generated_at: str


# ===============================================================================
# ENDPOINTS
# ===============================================================================

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "NeuroPath AI Backend"}


# ── Step 1 — Triage Chat ──────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Accepts a user message + full conversation history.
    Returns the assistant reply, updated history, and triage summary if generated.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Convert Pydantic models → plain dicts for gemini_engine
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    result = chat_with_triage(req.message, history_dicts)

    # Persist symptom text + triage summary in session
    _session["symptom_text"]   = extract_symptoms_from_history(result["history"])
    if result["triage_summary"]:
        _session["triage_summary"] = result["triage_summary"]

    return ChatResponse(
        reply=result["reply"],
        triage_summary=result["triage_summary"],
        history=[ChatMessage(**m) for m in result["history"]],
    )


# ── Step 2 — MRI Analysis ─────────────────────────────────────────────────────
@app.post("/api/analyse-mri")
async def analyse_mri_endpoint(
    image: UploadFile = File(...),
):
    """
    Accepts a brain MRI image (JPG / PNG).
    Step 1: Gatekeeper model validates the image is a brain MRI.
    Step 2: Xception CNN classifies + Grad-CAM explains.
    """
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are accepted."
        )

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # ── GATEKEEPER CHECK ──────────────────────────────────────
    is_brain, gate_confidence = is_valid_brain_mri(pil_image)

    if not is_brain:
        raise HTTPException(
            status_code=422,
            detail={
                "type":        "invalid_image",
                "title":       "Not a Brain MRI",
                "message":     (
                    "The uploaded image does not appear to be a brain MRI scan. "
                    "Please upload a valid T1 or T2-weighted axial brain MRI (JPG or PNG)."
                ),
                "suggestions": [
                    "Make sure the image is a brain MRI, not a photo or other scan type.",
                    "Use axial (top-down) T1 or T2-weighted MRI slices.",
                    "Chest, spine, or knee MRIs are not accepted — brain only.",
                    "Screenshots or photos of MRI reports are not valid — upload the scan image itself.",
                ],
                "gate_confidence": gate_confidence,
            }
        )
    # ── END GATEKEEPER ────────────────────────────────────────

    symptom_text = _session.get("symptom_text", "")
    result       = run_cnn_analysis(pil_image, symptom_text)

    # Persist MRI results in session for report generation
    _session["pred_class"]  = result["pred_class"]
    _session["confidence"]  = result["confidence"]
    _session["all_probs"]   = result["probabilities"]
    _session["correlation"] = result["correlation"]

    return {
        "pred_class":    result["pred_class"],
        "pred_label":    result["pred_label"],
        "confidence":    result["confidence"],
        "description":   result["description"],
        "risk_flag":     result["risk_flag"],
        "probabilities": result["probabilities"],
        "gradcam_b64":   result["gradcam_b64"],
        "heatmap_b64":   result["heatmap_b64"],
        "correlation":   result["correlation"],
    }


# ── Step 3 — Full Report ──────────────────────────────────────────────────────
@app.post("/api/report", response_model=ReportResponse)
def report_endpoint():
    """
    Combines session triage data + MRI results into a Gemini-written report.
    Both /api/chat and /api/analyse-mri must have been called first.
    """
    if _session["pred_class"] is None:
        raise HTTPException(
            status_code=400,
            detail="MRI analysis not completed. Please run Step 2 first."
        )

    correlation = correlate_symptoms(
        _session["symptom_text"],
        _session["pred_class"],
    )

    report_text = generate_report(
        pred_class=_session["pred_class"],
        confidence=_session["confidence"],
        all_probs=_session["all_probs"],
        symptom_summary=_session["symptom_text"],
        triage_summary=_session["triage_summary"],
        correlation_text=correlation,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    full_report = (
        f"NEUROPATH INTEGRATED DIAGNOSTIC REPORT\n"
        f"Generated : {timestamp}\n"
        f"{'=' * 55}\n\n"
        f"{report_text}\n\n"
        f"{'=' * 55}\n"
        f"Powered by NeuroPath AI  |  For screening purposes only\n"
        f"Must be reviewed by a licensed medical professional."
    )

    return ReportResponse(report=full_report, generated_at=timestamp)


# ── Clear Session ─────────────────────────────────────────────────────────────
@app.post("/api/clear")
def clear_session():
    """Resets the in-memory session (clears chat + MRI results)."""
    _session.update({
        "pred_class":     None,
        "confidence":     None,
        "all_probs":      None,
        "symptom_text":   "",
        "triage_summary": "",
        "correlation":    "",
    })
    return {"status": "session cleared"}