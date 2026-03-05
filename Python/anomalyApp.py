"""
Finfolder Anomaly Detection - Unified API Service
--------------------------------------------------
Combines:
  1. Audio anomaly detection (ONNX model)
  2. Visual image anomaly detection (Cosmos Reason 2)
  3. Visual video anomaly detection (Cosmos Reason 2)

Endpoints:
  POST /detect-anomaly/        - Audio anomaly detection (existing)
  POST /check-visual-image/    - Image anomaly detection via Cosmos
  POST /check-visual-video/    - Video anomaly detection via Cosmos
  GET  /health                 - Health check

Requirements:
    pip install fastapi uvicorn onnxruntime librosa httpx python-dotenv
"""

import io
import os
import base64
import logging
import time
from datetime import datetime

import httpx
import numpy as np
import librosa
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────

load_dotenv()

MODEL_PATH    = os.getenv("MODEL_PATH",    r"C:\Users\samy.timalsina.AUER\source\repos\VS2022AndLaterRepos\repos\Python\AnomalyApi\Model\anomaly_detector.onnx")
MIN_VALS_PATH = os.getenv("MIN_VALS_PATH", r"C:\Users\samy.timalsina.AUER\source\repos\VS2022AndLaterRepos\repos\Python\AnomalyApi\Model\min_vals.npy")
MAX_VALS_PATH = os.getenv("MAX_VALS_PATH", r"C:\Users\samy.timalsina.AUER\source\repos\VS2022AndLaterRepos\repos\Python\AnomalyApi\Model\max_vals.npy")
THRESHOLD     = float(os.getenv("THRESHOLD", "5.155"))
BREV_ENDPOINT = os.getenv("BREV_ENDPOINT", "http://localhost:8000/v1/chat/completions")
MODEL_NAME    = "nvidia/Cosmos-Reason2-8B"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load Audio Model ───────────────────────────────────────────────────────────

session  = ort.InferenceSession(MODEL_PATH)
min_vals = np.load(MIN_VALS_PATH)
max_vals = np.load(MAX_VALS_PATH)
logger.info("✅ Audio ONNX model loaded")

# ── FastAPI App ────────────────────────────────────────────────────────────────

anomalyApp = FastAPI(
    title="Finfolder Anomaly Detection API",
    description="Unified audio + visual anomaly detection for finfolder stamping machine",
    version="2.0.0"
)

# ── Response Model ─────────────────────────────────────────────────────────────

class VisualCheckResponse(BaseModel):
    status:     str
    plc_stop:   bool
    trigger:    str
    keywords:   list[str] = []
    reasoning:  str       = ""
    timestamp:  str       = ""
    latency_ms: int       = 0

# ── Anomaly Keywords ───────────────────────────────────────────────────────────

ANOMALY_KEYWORDS = [
    # Human intrusion
    "hand is visible", "hand visible", "hand appears", "hand detected",
    "finger is visible", "finger visible", "finger appears",
    "arm is visible", "arm visible", "arm appears",
    "person is visible", "person visible", "person appears",
    "human is visible", "human visible", "human appears",
    "body part is visible", "body part visible",
    "operator is visible", "worker is visible",
    "glove is visible", "glove appears", "sleeve is visible",
    # Pattern anomalies
    "flat zone", "flat spot", "irregular pattern",
    "inconsistent pattern", "pattern disrupted", "pattern interrupted",
    "pattern break", "missing pattern", "uneven pattern",
    "material bunching", "material folding", "material crushing",
    "sheet buckling", "jammed", "jam detected",
    "crease detected", "wrinkle detected",
    # General anomalies
    "foreign object detected", "foreign object is visible",
    "obstruction detected", "obstruction is visible",
    "anomaly detected", "defect detected",
    "damage detected", "unusual object",
    "something is blocking", "object is blocking"
]

NEGATION_PHRASES = [
    "no ", "not ", "without ", "none ", "absent",
    "free of", "cannot see", "can't see",
    "don't see", "doesn't show", "no sign of",
    "no visible", "no evidence"
]

# ── Helper Functions ───────────────────────────────────────────────────────────

def scan_reasoning_for_anomaly(reasoning: str) -> tuple[bool, list[str]]:
    reasoning_lower = (reasoning or "").lower()
    triggered = []
    for kw in ANOMALY_KEYWORDS:
        if kw in reasoning_lower:
            idx        = reasoning_lower.index(kw)
            context    = reasoning_lower[max(0, idx - 40):idx]
            is_negated = any(neg in context for neg in NEGATION_PHRASES)
            if not is_negated:
                triggered.append(kw)
    return len(triggered) > 0, triggered


def make_final_decision(cosmos_status: str, reasoning: str) -> dict:
    keyword_anomaly, triggered_keywords = scan_reasoning_for_anomaly(reasoning)
    status_anomaly = "ANOMALY" in cosmos_status.upper()
    if keyword_anomaly:
        return {"final_status": "ANOMALY", "trigger": "REASONING_KEYWORDS", "keywords": triggered_keywords, "plc_stop": True}
    elif status_anomaly:
        return {"final_status": "ANOMALY", "trigger": "COSMOS_STATUS",      "keywords": [],                "plc_stop": True}
    else:
        return {"final_status": "NORMAL",  "trigger": "NONE",               "keywords": [],                "plc_stop": False}


async def call_cosmos(content_payload: list) -> dict:
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": "You are an expert quality control inspector monitoring an industrial finfolder metal stamping machine. Detect any anomalies including human intrusion, pattern defects, or foreign objects. Always think step by step before answering."},
            {"role": "user",   "content": content_payload}
        ]
    }
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(BREV_ENDPOINT, json=payload)
        response.raise_for_status()
    data      = response.json()
    message   = data["choices"][0]["message"]
    reasoning = message.get("reasoning", "")
    content   = message.get("content", "") or ""
    cosmos_status = ""
    for line in content.splitlines():
        if "STATUS:" in line.upper():
            cosmos_status = line.strip()
            break
    return {"reasoning": reasoning, "cosmos_status": cosmos_status}


# ── Prompts ────────────────────────────────────────────────────────────────────

IMAGE_PROMPT = """Carefully examine this image of a finfolder metal stamping machine output sheet.

The metal sheet should show a perfectly uniform continuous corrugated wave pattern.
Look extremely carefully for:
- Any human hand, finger, arm or body part near or touching the sheet
- Any foreign object or obstruction on or near the sheet
- Any area where waves look very irregular, and super inconsistent
- Any material bunching, folding, or crushing

Be  sensitive — even partial or brief anomalies count.
When in doubt, choose ANOMALY.

<think>
Describe every object you see in this image in detail.
Is there any human body part visible anywhere?
Is there any foreign object near the machine or sheet?
Is the corrugated wave pattern perfectly uniform across the entire surface?
Is there anything at all that looks out of place?
</think>

Your DETAILED reasoning here. OK
</think>

STATUS: NORMAL or STATUS: ANOMALY
REASON: [exactly what you see, be specific about location]
CONFIDENCE: HIGH or MEDIUM or LOW"""

VIDEO_PROMPT = """Watch this entire video of a finfolder metal stamping machine output sheet carefully.

The metal sheet should show a perfectly uniform continuous corrugated wave pattern throughout.
Watch extremely carefully for:
- Any human hand, finger, arm or body part appearing near or touching the sheet at ANY moment
- Any foreign object or obstruction appearing on or near the sheet
- Any flat zone where the pattern is missing or interrupted
- Any area where waves look irregular at ANY point in the video
- Any material bunching, folding, or crushing
- Any shadow or object that does not belong near the machine

Be very sensitive — even a brief or partial anomaly at any single moment counts.
When in doubt, choose ANOMALY.

<think>
Watch every frame carefully.
Describe every object you see throughout the video.
Is there any human body part visible at any point?
Is there any foreign object near the machine or sheet at any point?
Is the corrugated wave pattern perfectly uniform throughout?
Is there anything at all that looks out of place at any moment?
</think>

STATUS: NORMAL or STATUS: ANOMALY
REASON: [exactly what you see, at what point in the video]
CONFIDENCE: HIGH or MEDIUM or LOW"""

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@anomalyApp.get("/health")
def health():
    return {"service": "finfolder-anomaly-api", "version": "2.0.0", "status": "ok",
            "endpoints": ["POST /detect-anomaly/", "POST /check-visual-image/", "POST /check-visual-video/"]}


# ── 1. Audio Anomaly Detection (unchanged) ────────────────────────────────────

def extract_features(y, sr=22050):
    """Extracts MFCC and spectral features from an audio signal."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    feature_vector = []
    for mfcc in mfccs:
        feature_vector.extend([np.mean(mfcc), np.std(mfcc)])

    for feat in [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]:
        feature_vector.extend([np.mean(feat), np.std(feat)])

    return np.array(feature_vector)

def compute_energy(audio_signal):
    return np.sum(audio_signal ** 2) / len(audio_signal)

threshold = 0.01  # Set threshold for anomaly detection
@anomalyApp.post("/detect-anomaly/")
async def detect_anomaly(file: UploadFile = File(...)):
    # Read the uploaded WAV file
    wav_data = await file.read()
    y, sr = librosa.load(io.BytesIO(wav_data), sr=22050, mono=True)
    
    # Extract features
    feature_vector = extract_features(y, sr)
    # Compute STE energy
    ste_energy = compute_energy(y)
    # Normalize features
    normalized_feature = 2 * (feature_vector - min_vals) / (max_vals - min_vals + 1e-10) - 1
    input_tensor = normalized_feature.astype(np.float32).reshape(1, -1)

    # Run ONNX inference
    output = session.run(None, {"input": input_tensor})[0]
    reconstruction_error = np.mean((output - input_tensor) ** 2)
  
    # Check if it's an anomaly
    is_anomaly = reconstruction_error > threshold
    # 
   # Log results
    logging.info(f"Reconstruction error: {reconstruction_error} | Anomaly detected: {is_anomaly} | STE Energy: {ste_energy}")
    # Return response with STE energy
    return {
        "reconstruction_error": float(reconstruction_error),
        "anomaly_detected": bool(is_anomaly),
        "ste_energy": float(ste_energy)  # Include STE energy in response
    }

# ── 2. Visual Image Anomaly Detection ─────────────────────────────────────────

@anomalyApp.post("/check-visual-image/", response_model=VisualCheckResponse,
          summary="C# sends captured image, Cosmos checks for anomaly")
async def check_visual_image(file: UploadFile = File(...)):
    """
    C# captures image from Arducam and POSTs it here.
    Sends to Cosmos Reason 2 on Brev GPU.
    If plc_stop=true → C# triggers PLC relay to stop machine.
    Call every 2-5 seconds during machine operation.
    """
    start        = time.time()
    image_data   = await file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    logger.info(f"Visual image — received {len(image_data)/1024:.1f}KB")

    content_payload = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        {"type": "text",      "text": IMAGE_PROMPT}
    ]

    try:
        result = await call_cosmos(content_payload)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach Cosmos. Check SSH tunnel.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Cosmos request timed out.")

    decision   = make_final_decision(result["cosmos_status"], result["reasoning"])
    latency_ms = int((time.time() - start) * 1000)
    logger.info(f"Visual image — status: {decision['final_status']} | latency: {latency_ms}ms {decision}")

    return VisualCheckResponse(
        status     = decision["final_status"],
        plc_stop   = decision["plc_stop"],
        trigger    = decision["trigger"],
        keywords   = decision["keywords"],
        reasoning  = result["reasoning"] or "",
        timestamp  = datetime.utcnow().isoformat(),
        latency_ms = latency_ms
    )


# ── 3. Visual Video Anomaly Detection ─────────────────────────────────────────

@anomalyApp.post("/check-visual-video/", response_model=VisualCheckResponse,
          summary="C# sends captured video clip, Cosmos checks for anomaly")
async def check_visual_video(file: UploadFile = File(...)):
    """
    C# captures short MP4 clip and POSTs it here.
    Sends to Cosmos Reason 2 on Brev GPU.
    Best for audit logging and richer analysis.
    Response takes 20-30 seconds — not for real-time PLC triggering.
    """
    start        = time.time()
    video_data   = await file.read()
    video_base64 = base64.b64encode(video_data).decode("utf-8")
    logger.info(f"Visual video — received {len(video_data)/1024/1024:.1f}MB")

    content_payload = [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}},
        {"type": "text",      "text": VIDEO_PROMPT}
    ]

    try:
        result = await call_cosmos(content_payload)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach Cosmos. Check SSH tunnel.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Cosmos request timed out. Try shorter clip.")

    decision   = make_final_decision(result["cosmos_status"], result["reasoning"])
    latency_ms = int((time.time() - start) * 1000)
    logger.info(f"Visual video — status: {decision['final_status']} | latency: {latency_ms}ms")

    return VisualCheckResponse(
        status     = decision["final_status"],
        plc_stop   = decision["plc_stop"],
        trigger    = decision["trigger"],
        keywords   = decision["keywords"],
        reasoning  = result["reasoning"] or "",
        timestamp  = datetime.utcnow().isoformat(),
        latency_ms = latency_ms
    )