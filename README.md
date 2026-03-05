# 🏭 Finfolder Stamping Machine — AI-Powered Safety System
### NVIDIA Cosmos Cookoff 2026 Submission

> **A dual-layer industrial safety system combining audio anomaly detection and NVIDIA Cosmos Reason 2 visual intelligence to protect workers and prevent machine damage — even during after-hours unattended operation.**

---

## 🎥 Demo Video
[Watch the full demo video here — https://1drv.ms/v/c/d73d578baf6e637c/IQDQpNMFA3pgQ6lPHwq9jpT5AV5ZpO84KE5jGaQiqT5u4kQ?e=aCitaI]
https://1drv.ms/v/c/d73d578baf6e637c/IQDQpNMFA3pgQ6lPHwq9jpT5AV5ZpO84KE5jGaQiqT5u4kQ?e=aCitaI
---

## 🏆 The Problem

The finfolder stamping machine is a legacy industrial machine that stamps corrugated wave patterns into thin metal foil sheets. It has **no built-in safety system** to detect jams or human intrusion. When something goes wrong, the machine continues running — potentially:

- 💀 **Injuring workers** whose hands enter the machine undetected
- 🔧 **Destroying expensive tooling** from undetected material jams
- 💸 **Halting production** from undetected pattern defects

Customer demand requires the machine to run **after hours with no operator present.** A human operator traditionally stands watch — listening and watching for anything unusual.

**We replaced that human operator with AI.**


## 🔧 Project Background

This project builds on a production system already deployed 
and running in an active manufacturing facility:

**Phase 1 — Already in Production (completed several months ago):**
- Electrically retrofitted a legacy out-of-support finfolder 
  stamping machine with a custom relay and Allen-Bradley PLC
- Designed and programmed the PLC logic from scratch to enable 
  emergency machine stop via software trigger
- Built audio anomaly detection using a custom-trained ONNX 
  autoencoder model listening to machine sounds
- Deployed C# Windows Forms control application used daily 
  in production

**Phase 2 — Added for Cosmos Cookoff (completed in 2 days):**
- Integrated NVIDIA Cosmos Reason 2 as a second visual layer
- Added /check-visual-image/ endpoint to existing FastAPI service
- Cosmos watches the output sheet and detects pattern anomalies 
  AND human intrusion in real time
- Reused existing PLC relay infrastructure — no hardware changes needed

**The result:** A battle-tested production safety system upgraded 
with state-of-the-art AI vision in just 2 days — because the 
foundation was already solid.

---

## 💡 The Solution — Virtual AI Operator

A two-layer safety system that mimics exactly how a human operator monitors the machine:

```
Human Operator                    AI Safety System
──────────────────                ──────────────────────────────
Listens for unusual sounds   →    Audio ONNX model
Watches output sheet pattern →    Cosmos Reason 2 visual check
Recognizes hand near machine →    Cosmos Reason 2 human intrusion
Pulls emergency stop         →    PLC relay triggered automatically
```

**Either layer detecting an anomaly stops the machine immediately.**

### Key Innovation — Unexpected Safety Discovery
During development we discovered that Cosmos Reason 2 naturally detects **human intrusion** — hands, arms, and body parts near the machine — even when the machine sounds perfectly normal. This turned our quality inspection tool into a **worker safety system** as well. The audio layer alone could never catch this.

---

## 🤖 How NVIDIA Cosmos Reason 2 Is Used

Cosmos Reason 2 (8B) acts as a virtual quality control inspector. It receives a camera frame and reasons through what it sees using physical world understanding:

### Example — Human Intrusion Detected
```
Cosmos Reasoning:
"I can see a human hand entering from the left side of the frame,
partially obscuring the corrugated wave pattern. The presence of
a human body part near the active stamping dies represents an
immediate safety hazard. The machine should be stopped immediately."

→ STATUS: ANOMALY
→ TRIGGER: REASONING_KEYWORDS — "hand is visible"
→ PLC STOP TRIGGERED ⚡
```

### Example — Normal Operation
```
Cosmos Reasoning:
"The corrugated wave pattern is uniform and consistent throughout
the visible area. Waves are evenly spaced from left to right and
top to bottom. No foreign objects, obstructions, or human body
parts are visible. Machine appears to be operating correctly."

→ STATUS: NORMAL
→ Machine continues running ✅
```

### Why Cosmos Reason 2 — Not a Traditional CV Model
| Approach | Why We Chose / Rejected |
|----------|------------------------|
| Traditional CV (YOLO etc.) | Needs labeled anomaly training data — we have almost none |
| Post-training Cosmos | Insufficient anomaly footage for reliable fine-tuning |
| **Zero-shot Cosmos Reason 2** | ✅ Physical world understanding out of the box |
| **Chain-of-thought reasoning** | ✅ Explainable decisions — operators know WHY machine stopped |

---

## ⚡ Demo Mode vs Production Mode

A key engineering decision in this system is balancing **AI reasoning richness** against **response speed**:

### Demo Mode (this submission)
```
Check interval:  Every 5 seconds
Reasoning:       Full chain-of-thought visible
Response time:   15-30 seconds
Purpose:         Shows AI capability to judges and operators
                 Creates explainable audit trail
```

### Production Mode (roadmap)
```
Check interval:  Every 500ms
Reasoning:       Disabled for speed
Response time:   1.5-2 seconds
Purpose:         Real-time safety trigger
                 Runs alongside audio check simultaneously
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Production Machine                     │
│   Arducam OV9281 Camera    SM57 dynamic microphone       │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│              C# Windows Forms Application                │
│                                                          │
│  On startup → runs start_fastanomaly_api.bat             │
│  └── launches anomalyApp.py via uvicorn on port 105     │
│                                                          │
│  VisualMonitorService.cs                                 │
│  ├── Captures frame from Arducam                         │
│  ├── POST /check-visual-image/ → anomalyApp.py           │
│  └── plc_stop=true → PLC relay → machine stop            │
│                                                          │
│  AudioMonitorService.cs (existing)                       │
│  ├── Captures audio chunk                                │
│  ├── POST /detect-anomaly/ → anomalyApp.py               │
│  └── anomaly_detected=true → PLC relay → machine stop    │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│         anomalyApp.py — FastAPI (localhost:105)         │
│  ├── POST /detect-anomaly/       Audio ONNX model        │
│  ├── POST /check-visual-image/   Cosmos Reason 2         │
│  └── POST /check-visual-video/   Cosmos Reason 2         │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────┐
│  Brev GPU (A100 40GB)    │
│  vLLM + Cosmos R2 8B     │
│  SSH Tunnel Port 8000    │
└──────────────────────────┘
```

---

## 🔍 Anomaly Detection Logic

The visual service uses a **dual-layer detection strategy** with negation-aware keyword parsing:

```python
Primary   → Keyword scan on Cosmos reasoning text
             Catches anomalies even when STATUS field misses them
             Negation-aware: "no hand visible" does NOT trigger
             "hand is visible" DOES trigger

Secondary → Cosmos STATUS field
             Direct ANOMALY verdict from the model
```

### Why Negation-Aware Parsing?
Without negation checking, phrases like *"No human body parts are present"* would falsely trigger on the word "human". Our parser checks 40 characters before each keyword for negation phrases like "no", "not", "without", "free of" — preventing false positives on normal operation.

---

## 📁 Repository Structure

```
cosmos-finfolder-safety/
├── README.md
├── Python/
│   ├── anomalyApp.py                ← Main FastAPI service (audio + visual)
│   ├── start_fastanomaly_api.bat    ← Bat file launched by C# app on startup
│   ├── test_cosmos_image.py         ← Image test script
│   ├── test_cosmos_video.py         ← Video test script
│   ├── chunk_video.py               ← Video chunking utility
│   ├── requirements.txt             ← Python dependencies
│   └── .env.template                ← Environment config template
├── CSharp/
│   └── VisualMonitorService.cs      ← Service layer between API and WinForms
└── samples/
    ├── normal_run.mp4               ← Normal machine operation
    ├── hand_detected.mp4            ← Human intrusion sample
    └── paper_thrown.mp4             ← Foreign object anomaly sample
```

---

## 🚀 Deployment Instructions

### Prerequisites
- Python 3.10+
- Brev.dev account (or any cloud GPU with A100/H100 40GB+ VRAM)
- HuggingFace account with access to `nvidia/Cosmos-Reason2-8B`
- Accept model license at: https://huggingface.co/nvidia/Cosmos-Reason2-8B

---

### Step 1 — Deploy Cosmos Reason 2 on Brev GPU

```bash
# 1. Create A100 40GB instance at brev.nvidia.com (~$1.25/hr)
# 2. Open Jupyter terminal and run:

pip install vllm
huggingface-cli login  # token needs "Access public gated repositories"

vllm serve nvidia/Cosmos-Reason2-8B \
  --max-model-len 16384 \
  --reasoning-parser qwen3 \
  --port 8000 \
  --host 0.0.0.0

# Wait for: "Application startup complete"
# First run downloads ~16GB model — takes 15-20 minutes
```

---

### Step 2 — SSH Tunnel (Local → Brev)

```bash
# On your local machine:
ssh -L 8000:localhost:8000 shadeform@<your-brev-ip>

# Verify connection:
curl http://localhost:8000/v1/models
# Should return: nvidia/Cosmos-Reason2-8B
```

---

### Step 3 — Python Service Setup

```bash
pip install -r Python/requirements.txt

cp Python/.env.template Python/.env
# Edit .env:
# BREV_ENDPOINT=http://localhost:8000/v1/chat/completions
# CAMERA_INDEX=0

# Start service manually (or let C# app launch it via bat file):
uvicorn anomalyApp:app --host 0.0.0.0 --port 5001
```

---

### Step 4 — Test With Sample Videos

```bash
cd Python

# Update VIDEO_PATH in script to point to samples folder
py test_cosmos_video.py
# Expected: STATUS: NORMAL for normal_run.mp4
# Expected: STATUS: ANOMALY for hand_detected.mp4 and paper_thrown.mp4

# Test image endpoint
py test_cosmos_image.py
```

---

## 📡 API Reference

### POST /check-visual-image/
```json
// Request: multipart/form-data, field "file" (JPEG)
// Response:
{
  "status": "ANOMALY",
  "plc_stop": true,
  "trigger": "REASONING_KEYWORDS",
  "keywords": ["hand is visible"],
  "reasoning": "I can see a human hand entering from the left...",
  "timestamp": "2026-03-05T18:30:00",
  "latency_ms": 4200
}
```

### POST /detect-anomaly/
```json
// Request: multipart/form-data, field "file" (WAV)
// Response:
{
  "reconstruction_error": 3.24,
  "anomaly_detected": false,
  "ste_energy": 0.000412
}
```

### POST /check-visual-video/
Same response schema as `/check-visual-image/` — accepts MP4 file.

### GET /health
Returns service status and available endpoints.

---

## 📊 Results

| Scenario | Audio | Visual | Machine Stops |
|----------|-------|--------|---------------|
| Normal operation | ✅ NORMAL | ✅ NORMAL | ❌ No |
| Machine jam | 🚨 ANOMALY | 🚨 ANOMALY | ✅ Yes |
| Hand intrusion | ✅ NORMAL | 🚨 ANOMALY | ✅ Yes |
| Paper/foreign object | ✅ NORMAL | 🚨 ANOMALY | ✅ Yes |
| After-hours operation | ✅ Monitoring | ✅ Monitoring | On detection |

---

## 💰 Cost Management

The Cosmos Reason 2 model was deployed on a Brev.dev A100 40GB GPU instance at **$1.25/hr**. After successful testing and demo recording, the GPU instance was **terminated immediately** to avoid unnecessary costs.

```
Total GPU usage for this project: ~5 hours
Total cost: ~$6.25
```

To reproduce: spin up a new Brev instance, run the vLLM startup command above, and the system is fully operational in ~20 minutes.

---

## 🔮 Future Roadmap

### Phase 2 — Performance
- **500ms polling** with fast mode for real-time PLC safety response
- **Edge deployment** on NVIDIA Jetson to eliminate cloud GPU dependency
- **Multi-camera** coverage for full machine perimeter monitoring

### Phase 3 — Intelligence
- **Post-training** on collected anomaly footage as labeled dataset grows
- **Few-shot prompting** with reference frames for higher pattern sensitivity
- **Confidence thresholds** — HIGH confidence required to stop, LOW triggers alert only

### Phase 4 — Operations
- **SMS/email alerts** for remote operator notification after hours
- **Anomaly logging dashboard** with full Cosmos reasoning history
- **Predictive maintenance** — detect gradual pattern degradation before jam occurs

---

## 🛠️ Hardware

| Component | Specification |
|-----------|---------------|
| Camera | Arducam OV9281 100fps Mono Global Shutter USB |
| Microphone | Shure SM57 Dynamic Microphone |
| Audio Interface | Behringer UMC404HD USB Audio Interface |
| Cloud GPU | NVIDIA A100 40GB via Brev.dev |
| AI Model | NVIDIA Cosmos Reason 2 8B |
| PLC Interface | Allen-Bradley Control Logix PLC + custom relay retrofit|

---

## 📦 Requirements

```
fastapi
uvicorn
onnxruntime
librosa
httpx
python-dotenv
python-multipart
pydantic
opencv-python
numpy
```

---

## ⚠️ Notes

- Keep SSH tunnel open while service is running
- Stop Brev instance when not in use — billed per hour
- Audio ONNX model paths must be updated in `.env` for your environment
- The production C# Windows Forms application is deployed in an active manufacturing environment and cannot be shared publicly. `VisualMonitorService.cs` demonstrates the API integration pattern used between the WinForms UI and the Python FastAPI service. The demo video shows the complete production system in operation.
- Camera index `0` assumes Arducam is the first USB camera — update `CAMERA_INDEX` in `.env` if needed

---

## 👤 Author

**Samy Timalsina**
NVIDIA Cosmos Cookoff 2026

---

## 📄 License

MIT License

---

*Built for the NVIDIA Cosmos Cookoff 2026 — Physical AI Challenge*
*Powered by NVIDIA Cosmos Reason 2 — Real deployment, real machine, real impact.*
