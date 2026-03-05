"""
Finfolder Visual Anomaly Detection - Video Test with Reasoning-Based Detection
------------------------------------------------------------------------------
Tests a video clip against Cosmos Reason 2.
Uses keyword scanning on reasoning field for reliable anomaly detection.

Requirements:
    pip install httpx

Usage:
    py test_cosmos_video.py
"""

import httpx
import base64
import asyncio

# ── Config ─────────────────────────────────────────────────────────────────────

VIDEO_PATH      = r"C:\Users\samy.timalsina.AUER\Pictures\Camera Roll\chunk2\chunk_010.mp4"
COSMOS_ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL_NAME      = "nvidia/Cosmos-Reason2-8B"

# ── Anomaly Keywords ───────────────────────────────────────────────────────────
# If ANY of these appear in Cosmos reasoning → trigger ANOMALY
# Add more keywords as you discover them from testing

ANOMALY_KEYWORDS = [
    # Human intrusion — only positive detections
    "hand is visible", "hand visible", "hand appears",
    "hand detected", "finger is visible", "arm is visible",
    "person is visible", "person appears", "human is visible",
    "body part is visible", "operator is visible",
    "glove is visible", "glove appears",

    # Pattern anomalies
    "flat zone", "flat spot", "irregular pattern",
    "inconsistent pattern", "pattern disrupted", "pattern interrupted",
    "pattern break", "missing pattern", "uneven pattern",
    "material bunching", "material folding", "material crushing",
    "sheet buckling", "jammed", "jam detected",

    # General anomalies
    "foreign object detected", "foreign object is visible",
    "obstruction detected", "obstruction is visible",
    "anomaly detected", "defect detected",
    "damage detected", "unusual object",
    "something is blocking", "object is blocking"
]

# ── Keyword Scanner ────────────────────────────────────────────────────────────

def scan_reasoning_for_anomaly(reasoning: str) -> tuple[bool, list[str]]:
    reasoning_lower = (reasoning or "").lower()
    triggered = [kw for kw in ANOMALY_KEYWORDS if kw in reasoning_lower]
    return len(triggered) > 0, triggered


def make_final_decision(cosmos_status: str, reasoning: str) -> dict:
    keyword_anomaly, triggered_keywords = scan_reasoning_for_anomaly(reasoning)
    status_anomaly = "ANOMALY" in cosmos_status.upper()

    if keyword_anomaly:
        return {
            "final_status": "ANOMALY",
            "trigger":      "REASONING_KEYWORDS",
            "keywords":     triggered_keywords,
            "plc_stop":     True
        }
    elif status_anomaly:
        return {
            "final_status": "ANOMALY",
            "trigger":      "COSMOS_STATUS",
            "keywords":     [],
            "plc_stop":     True
        }
    else:
        return {
            "final_status": "NORMAL",
            "trigger":      "NONE",
            "keywords":     [],
            "plc_stop":     False
        }

# ── Main ───────────────────────────────────────────────────────────────────────

async def test_video():
    print(f"\n{'='*60}")
    print(f"VIDEO TEST — Cosmos Reason 2 + Reasoning Detection")
    print(f"{'='*60}")
    print(f"Video: {VIDEO_PATH}")

    try:
        with open(VIDEO_PATH, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode()
        print("✅ Video loaded")
    except FileNotFoundError:
        print(f"❌ Video not found: {VIDEO_PATH}")
        return

    print("⏳ Sending to Cosmos Reason 2... (30-60 seconds for video)")

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                COSMOS_ENDPOINT,
                json={
                    "model": MODEL_NAME,
                    "max_tokens": 4096,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert quality control inspector monitoring an industrial finfolder metal stamping machine. Your job is to watch the video and detect any anomalies including human intrusion, pattern defects, or foreign objects."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": f"data:video/mp4;base64,{video_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": """Watch this entire video of a finfolder metal stamping machine output sheet carefully.

The metal sheet should show a perfectly uniform continuous corrugated wave pattern throughout.
Watch extremely carefully for:
- Any human hand, finger, arm or body part appearing near or touching the sheet at ANY moment
- Any foreign object or obstruction appearing on or near the sheet
- Any flat zone where the pattern is missing or interrupted
- Any area where waves look irregular, inconsistent, or different at ANY point
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
                                }
                            ]
                        }
                    ]
                }
            )
        response.raise_for_status()

    except httpx.ConnectError:
        print("❌ Cannot connect! Make sure SSH tunnel is running:")
        print("   ssh -L 8000:localhost:8000 shadeform@130.250.171.40")
        return
    except httpx.TimeoutException:
        print("❌ Request timed out - try a shorter video clip")
        return

    result    = response.json()
    message   = result["choices"][0]["message"]
    reasoning = message.get("reasoning", "")
    content   = message.get("content", "")

    cosmos_status = ""
    for line in (content or "").splitlines():
        if "STATUS:" in line.upper():
            cosmos_status = line.strip()
            break

    decision = make_final_decision(cosmos_status, reasoning)

    print(f"\n{'='*60}")
    print("🔍 COSMOS REASONING:")
    print(f"{'='*60}")
    print(reasoning if reasoning else "(no reasoning returned)")

    print(f"\n{'='*60}")
    print("📋 COSMOS RAW RESPONSE:")
    print(f"{'='*60}")
    print(content)

    print(f"\n{'='*60}")
    print("🎯 FINAL DECISION:")
    print(f"{'='*60}")

    if decision["final_status"] == "ANOMALY":
        print(f"🚨 STATUS    : ANOMALY")
        print(f"   TRIGGER   : {decision['trigger']}")
        if decision["keywords"]:
            print(f"   KEYWORDS  : {', '.join(decision['keywords'])}")
        print(f"   PLC STOP  : YES — Machine should be stopped!")
    else:
        print(f"✅ STATUS    : NORMAL")
        print(f"   PLC STOP  : NO")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(test_video())