"""
Finfolder Visual Anomaly Detection - Image Test with Reasoning-Based Detection
------------------------------------------------------------------------------
Tests a single image against Cosmos Reason 2.
Uses keyword scanning on reasoning field for reliable anomaly detection.

Requirements:
    pip install httpx

Usage:
    py test_cosmos_image.py
"""

import httpx
import base64
import asyncio

# ── Config ─────────────────────────────────────────────────────────────────────

IMAGE_PATH     = r"C:\Users\samy.timalsina.AUER\Desktop\cosmos\scripts\reference_1.jpg"
COSMOS_ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL_NAME      = "nvidia/Cosmos-Reason2-8B"

# ── Anomaly Keywords ───────────────────────────────────────────────────────────
# If ANY of these appear in Cosmos reasoning → trigger ANOMALY
# Add more keywords as you discover them from testing

ANOMALY_KEYWORDS = [
    # Human intrusion
    "hand", "finger", "arm", "person", "human", "body part",
    "operator", "worker", "glove", "sleeve",

    # Pattern anomalies
    "flat", "irregular", "inconsistent", "disruption", "disrupted",
    "break", "broken", "missing pattern", "gap", "uneven",
    "distortion", "distorted", "crush", "crushed", "fold", "folded",
    "buckle", "stuck", "jam", "jammed", "bunching", "bunched",
    "crease", "crumple", "wrinkle",

    # General anomalies
    "foreign object", "obstruction", "obstructed", "unusual",
    "abnormal", "anomaly", "defect", "damage", "damaged",
    "something", "object", "shadow", "blocked", "blocking"
]

# ── Keyword Scanner ────────────────────────────────────────────────────────────

def scan_reasoning_for_anomaly(reasoning: str) -> tuple[bool, list[str]]:
    """
    Scans Cosmos reasoning text for anomaly keywords.
    Returns (is_anomaly, list of triggered keywords)
    """
    reasoning_lower = reasoning.lower()
    triggered = [kw for kw in ANOMALY_KEYWORDS if kw in reasoning_lower]
    return len(triggered) > 0, triggered


def make_final_decision(cosmos_status: str, reasoning: str) -> dict:
    """
    Makes final NORMAL/ANOMALY decision by combining:
    1. Keyword scan on reasoning (primary — catches what STATUS field misses)
    2. Cosmos STATUS field (secondary)
    """
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

async def test_image():
    print(f"\n{'='*60}")
    print(f"IMAGE TEST — Cosmos Reason 2 + Reasoning Detection")
    print(f"{'='*60}")
    print(f"Image: {IMAGE_PATH}")

    # Load image
    try:
        with open(IMAGE_PATH, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        print("✅ Image loaded")
    except FileNotFoundError:
        print(f"❌ Image not found: {IMAGE_PATH}")
        return

    print("⏳ Sending to Cosmos Reason 2...")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                COSMOS_ENDPOINT,
                json={
                    "model": MODEL_NAME,
                    "max_tokens": 4096,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert quality control inspector monitoring an industrial finfolder metal stamping machine. Your job is to visually inspect the stamped metal sheet and detect any anomalies including human intrusion, pattern defects, or foreign objects."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": """Carefully examine this image of a finfolder metal stamping machine output.

The metal sheet should show a perfectly uniform continuous corrugated wave pattern.
Look extremely carefully for:
- Any human hand, finger, arm or body part near or touching the sheet
- Any foreign object or obstruction on or near the sheet
- Any flat zone where the pattern is missing or interrupted
- Any area where waves look irregular, inconsistent, or different
- Any material bunching, folding, or crushing
- Any shadow or object that does not belong

Be very sensitive — even partial or brief anomalies count.
When in doubt, choose ANOMALY.

<think>
Describe every object you see in this image in detail.
Is there any human body part visible anywhere?
Is there any foreign object near the machine or sheet?
Is the corrugated wave pattern perfectly uniform across the entire surface?
Is there anything at all that looks out of place?
</think>

STATUS: NORMAL or STATUS: ANOMALY
REASON: [exactly what you see, be specific about location]
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
        print("❌ Request timed out")
        return

    # Parse response
    result    = response.json()
    message   = result["choices"][0]["message"]
    reasoning = message.get("reasoning", "")
    content   = message.get("content", "")

    # Extract cosmos STATUS from content
    cosmos_status = ""
    for line in content.splitlines():
        if "STATUS:" in line.upper():
            cosmos_status = line.strip()
            break

    # Make final decision using reasoning keywords
    decision = make_final_decision(cosmos_status, reasoning)

    # ── Print Results ──────────────────────────────────────────────────────────

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
    asyncio.run(test_image())