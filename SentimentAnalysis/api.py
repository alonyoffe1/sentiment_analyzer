from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from sentiment_service import EmotionAnalyzer

app = FastAPI(title="Sentiment Analyzer API")

analyzer = EmotionAnalyzer()

class AnalyzeRequest(BaseModel):
    text: str

# --- UI Home (serves static/index.html) ---
@app.get("/")
def home():
    base_dir = os.path.dirname(__file__)
    return FileResponse(os.path.join(base_dir, "static", "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    return analyzer.analyze(text)

# --- Friendly endpoint for the UI ---
@app.post("/analyze-friendly")
def analyze_friendly(req: AnalyzeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please provide text")

    raw = analyzer.analyze(text)
    if "error" in raw:
        raise HTTPException(status_code=500, detail=raw["error"])

    # raw emotion comes from the model (lowercase)
    emotion = (raw.get("emotion") or "").lower()
    score = float(raw.get("score") or 0)

    # Low confidence -> be careful
    if score < 0.55:
        return {
            "emotion": "Unclear / Not sure",
            "interpretation": "It’s hard to tell the emotion from this message alone. Try adding one more sentence or some context.",
            "confidence": "Low",
            "raw_emotion": emotion,
            "raw_score": round(score, 4),
            # extra info (from Nitzan) - optional for the UI
            "emotions": raw.get("emotions", []),
            "social_interpretation": raw.get("social_interpretation", {}),
            "highlights": raw.get("highlights", []),
        }

    emotion_map = {
        "joy": "Joy / Positive",
        "anger": "Anger / Frustration",
        "sadness": "Sadness",
        "fear": "Worry / Fear",
        "disgust": "Discomfort / Dislike",
        "surprise": "Surprise",
        "neutral": "Neutral",
    }

    interpretation_map = {
        "joy": "The sender may feel happy, relieved, excited, or supportive.",
        "anger": "The sender may feel frustrated, annoyed, or upset. Sometimes it can also be criticism.",
        "sadness": "The sender may feel sad, disappointed, or hurt.",
        "fear": "The sender may feel worried, anxious, or uncertain about the situation.",
        "disgust": "The sender may feel uncomfortable or strongly dislike the topic.",
        "surprise": "The sender may be surprised or did not expect what happened.",
        "neutral": "The message might be factual or emotionally neutral.",
    }

    if score >= 0.80:
        confidence = "High"
    elif score >= 0.60:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "emotion": emotion_map.get(emotion, "Unclear / Not sure"),
        "interpretation": interpretation_map.get(
            emotion,
            "A message can be interpreted in more than one way. Try adding more context."
        ),
        "confidence": confidence,
        "raw_emotion": emotion,
        "raw_score": round(score, 4),

        # extra info (from Nitzan) - optional for UI but useful for future
        "emotions": raw.get("emotions", []),
        "social_interpretation": raw.get("social_interpretation", {}),
        "highlights": raw.get("highlights", []),
    }
