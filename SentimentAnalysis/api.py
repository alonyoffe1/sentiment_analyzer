from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentiment_service import EmotionAnalyzer

app = FastAPI(title="Sentiment Analyzer API")

analyzer = EmotionAnalyzer()

class AnalyzeRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    return analyzer.analyze(text)

