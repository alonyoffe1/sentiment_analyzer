from transformers import pipeline

class EmotionAnalyzer:
    def __init__(self):
        print("Loading emotion model...")
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=3  # Nitzan: return top 3 emotions
        )
        print("Model loaded successfully!")

    def _interpret_social(self, text: str, emotions):
        if not emotions:
            return {
                "tone": "unknown",
                "confidence": "low",
                "explanation": "Not enough information to infer tone.",
                "note": "This is an automated guess, not a fact."
            }

        top = emotions[0]
        t = text.lower()

        # Emoji / sarcasm hints
        has_laugh_emoji = any(e in text for e in ["😂", "😅", "🤣", "😉"])
        has_teasing_words = any(w in t for w in ["sure", "yeah right", "as if", "again"])

        if has_laugh_emoji and has_teasing_words:
            return {
                "tone": "joking_teasing",
                "confidence": "medium",
                "explanation": "Emoji + wording suggests a teasing/joking tone (possibly a light jab).",
                "note": "This is an automated guess (not a truth). If unsure, consider asking the sender."
            }

        second = emotions[1] if len(emotions) > 1 else None

        top_label = top["label"]
        top_score = float(top["score"])
        second_label = second["label"] if second else None
        second_score = float(second["score"]) if second else 0.0

        ambiguous = abs(top_score - second_score) < 0.12

        if top_label == "joy":
            tone = "friendly_positive"
            explanation = "Sounds positive or friendly."
            confidence = "high" if top_score >= 0.70 else "medium"

            if second_label in {"anger", "fear", "disgust"} and second_score >= 0.20:
                tone = "joking_teasing"
                explanation = "Could be joking/teasing (positive but with some tension)."
                confidence = "medium"

        elif top_label in {"anger", "disgust"}:
            tone = "critical_angry"
            explanation = "May sound frustrated, angry, or critical."
            confidence = "high" if top_score >= 0.70 else "medium"

        elif top_label == "sadness":
            tone = "hurt_sad"
            explanation = "May sound disappointed, sad, or hurt."
            confidence = "high" if top_score >= 0.70 else "medium"

        elif top_label == "fear":
            tone = "worried_anxious"
            explanation = "May sound worried or anxious."
            confidence = "high" if top_score >= 0.70 else "medium"

        elif top_label == "surprise":
            tone = "surprised_confused"
            explanation = "May sound surprised or confused (could be positive or negative)."
            confidence = "medium"

        else:
            tone = "neutral_unclear"
            explanation = "Tone is not clear."
            confidence = "low"

        result = {
            "tone": tone,
            "confidence": confidence,
            "explanation": explanation,
            "note": "This is an automated guess (not a truth). If unsure, consider asking the sender."
        }

        if ambiguous and second_label:
            result["confidence"] = "low"
            result["explanation"] = "Message may have multiple interpretations (model is uncertain)."
            result["alternatives"] = [
                {"emotion_hint": top_label, "score": round(top_score, 4)},
                {"emotion_hint": second_label, "score": round(second_score, 4)}
            ]

        return result

    def _extract_highlights(self, text: str):
        t = text.lower()
        highlights = []

        emoji_map = {
            "😂": "Laughing emoji often indicates humor/teasing.",
            "😅": "Nervous/laugh emoji can soften a message (possible teasing).",
            "🤣": "Strong laughter emoji indicates joking.",
            "😉": "Wink emoji can indicate teasing or sarcasm.",
            "😡": "Angry emoji may indicate frustration.",
            "❤️": "Heart emoji often indicates warmth/positivity.",
            "🙏": "May indicate request/thanks or emotional emphasis.",
        }

        for e, reason in emoji_map.items():
            if e in text:
                highlights.append({"span": e, "type": "emoji", "reason": reason})

        keyword_map = {
            "again": "May imply repetition/annoyance or a light jab.",
            "sure": "Sometimes used in sarcastic/teasing replies (depends on context).",
            "fine": "Can be passive-aggressive in short replies.",
            "whatever": "May indicate dismissal or frustration.",
            "always": "Generalization can intensify criticism.",
            "never": "Generalization can intensify criticism.",
            "sorry": "Often signals apology or regret.",
            "please": "Polite request; can soften tone.",
            "thanks": "Gratitude; usually positive.",
        }

        for kw, reason in keyword_map.items():
            if kw in t:
                highlights.append({"span": kw, "type": "keyword", "reason": reason})

        if "!!!" in text or text.count("!") >= 3:
            highlights.append({"span": "!!!", "type": "punctuation", "reason": "Many exclamation marks may indicate strong emotion."})

        if "???" in text or text.count("?") >= 3:
            highlights.append({"span": "???", "type": "punctuation", "reason": "Many question marks may indicate confusion or pressure."})

        words = [w for w in text.split() if len(w) >= 3]
        if any(w.isupper() for w in words):
            highlights.append({"span": "ALL_CAPS", "type": "style", "reason": "ALL CAPS can be perceived as shouting/emphasis."})

        return highlights[:8]

    def analyze(self, text: str):
        try:
            results = self.classifier(text)[0]  # top-3

            emotions = [
                {"label": r["label"], "score": round(float(r["score"]), 4)}
                for r in results
            ]

            top = emotions[0] if emotions else {"label": None, "score": None}

            social = self._interpret_social(text, emotions)
            highlights = self._extract_highlights(text)

            return {
                "text": text,
                "emotion": top["label"],   # e.g. "joy"
                "score": top["score"],     # float
                "emotions": emotions,      # list of top3
                "social_interpretation": social,
                "highlights": highlights,
            }

        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    analyzer = EmotionAnalyzer()
    print(analyzer.analyze("I finally finished the project and it works perfectly!"))
    print(analyzer.analyze("Why does this code keep crashing? I am so frustrated!"))
    print(analyzer.analyze("I am worried about the deadline tomorrow."))
