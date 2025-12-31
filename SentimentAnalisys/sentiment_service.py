from transformers import pipeline

class EmotionAnalyzer:
    def __init__(self):
        print("Loading emotion model...")
        # טעינת מודל ספציפי לזיהוי רגשות (במקום חיובי/שלילי)
        self.classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base", 
            top_k=1  # מחזיר רק את הרגש המוביל. שנה ל-None כדי לקבל את כולם
        )
        print("Model loaded successfully!")

    def analyze(self, text):
        try:
            # ביצוע הניתוח
            # הקריאה מחזירה רשימה של רשימות, לכן אנחנו לוקחים את [0]
            result = self.classifier(text)[0] 
            
            # שליפת התוצאה הראשונה (הכי חזקה)
            top_emotion = result[0]
            
            return {
                "text": text,
                "emotion": top_emotion['label'],  # למשל: 'joy', 'anger'
                "score": round(top_emotion['score'], 4) # רמת ביטחון (0-1)
            }
        except Exception as e:
            return {"error": str(e)}

# --- בדיקה עצמית ---
if __name__ == "__main__":
    analyzer = EmotionAnalyzer()
    
    # בדיקת שמחה
    print(analyzer.analyze("I finally finished the project and it works perfectly!"))
    
    # בדיקת כעס
    print(analyzer.analyze("Why does this code keep crashing? I am so frustrated!"))
    
    # בדיקת פחד
    print(analyzer.analyze("I am worried about the deadline tomorrow."))