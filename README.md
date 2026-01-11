# Message Emotion Helper (Sentiment Analyzer)

Message Emotion Helper is a lightweight tool that helps interpret the emotional tone of short text messages (e.g., WhatsApp messages).
The system analyzes a given message using a pre-trained NLP emotion model and returns:
- Top-3 predicted emotions with confidence scores  
- A simple social interpretation (e.g., friendly, teasing, critical)  
- Highlights of keywords, emojis, or punctuation that may influence interpretation  
- A minimal web UI and REST API built with FastAPI  

This project was developed as part of a final course project in *Introduction to Artificial Intelligence*.

## Motivation
People with difficulties in reading social cues (such as understanding sarcasm, teasing, or emotional tone in short messages) may misinterpret written communication and experience confusion or anxiety.
The goal of this project is to provide an accessible assistant that offers a **helpful emotional interpretation** of a message and explains which elements in the text may have influenced that interpretation.

> **Note:** The system provides an automated guess only. It is not a diagnostic tool and does not represent absolute truth.

## How It Works (High-Level)
1. The user pastes a text message into the UI (or sends it via the API).
2. The backend sends the text to a pre-trained emotion classification model (Hugging Face Transformers).
3. The model returns the top emotion labels with confidence scores (Top-3).
4. A lightweight rule-based layer maps these results into a higher-level social tone (e.g., joking/teasing).
5. An additional module extracts highlights such as emojis, keywords, or punctuation that may affect interpretation.
6. The results are returned as JSON and displayed in a friendly web interface.

## Tech Stack
- **Backend:** FastAPI, Uvicorn  
- **NLP Model:** `j-hartmann/emotion-english-distilroberta-base` (Hugging Face Transformers)  
- **Frontend:** Static HTML/CSS/JavaScript served by FastAPI  

## Installation & Running Locally

### 1. Create and activate a virtual environment
```
python -m venv venv
```
**Windows**

venv\Scripts\activate

**macOS / Linux**

source venv/bin/activate

### 2. Install dependencies
```
pip install fastapi uvicorn transformers torch
```
On first run, the emotion model will be downloaded automatically and cached.

### 3. Run the server
From the project root directory:
```
python -m uvicorn SentimentAnalysis.api:app --reload
```
### 4. Open in browser
Web UI: http://127.0.0.1:8000

