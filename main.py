from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import sys
import os

# Add ml-engine to path
sys.path.append(os.path.abspath("../ml-engine"))

from preprocess import clean_text
from database import SessionLocal, engine
from models import QueryLog, Base

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
model = joblib.load("../ml-engine/model.pkl")
vectorizer = joblib.load("../ml-engine/vectorizer.pkl")


class TextRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "TrustGuard AI Backend Running"}


@app.post("/analyze")
def analyze_text(request: TextRequest):

    # Input validation
    if not request.text or len(request.text.strip()) < 20:
        return {
            "score": 0,
            "category": "Invalid",
            "explanation": "Text too short to analyze properly."
        }

    cleaned_text = clean_text(request.text)
    text_vector = vectorizer.transform([cleaned_text])

    probabilities = model.predict_proba(text_vector)[0]
    real_probability = probabilities[1]
    trust_score = int(real_probability * 100)

    if trust_score >= 70:
        category = "Real"
        explanation = "Content aligns with legitimate reporting patterns."
    elif 40 <= trust_score < 70:
        category = "Suspicious"
        explanation = "Mixed credibility signals detected."
    else:
        category = "Fake"
        explanation = "Patterns associated with misinformation detected."

    # Save to database
    db = SessionLocal()
    log = QueryLog(
        text=request.text,
        score=trust_score,
        category=category
    )
    db.add(log)
    db.commit()
    db.close()

    return {
        "score": trust_score,
        "category": category,
        "explanation": explanation
    }


@app.get("/logs")
def get_logs():
    db = SessionLocal()
    logs = db.query(QueryLog).all()
    db.close()

    return [
        {
            "id": log.id,
            "text": log.text,
            "score": log.score,
            "category": log.category
        }
        for log in logs
    ]
    