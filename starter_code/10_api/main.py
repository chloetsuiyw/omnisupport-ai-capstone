"""FastAPI service exposing the escalation classifier and RAG assistant."""

import sys
import pickle
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_ml"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_rag"))

import train_classification
import rag_pipeline

app = FastAPI(title="OmniSupport AI API", version="1.0")

_model_state = {"pipeline": None}
_rag_state = {"index": None}


class EscalationRequest(BaseModel):
    support_channel: str
    customer_region: str
    customer_age_band: str
    preferred_language: str
    customer_tenure_months: int
    product_category: str
    order_value_capped: float
    order_value_was_capped: int
    delivery_delay_days: int
    previous_ticket_count: int
    issue_category: str
    priority: str
    attachment_available: int
    accessibility_support_flag: int
    ticket_hour: int
    ticket_day_of_week: int
    ticket_month: int
    issue_title_length: int
    issue_description_length: int


class EscalationResponse(BaseModel):
    escalation_probability: float
    predicted_escalation: bool
    requires_human_review: bool


class PolicyQuestionRequest(BaseModel):
    question: str


class PolicyQuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    top_retrieval_score: Optional[float]


@app.on_event("startup")
def load_resources():
    X, y = train_classification.load_model_ready_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train_classification.split_data(X, y)
    pipeline, _ = train_classification.train_rf_classifier(X_train, y_train, X_val, y_val)
    _model_state["pipeline"] = pipeline
    _rag_state["index"] = rag_pipeline.build_index()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_state["pipeline"] is not None}


@app.post("/predict/escalation", response_model=EscalationResponse)
def predict_escalation(request: EscalationRequest):
    if _model_state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    import pandas as pd
    row = pd.DataFrame([request.model_dump()])
    prob = _model_state["pipeline"].predict_proba(row)[0][1]
    return EscalationResponse(
        escalation_probability=round(float(prob), 4),
        predicted_escalation=prob >= 0.5,
        requires_human_review=prob >= 0.6,
    )


@app.post("/ask/policy", response_model=PolicyQuestionResponse)
def ask_policy(request: PolicyQuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    result = rag_pipeline.answer_question(request.question, _rag_state["index"], top_k=3)
    return PolicyQuestionResponse(
        answer=result["answer"],
        sources=result["retrieved_sources"],
        top_retrieval_score=result["retrieval_scores"][0] if result["retrieval_scores"] else None,
    )