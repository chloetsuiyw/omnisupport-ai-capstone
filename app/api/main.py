"""FastAPI starter contracts for the OmniSupport capstone.

Request schemas are supplied so students can test validation. The actual AI,
RAG, vision and agent behaviour remains intentionally unimplemented.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="OmniSupport AI - Student Prototype", version="0.2.0-starter")


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class RAGRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AgentRequest(BaseModel):
    request: str = Field(min_length=3, max_length=2000)


class VisionRequest(BaseModel):
    image_path: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "message": "Starter API is running; implement model/RAG/agent endpoints."}


@app.post("/tickets/extract")
def extract_ticket(payload: TextRequest):
    raise HTTPException(status_code=501, detail="Student task: implement validated structured extraction")


@app.post("/rag/ask")
def rag_ask(payload: RAGRequest):
    raise HTTPException(status_code=501, detail="Student task: implement grounded RAG with no-answer handling")


@app.post("/agent/run")
def agent_run(payload: AgentRequest):
    raise HTTPException(status_code=501, detail="Student task: implement controlled agent/tool workflow")


@app.post("/vision/predict")
def vision_predict(payload: VisionRequest):
    raise HTTPException(status_code=501, detail="Student task: implement image validation and prediction")
