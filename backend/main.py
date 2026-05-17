from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any

from .agent_runner import run_agent_turn

app = FastAPI(title="Dataset Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: str

class ChatResponse(BaseModel):
    session_id: Optional[str]
    turn_id: Optional[int]
    answer: Optional[str]
    generated_code: Optional[str]
    result_table: Optional[list[dict[str, Any]]]
    visualization: Optional[str]
    visualization_decision: Optional[dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/")
def root():
    return {"message": "Dataset Agent API is running. POST /chat to ask questions."}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return run_agent_turn(session_id=req.session_id, question=req.question)
    except Exception as exc:
        return ChatResponse(
            session_id=req.session_id,
            turn_id=None,
            answer=None,
            generated_code=None,
            result_table=None,
            visualization=None,
            visualization_decision=None,
            error=str(exc),
        )
