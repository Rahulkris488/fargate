from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from app.quiz import generate_quiz
from app.rag import (
    rag_answer,
    ingest_file,
    index_course_content,
    get_course_status
)
from app.embeddings import llm

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Moodle AI Backend", version="3.0.0-phase3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# MODELS
# =====================
class QuizRequest(BaseModel):
    content: str
    num_questions: int = 5
    difficulty: str = "medium"

class ChatRequest(BaseModel):
    course_id: int
    question: str
    course_name: Optional[str] = None
    user_id: Optional[int] = None

# =====================
# HEALTH
# =====================
@app.get("/health")
def health():
    return {"status": "ok", "phase": 3}

# =====================
# QUIZ (UNCHANGED)
# =====================
@app.post("/generate-quiz")
def quiz(req: QuizRequest):
    return generate_quiz(req.content, req.num_questions, req.difficulty)

# =====================
# CHAT (FIXED)
# =====================
@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    course_name = req.course_name or f"course_{req.course_id}"

    status = await get_course_status(req.course_id)

    if status["indexed"]:
        answer = await rag_answer(req.course_id, question)
        return {
            "success": True,
            "response": answer,
            "mode": "rag"
        }

    prompt = f"""
You are an AI tutor.

Question:
{question}

Give a helpful, general explanation.
"""
    return {
        "success": True,
        "response": llm(prompt),
        "mode": "ai"
    }

# =====================
# INDEX COURSE
# =====================
@app.post("/courses/{course_id}/index")
async def index_course(course_id: int):
    raise HTTPException(
        status_code=400,
        detail="Indexing must be triggered via Moodle extractor"
    )

# =====================
# STATUS
# =====================
@app.get("/courses/{course_id}/status")
async def status(course_id: int):
    return await get_course_status(course_id)
