from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.quiz import generate_quiz
from app.rag import rag_answer, ingest_file
import traceback

app = FastAPI()

# -----------------------------
# MODELS
# -----------------------------
class ChatRequest(BaseModel):
    course_id: int
    question: str

class QuizRequest(BaseModel):
    course_id: int
    topic: str
    num_questions: int = 5
    content: str | None = None

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"message": "Moodle AI Backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        answer = await rag_answer(req.course_id, req.question)
        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-quiz")
def quiz(req: QuizRequest):
    try:
        quiz = generate_quiz(
            course_id=req.course_id,
            topic=req.topic,
            count=req.num_questions,
            content=req.content
        )
        return {"quiz": quiz}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest(
    course_id: int = Form(...),
    chapter_id: int = Form(...),
    file: UploadFile = File(...)
):
    try:
        result = await ingest_file(course_id, chapter_id, file)
        return {"message": "Ingestion successful", "detail": result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
