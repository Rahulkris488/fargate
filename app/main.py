from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from app.quiz import generate_quiz
from app.rag import rag_answer, ingest_file
import traceback

app = FastAPI(title="Moodle AI Backend", version="1.0")

# -----------------------------
# MODELS
# -----------------------------

class ChatRequest(BaseModel):
    course_id: int = Field(..., gt=0)
    question: str = Field(..., min_length=3)

class QuizRequest(BaseModel):
    course_id: int = Field(..., gt=0)
    topic: str = Field(..., min_length=3)
    num_questions: int = Field(default=5, ge=1, le=50)
    content: str = Field(..., min_length=20)

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
def generate_quiz_api(req: QuizRequest):
    try:
        quiz = generate_quiz(
            course_id=req.course_id,
            topic=req.topic,
            count=req.num_questions,
            content=req.content
        )

        return {
            "status": "success",
            "count": len(quiz),
            "quiz": quiz
        }

    except ValueError as ve:
        # AI output / validation errors
        raise HTTPException(status_code=422, detail=str(ve))

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ingest")
async def ingest(
    course_id: int = Form(...),
    chapter_id: int = Form(...),
    file: UploadFile = File(...)
):
    try:
        result = await ingest_file(course_id, chapter_id, file)
        return {"status": "success", "detail": result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
