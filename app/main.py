from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from app.quiz import generate_quiz
from app.rag import rag_answer, ingest_file
import traceback
import logging

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = FastAPI(
    title="Moodle AI Backend",
    version="1.0.0",
    description="Enterprise AI backend for Moodle plugins (Quiz + Chat)"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------------------------
# MODELS
# -------------------------------------------------

class ChatRequest(BaseModel):
    course_id: int = Field(..., gt=0, description="Moodle course ID")
    question: str = Field(..., min_length=3, description="Student question")

class QuizRequest(BaseModel):
    course_id: int = Field(..., gt=0, description="Moodle course ID")
    topic: str = Field(..., min_length=3)
    num_questions: int = Field(default=5, ge=1, le=50)
    content: str = Field(
        ...,
        min_length=50,
        description="Extracted course content used to generate quiz"
    )

# -------------------------------------------------
# ROOT / HEALTH
# -------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Moodle AI Backend",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "moodle-ai-backend"
    }

# -------------------------------------------------
# CHAT (RAG)
# -------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        logging.info(f"[CHAT] course_id={req.course_id}")

        answer = await rag_answer(
            course_id=req.course_id,
            question=req.question
        )

        return {
            "status": "success",
            "answer": answer
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Internal AI error"
        )

# -------------------------------------------------
# QUIZ GENERATION
# -------------------------------------------------

@app.post("/generate-quiz")
def generate_quiz_api(req: QuizRequest):
    try:
        logging.info(
            f"[QUIZ] course_id={req.course_id}, "
            f"topic={req.topic}, "
            f"count={req.num_questions}"
        )

        quiz_data = generate_quiz(
            course_id=req.course_id,
            topic=req.topic,
            count=req.num_questions,
            content=req.content
        )

        return {
            "status": "success",
            "count": len(quiz_data),
            "quiz": quiz_data
        }

    except ValueError as e:
        # AI output / validation / schema errors
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Quiz generation failed"
        )

# -------------------------------------------------
# INGEST (RAG)
# -------------------------------------------------

@app.post("/ingest")
async def ingest(
    course_id: int = Form(..., gt=0),
    chapter_id: int = Form(..., gt=0),
    file: UploadFile = File(...)
):
    try:
        logging.info(
            f"[INGEST] course_id={course_id}, "
            f"chapter_id={chapter_id}, "
            f"file={file.filename}"
        )

        result = await ingest_file(
            course_id=course_id,
            chapter_id=chapter_id,
            file=file
        )

        return {
            "status": "success",
            "chunks": result.get("chunks", 0)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Ingestion failed"
        )
