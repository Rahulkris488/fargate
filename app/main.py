from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from app.quiz import generate_quiz
from app.rag import rag_answer, ingest_file, index_course_content, get_course_status
import traceback
import logging

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = FastAPI(
    title="Moodle AI Backend",
    version="1.0.1",
    description="Enterprise AI backend for Moodle plugins (Quiz + Chat + Indexing)"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------------------------
# MODELS
# -------------------------------------------------

# 🆕 UPDATED: Added course_name and user_id for chat
class ChatRequest(BaseModel):
    course_id: int = Field(..., gt=0, description="Moodle course ID")
    course_name: str = Field(..., description="Course name")
    user_id: int = Field(..., gt=0, description="User ID")
    question: str = Field(..., min_length=3, description="Student question")

# ✅ UNCHANGED: Quiz request model
class QuizRequest(BaseModel):
    course_id: int = Field(..., gt=0)
    topic: str = Field(..., min_length=3)
    num_questions: int = Field(default=5, ge=1, le=50)
    content: str = Field(
        ...,
        min_length=20,
        description="Extracted course content"
    )

# 🆕 NEW: Model for indexing course content
class IndexRequest(BaseModel):
    course_id: int = Field(..., gt=0, description="Course ID")
    course_name: str = Field(..., description="Course name")
    documents: list = Field(..., description="List of documents to index")

# -------------------------------------------------
# ROOT / HEALTH
# -------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Moodle AI Backend is running",
        "version": "1.0.1",
        "endpoints": {
            "chat": "POST /chat",
            "index": "POST /index",
            "course_status": "GET /course/{id}/status",
            "quiz": "POST /generate-quiz",
            "ingest": "POST /ingest"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# CHAT (RAG) - Updated
# -------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        logging.info(f"[CHAT] course_id={req.course_id}, user_id={req.user_id}")
        answer = await rag_answer(req.course_id, req.question)

        return {
            "answer": answer
        }

    except ValueError as e:
        logging.error(f"[CHAT ERROR] ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logging.error(f"[CHAT ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Chat service temporarily unavailable")

# -------------------------------------------------
# 🆕 NEW: INDEX COURSE CONTENT
# -------------------------------------------------

@app.post("/index")
async def index_course(req: IndexRequest):
    try:
        logging.info(
            f"[INDEX] course_id={req.course_id}, "
            f"documents={len(req.documents)}"
        )
        
        result = await index_course_content(
            course_id=req.course_id,
            course_name=req.course_name,
            documents=req.documents
        )
        
        return result
        
    except ValueError as e:
        logging.error(f"[INDEX ERROR] ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logging.error(f"[INDEX ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Indexing failed")

# -------------------------------------------------
# 🆕 NEW: GET COURSE STATUS
# -------------------------------------------------

@app.get("/course/{course_id}/status")
async def course_status(course_id: int):
    try:
        logging.info(f"[STATUS] course_id={course_id}")
        result = await get_course_status(course_id)
        return result
        
    except Exception as e:
        logging.error(f"[STATUS ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Status check failed")

# -------------------------------------------------
# ✅ UNCHANGED: QUIZ GENERATION
# -------------------------------------------------

@app.post("/generate-quiz")
def generate_quiz_api(req: QuizRequest):
    try:
        logging.info(
            f"[QUIZ] course_id={req.course_id}, "
            f"topic={req.topic}, "
            f"count={req.num_questions}"
        )

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
        raise HTTPException(status_code=422, detail=str(ve))

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Quiz generation failed"
        )

# -------------------------------------------------
# ✅ UNCHANGED: INGEST (RAG)
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

        result = await ingest_file(course_id, chapter_id, file)

        return {
            "status": "success",
            "chunks": result.get("chunks", 0)
        }

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Ingestion failed"
        )