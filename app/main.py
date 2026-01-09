from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    version="1.0.2",
    description="Enterprise AI backend for Moodle plugins (Quiz + Chat + Indexing)"
)

# CORS middleware for Moodle frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Moodle domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    course_name: str = Field(..., description="Course name")
    user_id: int = Field(..., gt=0, description="User ID")
    question: str = Field(..., min_length=3, description="Student question")

class QuizRequest(BaseModel):
    course_id: int = Field(..., gt=0)
    topic: str = Field(..., min_length=3)
    num_questions: int = Field(default=5, ge=1, le=50)
    content: str = Field(
        ...,
        min_length=20,
        description="Extracted course content"
    )

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
        "version": "1.0.2",
        "status": "healthy",
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
    return {"status": "ok", "version": "1.0.2"}

# -------------------------------------------------
# CHAT (RAG)
# -------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Handle student questions using RAG (Retrieval-Augmented Generation)
    """
    try:
        logging.info(
            f"[CHAT] course_id={req.course_id}, "
            f"user_id={req.user_id}, "
            f"question_len={len(req.question)}"
        )
        
        answer = await rag_answer(req.course_id, req.question)

        return {
            "success": True,
            "answer": answer
        }

    except ValueError as e:
        logging.error(f"[CHAT ERROR] ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logging.error(f"[CHAT ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail="Chat service temporarily unavailable"
        )

# -------------------------------------------------
# INDEX COURSE CONTENT
# -------------------------------------------------

@app.post("/index")
async def index_course(req: IndexRequest):
    """
    Index course content into vector database for RAG
    """
    try:
        logging.info(
            f"[INDEX] course_id={req.course_id}, "
            f"course_name='{req.course_name}', "
            f"documents={len(req.documents)}"
        )
        
        # Validate documents
        if not req.documents or len(req.documents) == 0:
            raise ValueError("No documents provided to index")
        
        result = await index_course_content(
            course_id=req.course_id,
            course_name=req.course_name,
            documents=req.documents
        )
        
        logging.info(
            f"[INDEX SUCCESS] course_id={req.course_id}, "
            f"chunks={result.get('chunks_indexed', 0)}"
        )
        
        return result
        
    except ValueError as e:
        logging.error(f"[INDEX ERROR] ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logging.error(f"[INDEX ERROR] Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Indexing failed: {str(e)}"
        )

# -------------------------------------------------
# GET COURSE STATUS
# -------------------------------------------------

@app.get("/course/{course_id}/status")
async def course_status(course_id: int):
    """
    Check if a course has been indexed and get statistics
    """
    try:
        logging.info(f"[STATUS] Checking course_id={course_id}")
        
        result = await get_course_status(course_id)
        
        logging.info(
            f"[STATUS] course_id={course_id}, "
            f"indexed={result.get('indexed', False)}, "
            f"chunks={result.get('chunks', 0)}"
        )
        
        return result
        
    except Exception as e:
        logging.error(f"[STATUS ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail="Status check failed"
        )

# -------------------------------------------------
# QUIZ GENERATION
# -------------------------------------------------

@app.post("/generate-quiz")
def generate_quiz_api(req: QuizRequest):
    """
    Generate quiz questions using AI
    """
    try:
        logging.info(
            f"[QUIZ] course_id={req.course_id}, "
            f"topic='{req.topic}', "
            f"count={req.num_questions}"
        )

        quiz = generate_quiz(
            course_id=req.course_id,
            topic=req.topic,
            count=req.num_questions,
            content=req.content
        )

        logging.info(
            f"[QUIZ SUCCESS] Generated {len(quiz)} questions "
            f"for course {req.course_id}"
        )

        return {
            "status": "success",
            "count": len(quiz),
            "quiz": quiz
        }

    except ValueError as ve:
        logging.error(f"[QUIZ ERROR] ValueError: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))

    except Exception as e:
        logging.error(f"[QUIZ ERROR] Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Quiz generation failed"
        )

# -------------------------------------------------
# INGEST FILE (Legacy RAG endpoint)
# -------------------------------------------------

@app.post("/ingest")
async def ingest(
    course_id: int = Form(..., gt=0),
    chapter_id: int = Form(..., gt=0),
    file: UploadFile = File(...)
):
    """
    Ingest a single file into the vector database (legacy endpoint)
    """
    try:
        logging.info(
            f"[INGEST] course_id={course_id}, "
            f"chapter_id={chapter_id}, "
            f"file='{file.filename}'"
        )

        result = await ingest_file(course_id, chapter_id, file)

        logging.info(
            f"[INGEST SUCCESS] Processed {result.get('chunks', 0)} chunks "
            f"from '{file.filename}'"
        )

        return {
            "status": "success",
            "chunks": result.get("chunks", 0),
            "file": file.filename
        }

    except Exception as e:
        logging.error(f"[INGEST ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )