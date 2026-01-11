from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from app.quiz import generate_quiz
from app.rag import rag_answer, ingest_file, index_course_content, get_course_status
from app.embeddings import llm
from app.moodle_extractor import moodle_extractor
import traceback
import logging

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = FastAPI(
    title="Moodle AI Backend",
    version="3.0.1-phase3",
    description="Shared backend for Moodle Quiz + AI Tutor (RAG)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    course_id: int = Field(..., gt=0)
    course_name: str = Field(...)
    user_id: int = Field(..., gt=0)
    question: str = Field(..., min_length=1)

class QuizRequest(BaseModel):
    course_id: int = Field(..., gt=0)
    topic: str = Field(..., min_length=3)
    num_questions: int = Field(default=5, ge=1, le=50)
    content: str = Field(..., min_length=20)

class MoodleExtractRequest(BaseModel):
    course_id: int = Field(..., gt=0)

# -------------------------------------------------
# ROOT / HEALTH
# -------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Moodle AI Backend is running",
        "version": "3.0.1-phase3",
        "status": "healthy",
        "phase": "3 - Course Content RAG"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.1-phase3",
        "moodle_extractor": "ready" if moodle_extractor else "disabled"
    }

# -------------------------------------------------
# CHAT (RAG + AI FALLBACK)
# -------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Smart chat with automatic RAG/AI routing
    - Indexed courses use RAG
    - Non-indexed courses use AI fallback
    """
    try:
        question = req.question.strip()
        
        logging.info(
            f"[CHAT] course_id={req.course_id}, user_id={req.user_id}, "
            f"course='{req.course_name}', question='{question[:100]}...'"
        )
        
        # Check if course is indexed
        status = await get_course_status(req.course_id)
        is_indexed = status.get("indexed", False)
        chunk_count = status.get("chunks", 0)
        
        # Try RAG if indexed
        if is_indexed and chunk_count > 0:
            logging.info(f"[CHAT] Using RAG mode ({chunk_count} chunks)")
            try:
                answer = await rag_answer(req.course_id, question)
                return {
                    "success": True,
                    "response": answer,  # ✅ For chat.js compatibility
                    "mode": "rag",
                    "phase": 3,
                    "sources": [],
                    "course_id": req.course_id,
                    "indexed_chunks": chunk_count
                }
            except Exception as e:
                logging.warning(f"[CHAT] RAG failed: {e}, falling back to AI")
        
        # AI Fallback
        logging.info("[CHAT] Using AI mode (course not indexed or RAG failed)")
        
        prompt = f"""You are an AI tutor helping a student in the course: "{req.course_name}".

The student asks: {question}

Provide a helpful, clear, and educational response. Be conversational and student-friendly.

Guidelines:
- Give accurate, helpful information
- Use simple language appropriate for students
- If it's a greeting, respond warmly and mention you're ready to help
- If asked what you can do, explain you can help with course questions, concepts, and learning
- Keep responses focused and concise (2-4 paragraphs unless more detail is needed)
- If you don't know something, be honest about it

Response:"""

        try:
            logging.info("[CHAT] Calling GROQ LLM...")
            answer = llm(prompt)
            logging.info(f"[CHAT] ✓ Response generated ({len(answer)} chars)")
        except Exception as e:
            logging.error(f"[CHAT ERROR] GROQ failed: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": "AI service temporarily unavailable. Please try again."
            }
        
        return {
            "success": True,
            "response": answer,  # ✅ For chat.js compatibility
            "mode": "ai",
            "phase": 3,
            "sources": [],
            "course_id": req.course_id,
            "note": "Course not indexed yet. Using general AI knowledge."
        }
        
    except Exception as e:
        logging.error(f"[CHAT ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Chat service error"
        )

# -------------------------------------------------
# INDEX COURSE (FROM MOODLE)
# -------------------------------------------------

@app.post("/courses/{course_id}/index")
async def index_course(course_id: int):
    """
    Extract and index course content from Moodle
    Returns immediately with status
    """
    try:
        logging.info(f"[INDEX] Starting for course_id={course_id}")
        
        if not moodle_extractor:
            raise HTTPException(
                status_code=503,
                detail="Moodle extractor not configured"
            )
        
        # Extract documents
        documents = moodle_extractor.extract_course_documents(course_id)
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No content found in course"
            )
        
        logging.info(f"[INDEX] Extracted {len(documents)} documents")
        
        # Get course name
        course_name = documents[0]["metadata"].get(
            "course_name",
            f"Course {course_id}"
        )
        
        # Index content
        result = await index_course_content(
            course_id=course_id,
            course_name=course_name,
            documents=documents
        )
        
        logging.info(f"[INDEX] ✓ Indexed {result.get('chunks_indexed', 0)} chunks")
        
        return {
            "success": True,
            "message": f"Successfully indexed course '{course_name}'",
            "course_id": course_id,
            "course_name": course_name,
            "documents_extracted": len(documents),
            "chunks_indexed": result.get("chunks_indexed", 0),
            "total_chars": result.get("total_content_chars", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[INDEX ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}"
        )

# -------------------------------------------------
# COURSE STATUS
# -------------------------------------------------

@app.get("/courses/{course_id}/status")
async def course_status(course_id: int):
    """Check if course is indexed"""
    try:
        result = await get_course_status(course_id)
        logging.info(
            f"[STATUS] course_id={course_id}, "
            f"indexed={result.get('indexed', False)}"
        )
        return result
    except Exception as e:
        logging.error(f"[STATUS ERROR] {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

# -------------------------------------------------
# QUIZ GENERATION (SHARED PLUGIN - UNCHANGED)
# -------------------------------------------------

@app.post("/generate-quiz")
def generate_quiz_api(req: QuizRequest):
    """Generate quiz questions using AI"""
    try:
        logging.info(
            f"[QUIZ] course_id={req.course_id}, "
            f"topic='{req.topic}', count={req.num_questions}"
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
        
    except Exception as e:
        logging.error(f"[QUIZ ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Quiz generation failed")

# -------------------------------------------------
# INGEST FILE (LEGACY - UNCHANGED)
# -------------------------------------------------

@app.post("/ingest")
async def ingest(
    course_id: int = Form(..., gt=0),
    chapter_id: int = Form(..., gt=0),
    file: UploadFile = File(...)
):
    """Ingest a single file into vector database"""
    try:
        logging.info(f"[INGEST] course_id={course_id}, file='{file.filename}'")
        result = await ingest_file(course_id, chapter_id, file)
        return {
            "status": "success",
            "chunks": result.get("chunks", 0),
            "file": file.filename
        }
    except Exception as e:
        logging.error(f"[INGEST ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Ingestion failed")

# -------------------------------------------------
# DEBUG ROUTES
# -------------------------------------------------

@app.get("/__debug/routes")
def debug_routes():
    """List all registered routes"""
    return [
        {
            "path": route.path,
            "methods": list(route.methods) if hasattr(route, 'methods') else []
        }
        for route in app.routes
    ]