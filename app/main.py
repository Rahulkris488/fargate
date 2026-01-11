from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from app.quiz import generate_quiz
from app.rag import rag_answer, ingest_file, index_course_content, get_course_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Moodle AI Tutor Backend", version="3.0.0-phase3")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class QuizRequest(BaseModel):
    content: str
    num_questions: int = 5
    difficulty: str = "medium"

class ChatRequest(BaseModel):
    course_id: int
    course_name: Optional[str] = None  # Made optional
    user_id: Optional[int] = None      # Made optional
    question: str

class FileUploadRequest(BaseModel):
    course_id: int
    course_name: str

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "3.0.0-phase3",
        "phase": 3,
        "moodle_extractor": "ready"
    }

# ============================================================================
# PHASE 1 & 2: QUIZ GENERATION (UNCHANGED)
# ============================================================================

@app.post("/generate-quiz")
async def generate_quiz_endpoint(req: QuizRequest):
    """Generate quiz from content"""
    try:
        logger.info(f"[QUIZ] Generating {req.num_questions} questions, difficulty={req.difficulty}")
        result = generate_quiz(req.content, req.num_questions, req.difficulty)
        logger.info(f"[QUIZ] Generated {len(result.get('questions', []))} questions")
        return result
    except Exception as e:
        logger.error(f"[QUIZ] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    course_id: int = Form(...),
    course_name: str = Form(...)
):
    """Upload and ingest file for quiz generation"""
    try:
        logger.info(f"[UPLOAD] file={file.filename}, course={course_id}")
        content = await file.read()
        result = await ingest_file(content, file.filename)
        logger.info(f"[UPLOAD] Ingested {len(result.get('text', ''))} chars")
        return result
    except Exception as e:
        logger.error(f"[UPLOAD] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PHASE 3: CHAT WITH RAG
# ============================================================================

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Smart chat with automatic RAG/AI routing
    - Indexed courses use RAG
    - Non-indexed courses use AI fallback
    
    FIXED: Made course_name and user_id optional
    """
    try:
        question = req.question.strip()
        
        # Default values if not provided
        course_name = req.course_name or f"course_{req.course_id}"
        user_id = req.user_id or 0
        
        logger.info(f"[CHAT] course_id={req.course_id}, question='{question[:50]}...'")
        
        # Check if course is indexed
        status = await get_course_status(req.course_id)
        
        if status["indexed"]:
            # Use RAG
            logger.info(f"[CHAT] Using RAG (indexed, {status['chunks']} chunks)")
            answer = await rag_answer(req.course_id, question)
            return {
                "success": True,
                "response": answer,  # Changed from "answer" to "response"
                "mode": "rag",
                "chunks_used": status['chunks']
            }
        else:
            # AI fallback
            logger.info(f"[CHAT] Using AI fallback (not indexed)")
            from app.rag import llm_provider
            
            fallback_prompt = f"""You are an AI tutor assistant. A student asked:

Question: {question}

Since this course hasn't been indexed yet, provide a helpful, general response based on your knowledge. 
If you need course-specific information, suggest that the instructor index the course content first.

Respond in a friendly, educational tone."""
            
            response = await llm_provider.get_completion(fallback_prompt)
            
            return {
                "success": True,
                "response": response,  # Changed from "answer" to "response"
                "mode": "ai",
                "note": "Course not indexed - using AI fallback"
            }
            
    except Exception as e:
        logger.error(f"[CHAT] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PHASE 3: COURSE INDEXING
# ============================================================================

@app.post("/courses/{course_id}/index")
async def index_course(course_id: int):
    """
    Index a Moodle course for RAG
    
    This endpoint:
    1. Connects to Moodle via web services
    2. Extracts all course content (HTML, files, activities)
    3. Chunks the content
    4. Stores embeddings in Qdrant
    
    FIXED: Returns flat JSON (no nested statistics)
    """
    try:
        logger.info(f"[INDEX] Starting indexing for course_id={course_id}")
        
        # Index the course
        result = await index_course_content(course_id)
        
        logger.info(f"[INDEX] Complete: {result.get('chunks_indexed', 0)} chunks indexed")
        
        # Return flat JSON structure (frontend expects this)
        return {
            "success": True,
            "course_id": course_id,
            "course_name": result.get("course_name", f"course_{course_id}"),
            "documents_extracted": result.get("documents_extracted", 0),
            "chunks_indexed": result.get("chunks_indexed", 0),
            "total_content_chars": result.get("total_content_chars", 0),
            "document_types": result.get("document_types", {}),
            "message": "Course indexed successfully"
        }
        
    except Exception as e:
        logger.error(f"[INDEX] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Indexing failed: {str(e)}"
        )

@app.get("/courses/{course_id}/status")
async def course_status(course_id: int):
    """
    Check if a course is indexed and get stats
    """
    try:
        logger.info(f"[STATUS] Checking course_id={course_id}")
        status = await get_course_status(course_id)
        logger.info(f"[STATUS] course_id={course_id}, indexed={status['indexed']}, chunks={status['chunks']}")
        return status
    except Exception as e:
        logger.error(f"[STATUS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ROOT
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Moodle AI Tutor Backend - Phase 3",
        "version": "3.0.0",
        "phase": 3,
        "features": [
            "quiz_generation",
            "file_upload",
            "rag_chat",
            "course_indexing",
            "ai_fallback"
        ],
        "endpoints": {
            "health": "GET /health",
            "quiz": "POST /generate-quiz",
            "upload": "POST /upload-file",
            "chat": "POST /chat",
            "index": "POST /courses/{course_id}/index",
            "status": "GET /courses/{course_id}/status"
        }
    }