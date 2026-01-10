from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.quiz import generate_quiz
from app.rag import rag_answer, ingest_file, index_course_content, get_course_status
from app.embeddings import llm
import traceback
import logging

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = FastAPI(
    title="Moodle AI Backend",
    version="2.0.0-phase2",
    description="Enterprise AI backend for Moodle - Phase 2: Real AI Chat with GROQ"
)

# CORS middleware
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
    course_id: int = Field(..., gt=0, description="Moodle course ID")
    course_name: str = Field(..., description="Course name")
    user_id: int = Field(..., gt=0, description="User ID")
    question: str = Field(..., min_length=1, description="Student question")

class QuizRequest(BaseModel):
    course_id: int = Field(..., gt=0)
    topic: str = Field(..., min_length=3)
    num_questions: int = Field(default=5, ge=1, le=50)
    content: str = Field(..., min_length=20, description="Extracted course content")

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
        "version": "2.0.0-phase2",
        "status": "healthy",
        "phase": "2 - Real AI Chat with GROQ",
        "mode": "production",
        "endpoints": {
            "chat": "POST /chat (Phase 2 - Real AI responses)",
            "chat_test": "POST /chat/test (Phase 1 - Static responses - DEPRECATED)",
            "index": "POST /index",
            "course_status": "GET /course/{id}/status",
            "quiz": "POST /generate-quiz",
            "ingest": "POST /ingest"
        },
        "phase_info": {
            "current": "Phase 2 - Real AI Chat",
            "status": "GROQ AI active - General knowledge Q&A",
            "next": "Phase 3 - Course Content RAG"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "version": "2.0.0-phase2",
        "phase": 2
    }

# -------------------------------------------------
# 🆕 PHASE 2: REAL AI CHAT (GROQ)
# -------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Phase 2: Real AI chat using GROQ
    Answers any question using AI - no course materials yet
    Phase 3 will add RAG with course content
    """
    try:
        question = req.question.strip()
        
        logging.info(
            f"[CHAT AI] course_id={req.course_id}, "
            f"user_id={req.user_id}, "
            f"course='{req.course_name}', "
            f"question='{question[:100]}...'"
        )
        
        # Build AI prompt with context
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

        # Get AI response from GROQ
        try:
            answer = llm(prompt)
            logging.info(f"[CHAT AI] ✓ Response generated ({len(answer)} chars)")
        except Exception as e:
            logging.error(f"[CHAT AI ERROR] GROQ failed: {e}")
            # Fallback error message
            return {
                "success": False,
                "error": "AI service temporarily unavailable. Please try again in a moment."
            }
        
        return {
            "success": True,
            "answer": answer,
            "mode": "ai",
            "phase": 2,
            "course_id": req.course_id,
            "course_name": req.course_name,
            "model": "groq-llama-3.1-8b"
        }
        
    except Exception as e:
        logging.error(f"[CHAT AI ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail="Chat service error"
        )

# -------------------------------------------------
# PHASE 1: CHAT TEST (Static - DEPRECATED but kept for compatibility)
# -------------------------------------------------

@app.post("/chat/test")
async def chat_test(req: ChatRequest):
    """
    Phase 1: Static test responses (DEPRECATED)
    Kept for backward compatibility only
    Use /chat endpoint for real AI
    """
    try:
        question = req.question.lower().strip()
        
        logging.info(
            f"[CHAT TEST - DEPRECATED] Using old endpoint, "
            f"question='{question}'"
        )
        
        # Simple responses
        if any(g in question for g in ['hi', 'hello', 'hey']):
            answer = f"👋 Hello! I'm your AI Tutor for **{req.course_name}**. Ask me anything!"
        elif 'help' in question:
            answer = "I can help with course questions, explanations, and learning. Just ask!"
        else:
            answer = f"⚠️ You're using the deprecated test endpoint. Please use the main /chat endpoint for real AI responses."

        return {
            "success": True,
            "answer": answer,
            "mode": "test-deprecated",
            "phase": 1,
            "warning": "This endpoint is deprecated. Use /chat for real AI responses."
        }
        
    except Exception as e:
        logging.error(f"[CHAT TEST ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# CHAT RAG - Phase 3+ (Course content search)
# -------------------------------------------------

@app.post("/chat/rag")
async def chat_rag(req: ChatRequest):
    """
    Phase 3+: RAG chat with course content
    Searches course materials and provides context-aware answers
    """
    try:
        logging.info(
            f"[CHAT RAG] course_id={req.course_id}, "
            f"user_id={req.user_id}, "
            f"question_len={len(req.question)}"
        )
        
        answer = await rag_answer(req.course_id, req.question)

        return {
            "success": True,
            "answer": answer,
            "mode": "rag",
            "phase": 3
        }

    except ValueError as e:
        logging.error(f"[CHAT RAG ERROR] ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logging.error(f"[CHAT RAG ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail="RAG service temporarily unavailable"
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
# INGEST FILE
# -------------------------------------------------

@app.post("/ingest")
async def ingest(
    course_id: int = Form(..., gt=0),
    chapter_id: int = Form(..., gt=0),
    file: UploadFile = File(...)
):
    """
    Ingest a single file into the vector database
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

# -------------------------------------------------
# DEBUG ROUTES
# -------------------------------------------------

@app.get("/__debug/routes")
def debug_routes():
    """List all registered routes (for debugging)"""
    return [
        {
            "path": route.path,
            "methods": list(route.methods) if hasattr(route, 'methods') else []
        }
        for route in app.routes
    ]