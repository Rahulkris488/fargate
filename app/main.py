from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
    version="3.0.0-phase3",
    description="Enterprise AI backend for Moodle - Phase 3: Course Content RAG"
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

class CourseIndexRequest(BaseModel):
    course_id: int = Field(..., gt=0, description="Moodle course ID to index")

# -------------------------------------------------
# ROOT / HEALTH
# -------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Moodle AI Backend is running",
        "version": "3.0.0-phase3",
        "status": "healthy",
        "phase": "3 - Course Content RAG",
        "mode": "production",
        "endpoints": {
            "chat": "POST /chat (Intelligent routing: AI or RAG)",
            "index_course": "POST /courses/{id}/index (Extract & index course)",
            "course_status": "GET /courses/{id}/status",
            "quiz": "POST /generate-quiz",
            "ingest": "POST /ingest (legacy file upload)"
        },
        "features": {
            "moodle_extraction": moodle_extractor is not None,
            "rag_enabled": True,
            "ai_fallback": True,
            "smart_routing": True
        },
        "phase_info": {
            "current": "Phase 3 - Course Content RAG",
            "status": "Full RAG with Moodle integration",
            "capabilities": [
                "Automatic course content extraction",
                "Intelligent RAG search",
                "AI fallback for non-indexed courses",
                "Multi-format content support"
            ]
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0-phase3",
        "phase": 3,
        "moodle_extractor": "ready" if moodle_extractor else "disabled"
    }

# -------------------------------------------------
# 🆕 PHASE 3: SMART CHAT (RAG + AI FALLBACK)
# -------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Phase 3: Intelligent chat with automatic RAG/AI routing
    
    - If course is indexed → Use RAG (course-specific answers)
    - If course not indexed → Use AI (general knowledge)
    - Seamless experience for students
    """
    try:
        question = req.question.strip()
        
        logging.info(
            f"[CHAT] course_id={req.course_id}, "
            f"user_id={req.user_id}, "
            f"course='{req.course_name}', "
            f"question='{question[:100]}...'"
        )
        
        # Check if course is indexed
        status = await get_course_status(req.course_id)
        is_indexed = status.get("indexed", False)
        chunk_count = status.get("chunks", 0)
        
        if is_indexed and chunk_count > 0:
            # Use RAG for indexed courses
            logging.info(
                f"[CHAT] Using RAG mode (course has {chunk_count} chunks)"
            )
            
            try:
                answer = await rag_answer(req.course_id, question)
                
                return {
                    "success": True,
                    "answer": answer,
                    "mode": "rag",
                    "phase": 3,
                    "course_id": req.course_id,
                    "course_name": req.course_name,
                    "indexed_chunks": chunk_count,
                    "model": "groq-llama-3.1-8b"
                }
                
            except Exception as rag_error:
                # RAG failed, fall back to AI
                logging.warning(
                    f"[CHAT] RAG failed, falling back to AI: {rag_error}"
                )
                # Continue to AI fallback below
        
        # Use AI for non-indexed courses or RAG failures
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
            logging.info(f"[CHAT] Calling GROQ LLM...")
            answer = llm(prompt)
            logging.info(f"[CHAT] ✓ Response generated ({len(answer)} chars)")
        except Exception as e:
            logging.error(f"[CHAT ERROR] GROQ failed: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": "AI service temporarily unavailable. Please try again in a moment."
            }
        
        return {
            "success": True,
            "answer": answer,
            "mode": "ai",
            "phase": 3,
            "course_id": req.course_id,
            "course_name": req.course_name,
            "model": "groq-llama-3.1-8b",
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
# 🆕 PHASE 3: INDEX COURSE FROM MOODLE
# -------------------------------------------------

@app.post("/courses/{course_id}/index")
async def index_course_from_moodle(course_id: int):
    """
    Phase 3: Extract and index course content from Moodle
    
    This endpoint:
    1. Connects to Moodle via Web Services
    2. Extracts all course content (pages, files, etc.)
    3. Chunks and embeds the content
    4. Stores in Qdrant vector database
    
    Returns progress and statistics
    """
    try:
        logging.info("\n" + "="*70)
        logging.info(f"[INDEX COURSE] Starting indexing for course_id={course_id}")
        logging.info("="*70)
        
        # Validate Moodle extractor is available
        if not moodle_extractor:
            raise HTTPException(
                status_code=503,
                detail="Moodle extractor not configured. Check MOODLE_URL and MOODLE_TOKEN in environment."
            )
        
        # Extract course content from Moodle
        logging.info(f"[INDEX COURSE] Step 1/3: Extracting content from Moodle...")
        
        try:
            documents = moodle_extractor.extract_course_documents(course_id)
        except ValueError as e:
            logging.error(f"[INDEX COURSE ERROR] Extraction failed: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logging.error(f"[INDEX COURSE ERROR] Unexpected extraction error: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract course content: {str(e)}"
            )
        
        # Validate we have content
        if not documents or len(documents) == 0:
            logging.warning(f"[INDEX COURSE] No content found for course {course_id}")
            raise HTTPException(
                status_code=400,
                detail="Course has no content to index. Add some content to the course first."
            )
        
        logging.info(
            f"[INDEX COURSE] ✓ Extracted {len(documents)} documents from Moodle"
        )
        
        # Get course name from first document
        course_name = documents[0]["metadata"].get(
            "course_name",
            f"Course {course_id}"
        )
        
        # Index the content
        logging.info(f"[INDEX COURSE] Step 2/3: Chunking and embedding content...")
        
        try:
            result = await index_course_content(
                course_id=course_id,
                course_name=course_name,
                documents=documents
            )
        except ValueError as e:
            logging.error(f"[INDEX COURSE ERROR] Indexing failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logging.error(f"[INDEX COURSE ERROR] Unexpected indexing error: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to index content: {str(e)}"
            )
        
        logging.info(f"[INDEX COURSE] Step 3/3: Storing in vector database...")
        logging.info(
            f"[INDEX COURSE] ✓ Successfully indexed {result.get('chunks_indexed', 0)} chunks"
        )
        
        # Build success response
        response = {
            "success": True,
            "message": f"Successfully indexed course '{course_name}'",
            "course_id": course_id,
            "course_name": course_name,
            "statistics": {
                "documents_extracted": len(documents),
                "chunks_indexed": result.get("chunks_indexed", 0),
                "total_content_chars": result.get("total_content_chars", 0),
                "collection": result.get("collection", "")
            },
            "document_types": {}
        }
        
        # Count document types
        for doc in documents:
            doc_type = doc["type"]
            response["document_types"][doc_type] = \
                response["document_types"].get(doc_type, 0) + 1
        
        logging.info("\n" + "="*70)
        logging.info(f"[INDEX COURSE SUCCESS] Course {course_id} indexed successfully!")
        logging.info(f"[INDEX COURSE] Documents: {len(documents)}")
        logging.info(f"[INDEX COURSE] Chunks: {result.get('chunks_indexed', 0)}")
        logging.info("="*70 + "\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[INDEX COURSE ERROR] Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Course indexing failed: {str(e)}"
        )

# -------------------------------------------------
# GET COURSE STATUS
# -------------------------------------------------

@app.get("/courses/{course_id}/status")
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
# LEGACY: INDEX WITH MANUAL DOCUMENTS
# -------------------------------------------------

@app.post("/index")
async def index_course(req: IndexRequest):
    """
    Index course content with manually provided documents (legacy)
    Use /courses/{id}/index for automatic Moodle extraction
    """
    try:
        logging.info(
            f"[INDEX LEGACY] course_id={req.course_id}, "
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
            f"[INDEX LEGACY SUCCESS] course_id={req.course_id}, "
            f"chunks={result.get('chunks_indexed', 0)}"
        )
        
        return result
        
    except ValueError as e:
        logging.error(f"[INDEX LEGACY ERROR] ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logging.error(f"[INDEX LEGACY ERROR] Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}"
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
# INGEST FILE (LEGACY)
# -------------------------------------------------

@app.post("/ingest")
async def ingest(
    course_id: int = Form(..., gt=0),
    chapter_id: int = Form(..., gt=0),
    file: UploadFile = File(...)
):
    """
    Ingest a single file into the vector database (legacy)
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