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
    version="1.1.0-phase1",
    description="Enterprise AI backend for Moodle - Phase 1: Basic Connectivity"
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
        "version": "1.1.0-phase1",
        "status": "healthy",
        "phase": "1 - Basic Connectivity Testing",
        "mode": "development",
        "endpoints": {
            "chat_test": "POST /chat/test (Phase 1 - Static responses for testing)",
            "chat": "POST /chat (Phase 2+ - AI with RAG)",
            "index": "POST /index",
            "course_status": "GET /course/{id}/status",
            "quiz": "POST /generate-quiz",
            "ingest": "POST /ingest"
        },
        "phase_info": {
            "current": "Phase 1 - Basic Connectivity",
            "status": "Testing static responses",
            "next": "Phase 2 - Direct AI Chat with GROQ"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "version": "1.1.0-phase1",
        "phase": 1
    }

# -------------------------------------------------
# 🆕 PHASE 1: CHAT TEST (Static Responses)
# -------------------------------------------------

@app.post("/chat/test")
async def chat_test(req: ChatRequest):
    """
    Phase 1: Simple test endpoint with static responses
    No AI, no RAG - just connectivity testing
    """
    try:
        question = req.question.lower().strip()
        
        logging.info(
            f"[CHAT TEST] course_id={req.course_id}, "
            f"user_id={req.user_id}, "
            f"course='{req.course_name}', "
            f"question='{question}'"
        )
        
        # Pattern matching for static responses
        if any(greeting in question for greeting in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon']):
            answer = f"""👋 **Hello there!**

I'm your AI Tutor for **{req.course_name}**. 

I'm here to help you learn and understand the course materials better!

Try asking me:
- "help" - to see what I can do
- "what can you do?" - to learn about my features"""

        elif 'help' in question or 'what can you do' in question or 'capabilities' in question:
            answer = """🎓 **I'm Your AI Learning Assistant!**

Here's what I can help you with:

📚 **Course Questions**
   Ask me anything about your course materials and I'll provide detailed explanations.

💡 **Concept Explanations**
   Need help understanding a topic? I'll break it down for you.

🎯 **Study Guidance**
   Get study tips, exam preparation help, and learning strategies.

❓ **Quick Answers**
   Fast, accurate answers to your questions anytime.

---

**Example questions to try:**
- "What is this course about?"
- "Explain [topic] to me"
- "Help me understand [concept]"
- "What should I study for the exam?"

**Current Status:** 🚧 Phase 1 Testing Mode
I'm currently in development mode with basic responses. Soon I'll have full AI capabilities!"""

        elif 'thank' in question or 'thanks' in question:
            answer = """You're very welcome! 😊

I'm always here to help you learn. Feel free to ask me anything about your course materials!

Happy studying! 📚"""

        elif 'bye' in question or 'goodbye' in question or 'see you' in question:
            answer = """Goodbye! 👋

Come back anytime you need help with your studies. I'm always here to assist you!

Good luck with your learning! 🌟"""

        elif 'who are you' in question or 'what are you' in question:
            answer = f"""🤖 **I'm Your AI Tutor!**

I'm an artificial intelligence assistant designed specifically to help students like you succeed in **{req.course_name}**.

**My Purpose:**
- Answer your questions about course content
- Explain difficult concepts in simple terms
- Provide study guidance and learning support
- Be available 24/7 whenever you need help

**Current Development Phase:** Phase 1 - Basic Testing
I'm being carefully developed to ensure I provide accurate, helpful responses based on your actual course materials."""

        elif 'test' in question or 'testing' in question:
            answer = """✅ **Connection Test Successful!**

Everything is working correctly:
- ✅ Moodle → Backend connection: **Active**
- ✅ Authentication: **Verified**
- ✅ Course context: **{course_name}**
- ✅ User ID: **{user_id}**
- ✅ Response system: **Operational**

**Phase 1 Status:** All systems nominal!

Try asking me a real question or say "help" to see what I can do!""".format(
                course_name=req.course_name,
                user_id=req.user_id
            )

        elif len(question) < 3:
            answer = """I received a very short message. 

Could you please ask me a complete question? For example:
- "Help" - to see what I can do
- "What is this course about?"
- "Explain [topic] to me"

I'm here to help! 😊"""

        else:
            # Default response for unrecognized questions
            answer = f"""I received your question: **"{req.question}"**

🚧 **Phase 1 Development Mode Active**

I'm currently in testing mode and can only respond to basic greetings and commands. 

**What works right now:**
- Say **"hi"** or **"hello"** - Get a greeting
- Say **"help"** - See my capabilities  
- Say **"test"** - Check connection status
- Say **"thank you"** - Get a friendly response

**Coming in Phase 2:**
✅ Full AI-powered responses using GROQ
✅ Answer any question you have
✅ Intelligent, context-aware assistance

**Coming in Phase 3+:**
✅ Search through your actual course materials
✅ Provide answers based on course content
✅ Citation of sources from your materials

For now, try one of the commands above to test the connection! 🚀"""

        logging.info(f"[CHAT TEST] Response generated, length={len(answer)}")
        
        return {
            "success": True,
            "answer": answer,
            "mode": "test",
            "phase": 1,
            "course_id": req.course_id,
            "course_name": req.course_name
        }
        
    except Exception as e:
        logging.error(f"[CHAT TEST ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail="Chat test service error"
        )

# -------------------------------------------------
# CHAT (RAG) - Phase 2+
# -------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Handle student questions using RAG (Retrieval-Augmented Generation)
    Phase 2+: Full AI with course content search
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
            "answer": answer,
            "mode": "rag",
            "phase": 2
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