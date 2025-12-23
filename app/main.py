from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from app.rag import rag_answer, ingest_file
from app.quiz import generate_quiz

app = FastAPI()

class ChatRequest(BaseModel):
    course_id: int
    question: str

class QuizRequest(BaseModel):
    course_id: int
    chapter_id: int | None = None
    topic: str
    num_questions: int = 5
    difficulty: str | None = "medium"
    content: str | None = None


@app.get("/")
def root():
    return {"message": "Moodle AI Backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    answer = await rag_answer(req.course_id, req.question)
    return {"answer": answer}

@app.post("/generate-quiz")
def quiz(req: QuizRequest):
    quiz = generate_quiz(
        course_id=req.course_id,
        topic=req.topic,
        count=req.num_questions,
        content=req.content
    )
    return {"quiz": quiz}


@app.post("/ingest")
async def ingest(
    course_id: int = Form(...),
    chapter_id: int = Form(...),
    file: UploadFile = File(...)
):
    result = await ingest_file(course_id, chapter_id, file)
    return {"message": "Ingestion successful", "detail": result}
