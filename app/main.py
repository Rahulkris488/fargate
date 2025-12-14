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
    topic: str
    count: int = 5

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
    quiz = generate_quiz(req.course_id, req.topic, req.count)
    return {"quiz": quiz}

@app.post("/ingest")
async def ingest(
    course_id: int = Form(...),
    chapter_id: int = Form(...),
    file: UploadFile = File(...)
):
    result = await ingest_file(course_id, chapter_id, file)
    return {"message": "Ingestion successful", "detail": result}
