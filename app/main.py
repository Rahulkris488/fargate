from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import rag_answer
from app.quiz import generate_quiz

app = FastAPI()

class ChatRequest(BaseModel):
    course_id: int
    question: str

class QuizRequest(BaseModel):
    course_id: int
    topic: str
    count: int = 5

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
