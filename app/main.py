from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from app.quiz import generate_quiz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class QuizRequest(BaseModel):
    course_id: int
    topic: str
    num_questions: int = 5
    content: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate-quiz")
def quiz(req: QuizRequest):
    try:
        logger.info("[API] /generate-quiz called")
        result = generate_quiz(
            course_id=req.course_id,
            topic=req.topic,
            count=req.num_questions,
            content=req.content
        )
        return {"quiz": result}
    except Exception as e:
        logger.exception("[API ERROR]")
        raise HTTPException(status_code=500, detail=str(e))
