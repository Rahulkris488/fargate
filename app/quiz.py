import logging
from app.embeddings import llm

logger = logging.getLogger(__name__)

def generate_quiz(course_id: int, topic: str, count: int, content: str):
    if not content:
        raise ValueError("Content is required for quiz generation")

    logger.info("[QUIZ] Generating quiz")
    logger.info(f"[QUIZ] course_id={course_id}, topic={topic}, count={count}")
    logger.info(f"[QUIZ] content_length={len(content)}")

    prompt = f"""
You are an expert UPSC educator.

Generate exactly {count} multiple-choice questions using ONLY the content below.

Rules:
- Each question must be factual
- 4 options (A, B, C, D)
- One correct answer
- Output STRICT JSON ARRAY ONLY
- No markdown, no explanation text

CONTENT:
\"\"\"
{content[:6000]}
\"\"\"
"""

    return llm(prompt)
