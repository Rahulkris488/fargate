from app.embeddings import llm

def generate_quiz(course_id: int, topic: str, count: int, content: str | None):
    if not content:
        raise ValueError("No content provided for quiz generation")

    prompt = f"""
You are an expert educator.

Using ONLY the content below, generate {count} high-quality MCQs.

Rules:
- Each question must be based directly on the content
- 4 options (A, B, C, D)
- One correct answer
- No placeholders
- Output STRICT JSON array only

CONTENT:
\"\"\"
{content[:6000]}
\"\"\"
"""

    return llm(prompt)
