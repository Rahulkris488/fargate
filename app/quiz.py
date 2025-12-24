from app.embeddings import llm

def generate_quiz(
    course_id: int,
    topic: str,
    count: int,
    content: str | None
):
    print("\n[QUIZ] START")
    print(f"[QUIZ] course_id={course_id}")
    print(f"[QUIZ] topic={topic}")
    print(f"[QUIZ] count={count}")

    if not content or not content.strip():
        raise ValueError("Quiz generation failed: content is empty")

    prompt = f"""
You are an expert exam question setter.

Using ONLY the content below, generate {count} MCQs.

Rules:
- Each question must be directly from content
- 4 options (A, B, C, D)
- Exactly ONE correct answer
- No placeholders
- Output STRICT JSON only

CONTENT:
\"\"\"
{content[:6000]}
\"\"\"
"""

    response = llm(prompt)
    print("[QUIZ] LLM response received")
    return response
