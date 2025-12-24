from app.embeddings import llm

def generate_quiz(course_id: int, topic: str, count: int, content: str):
    print("\n[QUIZ] START")
    print(f"[QUIZ] course_id={course_id}")
    print(f"[QUIZ] topic={topic}")
    print(f"[QUIZ] count={count}")
    print(f"[QUIZ] content length={len(content) if content else 0}")

    if not content or len(content.strip()) < 50:
        return {
            "error": "Content too short or missing for quiz generation"
        }

    prompt = f"""
You are an expert educator.

Using ONLY the content below, generate {count} multiple choice questions.

Rules:
- Each question must be strictly based on the content
- 4 options labeled A, B, C, D
- Clearly indicate the correct option
- DO NOT invent facts
- Output plain text (NOT JSON)

CONTENT:
\"\"\"
{content[:6000]}
\"\"\"
"""

    try:
        result = llm(prompt)
        print("[QUIZ] LLM success")
        return result
    except Exception as e:
        print("[QUIZ ERROR]", str(e))
        return {
            "error": "LLM generation failed",
            "detail": str(e)
        }
