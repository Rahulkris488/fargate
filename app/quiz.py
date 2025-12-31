import json
from app.embeddings import llm

def generate_quiz(course_id: int, topic: str, count: int, content: str):
    if not content or not content.strip():
        raise ValueError("Content is empty")

    prompt = f"""
You are an expert exam question setter.

Generate EXACTLY {count} MCQs in STRICT JSON ONLY.
NO text before or after JSON.

JSON SCHEMA:
[
  {{
    "question": "string",
    "options": {{
      "A": "string",
      "B": "string",
      "C": "string",
      "D": "string"
    }},
    "correct_answer": "A|B|C|D",
    "explanation": "string"
  }}
]

RULES:
- Use ONLY the content below
- One correct answer
- No hallucination
- No markdown

CONTENT:
\"\"\"
{content[:6000]}
\"\"\"
"""

    raw = llm(prompt)

    try:
        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == count
        return data
    except Exception as e:
        raise ValueError(f"Invalid AI output: {e}")
