import json
from app.embeddings import llm

REQUIRED_KEYS = {"question", "options", "correct_answer", "explanation"}
OPTION_KEYS = {"A", "B", "C", "D"}

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

        if not isinstance(data, list):
            raise ValueError("Quiz output must be a list")

        if len(data) != count:
            raise ValueError("Quiz count mismatch")

        for i, q in enumerate(data):
            if not REQUIRED_KEYS.issubset(q.keys()):
                raise ValueError(f"Question {i+1}: missing keys")

            if not isinstance(q["options"], dict):
                raise ValueError(f"Question {i+1}: options must be object")

            if set(q["options"].keys()) != OPTION_KEYS:
                raise ValueError(f"Question {i+1}: options must be A–D")

            if q["correct_answer"] not in OPTION_KEYS:
                raise ValueError(f"Question {i+1}: invalid correct_answer")

        return data

    except Exception as e:
        raise ValueError(f"Invalid AI output: {e}")
