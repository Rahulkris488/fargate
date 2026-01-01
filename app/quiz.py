import json
from app.embeddings import llm

def generate_quiz(course_id: int, topic: str, count: int, content: str):
    # -----------------------------
    # Hard validation
    # -----------------------------
    if course_id <= 0:
        raise ValueError("Invalid course_id")

    if not content or len(content.strip()) < 50:
        raise ValueError("Content is too short to generate a quiz")

    if count < 1 or count > 50:
        raise ValueError("Invalid number of questions requested")

    # -----------------------------
    # Prompt (STRICT JSON)
    # -----------------------------
    prompt = f"""
You are an expert exam question setter.

Generate EXACTLY {count} multiple‑choice questions in STRICT JSON ONLY.
DO NOT include any text before or after JSON.

JSON SCHEMA (must match exactly):
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
- Use ONLY the provided content
- Exactly ONE correct answer
- No repetition
- No hallucination
- No markdown
- No additional keys

CONTENT:
\"\"\"
{content[:6000]}
\"\"\"
"""

    raw = llm(prompt)

    # -----------------------------
    # Strict JSON validation
    # -----------------------------
    try:
        data = json.loads(raw)

        if not isinstance(data, list):
            raise ValueError("Quiz output is not a list")

        if len(data) != count:
            raise ValueError(
                f"Expected {count} questions, got {len(data)}"
            )

        for idx, q in enumerate(data, start=1):
            if not isinstance(q, dict):
                raise ValueError(f"Question {idx} is not an object")

            required_keys = {"question", "options", "correct_answer", "explanation"}
            if set(q.keys()) != required_keys:
                raise ValueError(f"Invalid keys in question {idx}")

            if not isinstance(q["options"], dict):
                raise ValueError(f"Options must be object in question {idx}")

            if set(q["options"].keys()) != {"A", "B", "C", "D"}:
                raise ValueError(f"Invalid options in question {idx}")

            if q["correct_answer"] not in {"A", "B", "C", "D"}:
                raise ValueError(f"Invalid correct_answer in question {idx}")

        return data

    except Exception as e:
        raise ValueError(f"Invalid AI output: {e}")
