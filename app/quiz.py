import json
import re
from app.embeddings import llm

REQUIRED_KEYS = {"question", "options", "correct_answer", "explanation"}
OPTION_KEYS = {"A", "B", "C", "D"}

def generate_quiz(course_id: int, topic: str, count: int, content: str):
    if not content or not content.strip():
        raise ValueError("Content is empty")

    prompt = f"""You are an expert exam question setter.

Generate EXACTLY {count} MCQs in STRICT JSON ONLY. NO text before or after JSON.

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
- Output MUST be valid JSON array

CONTENT:
\"\"\"
{content[:6000]}
\"\"\"
"""

    raw = llm(prompt)
    
    try:
        # Clean the response - remove markdown code blocks if present
        cleaned = raw.strip()
        
        # Remove ```json ... ``` if present
        if cleaned.startswith("```"):
            # Find the actual JSON content
            json_match = re.search(r'```(?:json)?\s*(\[.*\])\s*```', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1)
            else:
                # Try to extract anything between ``` markers
                cleaned = re.sub(r'```(?:json)?', '', cleaned).strip()
        
        # Try to parse JSON
        data = json.loads(cleaned)
        
        if not isinstance(data, list):
            raise ValueError("Quiz output must be a list")
        
        # More lenient count check - allow ±20% variance
        min_count = max(1, int(count * 0.8))
        max_count = int(count * 1.2)
        
        if not (min_count <= len(data) <= max_count):
            # Log the issue but continue if we have at least min_count questions
            print(f"⚠️ Warning: Expected {count} questions, got {len(data)}")
            if len(data) < min_count:
                raise ValueError(f"Too few questions: expected {count}, got {len(data)}")
        
        # Validate each question
        validated_questions = []
        for i, q in enumerate(data):
            try:
                # Check required keys
                if not REQUIRED_KEYS.issubset(q.keys()):
                    missing = REQUIRED_KEYS - set(q.keys())
                    raise ValueError(f"Missing keys: {missing}")
                
                # Validate options
                if not isinstance(q["options"], dict):
                    raise ValueError("Options must be an object")
                
                if set(q["options"].keys()) != OPTION_KEYS:
                    # Try to fix common issues
                    options = q["options"]
                    if all(k in options for k in ["a", "b", "c", "d"]):
                        # Convert lowercase to uppercase
                        q["options"] = {
                            "A": options["a"],
                            "B": options["b"],
                            "C": options["c"],
                            "D": options["d"]
                        }
                    else:
                        raise ValueError(f"Options must be A-D, got: {list(q['options'].keys())}")
                
                # Validate correct answer
                correct = q["correct_answer"].upper()
                if correct not in OPTION_KEYS:
                    raise ValueError(f"Invalid correct_answer: {q['correct_answer']}")
                q["correct_answer"] = correct  # Normalize to uppercase
                
                # Add validated question
                validated_questions.append(q)
                
            except Exception as qe:
                print(f"⚠️ Question {i+1} validation error: {qe}")
                # Skip invalid questions instead of failing completely
                continue
        
        # Final check - do we have enough valid questions?
        if len(validated_questions) < min_count:
            raise ValueError(f"Only {len(validated_questions)} valid questions out of {len(data)}")
        
        # Return only the number of questions requested (or what we have)
        return validated_questions[:count]
        
    except json.JSONDecodeError as je:
        # Better error message with the actual problematic content
        error_context = raw[:500] if len(raw) > 500 else raw
        raise ValueError(f"Invalid JSON from AI: {je}\n\nReceived:\n{error_context}")
    
    except Exception as e:
        raise ValueError(f"Invalid AI output: {e}")