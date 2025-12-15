from app.embeddings import llm

def generate_quiz(course_id: int, topic: str, count: int):
    prompt = f"""
Generate {count} high-quality multiple-choice questions
on the topic: {topic}

Each question must have:
- 4 options (A, B, C, D)
- Correct answer labelled
- Short explanation

Format as JSON list.
    """

    return llm(prompt)
