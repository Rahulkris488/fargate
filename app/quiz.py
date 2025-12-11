from app.embeddings import llm

def generate_quiz(course_id: int, topic: str, count: int):
    prompt = f"""
    Generate {count} high-quality multiple-choice questions
    for the topic: {topic}

    Each question should have:
    - 4 options (A, B, C, D)
    - Correct answer clearly labelled
    - Short explanation

    Format as JSON list.
    """

    return llm(prompt)
