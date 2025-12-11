from app.qdrant_client import client
from app.embeddings import embed_text, llm

async def rag_answer(course_id: int, question: str):
    query_emb = embed_text(question)

    results = client.search(
        collection_name=f"course_{course_id}_chunks",
        query_vector=query_emb,
        limit=5
    )

    context = "\n\n".join([r.payload.get("text", "") for r in results])

    prompt = f"""
    You are a helpful Moodle course assistant.
    Use ONLY the following course context to answer:

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Answer clearly and simply.
    """

    return llm(prompt)
