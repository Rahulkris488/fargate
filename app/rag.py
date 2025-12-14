from fastapi import UploadFile
from app.qdrant_client import client
from app.embeddings import embed_text, llm
from qdrant_client.http.models import PointStruct

# ---------------------------
# RAG ANSWER FUNCTION
# ---------------------------
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

# ---------------------------
# INGEST FILE FUNCTION
# ---------------------------
async def ingest_file(course_id: int, chapter_id: int, file: UploadFile):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    chunks = text.split("\n\n")

    points = []
    i = 0

    for chunk in chunks:
        if chunk.strip():
            emb = embed_text(chunk)

            points.append(
                PointStruct(
                    id=i,
                    vector=emb,
                    payload={
                        "text": chunk,
                        "course_id": course_id,
                        "chapter_id": chapter_id
                    }
                )
            )
            i += 1

    collection_name = f"course_{course_id}_chunks"

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    return {"chunks": len(points)}
