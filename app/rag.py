from fastapi import UploadFile
from qdrant_client.http import models
from app.qdrant_client import client
from app.embeddings import embed_text, llm

VECTOR_SIZE = 384  # MUST match embedding model


# =====================================================
# RAG ANSWER
# =====================================================
async def rag_answer(course_id: int, question: str):
    print("\n[RAG] START")
    print(f"[RAG] course_id={course_id}")
    print(f"[RAG] question={question}")

    collection = f"course_{course_id}_chunks"

    # Ensure collection exists
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        raise ValueError(
            f"No knowledge base found for course {course_id}. "
            f"Please ingest course content first."
        )

    query_emb = embed_text(question)
    print(f"[RAG] Embedding generated ({len(query_emb)})")

    results = client.query_points(
        collection_name=collection,
        query=query_emb,
        limit=5
    )

    context = "\n\n".join(
        p.payload.get("text", "") for p in results.points
    )

    print(f"[RAG] Context size = {len(context)}")

    prompt = f"""
You are an AI Moodle assistant.
Answer ONLY using the context below.

CONTEXT:
{context}

QUESTION:
{question}

Give a clear, student-friendly answer.
"""

    answer = llm(prompt)
    print("[RAG] LLM response OK")
    return answer


# =====================================================
# INGEST
# =====================================================
async def ingest_file(course_id: int, chapter_id: int, file: UploadFile):
    print("\n[INGEST] START")
    print(f"[INGEST] course_id={course_id}, chapter_id={chapter_id}, file={file.filename}")

    collection = f"course_{course_id}_chunks"

    # -------------------------------------------------
    # CREATE COLLECTION IF NOT EXISTS
    # -------------------------------------------------
    collections = [c.name for c in client.get_collections().collections]

    if collection not in collections:
        print(f"[INGEST] Creating collection {collection}")

        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

    # -------------------------------------------------
    # READ FILE
    # -------------------------------------------------
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    points = []

    for idx, chunk in enumerate(chunks):
        emb = embed_text(chunk)

        points.append(
            models.PointStruct(
                id=f"{chapter_id}_{idx}",
                vector=emb,
                payload={
                    "text": chunk,
                    "course_id": course_id,
                    "chapter_id": chapter_id
                }
            )
        )

    client.upsert(
        collection_name=collection,
        points=points
    )

    print(f"[INGEST] Stored {len(points)} chunks")
    return {"chunks": len(points)}
