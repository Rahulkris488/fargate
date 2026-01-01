from fastapi import UploadFile
from qdrant_client.http.models import VectorParams, Distance
from app.qdrant_client import client
from app.embeddings import embed_text, llm

VECTOR_DIM = 384  # MUST match embedding model

# =====================================================
# UTIL: Ensure collection exists
# =====================================================
def ensure_collection(collection_name: str):
    collections = client.get_collections().collections
    if collection_name not in [c.name for c in collections]:
        print(f"[QDRANT] Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE
            )
        )
    else:
        print(f"[QDRANT] Collection exists: {collection_name}")

# =====================================================
# RAG ANSWER
# =====================================================
async def rag_answer(course_id: int, question: str):
    print("\n[RAG] START")
    print(f"[RAG] course_id={course_id}")
    print(f"[RAG] question={question}")

    collection = f"course_{course_id}_chunks"

    # Ensure collection exists
    ensure_collection(collection)

    # Embed query
    query_emb = embed_text(question)
    print(f"[RAG] Embedding generated ({len(query_emb)})")

    # Query Qdrant
    results = client.query_points(
        collection_name=collection,
        query=query_emb,
        limit=5
    )

    if not results.points:
        return "I don't have enough information yet. Please upload course content."

    context = "\n\n".join(
        p.payload.get("text", "") for p in results.points
    )

    print(f"[RAG] Context size = {len(context)}")

    prompt = f"""
You are an AI Moodle assistant.
Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and concisely.
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

    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    print(f"[INGEST] Chunks found: {len(chunks)}")

    collection = f"course_{course_id}_chunks"

    # Ensure collection exists BEFORE upsert
    ensure_collection(collection)

    points = []
    for idx, chunk in enumerate(chunks):
        emb = embed_text(chunk)
        points.append({
            "id": f"{chapter_id}_{idx}",
            "vector": emb,
            "payload": {
                "text": chunk,
                "course_id": course_id,
                "chapter_id": chapter_id
            }
        })

    client.upsert(
        collection_name=collection,
        points=points
    )

    print("[INGEST] Upsert complete")
    return {"chunks": len(points)}
