from fastapi import UploadFile
from qdrant_client.http.models import VectorParams, Distance
from app.qdrant_client import client
from app.embeddings import embed_text, llm

# =====================================================
# RAG ANSWER
# =====================================================
async def rag_answer(course_id: int, question: str):
    print("\n[RAG] START")
    print(f"[RAG] course_id={course_id}")
    print(f"[RAG] question={question}")

    # Generate query embedding (384-dim)
    query_emb = embed_text(question)
    dim = len(query_emb)
    print(f"[RAG] Embedding generated (dim={dim})")

    collection = f"course_{course_id}_chunks"

    # Ensure collection exists with correct dimension
    ensure_collection(collection, dim)

    try:
        results = client.query_points(
            collection_name=collection,
            query=query_emb,
            limit=5
        )
    except Exception as e:
        print(f"[RAG ERROR] Qdrant query failed: {e}")
        raise

    if not results.points:
        return "No relevant course material found yet. Please ingest content first."

    context = "\n\n".join(
        p.payload.get("text", "") for p in results.points
    )

    print(f"[RAG] Context size={len(context)}")

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
    if not chunks:
        raise ValueError("Uploaded file contains no usable text")

    # Generate one embedding to detect dimension (384)
    test_emb = embed_text(chunks[0])
    dim = len(test_emb)

    collection = f"course_{course_id}_chunks"
    ensure_collection(collection, dim)

    points = []
    for idx, chunk in enumerate(chunks):
        emb = embed_text(chunk)
        points.append({
            "id": idx,
            "vector": emb,
            "payload": {
                "text": chunk,
                "chapter_id": chapter_id,
                "course_id": course_id
            }
        })

    client.upsert(
        collection_name=collection,
        points=points
    )

    print(f"[INGEST] Stored {len(points)} chunks")
    return {"chunks": len(points)}


# =====================================================
# HELPERS
# =====================================================
def ensure_collection(collection_name: str, dim: int):
    """
    Ensures the Qdrant collection exists with the correct vector size.
    Recreates only if missing.
    """
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if collection_name in existing:
        print(f"[QDRANT] Collection exists: {collection_name}")
        return

    print(f"[QDRANT] Creating collection {collection_name} (dim={dim})")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE
        )
    )
