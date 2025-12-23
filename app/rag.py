from fastapi import UploadFile
from app.qdrant_client import client
from app.embeddings import embed_text, llm

# =====================================================
# RAG ANSWER
# =====================================================
async def rag_answer(course_id: int, question: str):
    print("\n[RAG] START")
    print(f"[RAG] course_id={course_id}")
    print(f"[RAG] question={question}")

    query_emb = embed_text(question)
    print(f"[RAG] Embedding generated ({len(query_emb)})")

    collection = f"course_{course_id}_chunks"

    try:
        results = client.query_points(
            collection_name=collection,
            query=query_emb,
            limit=5
        )
    except Exception as e:
        print(f"[RAG ERROR] Qdrant query failed: {e}")
        raise

    context = "\n\n".join(p.payload.get("text", "") for p in results.points)
    print(f"[RAG] Context size = {len(context)}")

    prompt = f"""
You are an AI Moodle assistant. Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{question}

Respond clearly and concisely.
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

    chunks = text.split("\n\n")
    points = []

    for idx, chunk in enumerate(chunks):
        if chunk.strip():
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

    collection = f"course_{course_id}_chunks"
    client.upsert(collection_name=collection, points=points)

    return {"chunks": len(points)}
