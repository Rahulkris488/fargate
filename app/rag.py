from fastapi import UploadFile
from app.qdrant_client import client
from app.embeddings import embed_text, llm

# ---------------------------
# RAG ANSWER
# ---------------------------
async def rag_answer(course_id: int, question: str):

    print("\n[RAG] rag_answer() START")
    print(f"[RAG] course_id={course_id}")
    print(f"[RAG] question={question}")

    query_emb = embed_text(question)
    print(f"[RAG] Embedding generated, length={len(query_emb)}")

    collection = f"course_{course_id}_chunks"
    print(f"[RAG] Searching in collection: {collection}")

    try:
        results = client.query_points(
            collection_name=collection,
            query=query_emb,
            limit=5
        )
        print(f"[RAG] Qdrant returned {len(results.points)} results")
    except Exception as e:
        print(f"[RAG][ERROR] Qdrant search failed: {e}")
        raise

    # Extract text from payloads
    context_list = []
    for p in results.points:
        text_chunk = p.payload.get("text", "")
        context_list.append(text_chunk)

    context = "\n\n".join(context_list)
    print(f"[RAG] Context length: {len(context)}")

    prompt = f"""
You are a Moodle assistant. Use ONLY this context:

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and simply.
    """

    print("[RAG] Sending prompt to LLM")
    answer = llm(prompt)

    print("[RAG] LLM response received")
    return answer


# ---------------------------
# INGESTION
# ---------------------------
async def ingest_file(course_id: int, chapter_id: int, file: UploadFile):

    print("\n[INGEST] ingest_file() START")
    print(f"[INGEST] course_id={course_id}, chapter_id={chapter_id}")
    print(f"[INGEST] filename={file.filename}")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    print(f"[INGEST] File size={len(text)} chars")

    chunks = text.split("\n\n")
    print(f"[INGEST] Total chunks found: {len(chunks)}")

    points = []

    for idx, chunk in enumerate(chunks):
        if chunk.strip():
            print(f"[INGEST] Processing chunk #{idx}, length={len(chunk)}")

            emb = embed_text(chunk)
            print(f"[INGEST] Embedding generated for chunk #{idx}")

            points.append({
                "id": idx,
                "vector": emb,
                "payload": {
                    "text": chunk,
                    "course_id": course_id,
                    "chapter_id": chapter_id
                }
            })

    collection = f"course_{course_id}_chunks"
    print(f"[INGEST] Upserting into collection: {collection}")

    try:
        client.upsert(
            collection_name=collection,
            points=points
        )
        print("[INGEST] Qdrant upsert successful")
    except Exception as e:
        print(f"[INGEST][ERROR] Qdrant upsert failed: {e}")
        raise

    return {"chunks": len(points)}
