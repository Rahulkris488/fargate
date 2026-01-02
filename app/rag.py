from fastapi import UploadFile
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.qdrant_client import client
from app.embeddings import embed_text, llm

VECTOR_SIZE = 384  # MUST match SentenceTransformer all-MiniLM-L6-v2

# =====================================================
# HELPER: Ensure Collection Exists
# =====================================================
def ensure_collection_exists(collection_name: str):
    """
    Creates collection if it doesn't exist.
    Safe to call multiple times.
    """
    try:
        collections = [c.name for c in client.get_collections().collections]
        
        if collection_name not in collections:
            print(f"[QDRANT] Creating collection: {collection_name}")
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"[QDRANT] Collection created: {collection_name}")
        else:
            print(f"[QDRANT] Collection already exists: {collection_name}")
            
    except Exception as e:
        print(f"[QDRANT ERROR] Failed to ensure collection: {e}")
        raise


# =====================================================
# RAG ANSWER
# =====================================================
async def rag_answer(course_id: int, question: str):
    print("\n[RAG] START")
    print(f"[RAG] course_id={course_id}")
    print(f"[RAG] question={question}")

    collection = f"course_{course_id}_chunks"
    
    # Ensure collection exists (safe to call always)
    try:
        ensure_collection_exists(collection)
    except Exception as e:
        return f"Knowledge base error: {str(e)}"

    # Check if collection has content
    try:
        collection_info = client.get_collection(collection)
        if collection_info.points_count == 0:
            return (
                "No content has been uploaded for this course yet. "
                "Please ask your instructor to upload course materials."
            )
    except Exception:
        return "Knowledge base not available. Please contact support."

    # Generate embedding
    query_emb = embed_text(question)
    print(f"[RAG] Embedding generated (dim={len(query_emb)})")

    # Search
    try:
        results = client.search(
            collection_name=collection,
            query_vector=query_emb,
            limit=5
        )
    except Exception as e:
        print(f"[RAG ERROR] Search failed: {e}")
        raise

    if not results:
        return "I couldn't find relevant information to answer your question."

    # Build context
    context = "\n\n".join(
        hit.payload.get("text", "") for hit in results
    )

    print(f"[RAG] Context size = {len(context)}")

    # Generate answer
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

    # CRITICAL: Ensure collection exists BEFORE any operations
    ensure_collection_exists(collection)

    # Read file
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")

    # Split into chunks
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    
    if not chunks:
        raise ValueError("No valid text chunks found in file")

    print(f"[INGEST] Processing {len(chunks)} chunks")

    # Create points
    points = []
    for idx, chunk in enumerate(chunks):
        emb = embed_text(chunk)
        
        points.append(
            PointStruct(
                id=f"{chapter_id}_{idx}",
                vector=emb,
                payload={
                    "text": chunk,
                    "course_id": course_id,
                    "chapter_id": chapter_id
                }
            )
        )

    # Upsert to Qdrant
    print(f"[INGEST] Upserting {len(points)} points to collection {collection}")
    
    client.upsert(
        collection_name=collection,
        points=points
    )

    print(f"[INGEST] Successfully stored {len(points)} chunks")
    return {"chunks": len(points)}