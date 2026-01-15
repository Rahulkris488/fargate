from fastapi import UploadFile
from app.embeddings import embed_text, llm
import logging

logger = logging.getLogger(__name__)

# =========================
# QDRANT CLIENT (OPTIONAL)
# =========================
try:
    from app.qdrant_client import client
    QDRANT_AVAILABLE = client is not None
    
    if QDRANT_AVAILABLE:
        from qdrant_client.models import Distance, VectorParams, PointStruct
        logger.info("[RAG] ✅ Qdrant client available")
    else:
        logger.warning("[RAG] ⚠️ Qdrant client is None")
        
except Exception as e:
    logger.warning(f"[RAG] ⚠️ Qdrant not available: {e}")
    logger.warning("[RAG] Will use AI-only fallback mode")
    QDRANT_AVAILABLE = False
    client = None

VECTOR_SIZE = 384  # must match MiniLM

# =========================
# COLLECTION MANAGEMENT
# =========================
def ensure_collection_exists(name: str):
    """Create Qdrant collection if it doesn't exist"""
    if not QDRANT_AVAILABLE:
        raise RuntimeError("Qdrant is not available")
    
    collections = [c.name for c in client.get_collections().collections]
    if name not in collections:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        logger.info(f"[RAG] Created collection: {name}")

# =========================
# TEXT CHUNKING
# =========================
def chunk_text(text, size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# =========================
# INDEX COURSE CONTENT
# =========================
async def index_course_content(course_id, course_name, documents):
    """
    Index course documents into Qdrant
    """
    if not QDRANT_AVAILABLE:
        raise RuntimeError("Qdrant is not available. Cannot index content.")
    
    collection = f"course_{course_id}_chunks"

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection)
        logger.info(f"[RAG] Deleted existing collection: {collection}")
    except:
        pass

    ensure_collection_exists(collection)

    points = []
    pid = 0
    total_chars = 0

    for d in documents:
        content = d.get("content", "")
        if len(content) < 50:
            continue

        total_chars += len(content)
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            points.append(
                PointStruct(
                    id=pid,
                    vector=emb,
                    payload={
                        "text": chunk,
                        "course_id": course_id,
                        "course_name": course_name,
                        "source": d.get("source", "unknown"),
                        "type": d.get("type", "text"),
                    }
                )
            )
            pid += 1

    if not points:
        raise ValueError("No valid content to index")

    client.upsert(collection_name=collection, points=points)
    
    logger.info(f"[RAG] ✅ Indexed {len(points)} chunks for course {course_id}")

    return {
        "course_id": course_id,
        "course_name": course_name,
        "chunks_indexed": len(points),
        "total_content_chars": total_chars,
        "collection": collection
    }

# =========================
# COURSE STATUS
# =========================
async def get_course_status(course_id):
    """Check if a course has been indexed"""
    collection = f"course_{course_id}_chunks"
    
    if not QDRANT_AVAILABLE:
        return {
            "course_id": course_id,
            "indexed": False,
            "chunks": 0,
            "collection": collection,
            "message": "Qdrant not available"
        }
    
    try:
        info = client.get_collection(collection)
        return {
            "course_id": course_id,
            "indexed": True,
            "chunks": info.points_count,
            "collection": collection
        }
    except:
        return {
            "course_id": course_id,
            "indexed": False,
            "chunks": 0,
            "collection": collection
        }

# =========================
# RAG ANSWER (WITH AI FALLBACK)
# =========================
async def rag_answer(course_id, question):
    """
    Answer question using RAG if available, otherwise AI-only
    """
    # If Qdrant not available, use AI-only mode
    if not QDRANT_AVAILABLE:
        logger.info(f"[RAG] Using AI-only mode (Qdrant not available)")
        prompt = f"""
You are an AI tutor helping a student.

QUESTION:
{question}

Please provide a clear, helpful answer based on your knowledge.
"""
        return llm(prompt)
    
    # Try to use RAG
    collection = f"course_{course_id}_chunks"

    try:
        client.get_collection(collection)
    except:
        # Course not indexed - use AI-only mode
        logger.info(f"[RAG] Course {course_id} not indexed, using AI-only mode")
        prompt = f"""
You are an AI tutor helping a student.

QUESTION:
{question}

Please provide a clear, helpful answer based on your knowledge.
"""
        return llm(prompt)

    # Query vector database
    try:
        query_emb = embed_text(question)
        hits = client.query_points(
            collection_name=collection,
            query=query_emb,
            limit=5
        ).points

        if not hits:
            logger.info(f"[RAG] No relevant content found, using AI-only")
            prompt = f"""
You are an AI tutor helping a student.

QUESTION:
{question}

Please provide a clear, helpful answer.
"""
            return llm(prompt)

        # Build context from retrieved chunks
        context = "\n\n".join(h.payload["text"] for h in hits)

        prompt = f"""
You are an AI tutor. Answer ONLY using the course material provided below.

COURSE MATERIAL:
{context}

QUESTION:
{question}

ANSWER:
"""
        logger.info(f"[RAG] ✅ Using RAG mode with {len(hits)} context chunks")
        return llm(prompt)
        
    except Exception as e:
        logger.error(f"[RAG ERROR] {e}")
        prompt = f"""
You are an AI tutor helping a student.

QUESTION:
{question}

Please provide a clear, helpful answer.
"""
        return llm(prompt)

# =========================
# LEGACY INGEST (QUIZ SAFE)
# =========================
async def ingest_file(course_id, chapter_id, file: UploadFile):
    """
    Legacy file ingestion endpoint
    """
    if not QDRANT_AVAILABLE:
        raise RuntimeError("Qdrant is not available. Cannot ingest files.")
    
    collection = f"course_{course_id}_chunks"
    ensure_collection_exists(collection)

    text = (await file.read()).decode("utf-8", errors="ignore")
    chunks = chunk_text(text)

    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=chapter_id * 10000 + i,
                vector=embed_text(chunk),
                payload={"text": chunk}
            )
        )

    client.upsert(collection_name=collection, points=points)
    
    logger.info(f"[INGEST] ✅ Ingested {len(points)} chunks for course {course_id}")
    
    return {"chunks": len(points)}