from fastapi import UploadFile
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.qdrant_client import client
from app.embeddings import embed_text, llm
import logging

VECTOR_SIZE = 384  # must match MiniLM

# =========================
# COLLECTION MANAGEMENT
# =========================
def ensure_collection_exists(name: str):
    collections = [c.name for c in client.get_collections().collections]
    if name not in collections:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

# =========================
# TEXT CHUNKING
# =========================
def chunk_text(text, size=1000, overlap=200):
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
    collection = f"course_{course_id}_chunks"

    try:
        client.delete_collection(collection)
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
    collection = f"course_{course_id}_chunks"
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
# RAG ANSWER
# =========================
async def rag_answer(course_id, question):
    collection = f"course_{course_id}_chunks"

    try:
        client.get_collection(collection)
    except:
        return "This course has not been indexed yet."

    query_emb = embed_text(question)
    hits = client.query_points(
        collection_name=collection,
        query=query_emb,
        limit=5
    ).points

    if not hits:
        return "No relevant content found in course materials."

    context = "\n\n".join(h.payload["text"] for h in hits)

    prompt = f"""
You are an AI tutor. Answer ONLY using the course material.

COURSE MATERIAL:
{context}

QUESTION:
{question}

ANSWER:
"""

    return llm(prompt)

# =========================
# LEGACY INGEST (QUIZ SAFE)
# =========================
async def ingest_file(course_id, chapter_id, file: UploadFile):
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
    return {"chunks": len(points)}
