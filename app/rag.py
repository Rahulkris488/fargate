from fastapi import UploadFile
from app.qdrant_client import client
from app.embeddings import embed_text

async def ingest_file(course_id: int, chapter_id: int, file: UploadFile):
    # Read file
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    # Split text into chunks (simple split for now)
    chunks = text.split("\n\n")

    vectors = []
    payloads = []

    for chunk in chunks:
        if chunk.strip():
            emb = embed_text(chunk)
            vectors.append(emb)
            payloads.append({
                "text": chunk,
                "course_id": course_id,
                "chapter_id": chapter_id
            })

    collection_name = f"course_{course_id}_chunks"

    client.upsert(
        collection_name=collection_name,
        points=vectors,
        payloads=payloads
    )

    return {"chunks": len(vectors)}
