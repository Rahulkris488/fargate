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
# CHUNK TEXT
# =====================================================
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        
        if chunk and len(chunk) > 50:  # Skip very short chunks
            chunks.append(chunk)
        
        start = end - overlap
        
        if start >= text_len:
            break
    
    return chunks

# =====================================================
# 🆕 NEW: INDEX COURSE CONTENT
# =====================================================
async def index_course_content(course_id: int, course_name: str, documents: list):
    """
    Index course content from Moodle into Qdrant.
    
    Args:
        course_id: Moodle course ID
        course_name: Course name
        documents: List of dicts with keys: type, source, content
    
    Returns:
        dict with success status and stats
    """
    print(f"\n[INDEX] Starting indexing for course {course_id}")
    print(f"[INDEX] Course name: {course_name}")
    print(f"[INDEX] Documents to process: {len(documents)}")
    
    collection_name = f"course_{course_id}_chunks"
    
    # Create or recreate collection (avoids duplicates)
    try:
        # Delete if exists
        try:
            client.delete_collection(collection_name)
            print(f"[INDEX] Deleted old collection: {collection_name}")
        except:
            pass
        
        # Create fresh collection
        ensure_collection_exists(collection_name)
        
    except Exception as e:
        raise ValueError(f"Failed to create collection: {str(e)}")
    
    # Process all documents
    all_points = []
    point_id = 0
    
    for doc_idx, doc in enumerate(documents):
        content = doc.get('content', '').strip()
        source = doc.get('source', 'Unknown')
        doc_type = doc.get('type', 'text')
        
        if not content or len(content) < 50:
            print(f"[INDEX] Skipping empty/short document {doc_idx}")
            continue
        
        # Chunk the content
        chunks = chunk_text(content, chunk_size=1000, overlap=200)
        print(f"[INDEX] Document {doc_idx} ({doc_type}): {len(chunks)} chunks from '{source}'")
        
        for chunk_idx, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = embed_text(chunk)
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "course_id": course_id,
                        "course_name": course_name,
                        "source": source,
                        "type": doc_type,
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx
                    }
                )
                
                all_points.append(point)
                point_id += 1
                
            except Exception as e:
                print(f"[INDEX ERROR] Failed to process chunk {chunk_idx} of doc {doc_idx}: {e}")
                continue
    
    if not all_points:
        raise ValueError("No valid content to index")
    
    # Upsert all points
    print(f"[INDEX] Upserting {len(all_points)} points to {collection_name}")
    
    try:
        client.upsert(
            collection_name=collection_name,
            points=all_points
        )
        print(f"[INDEX] Successfully indexed {len(all_points)} chunks")
        
    except Exception as e:
        raise ValueError(f"Failed to store vectors: {str(e)}")
    
    return {
        "success": True,
        "course_id": course_id,
        "course_name": course_name,
        "documents_processed": len(documents),
        "chunks_indexed": len(all_points),
        "collection": collection_name
    }

# =====================================================
# 🆕 NEW: GET COURSE STATUS
# =====================================================
async def get_course_status(course_id: int):
    """
    Check if a course has been indexed and get stats.
    
    Args:
        course_id: Moodle course ID
    
    Returns:
        dict with indexing status and stats
    """
    collection_name = f"course_{course_id}_chunks"
    
    try:
        collection_info = client.get_collection(collection_name)
        
        return {
            "course_id": course_id,
            "indexed": True,
            "chunks": collection_info.points_count,
            "collection": collection_name,
            "message": f"Course has {collection_info.points_count} indexed chunks"
        }
        
    except Exception:
        return {
            "course_id": course_id,
            "indexed": False,
            "chunks": 0,
            "collection": collection_name,
            "message": "Course has not been indexed yet"
        }

# =====================================================
# RAG ANSWER - Updated to handle missing collections
# =====================================================
async def rag_answer(course_id: int, question: str):
    print("\n[RAG] START")
    print(f"[RAG] course_id={course_id}")
    print(f"[RAG] question={question}")

    collection = f"course_{course_id}_chunks"
    
    # Check if collection exists and has content
    try:
        collection_info = client.get_collection(collection)
        if collection_info.points_count == 0:
            return (
                "⚠️ This course content has not been indexed yet. "
                "Please contact your administrator to enable AI support for this course."
            )
    except Exception:
        return (
            "⚠️ This course content has not been indexed yet. "
            "Please contact your administrator to enable AI support for this course."
        )

    # Generate embedding
    query_emb = embed_text(question)
    print(f"[RAG] Embedding generated (dim={len(query_emb)})")

    # Search using query_points
    try:
        results = client.query_points(
            collection_name=collection,
            query=query_emb,
            limit=5
        )
        results = results.points
        print(f"[RAG] Found {len(results)} results")
    except Exception as e:
        print(f"[RAG ERROR] Search failed: {e}")
        raise

    if not results:
        return "I couldn't find relevant information in the course materials to answer your question. Could you rephrase or ask something else?"

    # Build context
    context = "\n\n---\n\n".join(
        f"Source: {hit.payload.get('source', 'Unknown')}\n{hit.payload.get('text', '')}" 
        for hit in results
    )

    print(f"[RAG] Context size = {len(context)} chars")

    # Generate answer
    prompt = f"""You are an AI tutor for a Moodle course. Answer the student's question using ONLY the course materials provided below.

COURSE MATERIALS:
{context}

STUDENT QUESTION:
{question}

INSTRUCTIONS:
- Give a clear, helpful answer based on the materials
- If the materials don't contain enough info, say so politely
- Be conversational and student-friendly
- Keep your answer focused and concise (2-4 paragraphs max)

ANSWER:"""

    answer = llm(prompt)
    print("[RAG] LLM response generated")
    return answer

# =====================================================
# ✅ UNCHANGED: INGEST FILE
# =====================================================
async def ingest_file(course_id: int, chapter_id: int, file: UploadFile):
    print("\n[INGEST] START")
    print(f"[INGEST] course_id={course_id}, chapter_id={chapter_id}, file={file.filename}")

    collection = f"course_{course_id}_chunks"

    # Ensure collection exists
    ensure_collection_exists(collection)

    # Read file
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")

    # Split into chunks
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    
    if not chunks:
        raise ValueError("No valid text chunks found in file")

    print(f"[INGEST] Processing {len(chunks)} chunks")

    # Create points
    points = []
    for idx, chunk in enumerate(chunks):
        emb = embed_text(chunk)
        
        points.append(
            PointStruct(
                id=chapter_id * 10000 + idx,
                vector=emb,
                payload={
                    "text": chunk,
                    "course_id": course_id,
                    "chapter_id": chapter_id,
                    "source": file.filename
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