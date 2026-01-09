from fastapi import UploadFile
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.qdrant_client import client
from app.embeddings import embed_text, llm
import logging

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
            logging.info(f"[QDRANT] Creating collection: {collection_name}")
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            logging.info(f"[QDRANT] ✓ Collection created: {collection_name}")
        else:
            logging.info(f"[QDRANT] Collection already exists: {collection_name}")
            
    except Exception as e:
        logging.error(f"[QDRANT ERROR] Failed to ensure collection: {e}")
        raise

# =====================================================
# CHUNK TEXT
# =====================================================
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Split text into overlapping chunks for better semantic search
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between consecutive chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        
        # Only keep chunks with meaningful content
        if chunk and len(chunk) > 50:
            chunks.append(chunk)
        
        start = end - overlap
        
        if start >= text_len:
            break
    
    return chunks

# =====================================================
# INDEX COURSE CONTENT
# =====================================================
async def index_course_content(course_id: int, course_name: str, documents: list):
    """
    Index course content from Moodle into Qdrant vector database.
    This replaces any existing indexed content for the course.
    
    Args:
        course_id: Moodle course ID
        course_name: Course name for metadata
        documents: List of dicts with keys: type, source, content
    
    Returns:
        dict with success status and indexing statistics
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"[INDEX] Starting indexing for course {course_id}")
    logging.info(f"[INDEX] Course name: {course_name}")
    logging.info(f"[INDEX] Documents to process: {len(documents)}")
    logging.info(f"{'='*60}")
    
    collection_name = f"course_{course_id}_chunks"
    
    # Create or recreate collection (this avoids duplicates)
    try:
        # Delete existing collection if it exists
        try:
            client.delete_collection(collection_name)
            logging.info(f"[INDEX] ✓ Deleted old collection: {collection_name}")
        except:
            logging.info(f"[INDEX] No existing collection to delete")
        
        # Create fresh collection
        ensure_collection_exists(collection_name)
        
    except Exception as e:
        raise ValueError(f"Failed to create collection: {str(e)}")
    
    # Process all documents
    all_points = []
    point_id = 0
    total_content_length = 0
    
    for doc_idx, doc in enumerate(documents):
        content = doc.get('content', '').strip()
        source = doc.get('source', 'Unknown')
        doc_type = doc.get('type', 'text')
        
        # Skip empty or very short documents
        if not content or len(content) < 50:
            logging.warning(
                f"[INDEX] Skipping document {doc_idx}: "
                f"too short ({len(content)} chars)"
            )
            continue
        
        total_content_length += len(content)
        
        # Chunk the content
        chunks = chunk_text(content, chunk_size=1000, overlap=200)
        logging.info(
            f"[INDEX] Document {doc_idx} ({doc_type}): "
            f"{len(chunks)} chunks from '{source}' "
            f"({len(content)} chars)"
        )
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            try:
                # Generate embedding vector
                embedding = embed_text(chunk)
                
                # Create point for Qdrant
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
                logging.error(
                    f"[INDEX ERROR] Failed to process chunk {chunk_idx} "
                    f"of document {doc_idx}: {e}"
                )
                continue
    
    # Validate we have content to index
    if not all_points:
        raise ValueError(
            "No valid content to index. All documents were empty or too short."
        )
    
    # Store all points in Qdrant
    logging.info(f"\n[INDEX] Upserting {len(all_points)} points to {collection_name}")
    
    try:
        client.upsert(
            collection_name=collection_name,
            points=all_points
        )
        logging.info(f"[INDEX] ✓ Successfully indexed {len(all_points)} chunks")
        
    except Exception as e:
        raise ValueError(f"Failed to store vectors in Qdrant: {str(e)}")
    
    # Return success with statistics
    result = {
        "success": True,
        "course_id": course_id,
        "course_name": course_name,
        "documents_processed": len(documents),
        "chunks_indexed": len(all_points),
        "total_content_chars": total_content_length,
        "collection": collection_name
    }
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[INDEX SUCCESS] {result}")
    logging.info(f"{'='*60}\n")
    
    return result

# =====================================================
# GET COURSE STATUS
# =====================================================
async def get_course_status(course_id: int):
    """
    Check if a course has been indexed and get statistics.
    
    Args:
        course_id: Moodle course ID
    
    Returns:
        dict with indexing status and chunk count
    """
    collection_name = f"course_{course_id}_chunks"
    
    try:
        # Try to get collection info
        collection_info = client.get_collection(collection_name)
        
        result = {
            "course_id": course_id,
            "indexed": True,
            "chunks": collection_info.points_count,
            "collection": collection_name,
            "message": f"Course has {collection_info.points_count} indexed chunks"
        }
        
        logging.info(f"[STATUS] Course {course_id}: indexed with {result['chunks']} chunks")
        return result
        
    except Exception as e:
        # Collection doesn't exist or other error
        result = {
            "course_id": course_id,
            "indexed": False,
            "chunks": 0,
            "collection": collection_name,
            "message": "Course has not been indexed yet"
        }
        
        logging.info(f"[STATUS] Course {course_id}: not indexed")
        return result

# =====================================================
# RAG ANSWER - With Better Error Handling
# =====================================================
async def rag_answer(course_id: int, question: str):
    """
    Answer student questions using RAG (Retrieval-Augmented Generation)
    
    Args:
        course_id: Moodle course ID
        question: Student's question
    
    Returns:
        AI-generated answer based on course materials
    """
    logging.info("\n" + "="*60)
    logging.info(f"[RAG] Starting RAG query")
    logging.info(f"[RAG] Course ID: {course_id}")
    logging.info(f"[RAG] Question: {question}")
    logging.info("="*60)

    collection = f"course_{course_id}_chunks"
    
    # Check if collection exists and has content
    try:
        collection_info = client.get_collection(collection)
        
        if collection_info.points_count == 0:
            logging.warning(f"[RAG] Collection {collection} exists but is empty")
            return (
                "⚠️ This course content has not been indexed yet. "
                "Please contact your administrator to enable AI support for this course."
            )
        
        logging.info(f"[RAG] Collection has {collection_info.points_count} chunks")
        
    except Exception as e:
        logging.warning(f"[RAG] Collection {collection} does not exist: {e}")
        return (
            "⚠️ This course content has not been indexed yet. "
            "Please contact your administrator to enable AI support for this course."
        )

    # Generate embedding for the question
    try:
        query_emb = embed_text(question)
        logging.info(f"[RAG] ✓ Embedding generated (dimension={len(query_emb)})")
    except Exception as e:
        logging.error(f"[RAG ERROR] Failed to generate embedding: {e}")
        raise

    # Search for relevant chunks in Qdrant
    try:
        results = client.query_points(
            collection_name=collection,
            query=query_emb,
            limit=5
        )
        results = results.points
        logging.info(f"[RAG] ✓ Found {len(results)} relevant chunks")
        
        # Log relevance scores
        for idx, hit in enumerate(results):
            logging.info(
                f"[RAG]   Result {idx+1}: score={hit.score:.4f}, "
                f"source={hit.payload.get('source', 'Unknown')}"
            )
            
    except Exception as e:
        logging.error(f"[RAG ERROR] Qdrant search failed: {e}")
        raise

    # Handle case where no relevant content was found
    if not results:
        logging.warning("[RAG] No relevant content found for question")
        return (
            "I couldn't find relevant information in the course materials "
            "to answer your question. Could you rephrase or ask something else?"
        )

    # Build context from search results
    context_parts = []
    for hit in results:
        source = hit.payload.get('source', 'Unknown')
        text = hit.payload.get('text', '')
        context_parts.append(f"Source: {source}\n{text}")
    
    context = "\n\n---\n\n".join(context_parts)
    logging.info(f"[RAG] Context assembled: {len(context)} characters")

    # Generate answer using LLM
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

    try:
        answer = llm(prompt)
        logging.info(f"[RAG] ✓ LLM response generated ({len(answer)} chars)")
        logging.info("="*60 + "\n")
        return answer
        
    except Exception as e:
        logging.error(f"[RAG ERROR] LLM generation failed: {e}")
        raise

# =====================================================
# INGEST FILE (Legacy endpoint)
# =====================================================
async def ingest_file(course_id: int, chapter_id: int, file: UploadFile):
    """
    Ingest a single file into the vector database (legacy endpoint).
    This adds to existing course content without replacing it.
    
    Args:
        course_id: Moodle course ID
        chapter_id: Chapter/section ID
        file: Uploaded file
    
    Returns:
        dict with number of chunks processed
    """
    logging.info("\n" + "="*60)
    logging.info(f"[INGEST] Starting file ingestion")
    logging.info(f"[INGEST] Course ID: {course_id}")
    logging.info(f"[INGEST] Chapter ID: {chapter_id}")
    logging.info(f"[INGEST] File: {file.filename}")
    logging.info("="*60)

    collection = f"course_{course_id}_chunks"

    # Ensure collection exists (won't recreate if exists)
    ensure_collection_exists(collection)

    # Read file content
    try:
        raw = await file.read()
        text = raw.decode("utf-8", errors="ignore")
        logging.info(f"[INGEST] File read: {len(text)} characters")
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

    # Split into chunks
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    
    if not chunks:
        raise ValueError("No valid text chunks found in file")

    logging.info(f"[INGEST] Processing {len(chunks)} chunks")

    # Create points for Qdrant
    points = []
    for idx, chunk in enumerate(chunks):
        try:
            emb = embed_text(chunk)
            
            # Use chapter_id in point ID to avoid conflicts
            point = PointStruct(
                id=chapter_id * 10000 + idx,
                vector=emb,
                payload={
                    "text": chunk,
                    "course_id": course_id,
                    "chapter_id": chapter_id,
                    "source": file.filename,
                    "type": "file_upload"
                }
            )
            points.append(point)
            
        except Exception as e:
            logging.error(f"[INGEST ERROR] Failed to process chunk {idx}: {e}")
            continue

    if not points:
        raise ValueError("Failed to process any chunks from file")

    # Store in Qdrant
    logging.info(f"[INGEST] Upserting {len(points)} points to {collection}")
    
    try:
        client.upsert(
            collection_name=collection,
            points=points
        )
        logging.info(f"[INGEST] ✓ Successfully stored {len(points)} chunks")
        
    except Exception as e:
        raise ValueError(f"Failed to store in Qdrant: {str(e)}")

    logging.info("="*60 + "\n")
    return {"chunks": len(points)}