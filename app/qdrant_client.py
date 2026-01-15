import os
import logging

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

try:
    from qdrant_client import QdrantClient
    
    logger.info(f"[QDRANT] Attempting to connect to: {QDRANT_URL}")
    
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=10  # Reduced timeout for faster failure
    )
    
    # Test connection
    client.get_collections()
    
    logger.info(f"[QDRANT] ✅ Connected successfully")
    
except Exception as e:
    logger.warning(f"[QDRANT] ⚠️ Not available: {e}")
    logger.warning("[QDRANT] RAG features will fall back to AI-only mode")
    client = None