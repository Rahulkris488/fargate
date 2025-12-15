from qdrant_client import QdrantClient
import os

QDRANT_URL = os.getenv("QDRANT_URL")  # Example: http://10.0.2.15:6333

print(f"[QDRANT] Loaded QDRANT_URL={QDRANT_URL}")

client = QdrantClient(
    url=QDRANT_URL,
    timeout=60  # prevent connection reset issues
)
