from qdrant_client import QdrantClient
import os

QDRANT_URL = os.getenv("QDRANT_URL")

if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL environment variable is not set")

client = QdrantClient(
    url=QDRANT_URL,
    timeout=60
)
