from qdrant_client import QdrantClient
import os

QDRANT_URL = os.getenv("QDRANT_URL")

if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL environment variable is not set")

print(f"[QDRANT] Connecting to: {QDRANT_URL}")

client = QdrantClient(
    url=QDRANT_URL,
    timeout=60
)

print(f"[QDRANT] Client initialized successfully")
