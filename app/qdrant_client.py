from qdrant_client import QdrantClient
import os

QDRANT_URL = os.getenv("QDRANT_URL")

client = QdrantClient(
    url=QDRANT_URL,
    timeout=60
)
