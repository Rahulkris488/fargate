from qdrant_client import QdrantClient
import os

# Example: http://10.0.2.15:6333  (must include http://)
QDRANT_URL = os.getenv("QDRANT_URL")

print(f"[QDRANT] Loaded QDRANT_URL={QDRANT_URL}")

# IMPORTANT: prefer port 6333, not 6334
client = QdrantClient(url=QDRANT_URL)
