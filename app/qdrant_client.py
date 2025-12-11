from qdrant_client import QdrantClient
import os

QDRANT_URL = os.getenv("QDRANT_URL")   # Examuuple: http://10.0.2.15:6333

client = QdrantClient(url=QDRANT_URL)
