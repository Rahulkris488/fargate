from qdrant_client import QdrantClient
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"))

for c in client.get_collections().collections:
    if c.name.startswith("course_"):
        print("Deleting:", c.name)
        client.delete_collection(c.name)

print("Done.")
