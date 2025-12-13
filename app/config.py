import os
from dotenv import load_dotenv

# Load .env locally (ignored in ECS – SSM provides env vars)
load_dotenv()

class Settings:
    AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
    
    # Bedrock models
    BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")
    BEDROCK_LLM_MODEL = os.getenv("BEDROCK_LLM_MODEL", "amazon.titan-text-lite-v1")

    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL")   # Example: http://10.0.2.15:6333

settings = Settings()
