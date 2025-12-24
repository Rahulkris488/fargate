import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")

    # ✅ UPDATED – ACTIVE MODELS
    BEDROCK_LLM_MODEL = "amazon.titan-text-lite-v1"
    BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v1"

    QDRANT_URL = os.getenv("QDRANT_URL")

settings = Settings()
