import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

    # Vector DB
    QDRANT_URL = os.getenv("QDRANT_URL")

settings = Settings()
