from abc import ABC, abstractmethod
from typing import List
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ===========================
# ABSTRACT INTERFACE
# ===========================
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embeddings from text"""
        pass


# ===========================
# GROQ PROVIDER
# ===========================
class GroqProvider(LLMProvider):
    def __init__(self):
        from groq import Groq
        from sentence_transformers import SentenceTransformer
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        self.client = Groq(api_key=api_key)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[GROQ] Provider initialized successfully")
    
    def generate(self, prompt: str) -> str:
        try:
            print(f"[GROQ] Generating response (prompt length: {len(prompt)})")
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            result = response.choices[0].message.content
            print(f"[GROQ] Response generated (length: {len(result)})")
            return result
        except Exception as e:
            print(f"[GROQ ERROR] {e}")
            raise
    
    def embed(self, text: str) -> List[float]:
        try:
            print(f"[EMBED] Generating embedding (text length: {len(text)})")
            embedding = self.embedder.encode(text).tolist()
            print(f"[EMBED] Embedding generated (dim: {len(embedding)})")
            return embedding
        except Exception as e:
            print(f"[EMBED ERROR] {e}")
            raise


# ===========================
# FACTORY
# ===========================
def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider"""
    provider_name = os.getenv("LLM_PROVIDER", "groq").lower()
    
    if provider_name == "groq":
        return GroqProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider_name}")


# ===========================
# GLOBAL INSTANCE
# ===========================
print("[LLM] Initializing provider...")
llm_provider = get_llm_provider()
print("[LLM] Provider ready")