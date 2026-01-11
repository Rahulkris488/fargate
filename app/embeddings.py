from sentence_transformers import SentenceTransformer
from app.llm_providers import llm_provider

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    return _model.encode(text).tolist()

# BACKWARD COMPATIBILITY
def llm(prompt: str) -> str:
    """
    DO NOT REMOVE.
    Quiz plugin and older code depend on this.
    """
    return llm_provider.get_completion(prompt)
