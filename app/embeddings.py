from app.llm_providers import llm_provider

def embed_text(text: str):
    """Generate embeddings - provider agnostic"""
    return llm_provider.embed(text)

def llm(prompt: str):
    """Generate text - provider agnostic"""
    return llm_provider.generate(prompt)