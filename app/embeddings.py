from app.llm_providers import llm_provider

def embed_text(text: str):
    return llm_provider.embed(text)

def llm(prompt: str):
    return llm_provider.generate(prompt)
