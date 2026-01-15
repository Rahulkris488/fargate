import os
from dotenv import load_dotenv
from groq import Groq

# CRITICAL: Load .env file
load_dotenv()

class LLMProvider:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        self.client = Groq(api_key=api_key)

    def get_completion(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()


# Singleton (important)
llm_provider = LLMProvider()