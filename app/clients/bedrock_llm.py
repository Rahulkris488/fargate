from langchain_aws import ChatBedrock
from app.config import settings

llm_client = ChatBedrock(
    credentials_profile_name=None,      # IAM Role
    region_name=settings.AWS_REGION,
    model_id=settings.BEDROCK_LLM_MODEL,
    max_tokens=4000,
    temperature=0.3
)

def generate_answer(prompt: str):
    """
    Generates a completion using Bedrock LLM.
    Used for:
    - Student Q&A chatbot
    - Test generation
    """
    response = llm_client.invoke(prompt)
    return response.content[0].text
