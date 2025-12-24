import boto3
import json
import os
from app.config import settings

REGION = settings.AWS_REGION

EMBED_MODEL = settings.BEDROCK_EMBED_MODEL
LLM_MODEL   = settings.BEDROCK_LLM_MODEL

client = boto3.client(
    "bedrock-runtime",
    region_name=REGION
)

# -----------------------------
# EMBEDDINGS
# -----------------------------
def embed_text(text: str):
    print("[EMBED] Using model:", EMBED_MODEL)

    payload = {"inputText": text}

    resp = client.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps(payload)
    )

    data = json.loads(resp["body"].read())
    return data["embedding"]

# -----------------------------
# LLM
# -----------------------------
def llm(prompt: str):
    print("[LLM] Using model:", LLM_MODEL)
    print("[LLM] Prompt length:", len(prompt))

    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 600,
            "temperature": 0.3,
            "topP": 0.9
        }
    }

    resp = client.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps(payload)
    )

    data = json.loads(resp["body"].read())
    return data["results"][0]["outputText"]
