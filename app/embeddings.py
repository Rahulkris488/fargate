import boto3
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_REGION", "ap-southeast-2")
EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")
LLM_MODEL = os.getenv("BEDROCK_LLM_MODEL", "amazon.titan-text-lite-v1")

client = boto3.client("bedrock-runtime", region_name=REGION)

# -----------------------------
# EMBEDDINGS
# -----------------------------
def embed_text(text: str):
    logger.info("[EMBED] Generating embedding")
    payload = {"inputText": text}

    try:
        resp = client.invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps(payload)
        )
        data = json.loads(resp["body"].read())
        return data["embedding"]
    except Exception as e:
        logger.exception("[EMBED ERROR]")
        raise RuntimeError(f"Embedding failed: {str(e)}")

# -----------------------------
# LLM
# -----------------------------
def llm(prompt: str):
    logger.info("[LLM] Calling Titan model")
    logger.info(f"[LLM] Prompt length: {len(prompt)}")

    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.3,
            "topP": 0.9
        }
    }

    try:
        resp = client.invoke_model(
            modelId=LLM_MODEL,
            body=json.dumps(payload)
        )
        data = json.loads(resp["body"].read())
        return data["results"][0]["outputText"]
    except Exception as e:
        logger.exception("[LLM ERROR]")
        raise RuntimeError(f"LLM failed: {str(e)}")
