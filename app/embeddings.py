import boto3
import json
import os
import traceback

REGION = os.getenv("AWS_REGION", "ap-southeast-2")
EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL")
LLM_MODEL = os.getenv("BEDROCK_LLM_MODEL")

client = boto3.client("bedrock-runtime", region_name=REGION)

# -----------------------------
# EMBEDDINGS
# -----------------------------
def embed_text(text: str):
    try:
        payload = {"inputText": text}

        resp = client.invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps(payload)
        )

        data = json.loads(resp["body"].read())
        return data["embedding"]

    except Exception as e:
        print("\n[EMBEDDING ERROR]")
        print(e)
        traceback.print_exc()
        raise


# -----------------------------
# LLM (Titan Text Lite)
# -----------------------------
def llm(prompt: str):
    try:
        payload = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 800,
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

    except Exception as e:
        print("\n[LLM ERROR]")
        print(e)
        traceback.print_exc()
        raise
