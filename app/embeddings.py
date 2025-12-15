import boto3
import json
import os

REGION = os.getenv("AWS_REGION", "ap-southeast-2")

# Titan models (FREE — NO MARKETPLACE SUBSCRIPTION REQUIRED)
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "amazon.titan-text-lite-v1"

print(f"[INIT] embeddings.py loaded. REGION={REGION}")
print(f"[INIT] EMBED_MODEL={EMBED_MODEL}")
print(f"[INIT] LLM_MODEL={LLM_MODEL}")

client = boto3.client("bedrock-runtime", region_name=REGION)

# -----------------------------------------------------------
# EMBEDDING FUNCTION (Titan Embeddings)
# -----------------------------------------------------------
def embed_text(text: str):
    print("\n[EMBED] Called embed_text()")
    print(f"[EMBED] Input text length: {len(text)}")

    payload = {"inputText": text}
    print(f"[EMBED] Payload: {payload}")

    try:
        resp = client.invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps(payload)
        )
        print("[EMBED] Bedrock API call succeeded")
    except Exception as e:
        print(f"[EMBED][ERROR] Bedrock embed model error: {e}")
        raise

    try:
        data = json.loads(resp["body"].read())
        print(f"[EMBED] Response keys: {list(data.keys())}")
    except Exception as e:
        print(f"[EMBED][ERROR] Failed parsing response: {e}")
        raise

    emb = data["embedding"]
    print(f"[EMBED] Embedding length: {len(emb)}")

    return emb


# -----------------------------------------------------------
# LLM FUNCTION (Titan Text Lite — FREE LLM)
# -----------------------------------------------------------
def llm(prompt: str):
    print("\n[LLM] Called llm()")
    print(f"[LLM] Prompt length: {len(prompt)}")

    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 400,
            "temperature": 0.4,
            "topP": 0.9
        }
    }

    print(f"[LLM] Titan Payload: {payload}")

    try:
        resp = client.invoke_model(
            modelId=LLM_MODEL,
            body=json.dumps(payload)
        )
        print("[LLM] Bedrock API call succeeded")
    except Exception as e:
        print(f"[LLM][ERROR] Titan LLM model error: {e}")
        raise

    try:
        data = json.loads(resp["body"].read())
        print(f"[LLM] Response keys: {list(data.keys())}")
    except Exception as e:
        print(f"[LLM][ERROR] Failed parsing LLM response: {e}")
        raise

    try:
        answer = data["results"][0]["outputText"]
        print("[LLM] Extracted Titan text successfully")
    except Exception as e:
        print(f"[LLM][ERROR] Missing outputText: {e}")
        raise

    return answer
