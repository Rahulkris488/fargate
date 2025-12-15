import boto3
import json
import os

REGION = os.getenv("AWS_REGION", "ap-southeast-2")

# ---- ACTIVE Bedrock Models ----
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL   = "amazon.titan-text-lite-v2:0"

print(f"[INIT] embeddings.py loaded. REGION={REGION}")
print(f"[INIT] EMBED_MODEL={EMBED_MODEL}")
print(f"[INIT] LLM_MODEL={LLM_MODEL}")

client = boto3.client("bedrock-runtime", region_name=REGION)

# =====================================================
# EMBEDDINGS
# =====================================================
def embed_text(text: str):
    print("\n[EMBED] Called embed_text()")
    print(f"[EMBED] Model: {EMBED_MODEL}")
    print(f"[EMBED] Input length: {len(text)}")

    payload = {"inputText": text}
    print(f"[EMBED] Payload JSON: {json.dumps(payload)}")

    try:
        resp = client.invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps(payload)
        )
        print("[EMBED] Bedrock API call succeeded")
    except Exception as e:
        print(f"[EMBED][ERROR] Embedding model failure: {e}")
        raise

    data = json.loads(resp["body"].read())

    emb = data["embedding"]
    print(f"[EMBED] Embedding length: {len(emb)}")

    return emb

# =====================================================
# LLM (TEXT GENERATION)
# =====================================================
def llm(prompt: str):
    print("\n[LLM] Called llm()")
    print(f"[LLM] Using model: {LLM_MODEL}")
    print(f"[LLM] Prompt length: {len(prompt)}")

    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 400,
            "temperature": 0.4,
            "topP": 0.9
        }
    }

    print(f"[LLM] Payload JSON: {json.dumps(payload)}")

    try:
        resp = client.invoke_model(
            modelId=LLM_MODEL,
            body=json.dumps(payload)
        )
        print("[LLM] Bedrock API call succeeded")
    except Exception as e:
        print(f"[LLM][ERROR] LLM model failure: {e}")
        print("[LLM][DEBUG]: Incorrect modelId or deprecated model")
        raise

    try:
        data = json.loads(resp["body"].read())
        answer = data["results"][0]["outputText"]
        print("[LLM] Extracted outputText successfully")
    except Exception as e:
        print(f"[LLM][ERROR] Failed parsing response: {e}")
        print("[LLM][DEBUG] RAW RESPONSE:")
        print(resp)
        raise

    return answer
