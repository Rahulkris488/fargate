import boto3
import json
import os

REGION = os.getenv("AWS_REGION", "ap-southeast-2")

# Correct working models
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "amazon.titan-text-lite-v1"   # <<< FIXED

print(f"[INIT] embeddings.py loaded. REGION={REGION}")
print(f"[INIT] EMBED_MODEL={EMBED_MODEL}")
print(f"[INIT] LLM_MODEL={LLM_MODEL}")

client = boto3.client("bedrock-runtime", region_name=REGION)

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
        print(f"[EMBED][ERROR] Bedrock embedding error: {e}")
        raise

    data = json.loads(resp["body"].read())
    print(f"[EMBED] Embedding length: {len(data['embedding'])}")

    return data["embedding"]


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
        print("[LLM][DEBUG] possible cause: WRONG modelId")
        raise

    try:
        data = json.loads(resp["body"].read())
        answer = data["results"][0]["outputText"]
        print("[LLM] Extracted outputText successfully")
    except Exception as e:
        print(f"[LLM][ERROR] Failed to parse LLM response: {e}")
        print("[LLM][DEBUG] Raw response:", resp)
        raise

    return answer
