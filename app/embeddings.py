import boto3
import json
import os

REGION = os.getenv("AWS_REGION", "ap-southeast-2")

# 🔥 Correct models for ap-southeast-2
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"

client = boto3.client("bedrock-runtime", region_name=REGION)

def embed_text(text: str):
    payload = {"inputText": text}
    resp = client.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps(payload)
    )
    data = json.loads(resp["body"].read())
    return data["embedding"]

def llm(prompt: str):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 400,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": prompt}]
    }

    resp = client.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps(payload)
    )
    data = json.loads(resp["body"].read())
    return data["content"][0]["text"]
