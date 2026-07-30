A containerized FastAPI backend that extracts course content (Moodle), builds embeddings, and provides retrieval-augmented QA plus automated quiz generation — designed to be run as a microservice (Dockerized; suitable for AWS Fargate-style deployment).
Stack

    Language(s): Python (primary), Dockerfile
    Framework / runtime: FastAPI (uvicorn), Python 3.11
    Notable libraries: sentence-transformers, torch (CPU wheels), qdrant-client, fastapi, uvicorn

How it's organized
Code

.gitignore
Dockerfile                # container image for the FastAPI app (EXPOSE 8080)
requirements.txt          # pip deps (sentence-transformers, torch, qdrant-client, etc.)
sample.txt                # sample content used by app/tools
app/
  __init__.py
  a.py                    # small helper / misc
  config.py               # configuration / environment handling (check for env var usage)
  embeddings.py           # embedding creation wrapper
  llm_providers.py        # abstraction for LLM providers
  main.py                 # FastAPI entrypoint: /, /health, /chat, /generate-quiz, /ingest
  moodle_extractor.py     # Moodle scraping/extraction logic (large module)
  qdrant_client.py        # thin client wrapper for Qdrant operations
  quiz.py                 # quiz generation logic (critical path)
  rag.py                  # retrieval-augmented generation: ingest_file, rag_answer
  routes/                 # (empty directory present — intended HTTP route organization)
  services/               # (empty directory present — intended business logic)
  utils/                  # (empty directory present — utilities)
.github/                   # repository metadata

How it fits together:

    app/main.py is the HTTP entrypoint. It exposes endpoints for chat (RAG), quiz generation, and ingestion.
    Ingestion: uploaded files are processed by moodle_extractor.py and stored (embeddings produced by embeddings.py and persisted via qdrant_client.py).
    Retrieval & QA: rag.py implements ingest and rag_answer that query Qdrant with sentence-transformers embeddings and use llm_providers.py to form answers.
    Quiz generation: quiz.py is the critical path for producing topic-based quizzes, likely using extracted content + LLMs.

How to run it

From the repo root (shortest path based on files present):

    Local (Python venv)

    pip install -r requirements.txt
    uvicorn app.main:app --host 0.0.0.0 --port 8080

    Docker (image defined by Dockerfile)

    docker build -t moodle-ai .
    docker run -p 8080:8080 -e <ENV_VARS> moodle-ai

Notes / required configuration

    Dockerfile sets OMP_NUM_THREADS and MKL_NUM_THREADS to 2 and exposes port 8080; container command runs uvicorn on port 8080.
    The app depends on a vector store (qdrant-client) and LLM provider(s) (llm_providers.py). Expect environment variables or secrets for:
        Qdrant endpoint / credentials
        LLM API key(s) for whichever provider is configured
        Any other settings defined in app/config.py (check that file)
    requirements.txt includes CPU PyTorch wheels and sentence-transformers — CPU-only inference is expected by default.

