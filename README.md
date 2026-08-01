# Moodle AI Backend
### Cloud-Native Retrieval-Augmented Learning Platform

A production-ready **FastAPI microservice** that extracts Moodle course content, generates semantic embeddings, performs Retrieval-Augmented Generation (RAG), and automatically creates quizzes using LLMs.

The application is fully **containerized with Docker** and deployed using a modern **AWS CI/CD pipeline** powered by **GitHub Actions, Amazon ECR, and Amazon ECS**.

---

# Features

- Moodle course content extraction
- Semantic embedding generation using Sentence Transformers
- Vector search with Qdrant
- Retrieval-Augmented Question Answering (RAG)
- AI-powered quiz generation
- FastAPI REST API
- Dockerized microservice architecture
- Automated AWS deployment pipeline
- Production-ready container deployment on Amazon ECS

---

# Tech Stack

## Backend

- Python 3.11
- FastAPI
- Uvicorn

## AI / ML

- Sentence Transformers
- PyTorch (CPU)
- Retrieval-Augmented Generation (RAG)

## Vector Database

- Qdrant

## Containerization

- Docker

## Cloud

- Amazon ECS
- Amazon Elastic Container Registry (ECR)
- AWS IAM

## DevOps

- GitHub Actions
- CI/CD Pipeline

---

# Project Structure

```
.
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── rag.py                 # Retrieval-Augmented Generation
│   ├── embeddings.py          # Embedding generation
│   ├── quiz.py                # AI Quiz Generator
│   ├── moodle_extractor.py    # Moodle content extraction
│   ├── llm_providers.py       # LLM abstraction layer
│   ├── qdrant_client.py       # Vector database interface
│   ├── config.py
│   └── ...
│
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        └── deploy.yml
```

---

# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Service status |
| GET | `/health` | Health check |
| POST | `/ingest` | Extract and embed Moodle content |
| POST | `/chat` | Retrieval-Augmented QA |
| POST | `/generate-quiz` | AI Quiz Generation |

---

# System Architecture

```text
                     Git Push
                        │
                        ▼
               GitHub Repository
                        │
                        ▼
              GitHub Actions CI/CD
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 Build Docker Image             Configure AWS
        │
        ▼
 Push Image to Amazon ECR
        │
        ▼
 Amazon ECS Service
        │
        ▼
  FastAPI Container
        │
        ▼
      REST API
        │
        ├──────────────┐
        ▼              ▼
   Qdrant DB        LLM Provider
        │
        ▼
 Retrieval-Augmented Responses
```

---

# AI Pipeline

```
Moodle Content
      │
      ▼
Content Extraction
      │
      ▼
Sentence Transformers
      │
      ▼
Embeddings
      │
      ▼
Qdrant Vector Store
      │
      ▼
Similarity Search
      │
      ▼
Retrieved Context
      │
      ▼
LLM
      │
      ▼
Answer / Quiz Generation
```

---

# Cloud Infrastructure

The application follows a cloud-native deployment model.

## Amazon ECS

Runs the FastAPI application as a scalable containerized service.

## Amazon ECR

Stores Docker images produced during every deployment.

## GitHub Actions

Automates build and deployment on every push to the `main` branch.

## Docker

Packages the application into a portable production-ready container.

---

# CI/CD Pipeline

Every push to the **main** branch automatically triggers the deployment workflow.

```
Developer
     │
 git push
     │
     ▼
GitHub Actions
     │
     ▼
Checkout Repository
     │
     ▼
Configure AWS Credentials
     │
     ▼
Login to Amazon ECR
     │
     ▼
Build Docker Image
     │
     ▼
Push Image to ECR
     │
     ▼
Force New Deployment
     │
     ▼
Amazon ECS
     │
     ▼
Updated Backend
```

### Workflow Steps

- Checkout source code
- Authenticate with AWS
- Login to Amazon ECR
- Build Docker image
- Push image to Amazon ECR
- Trigger Amazon ECS rolling deployment
- Zero manual deployment steps

---

# Docker

Build the image

```bash
docker build -t moodle-ai .
```

Run locally

```bash
docker run -p 8080:8080 moodle-ai
```

---

# Local Development

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

# Environment Variables

The application requires the following configuration.

```env
QDRANT_URL=

QDRANT_API_KEY=

LLM_API_KEY=

AWS_REGION=

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=
```

---

# DevOps Highlights

- Dockerized FastAPI microservice
- Automated GitHub Actions pipeline
- Continuous Integration
- Continuous Deployment
- Amazon Elastic Container Registry
- Amazon ECS rolling deployments
- Secure credential management using GitHub Secrets
- Infrastructure following cloud-native deployment practices

---

# Future Improvements

- Terraform Infrastructure as Code
- ECS Auto Scaling
- Application Load Balancer
- HTTPS using ACM
- CloudWatch Monitoring
- AWS Secrets Manager integration
- Multi-container deployment
- Blue/Green deployments
- GPU inference support

---

# Authors

Developed as a cloud-native AI backend demonstrating:

- Retrieval-Augmented Generation (RAG)
- LLM Integration
- Vector Search
- FastAPI Microservices
- Docker
- AWS ECS
- Amazon ECR
- GitHub Actions CI/CD
