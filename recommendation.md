# Production Deployment Recommendations
## WikiSage — Local Wikipedia RAG Assistant

This document outlines a path for taking WikiSage beyond single-machine development and into a reliable, scalable production environment.

---

## 1. Containerization

### Current State
The system is a collection of Python scripts relying on local file-based storage (a ChromaDB directory and a SQLite file).

### Recommendation
**Package everything with Docker Compose** to create a portable, reproducible deployment:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  app:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - ollama
    volumes:
      - ./data:/app/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
```

**Benefits:**
- Identical environment across any machine
- GPU passthrough for Ollama when a compatible GPU is available
- Straightforward path to orchestration and horizontal scaling

---

## 2. Vector Database Upgrade

### Current State
A custom vector store built on NumPy (cosine similarity) and JSON file persistence — adequate for ~40 entities and a few thousand chunks. No external C dependencies; fully portable.

### Recommendation
For production-grade scale (thousands of entities, millions of chunks):

| Scale | Recommendation | Notes |
|---|---|---|
| Small (< 10K chunks) | Custom NumPy store (current) or ChromaDB | Sufficient, minimal operational overhead |
| Medium (10K–1M chunks) | **Qdrant** or **Weaviate** | Client-server mode, better indexing options |
| Large (> 1M chunks) | **Pinecone** or **Milvus** | Managed or distributed, horizontal scale-out |

**Migration path:** Qdrant is the natural next step — it supports the same cosine similarity, ships a Docker image, and provides better HNSW tuning for production workloads.

---

## 3. Model Serving

### Current State
Ollama serves models on local CPU/GPU. Practical and simple for a single user.

### Recommendation
For multi-user production deployments:

- **vLLM** or **TGI (Text Generation Inference)** for high-throughput LLM serving
  - Supports batched inference, quantization (GPTQ, AWQ), and continuous batching
  - Substantial throughput improvement over Ollama under concurrent load
- **GPU requirements:** Minimum NVIDIA T4 (16 GB) for llama3.2:3b; A10G for larger models
- Consider upgrading the model for better quality:
  - `llama3.1:8b` — significantly stronger reasoning, fits on consumer GPUs
  - `mistral:7b` — excellent instruction-following performance
  - `phi3:14b` — strong quality-to-size ratio, comfortable on an A10G

---

## 4. Caching Layer

### Recommendation
Introduce a **Redis-based caching layer** targeting three scenarios:

1. **Embedding cache** — Avoid re-computing embeddings for repeated or similar queries
2. **Response cache** — Store LLM outputs for exact query matches with a configurable TTL
3. **Semantic cache** — Reuse responses for semantically near-identical queries (threshold-based embedding similarity)

**Expected impact:** 50–80% reduction in Ollama API calls for repeated or highly similar inputs.

---

## 5. API Layer

### Current State
Streamlit handles both the UI and the backend logic — the two concerns are tightly coupled.

### Recommendation
Introduce a **FastAPI backend** to separate responsibilities:

```
Frontend (Streamlit/React) ──▶ FastAPI ──▶ Retriever + Generator
                                  │
                                  ▼
                            Redis Cache
```

**Endpoints:**
- `POST /api/query` — Accept a question and return a streamed answer
- `POST /api/ingest` — Trigger ingestion of additional entities
- `GET /api/stats` — Return current system statistics
- `GET /api/health` — Health check for monitoring

**Benefits:**
- Enables multiple frontend clients (web, mobile, CLI)
- UI and API can scale independently
- Cleaner error handling and easier rate limiting

---

## 6. Observability

### Recommendation
Deploy a **monitoring stack** covering three layers:

| Component | Tool | Purpose |
|---|---|---|
| Metrics | Prometheus + Grafana | Latency, throughput, and error rates |
| Logging | ELK Stack or Loki | Centralized log aggregation and search |
| Tracing | LangSmith or Jaeger | End-to-end RAG pipeline tracing (retrieval quality, LLM latency) |

**Key metrics to instrument:**
- Query-to-answer latency (p50, p95, p99)
- Retrieval relevance scores (cosine similarity distribution)
- LLM token throughput (tokens/second)
- Cache hit rates
- Per-component error rates

---

## 7. Security

### Production Security Checklist

- [ ] **Authentication** — Protect the API with OAuth2 or JWT-based user authentication
- [ ] **Rate limiting** — Enforce per-user request limits to prevent abuse
- [ ] **Input validation** — Sanitize user queries to defend against prompt injection attacks
- [ ] **HTTPS** — Terminate TLS via Nginx or a cloud load balancer
- [ ] **Data encryption** — Encrypt the vector store and SQLite database at rest
- [ ] **Network isolation** — Keep the Ollama service off the public internet
- [ ] **Audit logging** — Record all queries for compliance and incident analysis

---

## 8. Scaling Strategy

### Horizontal Scaling

```
                    ┌─────────────┐
                    │ Load Balancer│
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐
         │ App Pod 1│ │ App Pod 2│ │ App Pod 3│
         └────┬────┘ └────┬────┘ └────┬────┘
              │            │            │
              ▼            ▼            ▼
         ┌─────────────────────────────────┐
         │    Shared Vector DB (Qdrant)     │
         └─────────────────────────────────┘
              │
              ▼
         ┌─────────────────────────────────┐
         │    GPU Pool (vLLM instances)     │
         └─────────────────────────────────┘
```

- **Kubernetes** orchestration with autoscaling driven by query load
- **Dedicated GPU pool** for LLM inference — scales independently of the app tier
- **Shared vector store** — a single Qdrant cluster accessible by all application pods

---

## 9. Data Pipeline

### Recommendation
For production-grade ingestion:

- **Scheduled ingestion** — A weekly cron job to refresh Wikipedia content
- **Incremental updates** — Re-ingest only articles that have changed (via the Wikipedia revision API)
- **Data validation** — Verify chunk quality before inserting into the vector store
- **Backup strategy** — Regular snapshots of the vector store and SQLite database

---

## 10. Cost Estimates (Cloud Deployment)

| Component | Instance Type | Monthly Cost |
|---|---|---|
| App servers (3×) | AWS t3.medium | ~$90 |
| GPU (LLM serving) | AWS g4dn.xlarge (T4) | ~$380 |
| Vector DB | Qdrant Cloud (1M vectors) | ~$50 |
| Redis Cache | AWS ElastiCache t3.micro | ~$15 |
| Total | | **~$535/month** |

*Estimates are approximate and based on US East (N. Virginia) pricing as of 2024.*

---

## Summary

The current architecture is well-suited for local development and demos. Moving to production requires focused investment in five areas:

1. **Containerization** (Docker Compose → Kubernetes) — delivers immediate, low-effort gains
2. **API decoupling** (FastAPI) — unlocks multi-client support and independent scaling
3. **Caching** (Redis) — cuts latency and reduces compute overhead significantly
4. **Vector DB upgrade** (Qdrant) — necessary for production data volumes
5. **Model serving** (vLLM) — required to handle concurrent user load

Each of these can be adopted incrementally without rewriting the core RAG pipeline logic.
