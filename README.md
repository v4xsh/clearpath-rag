
# Clearpath Support Assistant

## Overview

This project is a retrieval-augmented support assistant for a fictional SaaS product called **Clearpath**.

Given a user question, the system searches across the provided Clearpath documentation and generates a grounded answer. It is designed to avoid confident guesswork — if the system cannot find sufficient supporting context, it returns a safe fallback instead of hallucinating.

Each response surfaces internal telemetry in a debug panel, including routing decisions, token usage, and evaluator flags, so the behavior is transparent and easy to inspect.

---

## Architecture

### Chunking

Documents are split into ~450-token chunks with a 75-token overlap.
The chunker is structure-aware:
- Markdown headings are respected  
- Code blocks and tables are preserved  
- Mid-line splits are avoided  



### Retrieval

Hybrid retrieval combines:
- **FAISS semantic search** for paraphrase and intent matching  
- **BM25 keyword search** for exact technical terms (API names, error codes)

The fusion weighting shifts slightly based on how technical the query appears, so keyword-heavy queries lean more on BM25 while natural language queries lean more on embeddings.

---

### Sufficiency Gate

A coverage score determines whether generation should run at all.
- **LOW coverage** → safe fallback message (no LLM call)  
- **MEDIUM coverage** → retrieval expands by +2 chunks, then generate  
- **HIGH coverage** → normal generation  

---

### Router

The router is fully deterministic and rule-based — no LLM is used for routing. A query is classified as **SIMPLE** only if all of the following are true:
- Word count ≤ 12  
- Exactly one question  
- No reasoning keywords  
- No complaint tone  

**Routing behavior:**
- **SIMPLE** → `llama-3.1-8b-instant`  
- **COMPLEX** → `llama-3.3-70b-versatile`

---

### Evaluator

After generation, responses are checked using sentence-level attribution scoring. The evaluator flags:
- No-context responses / Refusals
- Low citation diversity  
- Competitor bleed (domain-specific hallucination check)  
- Prompt leakage / Missing domain entities  

---

## Live Deployment

The backend is publicly deployed on Railway:

**Base URL:** `https://clearpath-rag-production.up.railway.app`

| Endpoint | Method | Description |
| :--- | :--- | :--- |  
| `/health` | `GET` | System status check |
| `/query` | `POST` | Assignment-compliant response format |
| `/docs` | `GET` | Interactive API Documentation (Swagger UI) |

### Deployment Notes
The Railway instance runs on a shared CPU environment. Because sentence-transformer embeddings and the 70B model are CPU-bound at inference time, complex queries may exhibit higher latency in the hosted environment compared to local execution.

- **SIMPLE queries:** Fast and responsive.
- **COMPLEX queries:** Higher latency due to embedding generation and large-model orchestration on shared hardware.

---

## How to Run Locally

### Backend
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Setup environment variables
cp backend/.env.example backend/.env
# Open backend/.env and add your GROQ_API_KEY

python backend/scripts/ingest.py --docs-dir docs --output-dir backend/data/

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

```

### Frontend

```bash
cd frontend/react-app
npm install
npm run dev

```

---

## Known Limitations

* **Embedding recall ceiling:** Uses `all-MiniLM-L6-v2` for speed; retrieval recall can drop if user phrasing diverges heavily from documentation.
* **In-process session memory:** Resets on server restart; does not scale across multiple workers.
* **Attribution inflation:** Short chunks (headers/fragments) can sometimes inflate grounding scores.
* **Integration hallucination risk:** Mitigated via `COMPETITOR_BLEED` flags, but not 100% perfect.
* **Cold-start and CPU latency:** The public Railway deployment may introduce higher latency for 70B generations compared to a dedicated GPU-backed production environment.

---

## Bonus Features Implemented

* Conversational short-circuit for greetings
* Prompt injection sanitization layer
* Trust-boundary wrapping of retrieved context
* Adaptive top-k retrieval & Coverage-based sufficiency gate
* Streaming responses & Debug panel with full telemetry

