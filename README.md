# OCR Pipeline

Standalone Python/FastAPI microservice for AI-powered receipt OCR: structured field extraction, expense categorization, BIR VAT classification, and pgvector-based duplicate detection.

**Consumers:** SERMS and PRS (service-to-service webhook integration)

## Overview

Upstream services submit a receipt image or PDF URL. The API queues an async Celery job, runs OCR + LLM extraction, stores results, and POSTs a callback when done.

```
Consumer (SERMS/PRS)
        │
        ▼
   FastAPI API  ──queue──►  Celery Worker
        │                        │
        │                   PaddleOCR + Tesseract
        │                   Candidate selection
        │                   Google AI Studio (Gemini 2.0 Flash)
        │                   Embeddings (MiniLM)
        │                        │
   MongoDB (jobs)          PostgreSQL + pgvector
   Redis (broker)          Callback → Consumer
```

### Stack

| Component | Role |
|-----------|------|
| **FastAPI** | HTTP API, auth, job enqueue |
| **Celery + Redis** | Async receipt processing queue |
| **MongoDB** | Job status, OCR results, extracted fields |
| **PostgreSQL + pgvector** | Receipt text embeddings / similarity search |
| **Google AI Studio** | Cloud LLM for structured extraction (`gemini-2.0-flash`) |
| **PaddleOCR + Tesseract** | Candidate-based OCR selection with pinned Paddle 3.7-compatible runtime and full Tesseract variant cross-reference |
| **Sentence-Transformers** | Embeddings (`all-MiniLM-L6-v2`) |

## Prerequisites

- Docker Desktop (Compose V2)
- Git
- External Docker network `shared-capstone-network` (shared with SERMS/PRS compose stacks)

```bash
docker network create shared-capstone-network
```

## Quick Start

1. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Set consumer API keys (`SERMS_API_KEY`, `PRS_API_KEY`) and `CALLBACK_API_KEY` before exposing the service.

2. **Start the stack**

   ```bash
   docker compose up -d --build
   ```

   First build downloads system OCR libs, PyTorch, and caches the embedding model. Ensure `GEMINI_API_KEY` is configured in `.env`.

3. **Run Postgres migrations**

   ```bash
   docker compose exec api alembic upgrade head
   ```

4. **Verify**

   ```bash
   curl http://localhost:8010/api/health
   ```

### Services & ports

| Service | Container | Host port |
|---------|-----------|-----------|
| API | `ocr_api` | `8010` |
| Redis | `ocr_redis` | `6380` |
| MongoDB | `ocr_mongo` | `27017` |
| PostgreSQL | `ocr_postgres` | `5433` |

Worker: `ocr_worker` (no host port).

### Useful commands

```bash
docker compose ps
docker compose logs -f
docker compose logs -f worker
docker compose exec api pytest
docker compose down          # keep volumes
docker compose down -v       # wipe data
```

> [!NOTE]
> **Model downloads**
> - `all-MiniLM-L6-v2` is downloaded during the API/worker image build (`scripts/download_model.py`).

## Environment variables

Copy from `.env.example`. Important keys:

| Variable | Purpose |
|----------|---------|
| `SERMS_API_KEY` / `PRS_API_KEY` | Bearer tokens for inbound API auth |
| `CALLBACK_API_KEY` | Key used when calling consumer callback URLs |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Google AI Studio API key and model (`gemini-2.0-flash`) |
| `REDIS_URL` | Celery broker |
| `MONGODB_URL` / `MONGODB_DATABASE` | Job store |
| `POSTGRES_URL` | Async SQLAlchemy + pgvector connection |
| `EMBEDDING_MODEL` | Sentence-Transformers model id |
| `DUPLICATE_SIMILARITY_THRESHOLD` | Cosine similarity cutoff (default `0.85`) |
| `DUPLICATE_DAYS_WINDOW` | Lookback window in days (default `90`) |
| `OMP_NUM_THREADS` / `OMP_THREAD_LIMIT` | Recommended `1` in the worker container for Tesseract fallback processes |

The app loads `.env` via pydantic-settings (mounted into containers at `/app`).

## API (summary)

All endpoints except health require:

```http
Authorization: Bearer <SERMS_API_KEY|PRS_API_KEY>
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness (dependency checks currently stubbed) |
| `POST` | `/api/ocr/process` | Queue receipt OCR (returns `202` + `job_id`) |
| `POST` | `/api/duplicate-check` | Embedding similarity check |
| `GET` | `/api/jobs/{job_id}/status` | Job status |
| `GET` | `/api/metrics` | Aggregate processing metrics |

### Submit a receipt

```http
POST /api/ocr/process
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "receipt_id": 42,
  "file_url": "https://example.com/receipt.jpg",
  "callback_url": "https://serms.example/api/ocr/callback",
  "source_service": "serms"
}
```

```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Receipt queued for OCR processing."
}
```

When processing finishes, the worker POSTs results (or failure) to `callback_url`.

Full request/response shapes: [`docs/api-contract.md`](docs/api-contract.md).

Interactive OpenAPI docs (when API is up): http://localhost:8010/docs

## Processing pipeline

For each queued job the worker:

1. Downloads the file from `file_url` (image or PDF)
2. Preprocesses with OpenCV; **fast path (early exit)**: a single Tesseract pass ships the receipt when its anchor score is high and the money reconciles with a total present — otherwise the worker escalates to PaddleOCR + the Tesseract candidate pool, selecting the strongest reading by receipt-aware anchor scoring while retaining all candidates for reconciliation
3. Extracts structured fields via Google AI Studio (`gemini-2.0-flash`): **one consolidated assist call** answers vendor, location, tax/currency semantics, and subtotal questions; batch item normalization is enabled; the fast path skips the model entirely for clean reconciled receipts
4. Validates TIN and classifies VAT (BIR rules)
5. Computes a composite confidence score
6. Stores a text embedding in PostgreSQL (pgvector) — computed once per job and reused for duplicate detection
7. Persists job outcome in MongoDB
8. Sends the consumer webhook callback (with retries)

## Project layout

```
app/
  api/routes/       # FastAPI routers (ocr, jobs, duplicate, health, metrics)
  api/schemas/      # Pydantic request/response models
  core/             # Pipeline, OCR, preprocessing, BIR, confidence, callbacks
  db/               # MongoDB + Postgres clients and models
  embeddings/       # Sentence-Transformers + similarity
  llm/              # LLM provider abstraction (Google AI Studio Gemini)
  tasks/            # Celery app and process_receipt task
  config.py         # Settings from environment
alembic/            # Postgres migrations (embeddings table)
docs/               # Product, architecture, ops, and API docs
scripts/            # Model download and benchmark utilities
tests/              # pytest suite
```

## Documentation

| Doc | Description |
|-----|-------------|
| [`docs/index.md`](docs/index.md) | Documentation index |
| [`docs/PRD.md`](docs/PRD.md) | Product requirements |
| [`docs/SAD.md`](docs/SAD.md) | Architecture |
| [`docs/SDD.md`](docs/SDD.md) | System design (routes, tasks, auth) |
| [`docs/DSD.md`](docs/DSD.md) | Data model design |
| [`docs/Build.md`](docs/Build.md) | Build, run, and test guide |
| [`docs/OPS.md`](docs/OPS.md) | Operations runbook |
| [`docs/api-contract.md`](docs/api-contract.md) | Webhook / API contract |
| [`docs/AGENTS.md`](docs/AGENTS.md) | Notes for coding agents |

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `docker compose up` fails on network | Create `shared-capstone-network` (see Prerequisites) |
| Jobs stay `queued` | `docker compose logs -f worker`; confirm Redis is up |
| LLM / extraction errors | Inspect worker logs (`docker compose logs -f worker`); verify `GEMINI_API_KEY` is set in `.env` |
| API 401 | Bearer token must match `SERMS_API_KEY` or `PRS_API_KEY` in `.env` |
| pgvector / embedding errors | Run `alembic upgrade head` inside `api` |
| Callback never arrives | Worker logs; consumer URL reachable from `ocr_network` / shared network |
| PaddleOCR does not appear in worker logs | Rebuild the worker image after dependency changes: `docker compose up -d --build worker api`; verify with `docker compose exec -T worker python -c "import paddle, paddleocr; print('PaddleOCR installed')"` |

---

### Architectural Transition: Local Ollama to Gemini 2.0 Flash
The pipeline previously ran a local Ollama container (`ocr_ollama`) with `qwen2.5:1.5b` / `qwen3:1.7b` on CPU. This setup incurred significant cold-start latency (~20–85s per assist call), high memory consumption (~1.5–2GB RAM), and forced batch line item normalization to remain disabled. The project has fully transitioned to **Google AI Studio (`gemini-2.0-flash`)**:
- **Speed & Efficiency:** Near-instant inference replaces 20–85s CPU generations, and local RAM pressure is eliminated.
- **Enhanced Accuracy:** Line-item spelling normalization is enabled by default (`ENABLE_LLM_ITEM_NORMALIZATION=true`), and category tie-breaking is wired directly to Gemini.
- **Zero Local LLM Dependencies:** The `ocr_ollama` container, entrypoint scripts, and local weights have been completely removed.

### Financial semantics
The worker keeps amount extraction deterministic and uses an optional dedicated Gemini semantics call only for tax basis and currency context. Caller `country`, `currency`, and `location` metadata is accepted on submission and has highest contextual priority. The callback may include optional `tax_basis`, `financial_reconciliation_status`, and `needs_manual_review`; detailed evidence and reported/computed reconciliation remain in MongoDB. Gemini assist failures are soft and never invent totals.
