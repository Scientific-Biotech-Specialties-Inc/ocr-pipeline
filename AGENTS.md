# AGENTS.md — OCR Pipeline

> **Materialized from `README.md` and the full `docs/` suite. Edit the source docs, then re-materialize — do not hand-edit this file as the source of truth.**

OCR Pipeline is a standalone Python/FastAPI microservice providing AI-powered receipt OCR extraction, structured field parsing, expense categorization, BIR VAT classification, anomaly risk scoring, and pgvector-based duplicate detection.

---

## Read order (every session)

1. `README.md`
2. `docs/SUMMARY.md` — project summary & key user flows
3. `docs/api-contract.md`
4. `docs/SDD.md` — routes, tasks, env vars
5. `docs/DSD.md` — schemas (PostgreSQL, MongoDB, Redis)
6. `docs/OPS.md` — operational runbook & SLOs
7. `docs/QAD.md` — QA strategy, test scenarios, release criteria
8. `docs/sprint.md` — canonical task board (what to do next, task status)
9. `docs/AGENTS.md` (this guide)

**Full documentation index:** [`docs/index.md`](index.md)

### SERMS Documentation Connection

If the prompt or the AI model/agent requires context, constraints, or schemas from the main SERMS (Smart Expense Management System) repository, reference the external SERMS documentation located at:

- **SERMS AGENTS.md:** `c:\Projects\smart-expense-management-system\AGENTS.md`
- **SERMS Docs Directory:** `c:\Projects\smart-expense-management-system\docs\`
  Agents must actively read these files when making integration decisions (e.g., payloads, API headers, webhook rules) to ensure compliance with SERMS core rules.

---

## Pinned stack

| Layer                      | Technology                                                  | Role                                                                             |
| -------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **API Framework**          | FastAPI (Python 3.12+)                                      | HTTP API, auth, job enqueue                                                      |
| **Task Queue**             | Celery + Redis                                              | Async receipt processing queue                                                   |
| **Message Broker / Cache** | Redis                                                       | Celery broker; file-hash caching (`ocr:file_hash:<sha256>`, TTL 90d)             |
| **NoSQL Database**         | MongoDB (`motor`)                                           | Job status, raw OCR results, extracted fields, BIR validation, confidence scores |
| **Vector Database**        | PostgreSQL + `pgvector` (`sqlalchemy`, `asyncpg`)           | 384-dim receipt embeddings; cosine similarity duplicate detection                |
| **Local LLM**              | Ollama (`qwen3:1.7b`)                                       | Structured field extraction from raw OCR text                                    |
| **OCR Engine**             | PaddleOCR + Tesseract (`paddleocr`, `pytesseract`) + OpenCV | Image pre-processing + PaddleOCR primary + Tesseract fallback                    |
| **Embeddings**             | Sentence-Transformers (`all-MiniLM-L6-v2`)                  | 384-dim vector generation                                                        |
| **ML Anomaly Model**       | Scikit-learn (`RandomForestClassifier`)                     | Receipt anomaly risk scoring (0.0–1.0)                                           |
| **Settings**               | Pydantic Settings                                           | Environment variable management                                                  |
| **Migrations**             | Alembic                                                     | PostgreSQL schema migrations                                                     |

---

## Architecture & Infrastructure

- Docker Compose (`docker-compose.yml`) orchestrates all services.
- `ocr_ollama` container auto-pulls `qwen3:1.7b` via `scripts/ollama-entrypoint.sh` on first start.
- `ocr_api` caches `all-MiniLM-L6-v2` during image build via `scripts/download_model.py`.
- All containers communicate over a private Docker bridge network (`ocr_network`).
- An **external** Docker network `shared-capstone-network` is required so SERMS and PRS can route callbacks.

### Container roster

| Container      | Host port | Role            |
| -------------- | --------- | --------------- |
| `ocr_api`      | `8010`    | FastAPI API     |
| `ocr_worker`   | —         | Celery worker   |
| `ocr_redis`    | `6380`    | Broker / cache  |
| `ocr_mongo`    | `27017`   | Job store       |
| `ocr_postgres` | `5433`    | Embedding store |
| `ocr_ollama`   | `11434`   | Local LLM       |

---

## API surface

All endpoints except `/api/health` require:

```http
Authorization: Bearer <SERMS_API_KEY|PRS_API_KEY>
```

| Method | Path                        | Description                                                       |
| ------ | --------------------------- | ----------------------------------------------------------------- |
| `GET`  | `/api/health`               | Liveness check (dependency checks currently stubbed — see OPS.md) |
| `POST` | `/api/ocr/process`          | Queue receipt OCR; returns `202 Accepted` + `job_id`              |
| `POST` | `/api/duplicate-check`      | Embedding cosine-similarity check against pgvector                |
| `GET`  | `/api/jobs/{job_id}/status` | Job status from MongoDB                                           |
| `GET`  | `/api/metrics`              | Aggregate processing metrics (currently stub — returns zeros)     |

Full request/response shapes: [`docs/api-contract.md`](api-contract.md).
Interactive OpenAPI docs (stack must be running): http://localhost:8010/docs

---

## Processing pipeline (per job)

Each `process_receipt_task` Celery job runs these stages in order:

1. **Download & Ingest** — fetch file from `file_url` or parallel `file_urls` (multi-segment receipts up to 6); 3-zone blur verification for tall receipts
2. **Pre-process & Strip Tiling** — OpenCV (grayscale, deskew, adaptive thresholding, denoising); tall single-image receipts and tall `file_urls` segments (`height >= 1800` & `aspect >= 2.8`) automatically sliced into overlapping strips (capped at 10 strips)
3. **OCR & Seam Stitching** — PaddleOCR extracts primary text & geometry on upright images; single-pass OpenCV + Tesseract serves as fallback; middle item segments complete via density-aware fast path; multi-segment/strip texts merged via 30-line seam deduplication, 7-digit barcode matching, and margin-guarded auto-ordering; unified anchors scored; targeted CLAHE crop recovery on header/footer if fields missing
4. **LLM extraction** — Ollama `qwen3:1.7b` (non-thinking JSON mode, `num_ctx=2048`, integer `keep_alive`) parses vendor, location, tax basis, currency, and subtotal verification (prompt bounded to 2000 head + 2000 tail characters)
5. **BIR validation** — `app/core/bir_validator.py` validates TIN format, VAT classification
6. **Confidence scoring & Reconciliation** — `app/core/confidence.py` computes composite score (≥ 0.75 = acceptable); receipts (≥ 3 items) near-reconciled within 3% or lacking total target emit line items with score capped at 0.79 (`REASON_NEAR_RECONCILED`), and `needs_manual_review` is always surfaced on the callback payload
7. **Anomaly risk scoring** — `RandomForestClassifier` assigns a risk score (0.0–1.0) and category (`Low` / `Medium` / `High Risk`)
8. **Embedding** — `all-MiniLM-L6-v2` generates a 384-dim vector from compacted receipt text; stored in PostgreSQL (`receipt_embeddings`)
9. **Duplicate detection** — cosine similarity search within a 90-day window; threshold `0.85`
10. **Persist** — results written to MongoDB `ocr_jobs` collection; job status → `completed` or `failed`
11. **Callback** — webhook `POST` to `callback_url` with full payload; 3 retries with exponential backoff (10 s → 30 s → 60 s)

---

## Data layer

### PostgreSQL — `receipt_embeddings`

| Column           | Type           | Description              |
| ---------------- | -------------- | ------------------------ |
| `id`             | `Integer` PK   | Auto-increment           |
| `receipt_id`     | `Integer`      | Caller-system receipt ID |
| `source_service` | `String(50)`   | `"serms"` or `"prs"`     |
| `embedding`      | `Vector(384)`  | Dense text embedding     |
| `receipt_text`   | `Text`         | Raw combined OCR output  |
| `created_at`     | `DateTime(tz)` | UTC timestamp            |

Indices: `idx_embeddings_source_service`, `idx_embeddings_created_at`, `idx_embeddings_vector` (IVFFlat/HNSW, optional).

### MongoDB — `ocr_jobs`

Key fields: `job_id` (UUID v4), `receipt_id`, `file_url`, `callback_url`, `source_service`, `status` (`queued|processing|completed|failed`), `ocr_result` (all extracted fields + `confidence_score` + `line_items`), `duplicate_check` (`is_duplicate`, `similarity_score`, `matches_count`), `created_at`, `updated_at`.

### Redis cache

- **Celery queues:** standard Celery broker keys.
- **File-hash cache:** key `ocr:file_hash:<sha256>` → `job_id`; TTL 90 days. Cache hit bypasses OCR worker entirely.

Full schema: [`docs/DSD.md`](DSD.md).

---

## Key environment variables

| Variable                           | Default                     | Description                                |
| ---------------------------------- | --------------------------- | ------------------------------------------ |
| `APP_PORT`                         | `8010`                      | FastAPI listen port                        |
| `SERMS_API_KEY` / `PRS_API_KEY`    | —                           | Inbound Bearer token auth                  |
| `CALLBACK_API_KEY`                 | —                           | Key attached to outbound webhook callbacks |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://ocr_ollama:11434`   | Local LLM                                  |
| `REDIS_URL`                        | `redis://ocr_redis:6379/0`  | Celery broker                              |
| `MONGODB_URL` / `MONGODB_DATABASE` | `mongodb://ocr_mongo:27017` | Job store                                  |
| `POSTGRES_URL`                     | —                           | Async pgvector connection                  |
| `EMBEDDING_MODEL`                  | `all-MiniLM-L6-v2`          | Embedding model id                         |
| `DUPLICATE_SIMILARITY_THRESHOLD`   | `0.85`                      | Cosine similarity cutoff                   |
| `DUPLICATE_DAYS_WINDOW`            | `90`                        | Lookback window (days)                     |
| `RATE_LIMIT_ENABLED`               | `true`                      | Enable/disable API rate limiting           |
| `RATE_LIMIT_OCR_PER_MINUTE`        | `30`                        | Rate limit for `POST /api/ocr/process`     |
| `RATE_LIMIT_DUPLICATE_PER_MINUTE`  | `60`                        | Rate limit for `POST /api/duplicate-check` |
| `RATE_LIMIT_READ_PER_MINUTE`       | `120`                       | Rate limit for `/api/jobs/*` & `/metrics`  |
| `RATE_LIMIT_HEALTH_PER_MINUTE`     | `60`                        | Rate limit for `/api/health` & `/`         |
| `RATE_LIMIT_AUTH_FAIL_PER_MINUTE`  | `15`                        | Brute-force limit for failed auth attempts |

Full reference: [`docs/SDD.md § 3`](SDD.md).

---

## Conventions

- Use `async`/`await` throughout (`asyncpg`, `motor`, `httpx`).
- PostgreSQL schema changes must be scripted under `alembic/versions/` and applied via `alembic upgrade head`.
- Pydantic models for all request/response validation (`app/api/schemas/`).
- Secrets in `.env` only — never hardcoded; never logged.
- All blocking OCR/LLM work goes in Celery tasks via `.delay()` — never inside FastAPI route handlers.
- Always pre-process images (OpenCV) before passing to Tesseract.
- No CORS — this service is backend-to-backend only.
- **No PII in logs.** Log job IDs and statuses only; never raw receipt text at `INFO` level in production.
- **Documentation Sync:** All agents MUST update the corresponding `/docs` files immediately whenever they introduce a new technology, logic change, architectural shift, or new rules that aren't yet documented.

---

## Subagent roster (Antigravity / Claude Code)

Defined in `docs/SAD.md § 5`. Materialized to `.agents/subagents/` (Antigravity) and `.claude/agents/*.md` (Claude Code).

| Agent ID | Name                       | Role                                                 | Spawn trigger              |
| -------- | -------------------------- | ---------------------------------------------------- | -------------------------- |
| SAD-A1   | `laravel-endpoint-builder` | Scaffolds Laravel 13 controllers, migrations, queues | Backend task initiation    |
| SAD-A2   | `vue-component-scaffolder` | Scaffolds Vue 3 SPA components                       | Frontend task initiation   |
| SAD-A3   | `serms-compliance-auditor` | Audits immutable logs, RBAC, BIR logic               | On backend component diffs |
| SAD-A4   | `reusability-auditor`      | Enforces Axiom A-09 "Reuse Before You Write"         | On any component diffs     |

### Agent Skills

Whenever the `!skill` command is used, or if the user's prompt aligns with the use description of a skill, agents must automatically consider, find, and use the relevant skill located at `C:\Users\mobar\.agents\skills`.

---

## QA & testing

- **Unit tests:** `pytest` + `pytest-asyncio` under `tests/unit/`. All external deps (Ollama, MongoDB, Redis, PostgreSQL) mocked — no live containers required.
- **Static analysis:** `ruff check .` (lint); `mypy` (type-checking, if configured).
- **Manual / exploratory:** Postman against `http://localhost:8010` with a valid Bearer token.
- **Real receipt images:** `docs/receipts/` contains ~30 sample receipt images (jpg/png). Use these as reference for manual testing, validating OCR accuracy, and reproducing extraction issues against real images. These are for local/manual reference only — never add them to `tests/fixtures/` (see fixture policy below).
- **CI gate:** PRs must not break `pytest` or introduce new lint errors.
- **Fixture data policy:** Synthetic only — no real receipt images or customer PII in `tests/fixtures/`.

Key test scenarios: see [`docs/QAD.md`](QAD.md) for H-01 – H-06 (happy paths), S-01 – S-07 (error paths), AB-01 – AB-04 (adversarial/abuse paths).

---

## Definition of Done

- `pytest` passes (zero failures).
- `ruff check .` passes (zero new lint errors).
- Docker containers build and start successfully (`docker compose up --build`).
- `alembic upgrade head` applies cleanly.
- H-01 through H-06 happy paths verified against live local stack.
- `403 Forbidden` returned for invalid API keys (AB-01).
- MongoDB job records contain `status`, `ocr_result`, `bir_validation`, `composite_confidence_score`, `duplicate_check` on completion.
- Callback delivery confirmed at SERMS and PRS test endpoints.
- No API keys or PII appear in application log output.
- Webhook integrations align with [`docs/api-contract.md`](api-contract.md).

Full release criteria: [`docs/QAD.md § 6`](QAD.md).

---

## Operational health

SLOs and alerting thresholds are defined in [`docs/OPS.md`](OPS.md). Quick reference:

| Signal                           | Target                            |
| -------------------------------- | --------------------------------- |
| API availability (`/api/health`) | ≥ 99% uptime                      |
| Job processing time (p95)        | ≤ 10 s                            |
| Celery task success rate         | ≥ 97% (after 3 retries)           |
| Callback delivery                | 100% on `completed`/`failed` jobs |
| Composite confidence score       | 95% of receipts ≥ 0.75            |

> **Known stubs:** `/api/health` dependency checks and `/api/metrics` aggregations are not yet wired. Wire before production go-live (see OPS.md self-check).
