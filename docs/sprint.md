# OCR Pipeline — Sprint Board

**Created:** 2026-07-04 · **Updated:** 2026-07-29 · **Owner:** SERMS Core Team (Lead Engineer) · **Status:** active

This is the canonical working task board for the **OCR Pipeline** microservice, derived from the **Capstone 6-Week Deliverable Schedule** and system architecture specs ([`SDD.md`](SDD.md), [`SAD.md`](SAD.md), [`OPS.md`](OPS.md)). It is the single place agents and humans track *what to do next*, *which tasks are completed*, and *which code surfaces each task touches*.

- Product vision & non-goals: see [`PRD.md`](PRD.md).
- Architectural patterns & subagents: see [`SAD.md`](SAD.md) and [`AGENTS.md`](../AGENTS.md).
- System routes & env vars: see [`SDD.md`](SDD.md).
- Polyglot database schemas: see [`DSD.md`](DSD.md).
- Operational runbooks & SLOs: see [`OPS.md`](OPS.md).

---

## How to use this file

- Tasks are organized by the **6-Week Capstone Deliverable Schedule** (July 4 – August 8, 2026) and the **Production Roadmap**.
- Each task lists **Priority · Owner · Status** and a **References** block linking required docs and source files.
- **Status values:** `🔴 todo` · `🟡 in progress` · `✅ done` · `📋 committed` (roadmap item) · `❌ dropped`.
- **Priority values:** `P0` blocker · `P1` important · `P2` nice-to-have.

---

## Reference index

**System & Architecture Docs (`docs/`)**

| Doc | Description |
|---|---|
| [`index.md`](index.md) | Documentation suite master index & health check |
| [`PRD.md`](PRD.md) | Product Requirements Document — core features, target audience, non-goals |
| [`SAD.md`](SAD.md) | Software Architecture Document — tech stack, sequence flows, subagent roster |
| [`SDD.md`](SDD.md) | System Design Document — API routes, Celery tasks, env vars |
| [`DSD.md`](DSD.md) | Detailed System Design — PostgreSQL pgvector, MongoDB schemas, ERD, Redis cache |
| [`OPS.md`](OPS.md) | Operations & Observability Runbook — SLOs, container health, incident response |
| [`QAD.md`](QAD.md) | Quality Assurance Document — test scenarios H-01–H-06, S-01–S-07, AB-01–AB-04 |
| [`Build.md`](Build.md) | Build & Setup Guide — Docker Compose instructions and test suite execution |
| [`api-contract.md`](api-contract.md) | Service API Contract — request/response shapes, quality gate, callback payload |
| [`SUMMARY.md`](SUMMARY.md) | High-level project executive summary & key user flows |
| [`6-WEEK-DELIVERABLE-SCHEDULE.md`](6-WEEK-DELIVERABLE-SCHEDULE.md) | Master 6-week deliverable schedule & milestones (July 4 – August 8, 2026) |
| [`3rd-deliverable.md`](3rd-deliverable.md) | Polyglot persistence proof & pipeline integration report (Week 3 milestone) |
| [`4th-deliverable-final.md`](4th-deliverable-final.md) | AI integration, vector DB population & analytics queries (Week 4 milestone) |

**Engineering Docs & Infrastructure**

| File | Description |
|---|---|
| [`../AGENTS.md`](../AGENTS.md) | Repo guide & system operational rules for agents |
| [`../docker-compose.yml`](../docker-compose.yml) | Multi-container Docker orchestration (`api`, `worker`, `redis`, `mongo`, `postgres`, `ollama`) |
| [`../pyproject.toml`](../pyproject.toml) | Python dependencies, Celery, FastAPI, OpenCV, and Pytest configuration |
| [`../alembic.ini`](../alembic.ini) | Database migration configuration for PostgreSQL |
| [`../.env.example`](../.env.example) | Environment variable template & defaults |

**Key Code Surfaces**

| Area | Path |
|---|---|
| Fast API Routes & Gateway | [`../app/api/`](../app/api/) (`routes/`, `schemas/`, `middleware/`) |
| Asynchronous Celery Tasks | [`../app/tasks/`](../app/tasks/) (`pipeline.py`, `worker.py`) |
| Image Pre-processing & OCR | [`../app/core/preprocessing.py`](../app/core/preprocessing.py), [`../app/core/ocr_engine.py`](../app/core/ocr_engine.py), [`../app/core/textquality.py`](../app/core/textquality.py) |
| BIR Tax & Quality Validation | [`../app/core/bir_validator.py`](../app/core/bir_validator.py), [`../app/core/confidence.py`](../app/core/confidence.py) |
| Local LLM Extraction (Ollama) | [`../app/llm/`](../app/llm/) (`factory.py`, `ollama_provider.py`) |
| Embeddings & Vector Search | [`../app/embeddings/generator.py`](../app/embeddings/generator.py), [`../app/db/postgres.py`](../app/db/postgres.py) |
| Anomaly Risk Machine Learning | [`../app/eval/anomaly_detector.py`](../app/eval/anomaly_detector.py), [`../app/eval/train_model.py`](../app/eval/train_model.py) |
| Database Persistence Models | [`../app/db/models.py`](../app/db/models.py), [`../app/db/mongodb.py`](../app/db/mongodb.py), [`../alembic/versions/`](../alembic/versions/) |
| Test Suites | [`../tests/unit/`](../tests/unit/), [`../tests/fixtures/`](../tests/fixtures/) |

---

## Locked Architectural Decisions

| # | Decision | Detail |
|---|---|---|
| D1 | **Asynchronous Microservice Architecture** | All heavy OCR, LLM inference, embedding generation, and vector searches run out-of-band in Celery background workers. FastAPI returns `202 Accepted` immediately upon ingestion. |
| D2 | **Polyglot Persistence Layer** | Five specialized storage tiers: **PostgreSQL + pgvector** (384-dim embeddings & similarity search), **MongoDB** (job logs, extracted fields, BIR results), **Redis** (Celery broker & file hash cache), **Supabase Storage** (object storage), and **MySQL** (upstream SERMS core data). |
| D3 | **Local LLM Extraction (Ollama `qwen2.5:1.5b`)** | Field parsing (vendor, date, total, VAT, TIN, line items, expense category) is executed locally via Ollama to avoid external vendor API costs and ensure data privacy. |
| D4 | **Pre-OCR Image Enhancement** | OpenCV performs adaptive thresholding, binarization, deskewing, and noise reduction prior to PyTesseract character extraction to maximize OCR text quality. |
| D5 | **Pre-OCR Quality Gate** | Ingestion requests evaluate image dimensions, blur, and brightness synchronously. Bad images are rejected with `422 Unprocessable Entity` before queuing unless `force_process=true`. |
| D6 | **Service-to-Service Security & Webhooks** | API endpoints strictly authenticated via `SERMS_API_KEY` or `PRS_API_KEY` Bearer tokens. Webhooks sent back to callers append `CALLBACK_API_KEY` with exponential backoff (10s → 30s → 60s). |
| D7 | **BIR Compliance & Composite Confidence** | Receipts are validated against Philippine BIR TIN formats (`xxx-xxx-xxx-xxxV`) and classified for VAT/Non-VAT. Composite confidence score (≥ 0.75 threshold) combines OCR raw score (50%), field completeness (40%), and TIN validity (10%). |
| D8 | **Machine Learning Anomaly Risk Scoring** | `RandomForestClassifier` scores receipt anomaly risk (0.0–1.0) and assigns risk tiers (`Low`, `Medium`, `High Risk`) to catch fraudulent or inconsistent expense claims. |

---

## Sprint Breakdown — Capstone 6-Week Schedule

### Week 1 — Project Initiation & Requirements Definition (July 4, 2026)

> Goal: Establish project scope, core service charter, decoupled architecture diagrams, and risk register.

- **W1-1 · Project Charter & Scope Definition · `P0` · ✅ done**  
  Documented project vision, functional bounds, microservice decoupling strategy, and success metrics.  
  - **References:** [`PRD.md`](PRD.md) §1–2, [`SUMMARY.md`](SUMMARY.md)

- **W1-2 · High-Level System Architecture (Arch 1) · `P0` · ✅ done**  
  Drafted high-level business data flow showing asynchronous processing decoupling between SERMS backend and OCR Pipeline.  
  - **References:** [`SAD.md`](SAD.md) §1, [`6-WEEK-DELIVERABLE-SCHEDULE.md`](6-WEEK-DELIVERABLE-SCHEDULE.md) §Week 1

- **W1-3 · Initial Risk Register & Mitigation Strategy · `P1` · ✅ done**  
  Identified operational risks including Ollama OOM, worker queue blockage, Tesseract low confidence, and webhook callback retries.  
  - **References:** [`OPS.md`](OPS.md) §3–4

---

### Week 2 — Full Architecture Design & Project Planning (July 11, 2026)

> Goal: Finalize logical, physical, process, and detailed database architectures (SDD & DSD).

- **W2-1 · System Design & API Contract Specification · `P0` · ✅ done**  
  Defined REST endpoints (`/api/ocr/process`, `/api/duplicate-check`, `/api/jobs/{job_id}/status`, `/api/health`, `/api/metrics`) and JSON request/response schemas.  
  - **References:** [`SDD.md`](SDD.md) §1, [`api-contract.md`](api-contract.md)

- **W2-2 · Polyglot Database Design (pgvector + MongoDB + Redis) · `P0` · ✅ done**  
  Designed `receipt_embeddings` PostgreSQL schema with `vector(384)`, MongoDB `ocr_jobs` document schema, and Redis key namespace `ocr:file_hash:<sha256>`.  
  - **References:** [`DSD.md`](DSD.md) §1–2

- **W2-3 · Sequence Diagrams & Process Architecture (Arch 4) · `P1` · ✅ done**  
  Constructed Mermaid sequence diagrams illustrating ingestion, Celery worker processing, vector duplicate checks, and webhook callback execution.  
  - **References:** [`SDD.md`](SDD.md) §2.1, [`SAD.md`](SAD.md) §3

---

### Week 3 — Data Layer Implementation & Polyglot Persistence (July 18, 2026)

> Goal: Provision polyglot databases, run Alembic migrations, write Celery processing task, and validate data sync.

- **W3-1 · Polyglot Database Provisioning & Docker Setup · `P0` · ✅ done**  
  Spun up Docker Compose environment with `ocr_postgres` (PostgreSQL 16 + pgvector), `ocr_mongo` (MongoDB 7), and `ocr_redis` (Redis Alpine).  
  - **References:** [`Build.md`](Build.md) §1, [`3rd-deliverable.md`](3rd-deliverable.md) §1

- **W3-2 · PostgreSQL Alembic Migrations & Models · `P0` · ✅ done**  
  Applied Alembic migration creating `receipt_embeddings` table and associated indices (`source_service`, `created_at`).  
  - **References:** [`DSD.md`](DSD.md) §3, [`alembic/versions/`](../alembic/versions/)

- **W3-3 · Async Processing Core Pipeline Implementation · `P0` · ✅ done**  
  Implemented `process_receipt_task` in `app/tasks/pipeline.py` handling image downloading, OpenCV preprocessing, PyTesseract text extraction, MongoDB job logging, and pgvector storage.  
  - **References:** [`3rd-deliverable.md`](3rd-deliverable.md) §2.2, [`app/tasks/pipeline.py`](../app/tasks/pipeline.py)

- **W3-4 · Data Validation & Database Sync Verification · `P1` · ✅ done**  
  Verified 1:1 match across MySQL `receipts` (4,567 rows), MongoDB `ocr_jobs`, and PostgreSQL `receipt_embeddings`.  
  - **References:** [`3rd-deliverable.md`](3rd-deliverable.md) §3.3

---

### Week 4 — AI Integration & Analytics Engine (July 25, 2026)

> Goal: Integrate local LLM field extraction, pgvector duplicate detection, BIR tax validation, and anomaly risk scoring.

- **W4-1 · Ollama LLM Integration (`qwen2.5:1.5b`) · `P0` · ✅ done**  
  Configured Ollama provider to parse raw OCR text into structured JSON fields (vendor, transaction date, total amount, VAT amount, TIN, invoice number, line items, expense category).  
  - **References:** [`app/llm/ollama_provider.py`](../app/llm/ollama_provider.py), [`SAD.md`](SAD.md) §2

- **W4-2 · Sentence-Transformers Embedding Generation & Duplicate Check · `P0` · ✅ done**  
  Integrated `all-MiniLM-L6-v2` model for 384-dim vector generation; wired cosine similarity lookup within 90-day window (threshold `0.85`).  
  - **References:** [`app/embeddings/generator.py`](../app/embeddings/generator.py), [`DSD.md`](DSD.md) §1.1

- **W4-3 · BIR Tax Compliance Validator & Composite Confidence Scoring · `P0` · ✅ done**  
  Implemented BIR TIN pattern validation (`validate_tin`), VAT classification (`classify_vat`), and composite scoring formula (50% OCR + 40% fields + 10% TIN).  
  - **References:** [`app/core/bir_validator.py`](../app/core/bir_validator.py), [`app/core/confidence.py`](../app/core/confidence.py)

- **W4-4 · Machine Learning Anomaly Risk Classifier · `P1` · ✅ done**  
  Trained `RandomForestClassifier` on receipt features yielding an anomaly risk score (0.0–1.0) and risk category (`Low`, `Medium`, `High Risk`).  
  - **References:** [`app/eval/anomaly_detector.py`](../app/eval/anomaly_detector.py), [`app/eval/train_model.py`](../app/eval/train_model.py)

- **W4-5 · Pre-OCR Quality Gate Rejection Filter · `P1` · ✅ done**  
  Added synchronous quality evaluation (`too_small`, `blurry`, `too_dark`) returning `422 Unprocessable Entity` when image quality fails pre-checks.  
  - **References:** [`app/core/textquality.py`](../app/core/textquality.py), [`api-contract.md`](api-contract.md)

---

### Week 5 — System Integration, Testing & Ops Hardening (August 1, 2026)

> Goal: End-to-end integration testing, operational health check dependency resolution, and real metrics aggregation.

- **W5-1 · Pytest Unit & Integration Suite Coverage · `P0` · ✅ done**  
  Built mock-backed test suite under `tests/unit/` verifying happy path (H-01–H-06), error handling (S-01–S-07), and security scenarios (AB-01–AB-04).  
  - **References:** [`QAD.md`](QAD.md), [`tests/unit/`](../tests/unit/)

- **W5-2 · Direct Health Check Dependency Integration (`/api/health`) · `P0` · 🟡 in progress**  
  Wire actual ping checks to MongoDB, PostgreSQL, Redis, and Ollama inside `/api/health` (resolving current stub TODOs).  
  - **References:** [`OPS.md`](OPS.md) §1, [`app/api/routes/health.py`](../app/api/routes/health.py)

- **W5-3 · Metrics Endpoint Real Data Aggregation (`/api/metrics`) · `P1` · 🟡 in progress**  
  Aggregate real processing metrics (total job count, success rate, p95 processing duration, queue depth, average confidence score) from MongoDB.  
  - **References:** [`OPS.md`](OPS.md) §2, [`app/api/routes/metrics.py`](../app/api/routes/metrics.py)

- **W5-4 · Webhook Retry & Exponential Backoff Hardening · `P1` · ✅ done**  
  Configured Celery task retries with 10s → 30s → 60s backoff schedule on webhook callback HTTP errors.  
  - **References:** [`app/core/callback.py`](../app/core/callback.py), [`SDD.md`](SDD.md) §1.2

- **W5-5 · End-to-End System Integration Test Report · `P1` · 🔴 todo**  
  Document full flow execution (SERMS Ingestion → FastAPI → Celery Worker → OpenCV/Tesseract → Ollama → pgvector → MongoDB → Callback).  
  - **References:** [`6-WEEK-DELIVERABLE-SCHEDULE.md`](6-WEEK-DELIVERABLE-SCHEDULE.md) §Week 5

---

### Week 6 — Final Package, Review & Production Roadmap (August 8, 2026)

> Goal: Finalize documentation package, conduct final review, and establish production roadmap.

- **W6-1 · Complete Documentation Suite Reconciliation · `P0` · 📋 committed**  
  Reconcile version numbers (`1.0.0`) and schema definitions across `PRD.md`, `SDD.md`, `DSD.md`, `SAD.md`, `OPS.md`, `QAD.md`, and `Build.md`.  
  - **References:** [`index.md`](index.md) §3

- **W6-2 · Complete Final Submission Package · `P0` · 📋 committed**  
  Package repository source code, architecture blueprints, project management documents, and 5-minute video demonstration into final deliverable archive.  
  - **References:** [`6-WEEK-DELIVERABLE-SCHEDULE.md`](6-WEEK-DELIVERABLE-SCHEDULE.md) §Week 6

---

## Production Roadmap & Operational Debt

> Post-capstone production hardening and operational scale tasks.

| Task ID | Component | Priority | Status | Description | References |
|---|---|---|---|---|---|
| **PR-01** | Observability | `P1` | `📋 committed` | Implement Prometheus exporter and Grafana dashboard for Celery worker queue depth and processing latency. | [`OPS.md`](OPS.md) §2 |
| **PR-02** | Security | `P1` | `📋 committed` | Establish automated 90-day API Key rotation schedule for `SERMS_API_KEY` and `PRS_API_KEY`. | [`OPS.md`](OPS.md) §5 |
| **PR-03** | Data Management | `P2` | `📋 committed` | Configure automated TTL cleanup policies in MongoDB for raw OCR job logs older than 180 days. | [`DSD.md`](DSD.md) §1.2 |
| **PR-04** | Performance | `P2` | `📋 committed` | Apply HNSW / IVFFlat index tuning on PostgreSQL `receipt_embeddings` table for sub-millisecond vector similarity queries at scale. | [`DSD.md`](DSD.md) §1.1 |
| **PR-05** | CI/CD | `P1` | `✅ done` | Implement GitHub Actions workflow (`ci.yml`) for running `ruff check`, `pytest` (unit, integration, e2e), and gating Azure deployment. | [`AGENTS.md`](../AGENTS.md), [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) |
| **PR-06** | Performance / Accuracy | `P1` | `✅ done` | Long/tall grocery receipts (`docs/receipts/receipt 23-29.jpg`): Paddle `text_det_limit_side_len=512` for tall images (~30% faster), supporting-reading score floor (no Tesseract garbage in `combined_text`), price-first layout parsing with verbatim store codes, LLM batch normalization gate, Ollama assists timeout 90s→150s. Measured: receipt 24 full pipeline 381s→136s; receipts 24/25/26 extract 24–35 line items with prices/quantities. Receipt 23 (two-column photo) partially fixed — vendor/invoice extracted, items deferred to column-split pass. Totals of receipts 24–26 remain OCR-illegible in the tail region → `needs_manual_review` flag. | [`QAD.md`](QAD.md) §S-17–S-19, [`SDD.md`](SDD.md) §1.2 |
| **PR-07** | Multi-Segment & Long Receipt Handling | `P0` | `✅ done` | Parallel multi-part segment ingestion (`file_urls` up to 6), 3-zone vertical blur check, segment role scoring & auto-ordering, text-level seam deduplication, middle-segment density fast-path bypass, single-image virtual strip tiling, multi-line split grocery item extraction (quantity preceding/following price), and long receipt near-reconciliation (<= 3% discrepancy on >= 20 items, confidence capped at 0.79 with `REASON_NEAR_RECONCILED`). | [`SDD.md`](SDD.md) §2.4, [`api-contract.md`](api-contract.md) |
| **PR-08** | OCR Accuracy / Robustness | `P0` | `✅ done` | Enhanced receipt date extraction: broad format coverage (numeric with `/`, `-`, `.`, spaces, 2/4-digit years, textual month variants `Nov 11, 2026`, `11-Nov-2026`, `'26`, ordinals), adjacent POS timestamp extraction & boundary safety (`07/31/2026(Fri)`), scoped character-to-digit recovery (`O`/`o` $\to$ `0`, `I`/`l` $\to$ `1`, `S` $\to$ `5`, `B` $\to$ `8`), token-span proximity & same-line expiry discrimination, PHT timezone tolerance (+1 day ceiling), and verification/anchors/pipeline date grounding synchronization. | [`SDD.md`](SDD.md) §2.5, [`app/core/extractors.py`](../app/core/extractors.py) |

---

*Keep this board current. When a task completes, update its Status here and reconcile with system documents.*
