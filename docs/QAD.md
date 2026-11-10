# QA & Test Plan (QAD)

**Project:** OCR Pipeline
**Version:** 0.1
**Status:** Draft
**Last reconciled:** N/A — not yet reconciled with code
**Related docs:** [PRD.md](PRD.md) · [SDD.md](SDD.md) · [OPS.md](OPS.md)

---

> This is a **headless Python/FastAPI microservice**. There is no UI, no frontend, and no database used directly by end users. The QA surface covers: API endpoint validation, Celery background task correctness, OCR pipeline accuracy (PaddleOCR/Tesseract cross-reference + Gemini extraction), BIR validation logic, pgvector duplicate detection, and callback delivery. Consuming services are **SERMS** (expense management) and **PRS** (Productivity Report System).

---

## 1. Testing Strategy & Scope

**In Scope:**
- FastAPI route validation (authentication, request schemas, response codes).
- BIR TIN validation and VAT classification logic (`bir_validator.py`).
- Confidence score computation (`confidence.py`).
- Celery task retry logic (3 retries with exponential backoff).
- Embedding generation and pgvector similarity search (duplicate detection).
- Callback delivery logic (`callback.py`).
- Alembic migration correctness.

**Out of Scope:**
- SERMS and PRS application logic (separate projects).
- Google AI Studio (Gemini) model accuracy itself (treated as an external API dependency; only the integration contract is tested).
- Load / stress testing (not applicable for V1).
- UI / browser testing (no frontend).

**Testing levels:**

| Level | Tooling | Environment | Path | Owner |
|---|---|---|---|---|
| Lint & Static analysis | `ruff` | Local / GitHub Actions | Root | Developer |
| Unit tests | `pytest` + `pytest-asyncio` | Mocked (no live containers) | `tests/unit/` | Developer |
| Integration tests | `pytest` + live containers (`pgvector`, `mongo:7`, `redis:alpine`) | Local Docker / GitHub Actions `services:` | `tests/integration/` | Developer / QA |
| End-to-End tests | `pytest` + full microservice flow + mock Gemini & Webhook | Local / GitHub Actions `services:` | `tests/e2e/` | Developer / QA |
| Live LLM tests | Opt-in live `gemini-2.0-flash` via `RUN_LIVE_LLM=1` | External Google AI Studio API | `tests/live/` | Developer / AI Lead |
| Manual API testing | Postman / `httpx` against live stack | Docker Compose (`http://localhost:8010`) | Local | Developer / QA |

**Mocking & Isolation strategy:**
- **Unit tests (`tests/unit/`):** All external dependencies (Gemini, MongoDB, Redis, PostgreSQL) are **mocked** using `unittest.mock` / `pytest-mock`. No live containers or external API keys required.
- **Integration tests (`tests/integration/`):** PostgreSQL `pgvector`, MongoDB, and Redis run as live containers. Verifies Alembic migrations, pgvector cosine search, MongoDB job states, Redis caching, and real OpenCV/PaddleOCR/Tesseract engine passes.
- **End-to-End tests (`tests/e2e/`):** Runs the full asynchronous receipt lifecycle (HTTP ingestion $\to$ Celery task execution $\to$ OCR $\to$ Mock Gemini $\to$ Webhook callback $\to$ Duplicate detection) using a self-contained in-memory fixture server for receipts, callbacks, and deterministic LLM responses.

---

## 2. Test Environments & Data

**Local development:** `docker compose up -d` with `.env.local`.
**Test runner:** `pytest` — runs without Docker; all dependencies are mocked.

**Data policy:** No real receipt images or customer PII are to be used in tests. All test data must use **synthetic, programmatically generated fixtures** (e.g., `tests/fixtures/synthetic_receipt.png`, mocked raw OCR text strings). Pre-created fixture files should be committed to `tests/fixtures/` and reviewed for PII before merging.

| Fixture type | Location | Examples |
|---|---|---|
| Synthetic receipt images | `tests/fixtures/images/` | `valid_receipt.png`, `blurry_receipt.png`, `no_tin_receipt.png` |
| Mocked OCR text strings | Inline in test files or `tests/fixtures/text/` | `valid_tin_receipt.txt`, `empty_text.txt` |
| Mocked API payloads | Inline in test files | JSON for `/api/ocr/process` request body |

---

## 3. Core Test Scenarios

### Happy Paths (must all pass before release)

| ID | Scenario | Steps | Expected Result |
|----|---|---|---|
| H-01 | Submit valid receipt via webhook | `POST /api/ocr/process` with valid API key + file URL | Returns `202` with `job_id` and `status: queued` |
| H-02 | Celery task completes successfully | Task processes mocked pipeline steps | MongoDB job status transitions to `completed`; callback is sent |
| H-03 | BIR TIN validation — valid TIN | `validate_tin()` called with a properly formatted 9-digit TIN | Returns `True` |
| H-04 | Confidence score threshold | `compute_composite_score()` with high OCR confidence + full extracted data | Returns a composite score ≥ 0.75 |
| H-07 | Paddle/Tesseract candidate selection | Same fixture is processed with Paddle and Tesseract candidates | The highest-scoring candidate becomes primary; other readings remain available; failed engines degrade independently |
| H-05 | Duplicate detection | Submit two receipts with identical text | pgvector similarity search returns the existing match |
| H-06 | Callback delivery | Mocked `httpx` client; pipeline completes | `send_callback()` POSTs the correct payload to the `callback_url` |

### Sad Paths (edge cases & error handling)

| ID | Scenario | Input / Trigger | Expected Behavior |
|----|---|---|---|
| S-01 | Invalid API key | `POST /api/ocr/process` with missing or wrong API key | Returns `403 Forbidden` |
| S-02 | Malformed request body | Missing `receipt_id` or `file_url` | Returns `422 Unprocessable Entity` |
| S-03 | File download failure | `file_url` returns a non-200 response | Job marked `failed`; failure callback sent; Celery retries |
| S-04 | Gemini timeout / unstructured output | Mocked Gemini returns empty or malformed JSON | Pipeline logs error; job marked `failed`; Celery retries (up to 3x) |
| S-05 | BIR TIN invalid | `validate_tin()` with malformed TIN | Returns `False`; `tin_valid: false` in job record |
| S-06 | All Celery retries exhausted | Task fails 3 consecutive times | Final `failed` status written to MongoDB; failure callback dispatched |
| S-07 | Confidence score below threshold | Candidate scores low or extracted data is sparse | Composite score < 0.75; flagged in job record for review |
| S-08 | Fast-path single pass is insufficient | Clean OCR but unreconciled money or missing total | Pipeline escalates to full pooled OCR; hard receipts keep pooled accuracy |
| S-09 | Fast-path OCR failure | Preprocessing raises during `read_fast` | Fail-soft: worker falls back to pooled OCR; job completes normally |
| S-10 | Consolidated assist fails | Mocked Gemini returns malformed JSON for the single assist call | Deterministic baseline retained; vendor/location/semantics/subtotal fall back individually |
| S-11 | OCR separator variants break invoice-number lookup | Fast-path text reads `Invoice number; 45065` or `No, 38326` (semicolon/comma) | `INVOICE_LABEL_RE` accepts `;` and `,` separators; covered by `tests/unit/test_extractors.py` |
| S-12 | `Sdn. Bhd.` company suffix misread as currency | Malaysian `my_pos` receipt contains `Sdn. Bhd.` | `explicit_currency` skips `BHD` when preceded by `Sdn.`; locale falls back to `MYR` via country |
| S-13 | Malaysian SST receipt basis unresolved | `my_pos` prints `Subtotal` / `SST 6%` / `TOTAL` with no inclusive/exclusive wording | `infer_tax_basis` adds a MY SST/GST country fallback → exclusive, so `total_sales = net + tax` derives |
| S-14 | First fast-path pass too weak on a clean receipt | Preprocessed single pass scores below 6.0 (e.g. `syn-0000`, deskew/denoise degrades the rendering) | `read_fast_fallback` runs `flat` at PSM 11 before the pool; eligible fallback early-exits (~5s) instead of paying the ~200s pooled pass |
| S-15 | Pooled Tesseract grid without OpenMP caps | Escalated pool runs 15 Tesseract passes concurrently, each spawning a full 8-thread OpenMP team | Without `OMP_NUM_THREADS=1`/`OMP_THREAD_LIMIT=1` the grid measures 258-309s vs ~12s sequential for identical readings (26x thrash penalty); with caps it runs in ~4s. Worker compose env must set both vars |
| S-16 | Pool size tuning via `OCR_POOL_VARIANTS`/`OCR_POOL_PSMS` | Reduced pool (e.g. `raw,flat,clean` × `6,11`) on hard receipts | **Expected to regress**: full-pool 94.7% machine-printed vs reduced 94.2%, moderate 96.3% vs 92.6% (`syn-0155` invoice MISSED, `syn-0156` currency WRONG). Winning readings span all 5 variants and all 3 PSMs (e.g. `sauvola/psm4` on real photos), so the full pool stays the default; settings exist for deployment tuning |
| S-17 | Tall grocery-POS receipts (`receipt 24-26.jpg`): price-first layout | Lines read `64.00 V` / `1 × 64.00` / `4800016522533 DRD PEV-CUT EP025G` — barcode has no decimal so never a money token; names were emitted as `V` | Pending-item mechanism: money line with non-name remainder holds the item until the following lines supply quantity and name; summary lines flush. Measured: 24-35 line items per receipt with prices/quantities (`tests/unit/test_items.py` `TestGroceryPriceFirstLayout`) |
| S-18 | Store-code names mangled by correctors | `DRD PEV-CUT EP025G` → `ORD PEANUT EP025G` (dictionary distance-2), LLM batch rewrite | `looks_like_store_code()` (uppercase + mixed alnum token like `EP025G`) excludes codes from dictionary correction and skips the ~82s LLM batch call when codes are ≥ half of valid names (`tests/unit/test_items.py` `TestLooksLikeStoreCode`, `tests/unit/test_postprocessing.py`) |
| S-19 | Tesseract garbage in `combined_text` | Supporting Tesseract readings scored ~0.36 vs Paddle 4.46 but their text polluted the LLM prompt (wrong vendor/country) | `_supporting_readings` drops readings below `primary.score / 2.0`; combined text stays clean (`tests/unit/test_ocr_engine.py`) |
| S-20 | Ollama generation timeout on long prompts | 768-token generation at ~9.4 t/s takes ~82s; 90s timeout hit → 500 + retry → 172s for one answer | Assists timeout raised to 150s (`app/llm/ollama_provider.py`); receipt 24 LLM stage 309s → 62s; full pipeline 381s → 136s |
| S-21 | Multi-segment long receipts without footer total (`receipt 27-29.jpg`) | Faded or cut-off footers leave `total_amount` null; unpopulated anchors caused false `not_a_receipt` rejection | `merge_segment_readings` populates unified `anchors`; `is_likely_receipt` falls back to `score_text` and validates on `item_scan.count >= 2`; parsed line items (≥ 3 rows) emit with `near_reconciled = True` and `needs_manual_review = True` |
| S-22 | Ollama HTTP 400 string duration parse failure | `keep_alive: "-1"` string sent to Go backend fails `time.ParseDuration("-1")` with missing duration unit | Numeric strings (`"-1"`, `"0"`, `"300"`) cast to `int` in `OllamaProvider._generate`; context clamped to 2000 head + 2000 tail characters to fit `num_ctx=2048` safely |

### Abuse / Adversarial Paths

| ID | Attack | Trigger | Expected Defense |
|----|---|---|---|
| AB-01 | Unauthenticated job submission | `POST /api/ocr/process` with no API key | `403 Forbidden` — `verify_api_key` dependency blocks request |
| AB-02 | API key enumeration / brute force | Submit requests with guessed keys | Constant-time `secrets.compare_digest` check; requests exceeding `rate_limit_auth_fail_per_minute` (15/min) are blocked with `429 Too Many Requests` |
| AB-03 | Oversized payload / file | Submit a very large `file_url` pointing to a huge file | `httpx` 30s timeout; pipeline fails safely and retries |
| AB-04 | Secret leakage in logs | Inspect application log output | API keys, raw PII, and `.env` values must not appear in logs |
| AB-05 | Endpoint flooding / DoS | Exceed per-minute rate limit on any endpoint | Returns `429 Too Many Requests` with `Retry-After` header and `ocr:ratelimit:*` tracking |

---

## 4. Automation vs. Manual Testing

### Automated (run on every PR)

```bash
# 1. Static type / lint check
ruff check .

# 2. Unit test suite — no Docker required, all deps mocked
pytest tests/unit/ -v

# 3. Integration test suite — live pgvector, mongodb, redis containers
pytest tests/integration/ -v -m integration

# 4. End-to-End test suite — full pipeline + mock Gemini & Webhook
pytest tests/e2e/ -v -m e2e

# 5. Verify Alembic migrations
alembic check
alembic upgrade head
```

**CI/CD Pipeline (`.github/workflows/ci.yml`):**
Runs on every PR and push to `main`:
1. `lint`: `ruff check .`
2. `unit`: `pytest tests/unit/ -v`
3. `integration`: Runs against GitHub Actions service containers (`pgvector/pgvector:pg16`, `mongo:7`, `redis:alpine`) with real OCR packages (`tesseract-ocr`, `poppler-utils`).
4. `e2e`: Runs full asynchronous flow test suite against service containers + mock Gemini fixture server.

Azure Container Apps deployment (`.github/workflows/deploy.yml`) is strictly gated on the CI workflow passing.

Fast-path and consolidated-assist behavior is covered by unit tests: `tests/unit/test_fast_path.py` (sufficiency gate), `tests/unit/test_ocr_engine.py` (early-exit, fallback pass, escalation, fail-soft), `tests/unit/test_postprocessing.py` (`use_llm=False` dictionary-only normalization), `tests/unit/test_extractors.py` (semicolon/comma invoice separators), `tests/unit/test_locale.py` (`Sdn. Bhd.` not currency), `tests/unit/test_financial_semantics.py` (MY SST/GST exclusive fallback). No live containers required.

### Manual / Exploratory

- Submit a test receipt via Postman to `POST http://localhost:8010/api/ocr/process` using a valid API key.
- Verify MongoDB job record transitions: `queued` → `processing` → `completed`.
- Verify the callback payload is received at the configured `callback_url`.
- Inspect `composite_confidence_score` in the MongoDB job document.
- Test the `/api/health` and `/api/metrics` endpoints for correct response schemas.
- Test the duplicate detection endpoint (`POST /api/duplicate/check`) with the same receipt submitted twice.
- **Fast path:** run `scripts/try_receipt.py` over a clean receipt; the `ocr_selection` block reports a single fast-pass candidate (fast path), while hard receipts report the full pooled candidate set.
- **Offline timing benchmark:** `docker compose exec worker python scripts/benchmark.py` reports per-stage averages; compare `engine_total` before/after optimization work (target: average receipt < 60 s).

---

## 5. Bug Triage Protocol

| Severity | Definition | Action |
|---|---|---|
| **P0 — Blocker** | API won't start; Celery workers crashing on every task; callback never delivered; all jobs stuck in `queued` | Cannot release. Fix immediately. |
| **P1 — High** | Celery retry exhaustion on valid receipts; `validate_tin` returning wrong results; confidence score always 0; duplicate detection false positives | Cannot release. Fix before next deploy. |
| **P2 — Medium** | Confidence score threshold missed on edge-case receipts; non-critical fields missing from extracted data | Can release. Fix in next iteration. |
| **P3 — Low** | Log verbosity issues; cosmetic API response field ordering; minor doc inaccuracy | Backlog. Address in refactor cycle. |

**Launch-blocking threshold:** Zero open P0 and P1 bugs.

---

## 6. Release Criteria (Definition of Done)

A release is approved when all the following are true:

- [ ] All unit tests in `tests/unit/` pass (`pytest tests/unit/`).
- [ ] No open P0 or P1 bugs.
- [ ] H-01 through H-06 happy paths pass against the live local Docker Compose stack.
- [ ] Alembic migrations apply cleanly (`alembic upgrade head`).
- [ ] Docker containers build successfully (`docker compose up --build`).
- [ ] API key authentication returns `403` for invalid keys (AB-01 verified manually).
- [ ] MongoDB job records contain the correct fields (`status`, `extracted_data`, `bir_validation`, `composite_confidence_score`) on completion.
- [ ] Callback delivery confirmed to reach SERMS and PRS test endpoints.
- [ ] No API keys or PII present in application log output.

---

## 7. AI / LLM Evaluation

The pipeline uses **Google AI Studio (`gemini-2.0-flash`)** for structured field extraction assists, line-item spelling normalization, and category tie-breaking from raw OCR text.

> **Architectural Transition:** The project migrated from local CPU-based Ollama (`qwen2.5:1.5b` / `qwen3:1.7b`) to Google AI Studio (`gemini-2.0-flash`). This shift eliminated high local container RAM utilization and multi-minute Ollama wait times on receipt batches, while enabling live line-item normalization.

**Eval scope:**
- Structured extraction correctness is validated via unit tests with mocked Gemini responses (not live model evaluation).
- Real-world model quality (hallucinations, missing fields) is assessed during manual testing using synthetic fixture receipts.
- If extraction accuracy falls below acceptable thresholds in production (identified via `composite_confidence_score` distribution in MongoDB), an investigation into prompt engineering or model upgrade should be opened as a P1.

> **Red-team note:** If the Gemini model is ever exposed to an external API surface or user-provided prompts, a prompt injection assessment must be added here.

## 8. Financial semantics QA
Offline tests cover provider JSON parsing/failure, confidence and OCR-evidence grounding, caller/receipt/OCR/LLM currency precedence, inclusive/exclusive/unknown arithmetic, conditional computed totals, multiple tax-line aggregation, item target selection, request normalization, and optional callback fields. Live model cases use `@pytest.mark.live_llm` and an explicit opt-in command (`tests/live/test_financial_semantics_gemini.py`); they skip when `RUN_LIVE_LLM` is unset or `GEMINI_API_KEY` is unavailable, and assert stable semantic outcomes rather than exact model wording. The default suite never requires external live API keys.

Known limitation: tax aggregation relies on the existing geometry label scanner. A tax-like line that cannot be paired to a money token, or a layout where tax and discount labels are inseparable, is withheld rather than aggressively interpreted.
