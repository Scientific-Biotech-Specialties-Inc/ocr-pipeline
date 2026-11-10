# Operations & Observability Runbook (OPS)

**Project:** OCR Pipeline
**Version:** 0.1
**Status:** Draft
**Last reconciled:** N/A — not yet reconciled with prod
**Related docs:** [SDD.md](SDD.md) · [QAD.md](QAD.md) · [Build.md](Build.md)

---

> **Scope reality:** This is a **headless Python/FastAPI microservice** running on Docker Compose. There is no UI and no server-rendered frontend. The operational surface is: keeping the API and Celery workers healthy, monitoring LLM API availability and quota, and ensuring the two consuming services (SERMS and PRS) receive reliable callbacks. The primary operational risks are **bad deploys**, **LLM API quota/timeout failures**, and **Celery queue blockage**.

---

## 1. SLOs & SLIs

| SLI (what you measure) | SLO (target) | Measured by | Breach action |
|---|---|---|---|
| API Availability (`/api/health`) | ≥ 99% uptime | Manual spot-checks (no formal uptime monitor yet) | Inspect container logs; restart failing service |
| OCR Job Processing Time (p95) | ≤ 30s across all receipt types | MongoDB `jobs` collection timestamps | Investigate LLM API latency, detector limits, or queue depth |
| OCR Job Processing Time (fast path) | Clean receipts complete in ~1.5s | MongoDB `jobs` collection timestamps | Review fast-path anchor scoring threshold (`FAST_PATH_ANCHOR_SCORE`) |
| OCR Job Processing Time (cache hit) | Exact duplicate receipts complete in < 200ms | MongoDB `jobs` collection timestamps | Verify Redis `ocr:file_hash:<sha256>` hit rate |
| OCR Throughput (worker) | Fallback escalation ≤ 10s per receipt | `scripts/benchmark.py` | PaddleOCR is primary; escalation runs a single-pass OpenCV + Tesseract fallback without multi-variant grid thrashing. |
| Celery Task Success Rate | ≥ 97% (after 3 retries) | MongoDB job status field | Alert on consecutive failures; inspect worker logs |
| Callback Delivery | 100% on completed/failed jobs | Application logs (`app.core.callback`) | Re-trigger callback manually if SERMS/PRS reports gap |
| Composite Confidence Score | 95% of receipts score ≥ 0.75 | MongoDB `composite_confidence_score` field | Review Paddle/Tesseract cross-reference, preprocessing, or Gemini extraction quality |

> **Note:** No formal uptime monitoring is in place yet. The `/api/health` endpoint exists but its dependency checks (Gemini, Redis, MongoDB, PostgreSQL) are currently stub TODO implementations. These should be wired before production go-live.

---

## 2. Observability — Logs, Metrics, Traces

All observability is currently via **Docker container logs** and **MongoDB job records**.

| Pillar | Tool | What's captured | How to access |
|---|---|---|---|
| Logs | Docker Compose (`docker compose logs`) | FastAPI request logs, Celery task lifecycle, pipeline step errors | `docker compose logs -f api` / `docker compose logs -f worker` |
| Metrics | `/api/metrics` endpoint (stub) | Job counts, success rate, avg processing time, queue depth, avg confidence score — **currently returns zeros** | `GET /api/metrics` with valid API key |
| Job State | MongoDB `jobs` collection | Per-job status (`queued`, `processing`, `completed`, `failed`), raw OCR text, extracted data, BIR validation, confidence score | Query MongoDB directly or via `GET /api/jobs/{job_id}` |
| Traces | — | N/A (no distributed tracing configured) | — |

**No-PII rule:** No real receipt images or customer PII should appear in logs. The pipeline logs job IDs and statuses only — never raw extracted text at `INFO` level in production.

**Key external dependencies to watch:**
- **Google AI Studio (`gemini-2.0-flash`):** Cloud LLM for assists, item normalization, and semantics. Monitor for API rate limits (`HTTP 429`) and network reachability.
- **Redis (`ocr_redis`):** Celery broker. If Redis is down, all job submissions fail silently to the queue.
- **SERMS & PRS callback URLs:** If the downstream service is unreachable, the pipeline logs a failure and raises — the Celery task will retry up to 3 times.

> **Architectural Transition:** The pipeline migrated from a local CPU-bound Ollama container (`qwen2.5:1.5b` / `qwen3:1.7b`) to Google AI Studio (`gemini-2.0-flash`). This removes the need for local GPU/CPU model residency, eliminates container OOM risks from local LLM weight allocation, and drops job processing latency significantly.

---

## 3. Alerting & On-Call

> **No formal alerting is configured yet.** The following are the recommended alert definitions to implement.

| Alert | Condition | Severity | Recommended action |
|---|---|---|---|
| API container down | `ocr_api` container exits or stops responding | P0 | Restart immediately; inspect logs |
| Worker container down | `ocr_worker` container exits | P0 | Restart; check if Redis is reachable |
| Gemini API rate limit / 429 | Consecutive 429 errors from Google AI Studio | P1 | Inspect request throughput; consider tier upgrade or backoff tuning |
| Celery queue depth high | Redis queue length > 50 | P1 | Scale workers or investigate blocking tasks |
| High job failure rate | >3% of jobs in `failed` state in last hour | P1 | Inspect MongoDB `failed` jobs; verify Gemini API key validity |
| Callback failures | Any `callback failed` in worker logs | P2 | Re-trigger manually; verify SERMS/PRS endpoint health |

**On-call model:** Best-effort by the development team. Monitor `docker compose ps` and MongoDB job statuses during active receipt processing windows.

---

## 4. Incident Response

**Severity ladder:**
- **P0:** API or worker completely down; no jobs can be processed.
- **P1:** Gemini API unavailable; high failure rate; queue blocked.
- **P2:** Callback delivery failures; degraded confidence scores.
- **P3:** Metrics endpoint returning stubs; minor logging gaps.

**When an incident fires:**
1. **Acknowledge** — claim ownership in the dev channel.
2. **Assess** — run `docker compose ps` to check container health; check `docker compose logs -f` for the failing service.
3. **Mitigate first** — restart the failing container: `docker compose restart <service>`.
4. **Rollback strategy:** No CI/CD automated rollback is defined yet. Until formalised: `git revert` the offending commit, rebuild the image, and redeploy with `docker compose up --build -d`. Use the previous image tag if available.
5. **Resolve & verify** — submit a test receipt via `POST /api/ocr/process` and confirm a `completed` job status in MongoDB.
6. **Postmortem** — for any P0/P1, document in `docs/pm-NNN.md` within 48h and fold action items back into this runbook.

---

## 5. Routine Operations

- **Dependency updates:** Regularly run `pip list --outdated` inside the container and update `pyproject.toml`. Always re-run `docker compose up --build` and `pytest` after updates.
- **Worker process pre-warming:** Celery worker processes automatically initialize PaddleOCR, Sentence-Transformers (`all-MiniLM-L6-v2`), and category prototype embeddings during worker process initialization (`@worker_process_init.connect`). This eliminates cold-start penalties on initial OCR tasks.
- **Alembic migrations:** Any schema change to the PostgreSQL `receipt_embeddings` table must be applied via Alembic before deploying new code: `alembic upgrade head`.
- **MongoDB storage:** Monitor the `ocr_mongo_data` volume growth; the `jobs` collection accumulates all OCR results.
- **API key rotation:** SERMS and PRS API keys, as well as `GEMINI_API_KEY`, are stored in `.env`. Rotate on a regular schedule and notify the consuming teams before rotation.

---

## Self-Check

- [ ] `/api/health` dependency checks are wired (Redis, MongoDB, PostgreSQL stubs resolved).
- [ ] `/api/metrics` endpoint aggregates real data from MongoDB.
- [ ] Rollback procedure documented and rehearsed.
- [ ] Formal uptime monitoring configured for the API endpoint.
- [ ] Celery queue depth monitoring in place.

## 6. Financial semantics operations
Gemini is an optional assist for tax-basis and currency semantics. Timeouts, malformed JSON, low confidence, or ungrounded evidence do not fail a receipt; the deterministic baseline is retained and the audit record identifies unresolved/review state. Monitor Mongo `financial_semantics` and `reconciliation.needs_manual_review` when investigating disputed totals. Live Gemini tests (`tests/live/test_financial_semantics_gemini.py`) are opt-in via `RUN_LIVE_LLM=1` and must not be part of the default offline test gate.
