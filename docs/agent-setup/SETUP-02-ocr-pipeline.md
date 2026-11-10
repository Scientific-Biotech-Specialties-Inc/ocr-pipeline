# SETUP-02 — ocr-pipeline (One-Shot Subagent Guide)

> **How to use this file:** This guide ships **inside the repo** at `docs/agent-setup/SETUP-02-ocr-pipeline.md` — the repo is already cloned to the user's machine. Feed this entire file to a subagent (or AI agent) and say _"Follow this guide end-to-end starting from the repo root."_ The subagent must execute every phase in order and handle everything itself. It must **STOP and ask the user** only at the marked `USER-INPUT-GATE`s (secret `.env` values). Nothing else requires user intervention.
> **Last verified:** Sept 6, 2026 · **Guide version:** 1.1.0 (in-repo edition — assumes the already-cloned repo) · **Covers repo:** `ocr-pipeline` (`main` branch default)

---

## 1. Subagent Directive (read first, obey strictly)

You are the **setup subagent** for `ocr-pipeline` — the standalone Python/FastAPI receipt-OCR microservice consumed by SERMS and PRS over service-to-service webhooks.

Rules:

1. Execute phases **0 → 8 in order**. Do not skip verification phases.
2. **Assume Windows 11 + Windows PowerShell 5.1** unless `uname` proves otherwise. PowerShell commands first; Bash/WSL variants noted.
3. **Docker Compose is the only supported path.** Do not attempt bare-metal Python runs (system libs — Tesseract OSD data, Poppler, PyTorch, Paddle — only resolve reliably inside the provided image). If the user has no Docker, install it (Phase 1A) rather than working around it.
4. **Never invent secrets.** At each `USER-INPUT-GATE`, stop, show exact keys, accept user values, then continue. If the user says "use safe local defaults", keep the documented local defaults — but warn that default API keys must change before exposing the service beyond `localhost`.
5. Quote paths with spaces. Verify parents with `Test-Path -LiteralPath "<parent>"` before creating anything.
6. First build is **slow** (10–20 min on modest hardware for PyTorch, PaddleOCR, and MiniLM model cache). Narrate progress; do not abort early. Do not run destructive `down -v` without explicit approval.
7. Finish only when **Definition of Done (§9)** is fully green. Report evidence (command + output), never claims.
8. **Do not create or add new files, and do not change the codebase, unless it is necessary to finish setup.** Setup legitimately creates: the root `.env` copied from `.env.example`, the external Docker network, containers/volumes, and applied Alembic migrations. Anything beyond that — new source files, edits to app code or configs, dependency changes — is out of scope: STOP, explain why you believe it is necessary, and ask the user first.
9. **Use the repo's other docs as references whenever you need them.** If a step is ambiguous or fails, consult `AGENTS.md` and the `docs/` suite (`Build.md`, `OPS.md`, `api-contract.md`, `SDD.md` — see `docs/index.md`) before improvising — and cite which doc resolved it in your report. If any doc conflicts with this guide, STOP and ask the user instead of guessing.

---

## 2. What You Are Setting Up

> [!NOTE] Architectural Transition (Ollama/Qwen → Gemini 2.0 Flash)
> The receipt OCR pipeline has fully transitioned from the self-hosted local LLM container (Ollama + `qwen2.5:1.5b` / `qwen3:1.7b`) to Google AI Studio's cloud API using `gemini-2.0-flash`. All structured field parsing, assists, item normalization, and semantic tie-breaking are executed via Gemini 2.0 Flash. The local `ocr_ollama` container, `Dockerfile.ollama`, and entrypoint scripts have been eliminated from the stack.

| Item               | Value (verified from repo)                                                                                                                                                                                                                                                                     |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Repo URL           | `https://github.com/Zanti00/ocr-pipeline.git`                                                                                                                                                                                                                                                  |
| Default local path | This repo — you are already inside the clone (this guide ships at `docs/agent-setup/` in it; typical path `C:\Projects\ocr-pipeline`)                                                                                                                                                          |
| Stack              | Python `>=3.12`, FastAPI `>=0.111`, Celery `>=5.4` + Redis, MongoDB `7`, PostgreSQL `pgvector/pgvector:pg16`, Google AI Studio (`gemini-2.0-flash`), PaddlePaddle `3.3.1` + PaddleOCR `3.7.0`, Tesseract + OSD data + Poppler, Sentence-Transformers `all-MiniLM-L6-v2` — see `pyproject.toml`, `Dockerfile` |
| Pipeline           | `FastAPI enqueue → Celery worker → OpenCV preprocess → Tesseract fast-path / PaddleOCR+Tesseract pool → Gemini 2.0 Flash structuring → BIR VAT + confidence → pgvector embedding → MongoDB persist → webhook callback`                                                                                   |
| Host ports         | API `8010`, Redis `6380→6379`, Mongo `27017`, Postgres `5433→5432`; worker has no host port                                                                                                                                                                                    |
| External network   | `shared-capstone-network` (external, must exist before `up`) + internal `ocr_network`                                                                                                                                                                                                          |
| Time / disk / RAM  | 15–25 min first build (PyTorch + OCR libs + MiniLM cache); ~6–8 GB disk; 8 GB RAM minimum, 16 GB recommended                                                                                                                                                              |

---

## 3. Phase 0 — Host Triage (never skip, never fails)

```powershell
$PSVersionTable.PSVersion; [Environment]::OSVersion.VersionString
Get-Command git -ErrorAction SilentlyContinue; git --version
Get-Command docker -ErrorAction SilentlyContinue; docker --version; docker compose version
Get-Command python -ErrorAction SilentlyContinue; python --version
Get-Command winget -ErrorAction SilentlyContinue; winget --version
Test-Path -LiteralPath "C:\Projects"
docker network ls | Select-String "shared-capstone-network"
docker ps --format "{{.Names}} ({{.Status}})"
```

**Interpretation:**

- `docker` missing → install Docker Desktop (Phase 1A). Non-negotiable for this repo.
- `git` missing → install Git (Phase 1B).
- `python` on host is **optional** (containers carry Python 3.12). Install host Python only if the user wants local `pytest`/linting outside containers.
- `shared-capstone-network` absent → create in Phase 3.
- If ports `8010`/`6380`/`27017`/`5433` are already bound, note the conflict now (Phase 6 tells you how to resolve).

---

## 4. Phase 1 — Zero-State Prerequisite Installer

### 1A. Docker Desktop (required)

```powershell
winget install --id Docker.DockerDesktop -e --accept-package-agreements --accept-source-agreements
# Fallback (no winget): https://www.docker.com/products/docker-desktop/
# Enable "Use the WSL 2 based engine", reboot if prompted.
```

1. Start **Docker Desktop**, wait for steady tray icon + green bottom-left status.
2. Fresh PowerShell window, then:
   ```powershell
   docker --version; docker compose version; docker info
   ```
3. Socket error `open //./pipe/dockerDesktopLinuxEngine ...` → Docker Desktop is not running. Start/restart it.

### 1B. Git (required)

```powershell
winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements
# Fallback: https://git-scm.com/download/win
git --version
```

### 1C. Host Python (optional — containers already include Python 3.12)

```powershell
# Only if user wants host-side tooling:
winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
# Fallback: https://www.python.org/downloads/ (3.12.x, check "Add python.exe to PATH")
python --version  # expect 3.12.x
```

> Open a **new** PowerShell window after any install (stale `PATH` otherwise).

---

## 5. Phase 2 — Locate the Repo (already cloned — verify, do not re-clone)

This guide ships inside the clone. Start from the repo root (the directory containing `docker-compose.yml`, `app/`, and `pyproject.toml`):

```powershell
# Verify you are at the repo root:
Test-Path -LiteralPath ".\docker-compose.yml"
Test-Path -LiteralPath ".\app"
Test-Path -LiteralPath ".\pyproject.toml"
git status -sb; git branch --show-current; git log --oneline -3
```

Reuse this clone as-is; never wipe uncommitted work. Record branch/commit for the final report.

> **Fallback only:** if the repo is somehow missing from this machine, clone it with `git clone https://github.com/Zanti00/ocr-pipeline.git` and start over from Phase 0 in the fresh clone.

---

## 6. Phase 3 — Shared Network + Environment File

### 3A. External network (idempotent)

```powershell
docker network ls | Select-String "shared-capstone-network"
# If no output:
docker network create shared-capstone-network
```

`docker-compose.yml` attaches `api` + `worker` to both `ocr_network` (internal) and `shared-capstone-network` (for SERMS/PRS callbacks). `up` fails without the external network.

### 3B. USER-INPUT-GATE — `.env` values (STOP here if secrets are missing)

Single file: root `.env` from `.env.example` (loaded via `pydantic-settings`; mounted at `/app` in containers).

```powershell
Test-Path -LiteralPath ".\.env"
# If False:
Copy-Item -LiteralPath ".\.env.example" -Destination ".\.env"
```

| Key                                                                                                               | What to do                                                                                                                                                                                                                                                                                  |
| :---------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `SERMS_API_KEY` (default `serms-ocr-service-key-change-me`)                                                       | Inbound Bearer token SERMS must present. Local dev: default works **only if** SERMS `AI_SERVICE_API_KEY` matches it. For any shared/demo deployment, ask user for a fresh strong value and mirror it into SERMS (`AI_SERVICE_API_KEY`) and PRS equivalents. Never expose defaults publicly. |
| `PRS_API_KEY` (default `prs-ocr-service-key-change-me`)                                                           | Same rule for the PRS consumer. Keep default locally; rotate for shared envs.                                                                                                                                                                                                               |
| `CALLBACK_API_KEY` (default `ocr-callback-key-change-me`)                                                         | Outbound key the worker sends to consumer `callback_url`s. Must match what SERMS/PRS expect (`AI_SERVICE_API_KEY` on their side). Confirm consistency across repos.                                                                                                                         |
| `GEMINI_API_KEY` (required) / `LLM_PROVIDER` (`gemini`) / `GEMINI_MODEL` (`gemini-2.0-flash`)                     | Provide a valid Google AI Studio API key (`AIza...`). Cloud-hosted Gemini 2.0 Flash replaces the previous local Ollama container.                                                                                                                           |
| `ENABLE_LLM_ITEM_NORMALIZATION` (`true`)                                                                          | Enables batch semantic cleanup and normalization of line item descriptions via Gemini 2.0 Flash.                                                                                                                                                             |
| `REDIS_URL` / `MONGODB_URL` / `MONGODB_DATABASE` / `POSTGRES_URL` (+ `POSTGRES_USER/PASSWORD/DB`)                 | Keep compose defaults (`ocr_redis`, `ocr_mongo`, `ocr_postgres:5432/ocr_pipeline`). Change only if host ports collide (see §8).                                                                                                                                                             |
| `EMBEDDING_MODEL` (`all-MiniLM-L6-v2`), `DUPLICATE_SIMILARITY_THRESHOLD` (`0.85`), `DUPLICATE_DAYS_WINDOW` (`90`) | Keep defaults unless user tunes accuracy/recall.                                                                                                                                                                                                                                            |
| `OCR_POOL_VARIANTS` (`all`), `OCR_POOL_PSMS` (`6,4,11`), `OCR_LANGUAGE` (`eng`)                                   | Keep defaults (full accuracy pool). Shrinking trades accuracy for speed — only on user request (see `docs/SDD.md`).                                                                                                                                                                         |
| `OMP_NUM_THREADS` / `OMP_THREAD_LIMIT` (`1`)                                                                      | **Do not change.** Set in compose `worker` env; measured 26× slowdown if raised (Tesseract OpenMP thrash).                                                                                                                                                                                  |
| `CALLBACK_MAX_RETRIES` (`3`), `CALLBACK_BACKOFF_SECONDS` (`10,30,60`)                                             | Keep defaults.                                                                                                                                                                                                                                                                              |

> **STOP condition:** If the user runs SERMS/PRS against this OCR instance, collect the three API keys now and ensure both sides match. Mismatches surface as `401` on `/api/ocr/process` or silent callback rejections — the hardest failure to debug later.

---

## 7. Phase 4 — Build, Start, Migrate

```powershell
# From repo root:
docker compose up -d --build
docker compose ps
```

Expect a **first build**: system OCR libs + PyTorch + MiniLM download (`scripts/download_model.py`).

Then run Postgres migrations (required — pgvector embeddings table):

```powershell
docker compose exec api alembic upgrade head
docker compose ps
docker compose logs --tail=60 api
docker compose logs --tail=40 worker
```

Wait until `api`, `worker`, `redis`, `mongodb`, `postgres` are all `Up`/`running` (5 containers) and `api` serves `:8010`.

---

## 8. Phase 5 — Verify (must all pass before declaring done)

```powershell
docker compose ps
# Liveness (no auth required):
curl.exe http://localhost:8010/api/health
# Or PowerShell:
Invoke-WebRequest -Uri "http://localhost:8010/api/health" -UseBasicParsing | Select-Object StatusCode, Content
# Interactive docs:
Start-Process "http://localhost:8010/docs"
# Auth check (expect 401 without token, 202/422 with valid key — proves key wiring):
curl.exe http://localhost:8010/api/metrics
```

| Check                                                                                                                                          | Expected                                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| `docker compose ps`                                                                                                                            | 5 containers `Up`: `ocr_api`, `ocr_worker`, `ocr_redis`, `ocr_mongo`, `ocr_postgres`               |
| `GET /api/health`                                                                                                                              | `200` (dependency checks currently stubbed — liveness only)                                        |
| `GET /api/metrics` without `Authorization`                                                                                                     | `401` (auth enforced)                                                                              |
| `POST /api/ocr/process` with `Authorization: Bearer <SERMS_API_KEY>` + valid JSON (`receipt_id`, `file_url`, `callback_url`, `source_service`) | `202` + `job_id`; worker logs show download → OCR → Gemini → callback attempt                      |
| `GET /api/jobs/{job_id}/status`                                                                                                                | Transitions `queued → processing → done/failed`                                                    |
| `docker compose exec api alembic current`                                                                                                      | At `head` (migrations applied)                                                                     |
| Optional: `docker compose exec api pytest` / `pytest -v`                                                                                       | Suite passes (run if time permits)                                                                 |

Minimal authenticated smoke test (replace key + use a reachable `file_url`; for a no-callback probe, point `callback_url` at a local catcher or omit per `docs/api-contract.md` if your version allows):

```powershell
$key = "<SERMS_API_KEY from .env>"
$body = @{ receipt_id = 1; file_url = "https://example.com/receipt.jpg"; callback_url = "http://host.docker.internal:8010/api/health"; source_service = "serms" } | ConvertTo-Json
Invoke-WebRequest -Uri "http://localhost:8010/api/ocr/process" -Method Post -Headers @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" } -Body $body -UseBasicParsing | Select-Object StatusCode, Content
docker compose logs --tail=50 worker
```

If `202` + worker activity appears, the pipeline is functionally done. Full request/response shapes: `docs/api-contract.md`.

---

## 9. Phase 6 — Troubleshooting (consult before retrying blindly)

| Symptom                                                            | Cause → Fix                                                                                                                                                                                                                                   |
| :----------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `network shared-capstone-network ... not found`                    | Missed §6-3A → `docker network create shared-capstone-network`, `up -d` again.                                                                                                                                                                |
| `port is already allocated` (`8010`/`6380`/`27017`/`5433`)          | Another stack or local service holds it → `docker ps`, stop the holder or remap that service's host port in compose (keep container ports).                                                                                                   |
| First `up --build` seems stuck                                     | Normal: PyTorch + MiniLM download. Watch `docker compose logs -f api worker` — progress appears there, not in `up` output.                                                                                                                      |
| Jobs stay `queued`                                                 | Worker/Redis issue → `docker compose logs -f worker`, confirm `ocr_redis` Up; `docker compose restart worker redis`.                                                                                                                          |
| `401` on `/api/ocr/process`                                        | Bearer ≠ `SERMS_API_KEY`/`PRS_API_KEY` in `.env` → fix key or header; `docker compose up -d --force-recreate api worker` after `.env` edits.                                                                                                  |
| `pgvector` / embedding errors                                      | Migrations not applied → `docker compose exec api alembic upgrade head`.                                                                                                                                                                      |
| LLM / extraction errors                                            | Check `GEMINI_API_KEY` in `.env`; verify key is valid with Google AI Studio and network egress to `generativelanguage.googleapis.com` is open.                                                                                               |
| Callback never arrives at SERMS/PRS                                | Consumer URL unreachable from `ocr_network`/`shared-capstone-network` → worker logs show retries (`CALLBACK_MAX_RETRIES`/`BACKOFF`); use container-DNS names (`http://serms_api:8000/...`) for compose-to-compose callbacks, not `localhost`. |
| `PaddleOCR does not appear in worker logs`                         | Stale image after dep change → `docker compose up -d --build worker api`; verify `docker compose exec -T worker python -c "import paddle, paddleocr; print('PaddleOCR installed')"`.                                                          |
| `OMP` slowness (12s → 250s+ per receipt)                           | Someone raised `OMP_NUM_THREADS` → reset to `1` in compose worker env.                                                                                                                                                                        |

Daily commands:

```powershell
docker compose ps
docker compose logs -f
docker compose logs -f worker
docker compose exec api pytest
docker compose down        # stop, keep volumes
# docker compose down -v   # DESTRUCTIVE: wipes Mongo + Postgres data — ask user first
```

---

## 10. Phase 7 — Stop / Reset

- **Stop (safe):** `docker compose down` — preserves `ocr_mongo_data`, `ocr_postgres_data`.
- **Full reset (destructive):** `docker compose down -v` — wipes jobs and embeddings. Only with explicit user approval.
- **Rebuild after dep change:** `docker compose up -d --build worker api`.

---

## 11. Definition of Done (all boxes must be checked)

- [ ] Docker Desktop running, `shared-capstone-network` exists.
- [ ] Operating from the repo root of this clone, branch/commit recorded.
- [ ] Root `.env` present; `SERMS_API_KEY` / `PRS_API_KEY` / `CALLBACK_API_KEY` / `GEMINI_API_KEY` resolved with user (defaults explicitly accepted or rotated).
- [ ] `docker compose ps` shows all 5 containers `Up`.
- [ ] `alembic upgrade head` applied (`alembic current` at head).
- [ ] `GET http://localhost:8010/api/health` returns `200`; `GET /docs` loads.
- [ ] Authenticated `POST /api/ocr/process` returns `202` + `job_id` (or documented reason why skipped).
- [ ] `GEMINI_API_KEY` tested and functional against Google AI Studio `gemini-2.0-flash`.
- [ ] Report lists: repo path, branch/commit, container states, health/auth outputs, key-consistency note with SERMS/PRS, and any deviations.

---

## 12. Appendix — File Map (where things live)

```text
.
├── app/
│   ├── api/routes/     # FastAPI routers (ocr, jobs, duplicate, health, metrics)
│   ├── api/schemas/    # Pydantic request/response models
│   ├── core/           # Pipeline, OCR, preprocessing, BIR, confidence, callbacks
│   ├── db/             # MongoDB + Postgres clients and models
│   ├── embeddings/     # Sentence-Transformers + similarity
│   ├── llm/            # LLM provider abstraction (Gemini 2.0 Flash)
│   ├── tasks/          # Celery app + process_receipt task
│   └── config.py       # Settings from environment (.env via pydantic-settings)
├── alembic/            # Postgres migrations (embeddings table)
├── scripts/            # download_model.py (MiniLM cache)
├── tests/              # pytest suite (run inside api container)
├── docs/               # PRD/SAD/SDD/DSD/Build/OPS/api-contract/AGENTS
├── Dockerfile          # Python 3.12-slim + Tesseract/OSD/Poppler + deps + model cache
└── docker-compose.yml  # api + worker + redis + mongodb + postgres (5 containers)
```

Full API shapes: `docs/api-contract.md`. Build/run/test source: `docs/Build.md`. Ops runbook: `docs/OPS.md`.
