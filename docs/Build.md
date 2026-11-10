# Build & Setup Guide: OCR Pipeline

This guide details how to build, run, and test the OCR Pipeline microservice using Docker Compose. Since the application relies on specific system libraries (Tesseract OCR, OpenCV dependencies, Poppler, and local LLMs), running it via Docker is the recommended setup as it isolates all required operating system binaries automatically.

---

## 1. Prerequisites
Before setting up the project, make sure you have the following installed on your host system:
- **Docker Desktop** (version 20.10+ or equivalent with Compose V2 support)
- **Git** (for version control)

---

## 2. Dev Environment Setup & Commands

All development services (FastAPI, Redis, Celery, MongoDB, and PostgreSQL) are configured via the root `docker-compose.yml` file. LLM extraction is served via Google AI Studio (`gemini-2.0-flash`).

### Step 1: Clone and Configure Environment Variables
Copy the example environment file to `.env.local` to enable local overrides (note that the application loads `.env` by default, but local overrides should be placed here):
```powershell
cp .env.example .env
```
Ensure you update the placeholder values in `.env` (like `SERMS_API_KEY`, `POSTGRES_PASSWORD`) to secure your development instance.

### Step 2: Build and Launch Services
Spin up the service stack in detached mode. This command builds the custom FastAPI and Celery worker images:
```powershell
docker compose up -d --build
```
*Note: The initial build may take several minutes as it downloads PyTorch, Hugging Face models, and system OCR libraries.*

### Step 3: Run Database Migrations
Apply PostgreSQL database schema migrations using Alembic inside the API container:
```powershell
docker compose exec api alembic upgrade head
```

---

## 3. Operations & Logs Management

### Monitoring Container Status
Check that all 6 containers are running successfully:
```powershell
docker compose ps
```

### Viewing Container Logs
To follow live log outputs from all services, or filter by a specific container (e.g., Celery worker):
```powershell
# View all logs
docker compose logs -f

# View worker logs only
docker compose logs -f worker
```

### Stopping the Stack
To shut down the microservice stack and preserve database volumes:
```powershell
docker compose down
```
To wipe local databases and start clean, add the volumes flag:
```powershell
docker compose down -v
```

---

## 4. Running Automated Tests & Linting
The testing framework uses `pytest`, `pytest-asyncio`, and `ruff`.

### Linting & Static Analysis
```powershell
ruff check .
```

### Running Test Tiers Inside Docker
```powershell
# Unit tests (mocked dependencies)
docker compose exec api pytest tests/unit/ -v

# Integration tests (live pgvector, MongoDB, Redis)
docker compose exec api pytest tests/integration/ -v -m integration

# End-to-End tests (full microservice lifecycle)
docker compose exec api pytest tests/e2e/ -v -m e2e

# All automated tests
docker compose exec api pytest
```

### Running Locally (without Docker)
```powershell
# Unit test suite runs without containers
pytest tests/unit/ -v

# E2E test suite with built-in mock server
pytest tests/e2e/ -v -m e2e
```

---

## 5. Development Guardrails & Code Quality
- **Never Hardcode Secrets:** Place all keys, passwords, and tokens in your local environment file (`.env`).
- **Always Keep It Asynchronous:** Do not write blocking file processing logic inside API routes. Offload all intensive OCR computations to Celery tasks using `task.delay()`.
- **Pre-process Before OCR:** Always pre-process images (e.g., converting to grayscale, deskewing) via OpenCV before sending them to PyTesseract to maintain high extraction accuracy.
