from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.middleware.rate_limit import RateLimitMiddleware, rate_limit_backend
from app.api.routes import duplicate, health, jobs, metrics, ocr
from app.config import settings
from app.db.mongodb import MongoDBClient


import asyncio
import logging

logger = logging.getLogger(__name__)


def _run_migrations():
    try:
        from alembic.config import Config
        from alembic import command
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic database migrations applied successfully.")
    except Exception as exc:
        logger.warning("Alembic migration skipped or failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    MongoDBClient.connect()
    await asyncio.to_thread(_run_migrations)
    yield
    MongoDBClient.close()
    await rate_limit_backend.close()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI OCR Pipeline for SERMS",
    lifespan=lifespan
)

app.add_middleware(RateLimitMiddleware)

app.include_router(health.router)
app.include_router(ocr.router)
app.include_router(duplicate.router)
app.include_router(jobs.router)
app.include_router(metrics.router)

@app.get("/")
def root():
    return {"message": "OCR Pipeline API is running."}
