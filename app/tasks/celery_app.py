import logging

from celery import Celery
from celery.signals import worker_process_init

from app.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "ocr_pipeline",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.process_receipt"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    worker_concurrency=1,
)


@worker_process_init.connect
def warm_worker_process(**kwargs):
    """Pre-warm CPU-heavy models so the first receipt pays zero cold-start penalty."""
    logger.info("Pre-warming OCR worker models...")
    try:
        from app.core.ocr_engine import _get_paddle_ocr
        _get_paddle_ocr()
    except Exception as exc:  # noqa: BLE001
        logger.debug("PaddleOCR pre-warm skipped or failed: %s", exc)

    try:
        from app.embeddings.generator import EmbeddingGenerator
        EmbeddingGenerator.get_model()
    except Exception as exc:  # noqa: BLE001
        logger.debug("EmbeddingGenerator pre-warm skipped or failed: %s", exc)

    try:
        from app.core.pipeline import _embedder
        _embedder()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Category prototype embeddings pre-warm skipped or failed: %s", exc)
    logger.info("OCR worker models pre-warmed successfully.")
