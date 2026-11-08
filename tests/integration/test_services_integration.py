"""Integration tests for live backing services (PostgreSQL + pgvector, MongoDB, Redis).

These tests run against real service instances when configured (e.g. in CI or local Docker Compose).
If a backing service is unreachable, tests skip cleanly with a clear explanation.
"""

from __future__ import annotations

import socket
import urllib.parse
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.config import settings
from app.core.file_cache import (
    cache_downloaded_bytes,
    get_cached_ocr_result,
    pop_cached_bytes,
    set_cached_ocr_result,
)
from app.db.models import ReceiptEmbedding
from app.db.mongodb import MongoDBClient
from app.db.postgres import AsyncSessionLocal
from app.embeddings.similarity import find_similar_receipts
from app.main import app

pytestmark = pytest.mark.integration


def _is_service_reachable(url: str, default_port: int) -> bool:
    """Check TCP connectivity to a service URL host and port."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or default_port
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _is_postgres_up() -> bool:
    return _is_service_reachable(settings.postgres_url, 5432)


def _is_mongo_up() -> bool:
    return _is_service_reachable(settings.mongodb_url, 27017)


def _is_redis_up() -> bool:
    return _is_service_reachable(settings.redis_url, 6379)


@pytest.mark.asyncio
async def test_postgres_pgvector_lifecycle():
    """Verify storing, querying by cosine similarity, and deleting embeddings in pgvector."""
    if not _is_postgres_up():
        pytest.skip(f"PostgreSQL not reachable at {settings.postgres_url}")

    receipt_id = 999901
    source_service = "serms"
    embedding = [0.05] * 384
    receipt_text = "Sample test receipt for pgvector integration"

    async with AsyncSessionLocal() as session:
        # Cleanup any existing test row
        existing = await session.execute(
            select(ReceiptEmbedding).where(
                ReceiptEmbedding.receipt_id == receipt_id,
                ReceiptEmbedding.source_service == source_service,
            )
        )
        for row in existing.scalars().all():
            await session.delete(row)
        await session.commit()

        # Insert new embedding
        record = ReceiptEmbedding(
            receipt_id=receipt_id,
            source_service=source_service,
            embedding=embedding,
            receipt_text=receipt_text,
            created_at=datetime.now(timezone.utc),
        )
        session.add(record)
        await session.commit()

        # Query cosine similarity (exact match should yield similarity close to 1.0)
        matches = await find_similar_receipts(
            session=session,
            embedding=embedding,
            source_service=source_service,
            threshold=0.85,
            days_window=90,
        )
        assert len(matches) >= 1
        assert any(m.receipt_id == receipt_id for m in matches)

    # Test deletion via API endpoint
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        del_res = await client.delete(
            f"/api/duplicate/embeddings/{receipt_id}",
            headers={"Authorization": f"Bearer {settings.serms_api_key}"},
        )
        assert del_res.status_code == 204

    # Verify record was purged
    async with AsyncSessionLocal() as session:
        check = await session.execute(
            select(ReceiptEmbedding).where(
                ReceiptEmbedding.receipt_id == receipt_id,
                ReceiptEmbedding.source_service == source_service,
            )
        )
        assert check.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_mongodb_job_lifecycle():
    """Verify creating, updating, and querying job status from MongoDB."""
    if not _is_mongo_up():
        pytest.skip(f"MongoDB not reachable at {settings.mongodb_url}")

    job_id = "integration-mongo-job-1"
    receipt_id = 888801
    job_payload = {
        "job_id": job_id,
        "receipt_id": receipt_id,
        "source_service": "serms",
        "file_url": "https://example.test/receipt.jpg",
        "callback_url": "https://serms.test/callback",
        "status": "queued",
    }

    # Create job in MongoDB
    await MongoDBClient.create_job(job_payload)

    # Verify job retrieved
    job = await MongoDBClient.get_job(job_id)
    assert job is not None
    assert job["job_id"] == job_id
    assert job["status"] == "queued"

    # Update job status
    await MongoDBClient.update_job_status(job_id, "completed", {"total_amount": 150.0})
    updated = await MongoDBClient.get_job(job_id)
    assert updated["status"] == "completed"
    assert updated["total_amount"] == 150.0

    # Query via HTTP status route
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get(
            f"/api/jobs/{job_id}/status",
            headers={"Authorization": f"Bearer {settings.serms_api_key}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["job_id"] == job_id
        assert data["receipt_id"] == receipt_id
        assert data["status"] == "completed"


@pytest.mark.asyncio
async def test_redis_cache_and_hash_result():
    """Verify Redis byte caching handoff and 90-day OCR result hashing."""
    if not _is_redis_up():
        pytest.skip(f"Redis not reachable at {settings.redis_url}")

    test_url = "https://example.test/receipt_cache_test.jpg"
    test_bytes = b"FAKE_RECEIPT_IMAGE_BYTES_FOR_CACHE_TEST"

    # Test short-lived byte handoff
    await cache_downloaded_bytes(test_url, test_bytes, ttl_seconds=60)
    retrieved = await pop_cached_bytes(test_url)
    assert retrieved == test_bytes

    # Second pop should return None (GETDEL semantic)
    second_pop = await pop_cached_bytes(test_url)
    assert second_pop is None

    # Test 90-day file-hash result caching
    test_hash = "abcdef0123456789abcdef0123456789"
    cached_payload = {
        "vendor_name": "Cached Test Vendor",
        "total_amount": 99.50,
        "ocr_confidence_score": 0.95,
    }
    await set_cached_ocr_result(test_hash, cached_payload, ttl_seconds=60)
    result = await get_cached_ocr_result(test_hash)
    assert result is not None
    assert result["vendor_name"] == "Cached Test Vendor"
    assert result["total_amount"] == 99.50
