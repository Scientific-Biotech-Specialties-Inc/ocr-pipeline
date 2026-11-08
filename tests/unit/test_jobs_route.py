from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.db.mongodb import MongoDBClient
from app.main import app


@pytest.mark.asyncio
async def test_get_job_status_success():
    created = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 11, 8, 12, 0, 5, tzinfo=timezone.utc)
    fake_job = {
        "job_id": "test-job-123",
        "receipt_id": 42,
        "source_service": "serms",
        "status": "completed",
        "created_at": created,
        "completed_at": completed,
    }

    with patch.object(MongoDBClient, "get_job", new_callable=AsyncMock) as mock_get_job:
        mock_get_job.return_value = fake_job
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get(
                "/api/jobs/test-job-123/status",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["job_id"] == "test-job-123"
            assert data["receipt_id"] == 42
            assert data["source_service"] == "serms"
            assert data["status"] == "completed"
            assert data["duration_ms"] == 5000
            assert "created_at" in data
            assert "completed_at" in data


@pytest.mark.asyncio
async def test_get_job_status_not_found():
    with patch.object(MongoDBClient, "get_job", new_callable=AsyncMock) as mock_get_job:
        mock_get_job.return_value = None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get(
                "/api/jobs/non-existent-id/status",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
            )
            assert res.status_code == 404
            assert res.json()["detail"] == "Job not found"


@pytest.mark.asyncio
async def test_get_job_status_source_service_mismatch_returns_404():
    fake_job = {
        "job_id": "prs-job-123",
        "receipt_id": 99,
        "source_service": "prs",
        "status": "completed",
        "created_at": datetime.now(timezone.utc),
    }

    with patch.object(MongoDBClient, "get_job", new_callable=AsyncMock) as mock_get_job:
        mock_get_job.return_value = fake_job
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # serms client tries to access prs job
            res = await client.get(
                "/api/jobs/prs-job-123/status",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
            )
            assert res.status_code == 404
            assert res.json()["detail"] == "Job not found"


@pytest.mark.asyncio
async def test_get_job_status_unauthorized():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/jobs/test-job/status")
        assert res.status_code == 403
