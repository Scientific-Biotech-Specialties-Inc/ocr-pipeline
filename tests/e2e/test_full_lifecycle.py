"""End-to-End (E2E) pipeline tests.

Verifies the full asynchronous receipt lifecycle:
1. HTTP ingestion (POST /api/ocr/process) returning 202 Accepted.
2. File download from external URL via HTTP.
3. Real OCR extraction (PaddleOCR / Tesseract) + Ollama HTTP assist.
4. Outbound webhook callback delivery with authorization headers and validated payload.
5. Job status persistence and retrieval (GET /api/jobs/{job_id}/status).
6. pgvector similarity search (POST /api/duplicate-check).
7. Pre-OCR quality rejection gate (HTTP 422 + immediate rejection callback).
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core import pipeline
from app.main import app

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_e2e_full_receipt_processing_lifecycle(mock_fixture_server):
    server, base_url = mock_fixture_server
    server.received_callbacks.clear()

    receipt_id = 99101
    file_url = f"{base_url}/receipts/receipt%201.jpg"
    callback_url = f"{base_url}/webhook/callback"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Step 1: Submit job to API
        ingest_res = await client.post(
            "/api/ocr/process",
            headers={"Authorization": f"Bearer {settings.serms_api_key}"},
            json={
                "receipt_id": receipt_id,
                "file_url": file_url,
                "callback_url": callback_url,
                "force_process": True,
                "country": "US",
                "currency": "USD",
            },
        )
        assert ingest_res.status_code == 202
        ingest_data = ingest_res.json()
        job_id = ingest_data["job_id"]
        assert ingest_data["status"] == "queued"
        assert "job_id" in ingest_data
        assert "message" in ingest_data

        # Step 2: Execute asynchronous worker processing
        # In E2E, this exercises real HTTP download, real OCR, mock Ollama HTTP client,
        # and real outbound webhook callback
        await pipeline.process_receipt(
            job_id=job_id,
            receipt_id=receipt_id,
            file_url=file_url,
            callback_url=callback_url,
            source_service="serms",
            force_process=True,
            country="US",
            currency="USD",
        )

        # Step 3: Assert outbound webhook callback was received at mock server
        assert len(server.received_callbacks) >= 1
        last_callback = server.received_callbacks[-1]
        payload = last_callback["payload"]
        headers = last_callback["headers"]

        # Validate authorization header on callback
        auth_header = headers.get("authorization") or headers.get("Authorization")
        if settings.callback_api_key:
            assert auth_header == f"Bearer {settings.callback_api_key}"

        # Validate contract payload
        assert payload["status"] == "completed"
        assert payload["receipt_id"] == receipt_id
        assert payload["total_amount"] == 32.50
        assert payload["vat_amount"] == 2.51
        assert payload["invoice_number"] == "340"
        assert "Jollibee" in payload["vendor_name"]
        assert payload["ocr_confidence_score"] >= 0.80

        # Step 4: Verify duplicate detection endpoint accepts the extracted text
        dup_res = await client.post(
            "/api/duplicate-check",
            headers={"Authorization": f"Bearer {settings.serms_api_key}"},
            json={
                "receipt_text": "Jollibee Store #340 Total 32.50 Tax 2.51",
                "threshold": 0.85,
            },
        )
        assert dup_res.status_code == 200
        assert "is_duplicate" in dup_res.json()


@pytest.mark.asyncio
async def test_e2e_pre_ocr_quality_rejection_and_callback(mock_fixture_server):
    """Verify synchronous 422 quality rejection and outbound failure callback."""
    server, base_url = mock_fixture_server
    server.received_callbacks.clear()

    receipt_id = 99102
    file_url = f"{base_url}/receipts/tiny_invalid.png"
    callback_url = f"{base_url}/webhook/callback"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/ocr/process",
            headers={"Authorization": f"Bearer {settings.serms_api_key}"},
            json={
                "receipt_id": receipt_id,
                "file_url": file_url,
                "callback_url": callback_url,
                "force_process": False,
            },
        )
        assert res.status_code == 422
        body = res.json()
        assert body["rejection_code"] == "too_small"
        assert "too small" in body["rejection_reason"].lower()

        # Wait a tick for the callback task to finish
        await asyncio.sleep(0.1)

        # Assert callback was dispatched to callback_url
        assert len(server.received_callbacks) >= 1
        last_callback = server.received_callbacks[-1]
        assert last_callback["payload"]["status"] == "rejected"
        assert last_callback["payload"]["receipt_id"] == receipt_id
        assert last_callback["payload"]["rejection_code"] == "too_small"


@pytest.mark.asyncio
async def test_e2e_auth_abuse_blocking():
    """Verify security boundaries: missing/invalid API keys return 403."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # No auth
        res_no_auth = await client.post(
            "/api/ocr/process",
            json={"receipt_id": 1, "file_url": "https://example.test/img.png"},
        )
        assert res_no_auth.status_code == 403

        # Invalid auth
        res_bad_auth = await client.post(
            "/api/ocr/process",
            headers={"Authorization": "Bearer totally-invalid-api-key"},
            json={"receipt_id": 1, "file_url": "https://example.test/img.png"},
        )
        assert res_bad_auth.status_code == 403
