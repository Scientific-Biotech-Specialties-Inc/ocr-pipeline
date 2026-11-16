"""Unit tests for two-tier faded receipt detection and non-overridable rejection."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.config import settings
from app.core import pipeline
from app.core.anchors import AnchorScore
from app.core.extraction import Extraction
from app.core.items import ExtractedItem, ItemScan
from app.core.ocr_engine import OcrBundle, OcrReading
from app.core.reconcile import MoneyResult
from app.main import app


def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_faded_image(width: int = 400, height: int = 600) -> Image.Image:
    arr = np.full((height, width), 235, dtype=np.uint8)
    for y in range(50, height - 50, 40):
        arr[y : y + 5, 50 : width - 50] = 215
    return Image.fromarray(arr)


def _make_blurry_image(width: int = 400, height: int = 600) -> Image.Image:
    arr = np.full((height, width), 200, dtype=np.uint8)
    for y in range(50, height - 50, 40):
        arr[y : y + 5, 50 : width - 50] = 0
        arr[y + 10 : y + 15, 50 : width - 50] = 255
    return Image.fromarray(cv2.GaussianBlur(arr, (25, 25), 0))


def _make_partially_faded_image(width: int = 400, height: int = 900) -> Image.Image:
    arr = np.full((height, width), 225, dtype=np.uint8)
    for y in range(40, 560, 35):
        arr[y : y + 4, 40 : width - 40] = 0
        arr[y + 8 : y + 12, 40 : width - 40] = 255
    return Image.fromarray(arr)


def _make_bundle(text: str, *, confidence: float = 0.85, anchors: AnchorScore | None = None) -> OcrBundle:
    reading = OcrReading(
        text=text,
        words=[],
        confidence=confidence,
        variant="raw",
        psm=6,
        anchors=anchors,
    )
    return OcrBundle(primary=reading, supporting=[], all_readings=[reading])


@pytest.mark.asyncio
@pytest.mark.parametrize("force_process", [False, True])
async def test_api_rejects_faded_receipt_even_when_forced(force_process: bool):
    faded_bytes = _png_bytes(_make_faded_image())
    created_jobs: list[dict] = []
    callbacks: list[dict] = []

    async def fake_create_job(doc: dict):
        created_jobs.append(doc)

    async def fake_send_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.api.routes.ocr._download_file", new=AsyncMock(return_value=faded_bytes)),
        patch("app.api.routes.ocr.MongoDBClient.create_job", side_effect=fake_create_job),
        patch("app.api.routes.ocr.send_callback", side_effect=fake_send_callback),
        patch("app.api.routes.ocr.cache_downloaded_bytes", new=AsyncMock()),
        patch("app.api.routes.ocr.process_receipt_task.delay") as mock_delay,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/ocr/process",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
                json={
                    "receipt_id": 701,
                    "file_url": "https://example.test/faded.png",
                    "callback_url": "https://serms.test/callback",
                    "force_process": force_process,
                },
            )

    assert res.status_code == 422
    body = res.json()
    assert body["status"] == "rejected"
    assert body["rejection_code"] == "faded"
    assert "faded" in body["rejection_reason"].lower()
    mock_delay.assert_not_called()
    assert len(created_jobs) == 1
    assert created_jobs[0]["rejection_code"] == "faded"
    assert len(callbacks) == 1
    assert callbacks[0]["rejection_code"] == "faded"


@pytest.mark.asyncio
async def test_api_allows_blurry_receipt_when_force_process_true():
    blurry_bytes = _png_bytes(_make_blurry_image())

    with (
        patch("app.api.routes.ocr._download_file", new=AsyncMock(return_value=blurry_bytes)),
        patch("app.api.routes.ocr.MongoDBClient.create_job", new=AsyncMock()),
        patch("app.api.routes.ocr.cache_downloaded_bytes", new=AsyncMock()),
        patch("app.api.routes.ocr.process_receipt_task.delay") as mock_delay,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/ocr/process",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
                json={
                    "receipt_id": 702,
                    "file_url": "https://example.test/blurry.png",
                    "callback_url": "https://serms.test/callback",
                    "force_process": True,
                },
            )

    assert res.status_code == 202
    assert res.json()["status"] == "queued"
    mock_delay.assert_called_once()


@pytest.mark.asyncio
async def test_worker_rejects_faded_pdf_pre_ocr():
    faded_img = _make_faded_image()
    updates: list[tuple[str, dict]] = []
    callbacks: list[dict] = []

    async def fake_update(job_id: str, status: str, extra_fields: dict | None = None):
        updates.append((status, extra_fields or {}))

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=b"%PDF-1.4")),
        patch("app.core.pipeline.load_images", return_value=[faded_img]),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock()) as mock_ocr,
        patch("app.core.pipeline.MongoDBClient.update_job_status", side_effect=fake_update),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-faded-pdf",
            receipt_id=703,
            file_url="https://example.test/faded.pdf",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    mock_ocr.assert_not_called()
    assert updates[-1][0] == "rejected"
    assert updates[-1][1]["rejection_code"] == "faded"
    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "faded"


@pytest.mark.asyncio
async def test_worker_tier2_rejects_partially_faded_receipt_missing_total_even_when_forced():
    partial_img = _make_partially_faded_image()
    bundle = _make_bundle(
        "ACME STORE\nTIN 123-456-789-000\nDate 2026-03-01",
        confidence=0.82,
        anchors=AnchorScore(total=3.0, money_tokens=0, keyword_hits=1, has_tax_id=True, has_date=True),
    )
    extraction = Extraction(fields={"vendor_name": "ACME STORE", "total_amount": None})
    updates: list[tuple[str, dict]] = []
    callbacks: list[dict] = []

    async def fake_update(_job_id: str, status: str, extra_fields: dict | None = None):
        updates.append((status, extra_fields or {}))

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(partial_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", side_effect=fake_update),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-partial-fade-fail",
            receipt_id=704,
            file_url="https://example.test/partial.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    assert updates[-1][0] == "rejected"
    assert updates[-1][1]["rejection_code"] == "faded"
    assert callbacks[-1]["rejection_code"] == "faded"


@pytest.mark.asyncio
async def test_worker_tier2_reclassifies_faded_slip_with_no_signals_from_not_a_receipt():
    partial_img = _make_partially_faded_image()
    bundle = _make_bundle("... .. .", confidence=0.35, anchors=AnchorScore(total=0.0))
    extraction = Extraction(fields={"vendor_name": None, "total_amount": None})
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(partial_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-faded-no-signals",
            receipt_id=705,
            file_url="https://example.test/faded_slip.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "faded"


@pytest.mark.asyncio
async def test_worker_tier2_allows_partially_faded_receipt_when_clahe_recovers_total():
    partial_img = _make_partially_faded_image()
    bundle = _make_bundle(
        "ACME STORE\nOfficial Receipt\nTotal 150.00\nCash 150.00",
        confidence=0.88,
        anchors=AnchorScore(total=5.0, money_tokens=2, keyword_hits=2, has_tax_id=False, has_date=False),
    )
    extraction = Extraction(
        fields={"vendor_name": "ACME STORE", "total_amount": 150.00},
        money=MoneyResult(reconciled=True),
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(partial_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline._store_embedding", new=AsyncMock()),
        patch("app.core.pipeline._evaluate_duplicate", new=AsyncMock(return_value=(False, None, []))),
        patch("app.core.pipeline.set_cached_ocr_result", new=AsyncMock()),
        patch("app.core.pipeline.EmbeddingGenerator.generate", return_value=[0.1] * 384),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-partial-fade-ok",
            receipt_id=706,
            file_url="https://example.test/partial_ok.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    assert callbacks[-1]["status"] == "completed"
    assert callbacks[-1]["total_amount"] == 150.00


@pytest.mark.asyncio
async def test_worker_tier2_rejects_faded_receipt_with_3_unreconciled_items_and_missing_total():
    # Mirrors receipt 40.jpg: 3 candidate items scraped (reconciled=False) while total_amount is None
    partial_img = _make_partially_faded_image()
    bundle = _make_bundle(
        "Fisherman's Hut Aruba\nWhole Fish Red Snapper 7.00\nFried 1.00\nMargarita $12.00\nSubtotal 75.00",
        confidence=0.83,
        anchors=AnchorScore(total=4.2, money_tokens=4, keyword_hits=1, has_tax_id=False, has_date=False),
    )
    item_scan = ItemScan(
        items=[
            ExtractedItem(name="Whole Fish Red Snapper", quantity=1, price=7.00),
            ExtractedItem(name="Fried", quantity=1, price=1.00),
            ExtractedItem(name="Margarita", quantity=1, price=12.00),
        ],
        reconciled=False,
        near_reconciled=False,
    )
    extraction = Extraction(
        fields={"vendor_name": "Fisherman's Hut Aruba", "total_amount": None},
        item_scan=item_scan,
    )
    updates: list[tuple[str, dict]] = []
    callbacks: list[dict] = []

    async def fake_update(_job_id: str, status: str, extra_fields: dict | None = None):
        updates.append((status, extra_fields or {}))

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(partial_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", side_effect=fake_update),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-3-unreconciled-faded",
            receipt_id=707,
            file_url="https://example.test/receipt40.jpg",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    assert updates[-1][0] == "rejected"
    assert updates[-1][1]["rejection_code"] == "faded"
    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "faded"
