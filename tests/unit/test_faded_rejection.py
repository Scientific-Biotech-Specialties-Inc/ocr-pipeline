"""Unit tests for strict non-overridable image quality and post-OCR rejection."""

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
from app.core.file_cache import OCR_CACHE_VERSION
from app.core.image_quality import NOT_A_RECEIPT_REASON
from app.core.items import ExtractedItem, ItemScan
from app.core.ocr_engine import OcrBundle, OcrReading
from app.core.reconcile import MoneyResult
from app.main import app


def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_sharp_image(width: int = 400, height: int = 600) -> Image.Image:
    arr = np.full((height, width), 220, dtype=np.uint8)
    for y in range(50, height - 50, 40):
        arr[y : y + 5, 50 : width - 50] = 0
        arr[y + 10 : y + 15, 50 : width - 50] = 255
    return Image.fromarray(arr)


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


def _make_dark_image(width: int = 400, height: int = 600) -> Image.Image:
    return Image.fromarray(np.full((height, width), 20, dtype=np.uint8))


def _make_tiny_image() -> Image.Image:
    return Image.fromarray(np.zeros((100, 100), dtype=np.uint8))


def _make_non_receipt_image() -> Image.Image:
    arr = np.zeros((600, 400, 3), dtype=np.uint8)
    arr[:300, :, :] = [100, 180, 240]
    arr[300:, :, :] = [30, 160, 40]
    cv2.line(arr, (0, 300), (400, 300), (0, 0, 0), 5)
    cv2.circle(arr, (200, 150), 50, (255, 255, 0), -1)
    return Image.fromarray(arr, mode="RGB")


def _make_undersized_non_receipt_image() -> Image.Image:
    arr = np.zeros((180, 160, 3), dtype=np.uint8)
    arr[:90, :, :] = [100, 180, 240]
    arr[90:, :, :] = [30, 160, 40]
    cv2.line(arr, (0, 90), (160, 90), (0, 0, 0), 3)
    cv2.circle(arr, (80, 45), 20, (255, 255, 0), -1)
    return Image.fromarray(arr, mode="RGB")


def _make_all_gates_failed_image() -> Image.Image:
    return Image.fromarray(np.full((180, 160, 3), [35, 5, 5], dtype=np.uint8), mode="RGB")


def _make_partially_faded_image(width: int = 400, height: int = 900) -> Image.Image:
    arr = np.full((height, width), 225, dtype=np.uint8)
    for y in range(40, 560, 35):
        arr[y : y + 4, 40 : width - 40] = 0
        arr[y + 8 : y + 12, 40 : width - 40] = 255
    # Bottom 1/3 keeps sharp-edged faint strokes so Tier 1 blur check passes while Tier 2 detects faded ink
    for y in range(620, 870, 35):
        arr[y : y + 4, 40 : width - 40] = 195
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


def _image_for_code(code: str) -> Image.Image:
    builders = {
        "faded": _make_faded_image,
        "blurry": _make_blurry_image,
        "too_dark": _make_dark_image,
        "too_small": _make_tiny_image,
        "not_a_receipt": _make_non_receipt_image,
    }
    return builders[code]()


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_code", ["faded", "blurry", "too_dark", "too_small", "not_a_receipt"])
@pytest.mark.parametrize("force_process", [False, True])
async def test_api_rejects_all_quality_failures_even_when_forced(expected_code: str, force_process: bool):
    img_bytes = _png_bytes(_image_for_code(expected_code))
    created_jobs: list[dict] = []
    callbacks: list[dict] = []

    async def fake_create_job(doc: dict):
        created_jobs.append(doc)

    async def fake_send_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.api.routes.ocr._download_file", new=AsyncMock(return_value=img_bytes)),
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
                    "file_url": f"https://example.test/{expected_code}.png",
                    "callback_url": "https://serms.test/callback",
                    "force_process": force_process,
                },
            )

    assert res.status_code == 422
    body = res.json()
    assert body["status"] == "rejected"
    assert body["rejection_code"] == expected_code
    assert "continue" not in body["rejection_reason"].lower()
    assert "manually" not in body["rejection_reason"].lower()
    mock_delay.assert_not_called()
    assert len(created_jobs) == 1
    assert created_jobs[0]["rejection_code"] == expected_code
    assert len(callbacks) == 1
    assert callbacks[0]["rejection_code"] == expected_code


@pytest.mark.asyncio
async def test_api_prioritizes_faded_segment_over_other_failures():
    blurry_bytes = _png_bytes(_make_blurry_image())
    faded_bytes = _png_bytes(_make_faded_image())

    async def fake_download(url: str) -> bytes:
        return faded_bytes if "seg1" in url else blurry_bytes

    with (
        patch("app.api.routes.ocr._download_file", side_effect=fake_download),
        patch("app.api.routes.ocr.MongoDBClient.create_job", new=AsyncMock()),
        patch("app.api.routes.ocr.send_callback", new=AsyncMock(return_value=True)),
        patch("app.api.routes.ocr.cache_downloaded_bytes", new=AsyncMock()),
        patch("app.api.routes.ocr.process_receipt_task.delay") as mock_delay,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/ocr/process",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
                json={
                    "receipt_id": 702,
                    "file_urls": [
                        "https://example.test/seg0.png",
                        "https://example.test/seg1.png",
                    ],
                    "callback_url": "https://serms.test/callback",
                    "force_process": True,
                },
            )

    assert res.status_code == 422
    body = res.json()
    assert body["rejection_code"] == "faded"
    assert body["segment_index"] == 1
    mock_delay.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_code", ["faded", "blurry", "too_dark", "too_small", "not_a_receipt"])
async def test_worker_rejects_pre_ocr_quality_failures_even_when_forced(expected_code: str):
    bad_img = _image_for_code(expected_code)
    updates: list[tuple[str, dict]] = []
    callbacks: list[dict] = []

    async def fake_update(_job_id: str, status: str, extra_fields: dict | None = None):
        updates.append((status, extra_fields or {}))

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=b"%PDF-1.4")),
        patch("app.core.pipeline.load_images", return_value=[bad_img]),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock()) as mock_ocr,
        patch("app.core.pipeline.MongoDBClient.update_job_status", side_effect=fake_update),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id=f"job-pre-{expected_code}",
            receipt_id=703,
            file_url="https://example.test/receipt.pdf",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    mock_ocr.assert_not_called()
    assert updates[-1][0] == "rejected"
    assert updates[-1][1]["rejection_code"] == expected_code
    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == expected_code


@pytest.mark.asyncio
async def test_worker_allows_multipage_pdf_if_any_page_passes_quality():
    dark_cover = _make_dark_image()
    sharp_receipt = _make_sharp_image()
    bundle = _make_bundle(
        "ACME STORE\nOfficial Receipt\nDate 2026-03-01\nTotal 150.00",
        confidence=0.88,
        anchors=AnchorScore(total=5.0, money_tokens=1, keyword_hits=2, has_tax_id=False, has_date=True),
    )
    extraction = Extraction(
        fields={"vendor_name": "ACME STORE", "transaction_date": "2026-03-01", "total_amount": 150.00},
        money=MoneyResult(reconciled=True),
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=b"%PDF-1.4")),
        patch("app.core.pipeline.load_images", return_value=[dark_cover, sharp_receipt]),
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
            job_id="job-pdf-cover-ok",
            receipt_id=7031,
            file_url="https://example.test/multipage.pdf",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    assert callbacks[-1]["status"] == "completed"
    assert callbacks[-1]["total_amount"] == 150.00


@pytest.mark.asyncio
async def test_worker_rejects_multipage_pdf_when_all_pages_fail_prioritizing_faded():
    dark_page = _make_dark_image()
    faded_page = _make_faded_image()
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=b"%PDF-1.4")),
        patch("app.core.pipeline.load_images", return_value=[dark_page, faded_page]),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock()) as mock_ocr,
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-pdf-all-fail",
            receipt_id=7032,
            file_url="https://example.test/all_bad.pdf",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    mock_ocr.assert_not_called()
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
    assert "Total Amount" in updates[-1][1]["rejection_reason"]
    assert "Transaction Date" in updates[-1][1]["rejection_reason"]
    assert "Merchant Name" not in updates[-1][1]["rejection_reason"]
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
    assert "Total Amount, Transaction Date, Merchant Name" in callbacks[-1]["rejection_reason"]


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
    # Mirrors receipt 40.jpg: 3 candidate items scraped (target=None) while total_amount is None
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
        near_reconciled=True,
        target=None,
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


@pytest.mark.asyncio
async def test_worker_rejects_targetless_near_reconciled_items_as_unreadable_even_when_forced():
    sharp_img = _make_sharp_image()
    bundle = _make_bundle(
        "Corner Cafe\nCoffee 5.00\nToast 7.00\nJuice 8.00",
        confidence=0.86,
        anchors=AnchorScore(total=3.0, money_tokens=3, keyword_hits=0, has_tax_id=False, has_date=False),
    )
    item_scan = ItemScan(
        items=[
            ExtractedItem(name="Coffee", quantity=1, price=5.00),
            ExtractedItem(name="Toast", quantity=1, price=7.00),
            ExtractedItem(name="Juice", quantity=1, price=8.00),
        ],
        reconciled=False,
        near_reconciled=True,
        target=None,
    )
    extraction = Extraction(
        fields={"vendor_name": "Corner Cafe", "total_amount": None},
        item_scan=item_scan,
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-targetless-near-reconciled",
            receipt_id=708,
            file_url="https://example.test/torn.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "unreadable"
    assert "Total Amount" in callbacks[-1]["rejection_reason"]
    assert "Transaction Date" in callbacks[-1]["rejection_reason"]
    assert "Merchant Name" not in callbacks[-1]["rejection_reason"]


@pytest.mark.asyncio
async def test_worker_allows_real_target_reconciled_items_when_total_amount_none():
    sharp_img = _make_sharp_image()
    bundle = _make_bundle(
        "Corner Cafe\nDate 2026-03-01\nCoffee 5.00\nToast 7.00\nJuice 8.00\nSubtotal 20.00",
        confidence=0.88,
        anchors=AnchorScore(total=4.0, money_tokens=4, keyword_hits=1, has_tax_id=False, has_date=True),
    )
    item_scan = ItemScan(
        items=[
            ExtractedItem(name="Coffee", quantity=1, price=5.00),
            ExtractedItem(name="Toast", quantity=1, price=7.00),
            ExtractedItem(name="Juice", quantity=1, price=8.00),
        ],
        reconciled=False,
        near_reconciled=True,
        target=20.00,
    )
    extraction = Extraction(
        fields={"vendor_name": "Corner Cafe", "transaction_date": "2026-03-01", "total_amount": None},
        item_scan=item_scan,
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
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
            job_id="job-real-target-near-reconciled",
            receipt_id=709,
            file_url="https://example.test/subtotal_only.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    assert callbacks[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_worker_flags_legible_receipt_missing_only_total_when_substantive_items_present():
    # Mirrors QAD S-21: legible non-faded receipt with valid merchant + date + >=3 items
    # but cut-off/unreadable footer total completes with needs_manual_review=True (flagged).
    sharp_img = _make_sharp_image()
    bundle = _make_bundle(
        "Puregold Price Club Inc\nDate 2026-07-30\nCoffee 50.00 V\nToast 70.00 V\nJuice 80.00 V",
        confidence=0.88,
        anchors=AnchorScore(total=4.0, money_tokens=3, keyword_hits=1, has_tax_id=False, has_date=True),
    )
    item_scan = ItemScan(
        items=[
            ExtractedItem(name="Coffee", quantity=1, price=50.00),
            ExtractedItem(name="Toast", quantity=1, price=70.00),
            ExtractedItem(name="Juice", quantity=1, price=80.00),
        ],
        reconciled=False,
        near_reconciled=True,
        target=None,
    )
    extraction = Extraction(
        fields={
            "vendor_name": "Puregold Price Club Inc",
            "transaction_date": "2026-07-30",
            "total_amount": None,
        },
        item_scan=item_scan,
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
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
            job_id="job-missing-total-substantive-items",
            receipt_id=7091,
            file_url="https://example.test/no_footer.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    assert callbacks[-1]["status"] == "completed"
    assert callbacks[-1]["needs_manual_review"] is True
    assert callbacks[-1]["ocr_confidence_score"] <= 0.50


@pytest.mark.asyncio
async def test_worker_rejects_illegible_ocr_output_as_unreadable_even_when_forced():
    sharp_img = _make_sharp_image()
    bundle = _make_bundle(
        "8 /50.00 US Oeasn L} Osneok }~ =a 19 99.00",
        confidence=0.30,
        anchors=AnchorScore(total=1.5, money_tokens=2, keyword_hits=0, has_tax_id=False, has_date=False),
    )
    extraction = Extraction(fields={"total_amount": 99.00})
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-illegible",
            receipt_id=710,
            file_url="https://example.test/illegible.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "unreadable"
    assert "illegible" in callbacks[-1]["rejection_reason"].lower()
    assert "Transaction Date, Merchant Name" in callbacks[-1]["rejection_reason"]


@pytest.mark.asyncio
async def test_worker_rejects_non_receipt_ocr_output_even_when_forced():
    sharp_img = _make_sharp_image()
    bundle = _make_bundle(
        "Hello friend welcome to our newsletter today",
        confidence=0.90,
        anchors=AnchorScore(total=0.0, money_tokens=0, keyword_hits=0, has_tax_id=False, has_date=False),
    )
    extraction = Extraction(fields={"vendor_name": None, "total_amount": None})
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-non-receipt",
            receipt_id=711,
            file_url="https://example.test/letter.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "not_a_receipt"
    assert callbacks[-1]["rejection_reason"] == NOT_A_RECEIPT_REASON


@pytest.mark.asyncio
async def test_worker_ignores_stale_cache_missing_total_amount():
    sharp_img = _make_sharp_image()
    stale_cache = {
        "job_document": {"status": "completed"},
        "combined_text": "ACME STORE\nTotal",
        "confidence": 0.45,
        "callback_fields": {"vendor_name": "ACME STORE", "total_amount": None},
    }
    bundle = _make_bundle(
        "ACME STORE\nOfficial Receipt\nTotal",
        confidence=0.80,
        anchors=AnchorScore(total=2.0, money_tokens=0, keyword_hits=2, has_tax_id=False, has_date=False),
    )
    extraction = Extraction(fields={"vendor_name": "ACME STORE", "total_amount": None})
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=stale_cache)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))) as mock_ocr,
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-stale-cache",
            receipt_id=712,
            file_url="https://example.test/stale.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    mock_ocr.assert_awaited_once()
    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "unreadable"


@pytest.mark.asyncio
async def test_worker_serves_valid_cache_with_total_amount():
    sharp_img = _make_sharp_image()
    valid_cache = {
        "cache_version": OCR_CACHE_VERSION,
        "job_document": {"status": "completed", "verification": {"review_reasons": []}},
        "combined_text": "ACME STORE\nOfficial Receipt\nTotal 150.00",
        "confidence": 0.91,
        "callback_fields": {"vendor_name": "ACME STORE", "total_amount": 150.00},
    }
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=valid_cache)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock()) as mock_ocr,
        patch("app.core.pipeline._store_embedding", new=AsyncMock()),
        patch("app.core.pipeline._evaluate_duplicate", new=AsyncMock(return_value=(False, None, []))),
        patch("app.core.pipeline.EmbeddingGenerator.generate", return_value=[0.1] * 384),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-valid-cache",
            receipt_id=713,
            file_url="https://example.test/valid.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    mock_ocr.assert_not_called()
    assert callbacks[-1]["status"] == "completed"
    assert callbacks[-1]["total_amount"] == 150.00


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stale_override",
    [
        {"cache_version": None},
        {"cache_version": 1},
        {"callback_fields": {"vendor_name": None, "total_amount": 150.00}},
        {"callback_fields": {"vendor_name": "Swag t", "total_amount": 150.00}},
        {
            "job_document": {
                "status": "completed",
                "verification": {"review_reasons": ["handwriting_or_illegible"]},
            }
        },
        {
            "job_document": {
                "status": "completed",
                "verification": {"review_reasons": ["total_amount_missing"]},
            }
        },
    ],
)
async def test_worker_ignores_unversioned_or_incomplete_cache_hits(stale_override: dict):
    sharp_img = _make_sharp_image()
    cached_entry = {
        "cache_version": OCR_CACHE_VERSION,
        "job_document": {"status": "completed", "verification": {"review_reasons": []}},
        "combined_text": "ACME STORE\nOfficial Receipt\nTotal 150.00",
        "confidence": 0.91,
        "callback_fields": {"vendor_name": "ACME STORE", "total_amount": 150.00},
        **stale_override,
    }
    if cached_entry["cache_version"] is None:
        cached_entry.pop("cache_version", None)

    bundle = _make_bundle(
        "ACME STORE\nOfficial Receipt\nDate 2026-03-01\nTotal 150.00",
        confidence=0.90,
        anchors=AnchorScore(total=5.0, money_tokens=1, keyword_hits=2, has_tax_id=False, has_date=True),
    )
    extraction = Extraction(
        fields={"vendor_name": "ACME STORE", "transaction_date": "2026-03-01", "total_amount": 150.00},
        money=MoneyResult(reconciled=True),
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=cached_entry)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))) as mock_ocr,
        patch("app.core.pipeline._store_embedding", new=AsyncMock()),
        patch("app.core.pipeline._evaluate_duplicate", new=AsyncMock(return_value=(False, None, []))),
        patch("app.core.pipeline.set_cached_ocr_result", new=AsyncMock()),
        patch("app.core.pipeline.EmbeddingGenerator.generate", return_value=[0.1] * 384),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-stale-v1-or-incomplete",
            receipt_id=714,
            file_url="https://example.test/stale_v1.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    mock_ocr.assert_awaited_once()
    assert callbacks[-1]["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor_name,total_amount", [("Wet Pets a", None), ("Swag t", 20.00)])
async def test_worker_rejects_paddle_gobbly_receipt_even_when_combined_equals_primary(
    vendor_name: str,
    total_amount: float | None,
):
    sharp_img = _make_sharp_image()
    r41_paddle = (
        "5 / 21 / 2022   5   01   PM\nStore   1\nWET\nWet   Pets   a\n3695   W\nMcMuo\n72\n"
        "www   wetp\nBill   To : +  Walk-in\nCashier   Jarred\nDescription   1\n"
        "SF   BRINE   SHRIMP   E\nSUN   GARDEN   PAPA\nTAXES\nRECE\nAI\nint   Tendered  $ 20   00\n"
        "ange   Given : $ 4   12\nCash  $  n0\nGuarantees  &  return   policie\n"
        "posted   at   register   check\nThank   you !  Have   a   nice   da\n501962"
    )
    bundle = _make_bundle(
        r41_paddle,
        confidence=0.96,
        anchors=AnchorScore(total=2.68, money_tokens=1, keyword_hits=1, has_tax_id=False, has_date=True),
    )
    extraction = Extraction(
        fields={
            "vendor_name": vendor_name,
            "transaction_date": "2022-05-21",
            "total_amount": total_amount,
        },
        money=MoneyResult(reconciled=False),
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-r41-paddle",
            receipt_id=715,
            file_url="https://example.test/receipt41.jpg",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "unreadable"
    assert "Merchant Name" in callbacks[-1]["rejection_reason"]
    assert "garbled" in callbacks[-1]["rejection_reason"].lower()


@pytest.mark.asyncio
async def test_worker_rejects_post_verify_dropped_critical_fields():
    sharp_img = _make_sharp_image()
    bundle = _make_bundle(
        "ACME STORE\nOfficial Receipt\nDate 2026-03-01\nSubtotal 100.00",
        confidence=0.92,
        anchors=AnchorScore(total=4.0, money_tokens=1, keyword_hits=2, has_tax_id=False, has_date=True),
    )
    # total_amount=999.99 is ungrounded so verify() nulls it and logs it in verification.rejected
    extraction = Extraction(
        fields={"vendor_name": "ACME STORE", "transaction_date": "2026-03-01", "total_amount": 999.99},
        money=MoneyResult(reconciled=False),
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-post-verify-drop",
            receipt_id=716,
            file_url="https://example.test/ungrounded.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "unreadable"
    assert callbacks[-1]["total_amount"] is None
    assert "Total Amount" in callbacks[-1]["rejection_reason"]


@pytest.mark.asyncio
async def test_worker_rejects_incomplete_fields_even_when_reconciled():
    sharp_img = _make_sharp_image()
    bundle = _make_bundle(
        "Official Receipt\nSubtotal 50.00\nTotal 50.00",
        confidence=0.90,
        anchors=AnchorScore(total=4.0, money_tokens=2, keyword_hits=2, has_tax_id=False, has_date=False),
    )
    # Missing BOTH vendor_name and transaction_date (>= 2 critical fields missing)
    extraction = Extraction(
        fields={"vendor_name": None, "transaction_date": None, "total_amount": 50.00},
        money=MoneyResult(reconciled=True),
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock(return_value=(bundle, extraction, {}))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-missing-two-fields",
            receipt_id=717,
            file_url="https://example.test/incomplete.png",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    assert callbacks[-1]["status"] == "rejected"
    assert callbacks[-1]["rejection_code"] == "unreadable"
    assert "Transaction Date, Merchant Name" in callbacks[-1]["rejection_reason"]


@pytest.mark.asyncio
async def test_worker_allows_dateless_reconciled_receipt_like_receipt_11():
    sharp_img = _make_sharp_image()
    bundle = _make_bundle(
        "THANK YOU FOR RECOMMENDING US !\n"
        "1 TRKY BRGR 12.00\n1 SALAD 8.95\n1 SODA 3.00\n"
        "FOOD 23.95\nTAX 2.15\nTOTAL 26.10",
        confidence=0.99,
        anchors=AnchorScore(total=5.0, money_tokens=6, keyword_hits=2, has_tax_id=False, has_date=False),
    )
    item_scan = ItemScan(
        items=[
            ExtractedItem(name="TRKY BRGR", quantity=1, price=12.00),
            ExtractedItem(name="SALAD", quantity=1, price=8.95),
            ExtractedItem(name="SODA", quantity=1, price=3.00),
        ],
        reconciled=True,
        target=23.95,
    )
    extraction = Extraction(
        fields={
            "vendor_name": "THANK YOU FOR RECOMMENDING US !",
            "transaction_date": None,
            "total_amount": 26.10,
        },
        item_scan=item_scan,
        money=MoneyResult(reconciled=False),
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
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
            job_id="job-dateless-receipt-11",
            receipt_id=718,
            file_url="https://example.test/receipt11.jpg",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    assert callbacks[-1]["status"] == "completed"
    assert callbacks[-1]["total_amount"] == 26.10


@pytest.mark.asyncio
async def test_worker_completes_when_primary_is_clean_and_supporting_tesseract_is_noisy():
    sharp_img = _make_sharp_image()
    primary_text = (
        "The Continent DC\n"
        "1110 Vermont Avenue Northwest\n"
        "Washington, DC 20005\n"
        "Ordered: 10/19/24 3:06 PM\n"
        "1 Egusi $35.00\n"
        "1 Cassava Leaf $35.00\n"
        "1 Lagos $22.00\n"
        "Subtotal $191.00\n"
        "Service Charge $36.00\n"
        "Tax $7.00\n"
        "Total $234.00"
    )
    supporting_text = (
        "» oi ~~ eo —\" ny CaS\n"
        "- - weet\n"
        "' { The Cont inenit OC ok tex <\n"
        "j Heshingten, OC 20005 wt At fs\n"
        "ah SE | y cassave beat pede MM A cape"
    )
    anchors = AnchorScore(total=5.0, money_tokens=7, keyword_hits=3, has_tax_id=False, has_date=True)
    primary = OcrReading(
        text=primary_text,
        words=[],
        confidence=0.96,
        variant="paddle",
        psm=6,
        anchors=anchors,
    )
    supporting = OcrReading(
        text=supporting_text,
        words=[],
        confidence=0.72,
        variant="raw",
        psm=6,
        anchors=anchors,
    )
    bundle = OcrBundle(
        primary=primary,
        supporting=[supporting],
        all_readings=[primary, supporting],
    )
    extraction = Extraction(
        fields={
            "vendor_name": "The Continent DC",
            "transaction_date": "2024-10-19",
            "total_amount": 234.00,
        },
        money=MoneyResult(reconciled=True),
    )
    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=_png_bytes(sharp_img))),
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
            job_id="job-clean-primary-noisy-supporting",
            receipt_id=719,
            file_url="https://example.test/receipt14.jpg",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=False,
        )

    assert callbacks[-1]["status"] == "completed"
    assert callbacks[-1]["total_amount"] == 234.00


@pytest.mark.asyncio
async def test_api_prioritizes_not_a_receipt_over_faded_and_other_failures():
    faded_bytes = _png_bytes(_make_faded_image())
    non_receipt_bytes = _png_bytes(_make_non_receipt_image())
    blurry_bytes = _png_bytes(_make_blurry_image())
    quality_failed_bytes = _png_bytes(_make_all_gates_failed_image())

    async def fake_download(url: str) -> bytes:
        if "seg1" in url:
            return non_receipt_bytes
        if "seg2" in url:
            return quality_failed_bytes
        if "seg3" in url:
            return blurry_bytes
        return faded_bytes

    with (
        patch("app.api.routes.ocr._download_file", side_effect=fake_download),
        patch("app.api.routes.ocr.MongoDBClient.create_job", new=AsyncMock()),
        patch("app.api.routes.ocr.send_callback", new=AsyncMock(return_value=True)),
        patch("app.api.routes.ocr.cache_downloaded_bytes", new=AsyncMock()),
        patch("app.api.routes.ocr.process_receipt_task.delay") as mock_delay,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/ocr/process",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
                json={
                    "receipt_id": 720,
                    "file_urls": [
                        "https://example.test/seg0.png",
                        "https://example.test/seg1.png",
                        "https://example.test/seg2.png",
                        "https://example.test/seg3.png",
                    ],
                    "callback_url": "https://serms.test/callback",
                    "force_process": True,
                },
            )

    assert res.status_code == 422
    body = res.json()
    assert body["rejection_code"] == "not_a_receipt"
    assert body["segment_index"] == 1
    mock_delay.assert_not_called()


@pytest.mark.asyncio
async def test_api_prioritizes_quality_failed_over_faded_segment():
    faded_bytes = _png_bytes(_make_faded_image())
    quality_failed_bytes = _png_bytes(_make_all_gates_failed_image())

    async def fake_download(url: str) -> bytes:
        return quality_failed_bytes if "seg1" in url else faded_bytes

    with (
        patch("app.api.routes.ocr._download_file", side_effect=fake_download),
        patch("app.api.routes.ocr.MongoDBClient.create_job", new=AsyncMock()),
        patch("app.api.routes.ocr.send_callback", new=AsyncMock(return_value=True)),
        patch("app.api.routes.ocr.cache_downloaded_bytes", new=AsyncMock()),
        patch("app.api.routes.ocr.process_receipt_task.delay") as mock_delay,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/ocr/process",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
                json={
                    "receipt_id": 724,
                    "file_urls": [
                        "https://example.test/seg0.png",
                        "https://example.test/seg1.png",
                    ],
                    "callback_url": "https://serms.test/callback",
                    "force_process": True,
                },
            )

    assert res.status_code == 422
    body = res.json()
    assert body["rejection_code"] == "quality_failed"
    assert body["segment_index"] == 1
    mock_delay.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("other_code", ["blurry", "too_dark", "too_small"])
async def test_api_prioritizes_faded_over_other_quality_failures(other_code: str):
    other_bytes = _png_bytes(_image_for_code(other_code))
    faded_bytes = _png_bytes(_make_faded_image())

    async def fake_download(url: str) -> bytes:
        return faded_bytes if "seg1" in url else other_bytes

    with (
        patch("app.api.routes.ocr._download_file", side_effect=fake_download),
        patch("app.api.routes.ocr.MongoDBClient.create_job", new=AsyncMock()),
        patch("app.api.routes.ocr.send_callback", new=AsyncMock(return_value=True)),
        patch("app.api.routes.ocr.cache_downloaded_bytes", new=AsyncMock()),
        patch("app.api.routes.ocr.process_receipt_task.delay") as mock_delay,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/ocr/process",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
                json={
                    "receipt_id": 721,
                    "file_urls": [
                        "https://example.test/seg0.png",
                        "https://example.test/seg1.png",
                    ],
                    "callback_url": "https://serms.test/callback",
                    "force_process": True,
                },
            )

    assert res.status_code == 422
    body = res.json()
    assert body["rejection_code"] == "faded"
    assert body["segment_index"] == 1
    mock_delay.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("img_factory", "expected_code", "expected_issues"),
    [
        (_make_tiny_image, "too_small", {"too_small", "too_dark", "blurry"}),
        (_make_undersized_non_receipt_image, "not_a_receipt", {"too_small", "not_a_receipt"}),
    ],
)
async def test_api_422_persists_detected_issues_without_exposing_in_http_or_callback(
    img_factory,
    expected_code: str,
    expected_issues: set[str],
):
    img_bytes = _png_bytes(img_factory())
    created_jobs: list[dict] = []
    callbacks: list[dict] = []

    async def fake_create_job(doc: dict):
        created_jobs.append(doc)

    async def fake_send_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.api.routes.ocr._download_file", new=AsyncMock(return_value=img_bytes)),
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
                    "receipt_id": 722,
                    "file_url": "https://example.test/multi-issue.png",
                    "callback_url": "https://serms.test/callback",
                    "force_process": False,
                },
            )

    assert res.status_code == 422
    body = res.json()
    assert body["rejection_code"] == expected_code
    assert "detected_issues" not in body
    mock_delay.assert_not_called()

    assert len(callbacks) == 1
    assert callbacks[0]["rejection_code"] == expected_code
    assert "detected_issues" not in callbacks[0]

    assert len(created_jobs) == 1
    qc = created_jobs[0]["quality_check"]
    assert qc["rejection_code"] == expected_code
    assert expected_issues.issubset(qc["detected_issues"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("img_factory", "expected_code", "expected_issues"),
    [
        (_make_tiny_image, "too_small", {"too_small", "too_dark", "blurry"}),
        (_make_dark_image, "too_dark", {"too_dark", "blurry"}),
    ],
)
async def test_worker_pre_ocr_rejection_persists_detected_issues_in_mongodb(
    img_factory,
    expected_code: str,
    expected_issues: set[str],
):
    bad_img = img_factory()
    updates: list[tuple[str, dict]] = []
    callbacks: list[dict] = []

    async def fake_update(_job_id: str, status: str, extra_fields: dict | None = None):
        updates.append((status, extra_fields or {}))

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=b"%PDF-1.4")),
        patch("app.core.pipeline.load_images", return_value=[bad_img]),
        patch("app.core.pipeline._read_and_extract", new=AsyncMock()) as mock_ocr,
        patch("app.core.pipeline.MongoDBClient.update_job_status", side_effect=fake_update),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id=f"job-pre-issues-{expected_code}",
            receipt_id=723,
            file_url="https://example.test/receipt.pdf",
            callback_url="https://serms.test/callback",
            source_service="serms",
            force_process=True,
        )

    mock_ocr.assert_not_called()
    assert updates[-1][0] == "rejected"
    extra = updates[-1][1]
    assert extra["rejection_code"] == expected_code
    assert "quality_check" in extra
    qc = extra["quality_check"]
    assert qc["rejection_code"] == expected_code
    assert expected_issues.issubset(qc["detected_issues"])
    assert callbacks[-1]["rejection_code"] == expected_code
    assert "detected_issues" not in callbacks[-1]
