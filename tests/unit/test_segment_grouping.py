"""Unit tests for smart multi-receipt segment grouping, isolated rejections, and split callbacks."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.api.schemas.ocr import (
    SplitReceiptPayload,
    build_callback_payload,
    build_split_receipt_payload,
)
from app.config import settings
from app.core import callback as callback_mod
from app.core import pipeline
from app.core.ocr_engine import OcrReading, Word
from app.core.segment_merger import (
    SegmentAnalysis,
    _evaluate_item_seam_overlap,
    _has_identity_conflict,
    analyze_segment,
    group_segments_into_receipts,
    split_tall_image_into_strips,
)
from app.main import app


def _make_reading(
    lines_of_text: list[str],
    start_top: int = 10,
    line_height: int = 20,
    mean_conf: float = 0.92,
) -> OcrReading:
    words: list[Word] = []
    lines: list[list[Word]] = []
    current_y = start_top

    for line_idx, line_str in enumerate(lines_of_text):
        tokens = line_str.split()
        line_words: list[Word] = []
        current_x = 10
        for token in tokens:
            w = Word(
                text=token,
                confidence=mean_conf,
                left=current_x,
                top=current_y,
                width=max(len(token) * 8, 10),
                height=line_height,
                block=1,
                par=1,
                line=line_idx + 1,
            )
            line_words.append(w)
            words.append(w)
            current_x += w.width + 5
        lines.append(line_words)
        current_y += line_height + 5

    text = "\n".join(" ".join(w.text for w in ln) for ln in lines)
    return OcrReading(
        text=text,
        words=words,
        confidence=mean_conf,
        variant="paddle",
        psm=0,
        engine="paddle",
        lines=lines,
    )


def _make_sharp_image(width: int = 400, height: int = 600) -> Image.Image:
    arr = np.full((height, width), 220, dtype=np.uint8)
    for y in range(50, height - 50, 40):
        arr[y : y + 5, 50 : width - 50] = 0
        arr[y + 10 : y + 15, 50 : width - 50] = 255
    return Image.fromarray(arr)


def _make_blurry_image(width: int = 400, height: int = 600) -> Image.Image:
    arr = np.full((height, width), 200, dtype=np.uint8)
    for y in range(50, height - 50, 40):
        arr[y : y + 5, 50 : width - 50] = 0
        arr[y + 10 : y + 15, 50 : width - 50] = 255
    return Image.fromarray(cv2.GaussianBlur(arr, (25, 25), 0))


def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Segment analysis, seam hardening, identity conflict, and grouping
# ---------------------------------------------------------------------------


def test_analyze_segment_distinguishes_header_middle_footer_and_complete():
    img = Image.new("RGB", (450, 600))

    r_header = _make_reading([
        "PUREGOLD SUPERMARKET, INC.",
        "TIN: 201-277-095-00040",
        "Date: 2026-03-30",
        "1 x 88.50 88.50 V",
        "4800011812047 SUPEHO G/BHON 500G",
    ])
    a_header = analyze_segment(0, img, r_header)
    assert a_header.has_header is True
    assert a_header.has_footer is False
    assert a_header.is_self_contained is False
    assert a_header.vendor_tax_id == "20127709500040"
    assert a_header.transaction_date == "2026-03-30"
    assert "PUREGOLD" in (a_header.vendor_name or "")

    r_acme = _make_reading([
        "ACME STORE",
        "1 Coffee 100.00",
        "Total 100.00",
    ])
    a_acme = analyze_segment(0, img, r_acme)
    assert a_acme.has_header is True
    assert a_acme.has_footer is True
    assert a_acme.is_self_contained is True
    assert a_acme.vendor_name == "ACME STORE"

    r_middle = _make_reading([
        "SUPEHO G/BHON 500G",
        "1 x 46.25 46.25 V",
        "4800024588963 DM CC S/S P/P900G",
        "2 x 14.75 29.50 V",
        "4800361415347 NSCFORGTP52/40GX10",
        "1 x 50.00 50.00 V",
        "1 x 30.00 30.00 V",
        "1 x 20.00 20.00 V",
    ])
    a_middle = analyze_segment(1, img, r_middle)
    assert a_middle.has_header is False
    assert a_middle.has_footer is False
    assert a_middle.is_self_contained is False
    assert a_middle.vendor_name is None

    r_footer = _make_reading([
        "2 x 14.75 29.50 V",
        "4800361415347 NSCFORGTP52/40GX10",
        "Subtotal 6,213.70",
        "Total Amt Due 6,193.70",
        "CASH 6,500.00",
        "CHANGE 306.30",
        "VATable Sale 5,530.09",
        "VAT (12%) 663.61",
        "THANK YOU! COME AGAIN",
        "PTU No: FP012020",
        "Date Issued: 01/15/2020",
    ])
    a_footer = analyze_segment(2, img, r_footer)
    assert a_footer.has_header is False
    assert a_footer.has_footer is True
    assert a_footer.is_self_contained is False
    assert a_footer.total_amount == 6193.70
    assert a_footer.vendor_name is None

    r_complete = _make_reading([
        "JOLLIBEE FOODS CORPORATION",
        "VAT REG TIN: 123-456-789-00000",
        "Invoice No: 90123",
        "Date: 2026-04-01",
        "1 Chickenjoy 110.00",
        "Subtotal 110.00",
        "Total Amount Due 110.00",
        "CASH 200.00",
    ])
    a_complete = analyze_segment(3, img, r_complete)
    assert a_complete.has_header is True
    assert a_complete.has_footer is True
    assert a_complete.is_self_contained is True
    assert a_complete.invoice_number == "90123"
    assert a_complete.total_amount == 110.00


def test_evaluate_item_seam_overlap_ignores_generic_boilerplate():
    r1 = _make_reading([
        "OFFICIAL RECEIPT",
        "SALES INVOICE",
        "TIN: 111-222-333-000",
        "POS TERMINAL 1",
        "BIR PERMIT NO 12345678",
        "VATABLE SALE 100.00",
        "VAT (12%) 12.00",
        "VAT EXEMPT 0.00",
        "ZERO RATED 0.00",
        "CUSTOMER COPY",
        "THANK YOU! COME AGAIN",
    ])
    r2 = _make_reading([
        "OFFICIAL RECEIPT",
        "SALES INVOICE",
        "TIN: 111-222-333-000",
        "POS TERMINAL 1",
        "BIR PERMIT NO 12345678",
        "VATABLE SALE 100.00",
        "VAT (12%) 12.00",
        "VAT EXEMPT 0.00",
        "ZERO RATED 0.00",
        "CUSTOMER COPY",
        "THANK YOU! COME AGAIN",
    ])
    assert _evaluate_item_seam_overlap(r1.lines, r2.lines) == 0.0

    # Genuine barcode/item seam still scores positively
    r_item_a = _make_reading([
        "THANK YOU",
        "4800011812047 SUPEHO G/BHON 500G",
        "1 x 46.25 46.25 V",
    ])
    r_item_b = _make_reading([
        "4800011812047 SUPEHO G/BHON 500G",
        "1 x 46.25 46.25 V",
        "OFFICIAL RECEIPT",
    ])
    assert _evaluate_item_seam_overlap(r_item_a.lines, r_item_b.lines) >= 2.0


def test_has_identity_conflict_all_dimensions():
    base = SegmentAnalysis(
        index=0,
        vendor_name="Jollibee Foods Corp",
        vendor_tax_id="20127709500040",
        invoice_number="INV-1001",
        transaction_date="2026-04-01",
        currency="PHP",
        total_amount=250.00,
        has_header=True,
        has_footer=True,
        is_self_contained=True,
    )

    same = SegmentAnalysis(
        index=1,
        vendor_name="Jollibee Foods",
        vendor_tax_id="201277095000",  # Same first 9 digits, 12 vs 14 digit branch
        invoice_number="INV-1001",
        transaction_date="2026-04-01",
        currency="PHP",
        total_amount=250.00,
        has_header=True,
        has_footer=True,
        is_self_contained=True,
    )
    assert _has_identity_conflict(base, same) is False

    # Different TIN first 9 digits
    assert _has_identity_conflict(
        base, SegmentAnalysis(index=1, vendor_tax_id="999888777000")
    ) is True
    # Different invoice
    assert _has_identity_conflict(
        base, SegmentAnalysis(index=1, invoice_number="INV-1002")
    ) is True
    # Different date
    assert _has_identity_conflict(
        base, SegmentAnalysis(index=1, transaction_date="2026-04-02")
    ) is True
    # Different currency
    assert _has_identity_conflict(
        base, SegmentAnalysis(index=1, currency="USD")
    ) is True
    # Distinct vendor
    assert _has_identity_conflict(
        base, SegmentAnalysis(index=1, vendor_name="Mercury Drug Corporation")
    ) is True
    # Distinct footer total
    assert _has_identity_conflict(
        base,
        SegmentAnalysis(index=1, total_amount=251.50, has_footer=True),
    ) is True


def test_group_segments_one_long_overlapping_receipt_stays_together():
    img = Image.new("RGB", (450, 600))
    r_h = _make_reading([
        "PUREGOLD PRICE CLUB, INC.",
        "TIN: 201-277-095-00040",
        "Date: 2026-03-30",
        "1 x 88.50 88.50 V",
        "4800011812047 SUPEHO G/BHON 500G",
    ])
    r_m = _make_reading([
        "4800011812047 SUPEHO G/BHON 500G",
        "1 x 46.25 46.25 V",
        "4800024588963 DM CC S/S P/P900G",
        "2 x 14.75 29.50 V",
        "4800361415347 NSCFORGTP52/40GX10",
    ])
    r_f = _make_reading([
        "2 x 14.75 29.50 V",
        "4800361415347 NSCFORGTP52/40GX10",
        "Subtotal 164.25",
        "Total Amt Due 164.25",
        "CASH 200.00",
    ])

    # Out-of-order upload [footer, middle, header]
    groups = group_segments_into_receipts([(0, img, r_f), (1, img, r_m), (2, img, r_h)])
    assert len(groups) == 1
    assert groups[0].segment_indices == [0, 1, 2]
    assert groups[0].is_self_contained is True
    assert groups[0].total_amount == 164.25
    # Internal order should be [header, middle, footer]
    assert groups[0].segments[0][1] is r_h
    assert groups[0].segments[-1][1] is r_f


def test_group_segments_one_long_complementary_non_overlapping_receipt():
    img = Image.new("RGB", (450, 600))
    r_h = _make_reading([
        "SM SUPERMARKET",
        "TIN: 201-277-095-00040",
        "Date: 2026-04-02",
        "1 Apple Juice 85.00",
    ])
    r_m = _make_reading([
        "1 x 45.00 45.00 V",
        "4800111222333 MILK POWDER",
        "1 x 120.00 120.00 V",
        "4800222333444 COFFEE PACK",
        "1 x 30.00 30.00 V",
        "1 x 20.00 20.00 V",
    ])
    r_f = _make_reading([
        "1 Bread Loaf 65.00",
        "Subtotal 365.00",
        "Total Amt Due 365.00",
        "CASH 400.00",
    ])

    groups = group_segments_into_receipts([(0, img, r_h), (1, img, r_m), (2, img, r_f)])
    assert len(groups) == 1
    assert groups[0].segment_indices == [0, 1, 2]
    assert groups[0].is_self_contained is True
    assert groups[0].total_amount == 365.00


def test_group_segments_splits_distinct_receipts_different_and_same_vendor():
    img = Image.new("RGB", (450, 600))
    r_a = _make_reading([
        "JOLLIBEE FOODS CORP",
        "TIN: 111-222-333-00000",
        "Invoice No: 101",
        "Date: 2026-04-01",
        "1 Chickenjoy 99.00",
        "Total Amount Due 99.00",
        "THANK YOU",
    ])
    r_b = _make_reading([
        "STARBUCKS COFFEE PH",
        "TIN: 444-555-666-00000",
        "Invoice No: 202",
        "Date: 2026-04-01",
        "1 Caffe Latte 180.00",
        "Total Amount Due 180.00",
        "THANK YOU",
    ])
    # Same vendor as A, same day, even identical line items -> must still split!
    r_a2 = _make_reading([
        "JOLLIBEE FOODS CORP",
        "TIN: 111-222-333-00000",
        "Invoice No: 102",
        "Date: 2026-04-01",
        "1 Chickenjoy 99.00",
        "Total Amount Due 99.00",
        "THANK YOU",
    ])

    groups = group_segments_into_receipts([(0, img, r_a), (1, img, r_b), (2, img, r_a2)])
    assert len(groups) == 3
    assert [g.segment_indices for g in groups] == [[0], [1], [2]]
    assert groups[0].total_amount == 99.00
    assert groups[1].total_amount == 180.00
    assert groups[2].total_amount == 99.00


def test_group_segments_mixed_batch_long_receipt_plus_separate_receipt():
    img = Image.new("RGB", (450, 600))
    long_h = _make_reading([
        "MERCURY DRUG CORPORATION",
        "TIN: 000-111-222-00000",
        "Date: 2026-04-05",
        "1 x 50.00 50.00 V",
        "4800999888777 VITAMIN C",
    ])
    long_f = _make_reading([
        "4800999888777 VITAMIN C",
        "1 x 50.00 50.00 V",
        "1 x 100.00 100.00 V",
        "Subtotal 150.00",
        "Total Amt Due 150.00",
    ])
    sep_c = _make_reading([
        "STARBUCKS COFFEE",
        "TIN: 888-777-666-00000",
        "Date: 2026-04-05",
        "1 Mocha 195.00",
        "Total Amount Due 195.00",
    ])

    # [long_h, long_f, sep_c]
    groups = group_segments_into_receipts([(0, img, long_h), (1, img, long_f), (2, img, sep_c)])
    assert len(groups) == 2
    assert groups[0].segment_indices == [0, 1]
    assert groups[1].segment_indices == [2]

    # [sep_c, long_h, long_f]
    groups_rev = group_segments_into_receipts([(0, img, sep_c), (1, img, long_h), (2, img, long_f)])
    assert len(groups_rev) == 2
    assert groups_rev[0].segment_indices == [0]
    assert groups_rev[1].segment_indices == [1, 2]


def test_virtual_strips_of_single_tall_image_stay_together():
    tall_img = Image.new("RGB", (450, 2700))
    strips = split_tall_image_into_strips(tall_img)
    assert len(strips) >= 2

    readings = [
        _make_reading(["ACME STORE", "TIN: 111-222-333-000", "Date: 2026-04-01", "Item 1 10.00"]),
        _make_reading(["Item 2 20.00", "Total Amount Due 30.00"]),
    ]
    indexed = [(i, strips[i], readings[i]) for i in range(2)]
    groups = group_segments_into_receipts(indexed)
    assert len(groups) == 1
    assert len(groups[0].segments) == 2


# ---------------------------------------------------------------------------
# Schemas & callback response helper
# ---------------------------------------------------------------------------


def test_callback_payload_omits_split_fields_when_none_and_includes_when_set():
    legacy = build_callback_payload(
        receipt_id=42,
        fields={"vendor_name": "Acme", "total_amount": 50.0},
        confidence=0.9,
    )
    dumped_legacy = legacy.model_dump()
    assert "segment_indices" not in dumped_legacy
    assert "split_receipts" not in dumped_legacy

    child = build_split_receipt_payload(
        segment_indices=[1],
        fields={"vendor_name": "Cafe B", "total_amount": 80.0},
        confidence=0.88,
        status="completed",
    )
    assert isinstance(child, SplitReceiptPayload)
    assert child.receipt_id is None

    split_parent = build_callback_payload(
        receipt_id=42,
        fields={"vendor_name": "Store A", "total_amount": 50.0},
        confidence=0.92,
        segment_indices=[0],
        split_receipts=[child],
    )
    dumped_split = split_parent.model_dump()
    assert dumped_split["segment_indices"] == [0]
    assert len(dumped_split["split_receipts"]) == 1
    assert dumped_split["split_receipts"][0]["segment_indices"] == [1]
    assert dumped_split["split_receipts"][0]["total_amount"] == 80.0
    assert "tax_basis" not in dumped_split["split_receipts"][0]


@pytest.mark.asyncio
async def test_send_callback_with_response_returns_json_and_keeps_send_callback_bool():
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": "ok", "data": {"split_receipt_ids": [102, 103]}}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return FakeResponse()

    with patch("app.core.callback.httpx.AsyncClient", return_value=FakeClient()):
        ok, body = await callback_mod.send_callback_with_response(
            "https://serms.test/cb", {"receipt_id": 101}
        )
        assert ok is True
        assert body == {"message": "ok", "data": {"split_receipt_ids": [102, 103]}}

        ok_bool = await callback_mod.send_callback("https://serms.test/cb", {"receipt_id": 101})
        assert ok_bool is True


# ---------------------------------------------------------------------------
# FastAPI route partial quality failure handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_queues_202_when_multipart_has_passing_and_failing_segments():
    sharp_bytes = _png_bytes(_make_sharp_image())
    blurry_bytes = _png_bytes(_make_blurry_image())
    cached_urls: list[str] = []

    async def fake_download(url: str) -> bytes:
        return sharp_bytes if "good" in url else blurry_bytes

    async def fake_cache(url: str, data: bytes, ttl_seconds: int | None = None):
        cached_urls.append(url)

    with (
        patch("app.api.routes.ocr._download_file", side_effect=fake_download),
        patch("app.api.routes.ocr.cache_downloaded_bytes", side_effect=fake_cache),
        patch("app.api.routes.ocr.MongoDBClient.create_job", new=AsyncMock()),
        patch("app.api.routes.ocr.process_receipt_task.delay") as mock_delay,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/ocr/process",
                headers={"Authorization": f"Bearer {settings.serms_api_key}"},
                json={
                    "receipt_id": 801,
                    "file_urls": [
                        "https://example.test/good.png",
                        "https://example.test/bad.png",
                    ],
                    "callback_url": "https://serms.test/callback",
                },
            )

    assert res.status_code == 202
    assert res.json()["status"] == "queued"
    assert len(cached_urls) == 2
    mock_delay.assert_called_once()


# ---------------------------------------------------------------------------
# Worker pipeline integration scenarios (mocked OCR/DB/callback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_keeps_one_long_receipt_callback_unchanged():
    img = _make_sharp_image()
    img_bytes = _png_bytes(img)

    r_header = _make_reading([
        "PUREGOLD PRICE CLUB, INC.",
        "TIN: 201-277-095-00040",
        "Date: 2026-03-30",
        "1 x 50.00 50.00 V",
        "4800011812047 ITEM ONE",
    ])
    r_footer = _make_reading([
        "4800011812047 ITEM ONE",
        "1 x 50.00 50.00 V",
        "1 x 50.00 50.00 V",
        "4800022233344 ITEM TWO",
        "Subtotal 100.00",
        "Total Amount Due 100.00",
    ])

    call_idx = {"n": 0}

    def fake_read_fast(_image: Image.Image, _lang: str = "eng") -> OcrReading:
        idx = call_idx["n"]
        call_idx["n"] += 1
        return r_header if idx == 0 else r_footer

    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    store_calls: list[tuple[int, str]] = []

    async def fake_store(rid: int, src: str, text: str, emb=None):
        store_calls.append((rid, text))

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=img_bytes)),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline.set_cached_ocr_result", new=AsyncMock()),
        patch("app.core.pipeline.read_segment_fast", side_effect=fake_read_fast),
        patch("app.core.pipeline.read_region_recovery", return_value=[]),
        patch("app.core.pipeline._ask_model_for_assists", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._embedder", return_value=None),
        patch("app.core.pipeline.EmbeddingGenerator.generate", return_value=[0.1] * 384),
        patch("app.core.pipeline._store_embedding", side_effect=fake_store),
        patch("app.core.pipeline._evaluate_duplicate", new=AsyncMock(return_value=(False, None, []))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-long-1",
            receipt_id=901,
            file_url="https://example.test/seg0.png",
            file_urls=["https://example.test/seg0.png", "https://example.test/seg1.png"],
            callback_url="https://serms.test/cb",
            source_service="serms",
        )

    assert len(callbacks) == 1
    cb = callbacks[0]
    assert cb["status"] == "completed"
    assert cb["total_amount"] == 100.0
    assert "segment_indices" not in cb
    assert "split_receipts" not in cb
    assert len(store_calls) == 1 and store_calls[0][0] == 901


@pytest.mark.asyncio
async def test_worker_splits_multiple_receipts_and_stores_child_embeddings():
    img = _make_sharp_image()
    img_bytes = _png_bytes(img)

    r_a = _make_reading([
        "JOLLIBEE FOODS CORP",
        "VAT REG TIN: 111-222-333-00000",
        "Invoice No: 101",
        "Date: 2026-04-01",
        "1 Chickenjoy 99.00",
        "Subtotal 99.00",
        "Total Amount Due 99.00",
    ])
    r_b = _make_reading([
        "STARBUCKS COFFEE PH",
        "VAT REG TIN: 444-555-666-00000",
        "Invoice No: 202",
        "Date: 2026-04-02",
        "1 Caffe Latte 180.00",
        "Subtotal 180.00",
        "Total Amount Due 180.00",
    ])

    call_idx = {"n": 0}

    def fake_read_fast(_image: Image.Image, _lang: str = "eng") -> OcrReading:
        idx = call_idx["n"]
        call_idx["n"] += 1
        return r_a if idx == 0 else r_b

    def fake_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "JOLLIBEE" in text else [0.0, 1.0, 0.0]

    sent_payloads: list[dict] = []

    async def fake_cb_with_resp(_url: str, payload: dict):
        sent_payloads.append(payload)
        return True, {"data": {"split_receipt_ids": [9002]}}

    stored_ids: list[int] = []

    async def fake_store(rid: int, _src: str, _text: str, _emb=None):
        stored_ids.append(rid)

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=img_bytes)),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline.read_segment_fast", side_effect=fake_read_fast),
        patch("app.core.pipeline.read_region_recovery", return_value=[]),
        patch("app.core.pipeline._ask_model_for_assists", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._embedder", return_value=None),
        patch("app.core.pipeline.EmbeddingGenerator.generate", side_effect=fake_embed),
        patch("app.core.pipeline._store_embedding", side_effect=fake_store),
        patch("app.core.pipeline._evaluate_duplicate", new=AsyncMock(return_value=(False, None, []))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback_with_response", side_effect=fake_cb_with_resp),
    ):
        await pipeline.process_receipt(
            job_id="job-split-2",
            receipt_id=9001,
            file_url="https://example.test/a.png",
            file_urls=["https://example.test/a.png", "https://example.test/b.png"],
            callback_url="https://serms.test/cb",
            source_service="serms",
        )

    assert len(sent_payloads) == 1
    cb = sent_payloads[0]
    assert cb["status"] == "completed"
    assert cb["segment_indices"] == [0]
    assert cb["total_amount"] == 99.0
    assert len(cb["split_receipts"]) == 1
    assert cb["split_receipts"][0]["segment_indices"] == [1]
    assert cb["split_receipts"][0]["total_amount"] == 180.0
    assert stored_ids == [9001, 9002]


@pytest.mark.asyncio
async def test_worker_mixed_batch_isolates_blurry_receipt_and_keeps_valid_long_receipt():
    sharp_bytes = _png_bytes(_make_sharp_image())
    blurry_bytes = _png_bytes(_make_blurry_image())

    r_long_h = _make_reading([
        "MERCURY DRUG CORPORATION",
        "VAT REG TIN: 123-456-789-00000",
        "Date: 2026-04-05",
        "1 x 50.00 50.00 V",
        "4800999888777 VITAMIN C",
    ])
    r_long_f = _make_reading([
        "4800999888777 VITAMIN C",
        "1 x 50.00 50.00 V",
        "1 x 100.00 100.00 V",
        "Subtotal 150.00",
        "Total Amount Due 150.00",
    ])

    async def fake_download(url: str) -> bytes:
        return blurry_bytes if "blurry" in url else sharp_bytes

    call_idx = {"n": 0}

    def fake_read_fast(_image: Image.Image, _lang: str = "eng") -> OcrReading:
        idx = call_idx["n"]
        call_idx["n"] += 1
        return r_long_h if idx == 0 else r_long_f

    sent_payloads: list[dict] = []

    async def fake_cb_with_resp(_url: str, payload: dict):
        sent_payloads.append(payload)
        return True, {"data": {"split_receipt_ids": [9102]}}

    stored_ids: list[int] = []

    async def fake_store(rid: int, _src: str, _text: str, _emb=None):
        stored_ids.append(rid)

    with (
        patch("app.core.pipeline.download_file", side_effect=fake_download),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline.read_segment_fast", side_effect=fake_read_fast),
        patch("app.core.pipeline.read_region_recovery", return_value=[]),
        patch("app.core.pipeline._ask_model_for_assists", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._embedder", return_value=None),
        patch("app.core.pipeline.EmbeddingGenerator.generate", return_value=[0.5, 0.5]),
        patch("app.core.pipeline._store_embedding", side_effect=fake_store),
        patch("app.core.pipeline._evaluate_duplicate", new=AsyncMock(return_value=(False, None, []))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback_with_response", side_effect=fake_cb_with_resp),
    ):
        await pipeline.process_receipt(
            job_id="job-mixed-partial-quality",
            receipt_id=9101,
            file_url="https://example.test/h.png",
            file_urls=[
                "https://example.test/h.png",
                "https://example.test/f.png",
                "https://example.test/blurry.png",
            ],
            callback_url="https://serms.test/cb",
            source_service="serms",
        )

    assert len(sent_payloads) == 1
    cb = sent_payloads[0]
    assert cb["status"] == "completed"
    assert cb["segment_indices"] == [0, 1]
    assert cb["total_amount"] == 150.0
    assert len(cb["split_receipts"]) == 1
    bad_child = cb["split_receipts"][0]
    assert bad_child["segment_indices"] == [2]
    assert bad_child["status"] == "rejected"
    assert bad_child["rejection_code"] == "blurry"
    # Only the completed primary receipt should have its embedding stored
    assert stored_ids == [9101]


@pytest.mark.asyncio
async def test_worker_rejects_whole_job_when_incomplete_fragment_paired_with_blurry_segment():
    sharp_bytes = _png_bytes(_make_sharp_image())
    blurry_bytes = _png_bytes(_make_blurry_image())

    r_header_only = _make_reading([
        "MERCURY DRUG CORPORATION",
        "VAT REG TIN: 123-456-789-00000",
        "Date: 2026-04-05",
        "1 x 50.00 50.00 V",
    ])

    async def fake_download(url: str) -> bytes:
        return blurry_bytes if "blurry" in url else sharp_bytes

    callbacks: list[dict] = []

    async def fake_callback(_url: str, payload: dict) -> bool:
        callbacks.append(payload)
        return True

    with (
        patch("app.core.pipeline.download_file", side_effect=fake_download),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline.read_segment_fast", return_value=r_header_only),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback", side_effect=fake_callback),
    ):
        await pipeline.process_receipt(
            job_id="job-broken-long",
            receipt_id=9201,
            file_url="https://example.test/h.png",
            file_urls=["https://example.test/h.png", "https://example.test/blurry.png"],
            callback_url="https://serms.test/cb",
            source_service="serms",
        )

    assert len(callbacks) == 1
    assert callbacks[0]["status"] == "rejected"
    assert callbacks[0]["rejection_code"] == "blurry"
    assert "split_receipts" not in callbacks[0]


@pytest.mark.asyncio
async def test_worker_isolates_post_ocr_and_intra_batch_duplicate_rejections():
    img_bytes = _png_bytes(_make_sharp_image())

    r_valid = _make_reading([
        "JOLLIBEE FOODS CORP",
        "VAT REG TIN: 111-222-333-00000",
        "Invoice No: 101",
        "Date: 2026-04-01",
        "1 Chickenjoy 99.00",
        "Subtotal 99.00",
        "Total Amount Due 99.00",
    ])
    r_dup_same = _make_reading([
        "JOLLIBEE FOODS CORP",
        "VAT REG TIN: 111-222-333-00000",
        "Invoice No: 101",
        "Date: 2026-04-01",
        "1 Chickenjoy 99.00",
        "Subtotal 99.00",
        "Total Amount Due 99.00",
    ])
    r_non_receipt = _make_reading([
        "Hello friend welcome to our newsletter",
        "Here is your weekly reading digest",
    ])

    readings = [r_valid, r_dup_same, r_non_receipt]
    call_idx = {"n": 0}

    def fake_read_fast(_image: Image.Image, _lang: str = "eng") -> OcrReading:
        idx = call_idx["n"]
        call_idx["n"] += 1
        return readings[idx]

    sent_payloads: list[dict] = []

    async def fake_cb_with_resp(_url: str, payload: dict):
        sent_payloads.append(payload)
        return True, {"data": {"split_receipt_ids": [9302, 9303]}}

    stored_ids: list[int] = []

    async def fake_store(rid: int, _src: str, _text: str, _emb=None):
        stored_ids.append(rid)

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=img_bytes)),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline.read_segment_fast", side_effect=fake_read_fast),
        patch("app.core.pipeline.read_region_recovery", return_value=[]),
        patch("app.core.pipeline._ask_model_for_assists", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._embedder", return_value=None),
        patch("app.core.pipeline.EmbeddingGenerator.generate", return_value=[1.0, 0.0, 0.0]),
        patch("app.core.pipeline._store_embedding", side_effect=fake_store),
        patch("app.core.pipeline._evaluate_duplicate", new=AsyncMock(return_value=(False, None, []))),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback_with_response", side_effect=fake_cb_with_resp),
    ):
        await pipeline.process_receipt(
            job_id="job-intra-dup-and-nonreceipt",
            receipt_id=9301,
            file_url="https://example.test/0.png",
            file_urls=[
                "https://example.test/0.png",
                "https://example.test/1.png",
                "https://example.test/2.png",
            ],
            callback_url="https://serms.test/cb",
            source_service="serms",
        )

    assert len(sent_payloads) == 1
    cb = sent_payloads[0]
    assert cb["status"] == "completed"
    assert cb["segment_indices"] == [0]
    assert cb["is_duplicate"] is False

    assert len(cb["split_receipts"]) == 2
    dup_child = cb["split_receipts"][0]
    assert dup_child["segment_indices"] == [1]
    assert dup_child["status"] == "rejected"
    assert dup_child["is_duplicate"] is True
    assert dup_child["duplicate_similarity"] >= 0.85
    assert dup_child["rejection_code"] == "duplicate"

    bad_child = cb["split_receipts"][1]
    assert bad_child["segment_indices"] == [2]
    assert bad_child["status"] == "rejected"
    assert bad_child["rejection_code"] == "not_a_receipt"

    # Only the non-rejected, non-duplicate Group 0 has its embedding persisted
    assert stored_ids == [9301]


@pytest.mark.asyncio
async def test_worker_isolates_db_duplicate_on_one_split_receipt_and_persists_valid_siblings():
    img_bytes = _png_bytes(_make_sharp_image())

    r_a = _make_reading([
        "JOLLIBEE FOODS CORP",
        "VAT REG TIN: 111-222-333-00000",
        "Invoice No: 501",
        "Date: 2026-04-01",
        "1 Chickenjoy 99.00",
        "Subtotal 99.00",
        "Total Amount Due 99.00",
    ])
    r_b = _make_reading([
        "STARBUCKS COFFEE PH",
        "VAT REG TIN: 444-555-666-00000",
        "Invoice No: 502",
        "Date: 2026-04-02",
        "1 Mocha 180.00",
        "Subtotal 180.00",
        "Total Amount Due 180.00",
    ])
    r_c = _make_reading([
        "JOLLIBEE FOODS CORP",
        "VAT REG TIN: 111-222-333-00000",
        "Invoice No: 503",
        "Date: 2026-04-03",
        "1 Burger Steak 115.00",
        "Subtotal 115.00",
        "Total Amount Due 115.00",
    ])

    readings = [r_a, r_b, r_c]
    call_idx = {"n": 0}

    def fake_read_fast(_image: Image.Image, _lang: str = "eng") -> OcrReading:
        idx = call_idx["n"]
        call_idx["n"] += 1
        return readings[idx]

    def fake_embed(text: str) -> list[float]:
        if "Mocha" in text:
            return [0.0, 1.0, 0.0]
        if "Burger Steak" in text:
            return [0.0, 0.0, 1.0]
        return [1.0, 0.0, 0.0]

    async def fake_eval_dup(_rid: int, _src: str, text: str, _emb=None):
        if "STARBUCKS" in text:
            return True, 0.9312, ["match"]
        return False, None, []

    sent_payloads: list[dict] = []

    async def fake_cb_with_resp(_url: str, payload: dict):
        sent_payloads.append(payload)
        return True, {"data": {"split_receipt_ids": [9402, 9403]}}

    stored_ids: list[int] = []

    async def fake_store(rid: int, _src: str, _text: str, _emb=None):
        stored_ids.append(rid)

    with (
        patch("app.core.pipeline.download_file", new=AsyncMock(return_value=img_bytes)),
        patch("app.core.pipeline.get_cached_ocr_result", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline.read_segment_fast", side_effect=fake_read_fast),
        patch("app.core.pipeline.read_region_recovery", return_value=[]),
        patch("app.core.pipeline._ask_model_for_assists", new=AsyncMock(return_value=None)),
        patch("app.core.pipeline._embedder", return_value=None),
        patch("app.core.pipeline.EmbeddingGenerator.generate", side_effect=fake_embed),
        patch("app.core.pipeline._store_embedding", side_effect=fake_store),
        patch("app.core.pipeline._evaluate_duplicate", side_effect=fake_eval_dup),
        patch("app.core.pipeline.MongoDBClient.update_job_status", new=AsyncMock()),
        patch("app.core.pipeline.send_callback_with_response", side_effect=fake_cb_with_resp),
    ):
        await pipeline.process_receipt(
            job_id="job-3-split-db-dup",
            receipt_id=9401,
            file_url="https://example.test/0.png",
            file_urls=[
                "https://example.test/0.png",
                "https://example.test/1.png",
                "https://example.test/2.png",
            ],
            callback_url="https://serms.test/cb",
            source_service="serms",
        )

    assert len(sent_payloads) == 1
    cb = sent_payloads[0]
    assert cb["status"] == "completed"
    assert cb["total_amount"] == 99.0
    assert len(cb["split_receipts"]) == 2
    assert cb["split_receipts"][0]["status"] == "rejected"
    assert cb["split_receipts"][0]["rejection_code"] == "duplicate"
    assert cb["split_receipts"][0]["duplicate_similarity"] == 0.9312
    assert cb["split_receipts"][1]["status"] == "completed"
    assert cb["split_receipts"][1]["total_amount"] == 115.0
    assert stored_ids == [9401, 9403]


def test_group_segments_keeps_tall_segment_virtual_strips_grouped_in_mixed_batch():
    normal_img = Image.new("RGB", (450, 600))
    tall_img = Image.new("RGB", (450, 2700))
    strips = split_tall_image_into_strips(tall_img, max_strips=3)
    assert len(strips) >= 2
    for s in strips:
        s._parent_segment_index = 1

    r_sep = _make_reading([
        "STARBUCKS COFFEE PH",
        "VAT REG TIN: 444-555-666-00000",
        "Date: 2026-04-02",
        "1 Latte 180.00",
        "Total Amount Due 180.00",
    ])
    r_tall_top = _make_reading([
        "MERCURY DRUG CORPORATION",
        "VAT REG TIN: 123-456-789-00000",
        "Date: 2026-04-05",
        "1 x 50.00 50.00 V",
    ])
    r_tall_bot = _make_reading([
        "1 x 100.00 100.00 V",
        "Subtotal 150.00",
        "Total Amount Due 150.00",
    ])

    indexed = [(0, normal_img, r_sep), (1, strips[0], r_tall_top), (1, strips[1], r_tall_bot)]
    groups = group_segments_into_receipts(indexed)
    assert len(groups) == 2
    assert groups[0].segment_indices == [0]
    assert groups[1].segment_indices == [1]
    assert len(groups[1].segments) == 2
    assert groups[1].is_self_contained is True
    assert groups[1].total_amount == 150.0


def test_group_segments_merges_footer_first_grocery_segments_with_trailing_bir_block():
    img_bot = Image.new("RGB", (450, 800))
    img_top = Image.new("RGB", (450, 800))
    img_mid = Image.new("RGB", (450, 800))

    r_bot = _make_reading([
        "64.00 V",
        "1 x 64.00",
        "DRD PEV-CUT EP025G",
        "125.50 V",
        "1 x 125.50",
        "MCBY CARAM PUF69G",
        "88.00 V",
        "1 x 88.00",
        "SMART DW P-LEMON 500ML",
        "VAT REG TIN 000-123-456-000",
        "POS PTU MIN BIR PERMIT",
    ])
    r_top = _make_reading([
        "Puregold Price Club Inc",
        "VAT REG TIN 000-123-456-000",
        "SALES INVOICE",
        "Date: 2026-07-30",
        "45.00 V",
        "1 x 45.00",
        "JERSEY EVAP 370ML",
    ])
    r_mid = _make_reading([
        "55.00 V",
        "1 x 55.00",
        "MAGGI MGC ERP8GX16",
        "72.00 V",
        "1 x 72.00",
        "MAYA HCM 590G",
    ])

    groups = group_segments_into_receipts([
        (0, img_bot, r_bot),
        (1, img_top, r_top),
        (2, img_mid, r_mid),
    ])
    assert len(groups) == 1
    assert groups[0].segment_indices == [0, 1, 2]
