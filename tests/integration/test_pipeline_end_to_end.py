"""End-to-end pipeline runs against real receipt images.

The unit tests and the eval harness both bypass ``pipeline.process_receipt``, so
without these the wiring itself is unverified - which is exactly where the old
code silently kept using the single-pass path.

Only genuinely external things are stubbed: HTTP download, Mongo, Postgres,
embeddings and the outbound callback. OCR, extraction, verification and confidence
all run for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core import ocr_engine, pipeline

RECEIPTS = Path(__file__).resolve().parents[2] / "docs" / "receipts"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RECEIPTS.exists(), reason="sample receipts not available"),
]


class Harness:
    """Captures everything the pipeline tries to send outward."""

    def __init__(self) -> None:
        self.job_updates: list[tuple[str, dict]] = []
        self.callbacks: list[dict] = []

    @property
    def final_status(self) -> str:
        return self.job_updates[-1][0] if self.job_updates else ""

    @property
    def job_document(self) -> dict:
        return self.job_updates[-1][1] if self.job_updates else {}

    @property
    def payload(self) -> dict:
        assert self.callbacks, "pipeline sent no callback"
        return self.callbacks[-1]


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    captured = Harness()

    async def fake_update(job_id, status, extra_fields=None):
        captured.job_updates.append((status, extra_fields or {}))

    async def fake_callback(url, payload):
        captured.callbacks.append(payload)
        return True

    async def fake_store_embedding(*_args, **_kwargs):
        return None

    monkeypatch.setattr(pipeline.MongoDBClient, "update_job_status", fake_update)
    monkeypatch.setattr(pipeline, "send_callback", fake_callback)
    monkeypatch.setattr(pipeline, "_store_embedding", fake_store_embedding)
    # No external Gemini LLM, sentence-transformers, or dynamic PaddleOCR model downloads in test environment;
    # assists must degrade deterministically to the baseline verified by these fixtures.
    monkeypatch.setattr(pipeline, "_embedder", lambda: None)
    monkeypatch.setattr(ocr_engine, "_get_paddle_ocr", lambda: None)
    return captured


async def _run(harness: Harness, monkeypatch: pytest.MonkeyPatch, image: str,
               receipt_id: int = 1) -> Harness:
    data = (RECEIPTS / image).read_bytes()

    async def fake_download(url):
        return data

    monkeypatch.setattr(pipeline, "download_file", fake_download)
    await pipeline.process_receipt(
        job_id=f"job-{receipt_id}",
        receipt_id=receipt_id,
        file_url=f"https://example.test/{image}",
        callback_url="https://serms.test/api/ocr/callback",
        source_service="serms",
    )
    return harness


@pytest.mark.asyncio
async def test_printed_receipt_produces_a_confident_payload(harness, monkeypatch):
    await _run(harness, monkeypatch, "receipt 1.jpg", receipt_id=101)

    assert harness.final_status == "completed"
    payload = harness.payload

    assert payload["receipt_id"] == 101
    assert payload["status"] == "completed"
    assert payload["total_amount"] == 32.50
    assert payload["vat_amount"] == 2.51
    assert payload["transaction_date"] == "2025-05-24"   # not the card date
    assert payload["invoice_number"] == "340"
    assert "Jollibee" in payload["vendor_name"]
    # US receipt: no PH TIN, and vat_classification must be null rather than a
    # third enum value SERMS would reject.
    assert payload["tin"] is None
    assert payload["vat_classification"] is None
    # Clean read must clear the consumer's manual-review threshold.
    assert payload["ocr_confidence_score"] >= 0.80


@pytest.mark.asyncio
async def test_handwritten_receipt_abstains_and_is_flagged(harness, monkeypatch):
    await _run(harness, monkeypatch, "receipt 3.jpg", receipt_id=103)

    assert harness.final_status == "completed"
    payload = harness.payload

    # The handwritten amounts are absent from the OCR text, so no total may be
    # invented. SERMS routes anything under 0.80 to 'flagged'.
    assert payload["total_amount"] is None
    assert payload["ocr_confidence_score"] < 0.80
    # The printed header is still readable, including the five-digit branch code.
    assert payload["tin"] == "303-124-202-00000"
    assert payload["vat_classification"] == "vat"

    verification = harness.job_document["verification"]
    assert verification["needs_manual_review"] is True
    assert "total_amount_missing" in verification["review_reasons"]


@pytest.mark.asyncio
async def test_customer_tax_id_is_never_reported_as_the_vendor(harness, monkeypatch):
    await _run(harness, monkeypatch, "receipt 3.jpg", receipt_id=203)
    # 201-841-917-000 belongs to Scientific Biotech Specialties Inc., the customer.
    assert harness.payload["tin"] != "201-841-917-000"


@pytest.mark.asyncio
async def test_payload_only_contains_contract_fields(harness, monkeypatch):
    await _run(harness, monkeypatch, "receipt 1.jpg", receipt_id=104)
    allowed = {
        "receipt_id", "vendor_name", "transaction_date", "total_amount",
        "subtotal", "vat_amount", "tin", "invoice_number", "vat_classification",
        "currency", "expense_category", "location", "ocr_confidence_score",
        "is_duplicate", "duplicate_similarity", "items", "status", "error",
        "rejection_code", "rejection_reason", "needs_manual_review",
        "tax_basis", "financial_reconciliation_status", "discount_amount", "discount_type",
    }
    assert set(harness.payload) <= allowed


@pytest.mark.asyncio
async def test_audit_document_retains_the_wider_internal_model(harness, monkeypatch):
    await _run(harness, monkeypatch, "receipt 1.jpg", receipt_id=105)
    document = harness.job_document

    assert document["ocr_selection"]["candidates_evaluated"] >= 1
    assert document["ocr_selection"]["engine"] in {"paddle", "tesseract"}
    assert document["ocr_selection"]["confidence"] >= 0.0
    assert "candidate_summary" in document["ocr_selection"]
    assert document["ocr_confidence"] >= 0.0
    assert document["locale"]["country"] == "US"
    assert document["locale"]["currency"] == "USD"
    assert document["reconciliation"]["reconciled"] is True
    assert "grounding_pass_rate" in document["verification"]
    # Fields SERMS would silently discard are kept on our side.
    assert "net_sales" in document["extracted_data"]


@pytest.mark.asyncio
async def test_download_failure_reports_a_failed_callback(harness, monkeypatch):
    async def broken_download(url):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(pipeline, "download_file", broken_download)

    with pytest.raises(RuntimeError):
        await pipeline.process_receipt(
            job_id="job-fail",
            receipt_id=999,
            file_url="https://example.test/missing.jpg",
            callback_url="https://serms.test/api/ocr/callback",
            source_service="serms",
        )

    assert harness.final_status == "failed"
    assert harness.payload["status"] == "failed"
    assert harness.payload["receipt_id"] == 999
