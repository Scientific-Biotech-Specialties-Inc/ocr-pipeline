"""Outbound callback contract.

Mirrors SERMS' OcrCallbackRequest rules. Each case here is a 422 that would
otherwise reject an entire correctly-extracted receipt.
"""

import pytest
from pydantic import ValidationError

from app.api.schemas.ocr import (
    OcrCallbackPayload,
    ReceiptItem,
    build_callback_payload,
)

FIELDS = {
    "vendor_name": "Yamachan Japanese Restaurant",
    "transaction_date": "2026-03-07",
    "total_amount": 1422.61,
    "tax_amount": 139.93,
    "net_sales": 1166.07,
    "service_charge": 116.61,
    "vendor_tax_id": "303-124-202-00000",
    "vat_classification": "vat",
    "expense_category": "Meals",
    "country": "PH",
    "currency": "PHP",
}


class TestFieldMapping:
    def test_internal_names_map_to_contract_names(self):
        payload = build_callback_payload(42, FIELDS, confidence=0.91)
        assert payload.tin == "303-124-202-00000"      # from vendor_tax_id
        assert payload.vat_amount == 139.93            # from tax_amount
        assert payload.total_amount == 1422.61

    def test_internal_only_fields_are_not_transmitted(self):
        # SERMS drops undeclared keys silently, so they are persisted our side.
        payload = build_callback_payload(42, FIELDS, confidence=0.91)
        transmitted = payload.model_dump()
        # raw net_sales and service_charge are internal keys; net_sales maps to subtotal.
        for internal in ("net_sales", "service_charge", "country"):
            assert internal not in transmitted
        assert transmitted.get("subtotal") == 1166.07
        assert payload.subtotal == 1166.07

    def test_currency_is_now_transmitted(self):
        # currency is in CALLBACK_FIELDS since the locale-propagation change.
        payload = build_callback_payload(42, FIELDS, confidence=0.91)
        assert payload.currency == "PHP"


class TestVatClassification:
    def test_accepted_values_pass(self):
        assert OcrCallbackPayload(receipt_id=1, vat_classification="vat"
                                  ).vat_classification == "vat"
        assert OcrCallbackPayload(receipt_id=1, vat_classification="non-vat"
                                  ).vat_classification == "non-vat"

    def test_third_values_become_null_rather_than_422(self):
        # 'not_applicable' is semantically right for a US receipt but SERMS
        # validates in:vat,non-vat and would reject the whole callback.
        payload = OcrCallbackPayload(receipt_id=1, vat_classification="not_applicable")
        assert payload.vat_classification is None


class TestCurrency:
    def test_known_codes_pass_through(self):
        for code in ("PHP", "USD", "JPY", "EUR", "GBP", "SGD"):
            payload = OcrCallbackPayload(receipt_id=1, currency=code)
            assert payload.currency == code

    def test_unknown_code_becomes_none_rather_than_422(self):
        # An unexpected string must not 422 the whole callback.
        payload = OcrCallbackPayload(receipt_id=1, currency="XYZ")
        assert payload.currency is None

    def test_none_passthrough(self):
        # Locale detection can legitimately return None (unresolved locale).
        payload = OcrCallbackPayload(receipt_id=1, currency=None)
        assert payload.currency is None

    def test_lowercase_code_is_normalised(self):
        # Defensive: normalise lowercase input even though OCR always emits uppercase.
        payload = OcrCallbackPayload(receipt_id=1, currency="usd")
        assert payload.currency == "USD"


class TestItemQuantityClamping:
    def test_fractional_quantity_is_rounded_up_to_one(self):
        assert ReceiptItem(name="Rice 0.5kg", quantity=0.5, price=25.0).quantity == 1

    def test_zero_quantity_is_clamped(self):
        assert ReceiptItem(name="x", quantity=0, price=1.0).quantity == 1

    def test_unparseable_items_are_dropped_not_fatal(self):
        payload = build_callback_payload(
            7, FIELDS, confidence=0.9,
            items=[
                {"name": "Chickenjoy", "quantity": 1, "price": 29.99},
                {"name": "broken", "price": "not a number"},
            ],
        )
        assert len(payload.items) == 1
        assert payload.items[0].name == "Chickenjoy"


class TestConfidenceBounds:
    def test_score_is_clamped_into_range(self):
        # SERMS requires 0..1; its own ExpenseService compares against 80 on the
        # same column, so sending a 0-100 value here would be doubly wrong.
        assert build_callback_payload(1, FIELDS, confidence=1.4).ocr_confidence_score == 1.0
        assert build_callback_payload(1, FIELDS, confidence=-2).ocr_confidence_score == 0.0

    def test_out_of_range_is_rejected_when_constructed_directly(self):
        with pytest.raises(ValidationError):
            OcrCallbackPayload(receipt_id=1, ocr_confidence_score=85.0)


class TestAbstention:
    def test_null_total_is_contract_valid(self):
        # The abstention design depends on this: total_amount is nullable in
        # OcrCallbackRequest, so abstaining needs no contract change.
        payload = build_callback_payload(
            9, {"total_amount": None, "vendor_name": None}, confidence=0.45
        )
        assert payload.total_amount is None
        assert payload.ocr_confidence_score == 0.45


class TestQualityRejection:
    def test_rejection_payload_fields(self):
        payload = build_callback_payload(
            receipt_id=42,
            status="rejected",
            error="Image is too blurry for accurate OCR. Please retake the photo with better focus and steady lighting.",
            rejection_code="blurry",
            rejection_reason="Image is too blurry for accurate OCR. Please retake the photo with better focus and steady lighting.",
        )
        assert payload.receipt_id == 42
        assert payload.status == "rejected"
        assert payload.rejection_code == "blurry"
        assert payload.error == "Image is too blurry for accurate OCR. Please retake the photo with better focus and steady lighting."
        assert payload.rejection_reason == payload.error

    def test_not_a_receipt_rejection_payload(self):
        from app.core.image_quality import NOT_A_RECEIPT_REASON
        payload = build_callback_payload(
            receipt_id=42,
            status="rejected",
            error=NOT_A_RECEIPT_REASON,
            rejection_code="not_a_receipt",
            rejection_reason=NOT_A_RECEIPT_REASON,
        )
        assert payload.receipt_id == 42
        assert payload.status == "rejected"
        assert payload.rejection_code == "not_a_receipt"
        assert "does not appear to be a receipt" in payload.rejection_reason

    def test_is_likely_receipt_logic(self):
        from app.core.anchors import AnchorScore
        from app.core.extraction import Extraction
        from app.core.ocr_engine import OcrBundle, OcrReading
        from app.core.pipeline import is_likely_receipt

        reading = OcrReading(
            text="Hello world this is a random portrait caption without prices",
            words=[],
            confidence=0.9,
            variant="raw",
            psm=6,
            anchors=AnchorScore(total=0.0, money_tokens=0, keyword_hits=0, has_tax_id=False, has_date=False),
        )
        bundle = OcrBundle(primary=reading, supporting=[], all_readings=[reading])
        extraction = Extraction(fields={"vendor_name": None, "total_amount": None, "vendor_tax_id": None})
        assert is_likely_receipt(bundle, extraction) is False

        # With total amount
        extraction_with_total = Extraction(fields={"total_amount": 125.50})
        assert is_likely_receipt(bundle, extraction_with_total) is True

        # With keyword hits
        reading_receipt = OcrReading(
            text="Official Receipt Total Amount",
            words=[],
            confidence=0.9,
            variant="raw",
            psm=6,
            anchors=AnchorScore(total=1.0, money_tokens=0, keyword_hits=2, has_tax_id=False, has_date=False),
        )
        bundle_receipt = OcrBundle(primary=reading_receipt, supporting=[], all_readings=[reading_receipt])
        assert is_likely_receipt(bundle_receipt, extraction) is True

        # Anchors is None on primary reading, but text has money tokens
        reading_none_anchors = OcrReading(
            text="Item description\n1 x 46.25 46.25 V\nItem two 14.75",
            words=[],
            confidence=0.9,
            variant="raw",
            psm=6,
            anchors=None,
        )
        bundle_none_anchors = OcrBundle(
            primary=reading_none_anchors, supporting=[], all_readings=[reading_none_anchors]
        )
        assert is_likely_receipt(bundle_none_anchors, extraction) is True

        # Parsed item_scan with >= 2 items
        from app.core.items import ExtractedItem, ItemScan
        item_scan = ItemScan(
            items=[
                ExtractedItem(name="Product A", quantity=1, price=10.0),
                ExtractedItem(name="Product B", quantity=1, price=20.0),
            ]
        )
        extraction_with_items = Extraction(fields={}, item_scan=item_scan)
        assert is_likely_receipt(bundle, extraction_with_items) is True



