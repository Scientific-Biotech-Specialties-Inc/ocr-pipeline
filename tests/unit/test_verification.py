"""The verification gate.

Cases mirror real corpus behaviour: ungrounded values must be nulled rather than
downgraded, and receipt 3's misread tax figure must not ship.
"""

from app.core.verification import (
    REASON_HANDWRITTEN,
    REASON_NO_TOTAL,
    REASON_RECONCILE,
    REASON_REJECTED,
    coerce_category,
    looks_illegible,
    verify,
)

PRINTED_TEXT = (
    "Jollibee Las Vegas\n3890 S Maryland Parkway Suite 137\nLas Vegas NV 89119\n"
    "5/24/2025, 11:47:32 AM\nInvoice number: 340\nSubtotal $29 ,99\nTax $2,51\n$32,50"
)


class TestGroundingRejection:
    def test_ungrounded_total_is_nulled_not_downgraded(self):
        result = verify(
            {"total_amount": 999.99, "invoice_number": "340"},
            PRINTED_TEXT,
            reconciled=True,
            locale_resolved=True,
            mean_word_confidence=0.9,
        )
        assert result.fields["total_amount"] is None
        assert "total_amount" in result.rejected
        assert REASON_REJECTED in result.reasons

    def test_grounded_total_survives(self):
        result = verify(
            {"total_amount": 32.50},
            PRINTED_TEXT,
            reconciled=True,
            locale_resolved=True,
            mean_word_confidence=0.9,
        )
        assert result.fields["total_amount"] == 32.50
        assert not result.rejected

    def test_nulls_are_never_rejected(self):
        result = verify(
            {"total_amount": None, "vendor_tax_id": None},
            PRINTED_TEXT,
            reconciled=True,
            locale_resolved=True,
        )
        assert result.rejected == {}

    def test_grounded_date_survives(self):
        result = verify(
            {"transaction_date": "2025-05-24"},
            PRINTED_TEXT,
            reconciled=True,
            locale_resolved=True,
            mean_word_confidence=0.9,
        )
        assert result.fields["transaction_date"] == "2025-05-24"
        assert not result.rejected

    def test_multi_line_joined_location_is_grounded(self):
        result = verify(
            {"location": "3890 S Maryland Parkway Suite 137, Las Vegas NV 89119"},
            PRINTED_TEXT,
            reconciled=True,
            locale_resolved=True,
            mean_word_confidence=0.9,
        )
        assert result.fields["location"] == "3890 S Maryland Parkway Suite 137, Las Vegas NV 89119"
        assert not result.rejected

    def test_fabricated_location_is_rejected(self):
        result = verify(
            {"location": "999 Hollywood Blvd, Los Angeles CA 90028"},
            PRINTED_TEXT,
            reconciled=True,
            locale_resolved=True,
            mean_word_confidence=0.9,
        )
        assert result.fields["location"] is None
        assert "location" in result.rejected

    def test_ocr_recovered_date_is_grounded(self):
        result = verify(
            {"transaction_date": "2026-08-11"},
            "Store #101\nDate: 08/ll/2026\nTotal: 10.00",
            reconciled=True,
            locale_resolved=True,
            mean_word_confidence=0.9,
        )
        assert result.fields["transaction_date"] == "2026-08-11"
        assert "transaction_date" not in result.rejected

    def test_ungrounded_date_is_rejected(self):
        result = verify(
            {"transaction_date": "2024-01-01"},
            PRINTED_TEXT,
            reconciled=True,
            locale_resolved=True,
            mean_word_confidence=0.9,
        )
        assert result.fields["transaction_date"] is None
        assert "transaction_date" in result.rejected
        assert REASON_REJECTED in result.reasons


class TestUnreconciledMoney:
    def test_misread_tax_figure_is_withheld(self):
        # Receipt 3: OCR drops the leading digit of 139.93. The value IS on the
        # page, so grounding passes - only the arithmetic reveals the error.
        text = "YAMACHAN JAPANESE RESTAURANT\nVAT Reg. TIN 303-124-202-00000\nVAT 39.93"
        result = verify(
            {"tax_amount": 39.93, "total_amount": None},
            text,
            reconciled=False,
            locale_resolved=True,
            mean_word_confidence=0.8,
        )
        assert result.fields["tax_amount"] is None
        assert REASON_RECONCILE in result.reasons

    def test_amounts_are_kept_when_a_total_was_read(self):
        result = verify(
            {"tax_amount": 2.51, "total_amount": 32.50},
            PRINTED_TEXT,
            reconciled=False,
            locale_resolved=True,
            mean_word_confidence=0.9,
        )
        assert result.fields["tax_amount"] == 2.51


class TestReviewReasons:
    def test_missing_total_is_flagged(self):
        result = verify({"total_amount": None}, PRINTED_TEXT,
                        reconciled=True, locale_resolved=True)
        assert REASON_NO_TOTAL in result.reasons
        assert result.needs_manual_review

    def test_cap_is_the_strictest_applicable(self):
        result = verify({"total_amount": None}, "x",
                        reconciled=False, locale_resolved=False,
                        mean_word_confidence=0.1)
        assert REASON_HANDWRITTEN in result.reasons
        assert result.cap == 0.45


class TestLegibility:
    def test_noise_is_illegible(self):
        assert looks_illegible("8 /50.00 US Oeasn L} Osneok }~ =a 19", 0.3)

    def test_clean_printed_text_is_legible(self):
        assert not looks_illegible(PRINTED_TEXT, 0.88)

    def test_empty_text_is_illegible(self):
        assert looks_illegible("", 0.0)


class TestCategoryCoercion:
    def test_canonical_names_pass_through(self):
        assert coerce_category("Meals") == "Meals"
        assert coerce_category("transportation") == "Transportation"

    def test_off_list_strings_are_coerced(self):
        # Both were emitted by the original prompt and are absent from SERMS'
        # DEFAULT_NAMES, so firstOrCreate() would add rows for them.
        assert coerce_category("Meals & Entertainment") == "Meals"
        assert coerce_category("Office Supplies") == "Supplies"

    def test_unknown_and_empty_become_others(self):
        assert coerce_category("Cryptocurrency") == "Others"
        assert coerce_category(None) == "Others"
