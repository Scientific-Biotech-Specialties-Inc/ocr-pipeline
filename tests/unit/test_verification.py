"""The verification gate.

Cases mirror real corpus behaviour: ungrounded values must be nulled rather than
downgraded, and receipt 3's misread tax figure must not ship.
"""

from app.core.textquality import is_gobbly_vendor_name, looks_gobbly_ocr
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

    def test_numeric_heavy_paddle_receipt_is_legible(self):
        paddle_text = (
            "THANK YOU FOR RECOMMENDING US !\n"
            "STATION # 5 SERV # 648\n"
            "ORDER C10 A008763 18 : 42\n"
            "1 TRKY BRGR / NO BUN 12.00\n"
            "1 CHCKN POT PIE 11.50\n"
            "1 SALAD 8.95\n"
            "1 ICED TEA 3.00\n"
            "UPUP HOUSEH $1.94\nT\n"
            "OREO COOKIE $5.98\nFC\nCartwheel MfrCpn 0.50-\nFC\n"
            "FOOD 35.45\n"
            "TAX 3.19\n"
            "TOTAL 38.64"
        )
        assert not looks_illegible(paddle_text, 0.99)
        assert not looks_gobbly_ocr(paddle_text, 0.99)

    def test_noisy_supporting_pass_does_not_mark_clean_receipt_illegible(self):
        combined = (
            "The Continent DC\n1110 Vermont Avenue Northwest\nWashington, DC 20005\n"
            "Ordered: 10/19/24 3:06 PM\n1 Egusi $35.00\n1 Cassava Leaf $35.00\n"
            "Subtotal $191.00\nTax $7.00\nTotal $234.00\n"
            "» oi ~~ eo —\" ny CaS\n' { The Cont inenit OC ok tex <"
        )
        assert not looks_illegible(combined, 0.96)

    def test_truncated_gobbly_paddle_receipt_detected(self):
        r41_paddle = (
            "5 / 21 / 2022   5   01   PM\nStore   1\nWET\nWet   Pets   a\n3695   W\nMcMuo\n72\n"
            "www   wetp\nBill   To : +  Walk-in\nCashier   Jarred\nDescription   1\n"
            "SF   BRINE   SHRIMP   E\nSUN   GARDEN   PAPA\nTAXES\nRECE\nAI\nint   Tendered  $ 20   00\n"
            "ange   Given : $ 4   12\nCash  $  n0\nGuarantees  &  return   policie\n"
            "posted   at   register   check\nThank   you !  Have   a   nice   da\n501962"
        )
        assert looks_gobbly_ocr(r41_paddle, 0.96)

    def test_grocery_pos_store_codes_and_long_receipt_isolated_noise_stay_legible(self):
        # Mirrors Puregold grocery POS layout: '×', 'V' VAT flags, SKU codes, and
        # isolated dot-matrix artifacts across a long receipt (>25 lines).
        item_lines = [
            "64.00 V",
            "1 × 64.00",
            "4800016522533 DRD PEV-CUT EP025G",
            "125.50 V",
            "1 × 125.50",
            "480G523220987 MCBY CARAM PUF69G",
            "88.00 V",
            "1 × 88.00",
            "SMART DW P-LEMON 500ML",
        ] * 4
        grocery_text = "\n".join(
            [
                "Puregold Price Club Inc",
                "FAIRVIEW QUEZON CITY",
                "VAT REG TIN 000-123-456-000",
                "SALES INVOICE",
                "Date 2026-07-30",
                *item_lines,
                "TffwyFinoyT ty450g",
                "CAMPrwD",
                "Subtotal 6213.70",
                "Total Amt Due 6193.70",
            ]
        )
        assert not looks_illegible(grocery_text, 0.80)
        assert not looks_gobbly_ocr(grocery_text, 0.80)

    def test_is_gobbly_vendor_name(self):
        for bad in (
            "Swag t",
            "Wet Pets a",
            "SF BRINE SHRIMP E",
            "' { The Cont inenit DC Ya YA",
            "Za SeyEN PEREZ CORN. ey",
            "= SSS SSS SS Nm re eee a",
            "",
            None,
        ):
            assert is_gobbly_vendor_name(bad) is True, f"Expected gobbly: {bad!r}"

        for good in (
            "Jollibee",
            "Jollibee Las Vegas",
            "Ma & Pa’s Bake Shop",
            "THANK YOU FOR RECOMMENDING US !",
            "The Cont Iment DC",
            "Entrek (B) Sdn.Bhd.",
            "Safeway",
            "Target",
            "Corner Cafe",
        ):
            assert is_gobbly_vendor_name(good) is False, f"Expected valid: {good!r}"


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
