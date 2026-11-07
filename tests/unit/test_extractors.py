"""Deterministic extraction rules.

These cases are drawn from real failures observed in the sample corpus rather
than invented, so a regression here means a receipt that used to work has broken.
"""

from app.core.extractors import (
    find_address_candidates,
    find_dates,
    find_invoice_number,
    find_tax_ids,
    format_tax_id,
    select_vendor_tax_id,
)


class TestTaxIdFormatting:
    def test_five_digit_branch_code_is_preserved(self):
        # Receipt 3's vendor TIN, which the original 3-3-3-3 regex rejected.
        assert format_tax_id("30312420200000") == "303-124-202-00000"

    def test_three_digit_branch_code(self):
        assert format_tax_id("000774471025") == "000-774-471-025"


class TestTaxIdSelection:
    def test_vendor_wins_over_customer(self):
        # Receipt 3 carries both; picking the customer's is a compliance failure.
        lines = [
            "YAMACHAN JAPANESE RESTAURANT",
            "VAT Reg. TIN 303-124-202-00000 * Mark Anthony R. Alvaro - Prop.",
            "Registered Name : Scientific Biotech Specialties, Inc",
            "TIN : 201-841-917-000",
        ]
        selected = select_vendor_tax_id(find_tax_ids(lines))
        assert selected is not None
        assert selected.formatted == "303-124-202-00000"
        assert selected.role == "vendor"

    def test_ocr_split_separator_is_recovered(self):
        # Receipt 6: Tesseract splits '471' as '47 1'.
        lines = ["VAT REG. TIN 000-774-47 1-025"]
        selected = select_vendor_tax_id(find_tax_ids(lines))
        assert selected is not None
        assert selected.formatted == "000-774-471-025"

    def test_card_reference_is_not_a_tax_id(self):
        # Receipt 1's payment terminal string contains a 12-digit window.
        lines = ["Reference # 62845289260246240685c", "Card number XXXX 4922"]
        assert select_vendor_tax_id(find_tax_ids(lines)) is None

    def test_phone_number_block_is_not_a_tax_id(self):
        # Receipt 2: '223 2890 223 2891' is fourteen digits with single spaces.
        lines = ["Jollibee Yayasan Complex BO6", "223 2890 223 2891"]
        assert select_vendor_tax_id(find_tax_ids(lines)) is None


class TestDateSelection:
    def test_transaction_date_beats_card_terminal_date(self):
        # Receipt 1: the card block sits next to the word 'Date', so positive
        # label matching alone picks the wrong one.
        lines = [
            "5/24/2025, 11:47:32 AM",
            "Invoice number: 340",
            "Card type Debit",
            "Date/t ime 11/20/2019 11:09 AM",
        ]
        best = find_dates(lines)[0]
        assert best.value.isoformat() == "2025-05-24"

    def test_atp_print_date_is_rejected(self):
        lines = ["0500 Date lssued 03/02/20", "Valid until 03/02/25"]
        candidates = [c for c in find_dates(lines) if c.score > 0]
        assert candidates == []

    def test_month_name_format(self):
        best = find_dates(["Date__ Aug. 9, 2012"])[0]
        assert best.value.isoformat() == "2012-08-09"

    def test_future_dates_are_rejected(self):
        assert find_dates(["Date: 01/01/2099"]) == []

    def test_ordinal_date_formats(self):
        best1 = find_dates(["Date: May 24th, 2025"])[0]
        assert best1.value.isoformat() == "2025-05-24"
        best2 = find_dates(["Date: 24th May 2025"])[0]
        assert best2.value.isoformat() == "2025-05-24"

    def test_timestamp_format(self):
        best = find_dates(["Txn Date: 2025-05-24 14:30:00"])[0]
        assert best.value.isoformat() == "2025-05-24"

    def test_compact_date_format_with_label(self):
        best = find_dates(["Date: 20250524"])[0]
        assert best.value.isoformat() == "2025-05-24"

    def test_spaced_date_format_with_label(self):
        best = find_dates(["dt: 2025 05 24"])[0]
        assert best.value.isoformat() == "2025-05-24"

    def test_compact_date_without_label_is_ignored(self):
        candidates = find_dates(["Reference: 20250524"])
        assert candidates == []

    def test_extended_date_labels_boost_score(self):
        c1 = find_dates(["Txn Date: 05/24/2025"])[0]
        assert c1.score > 2.0
        c2 = find_dates(["Billing Date: 05/24/2025"])[0]
        assert c2.score > 2.0

    def test_numeric_month_first_and_day_first(self):
        from datetime import date
        ref_today = date(2026, 12, 31)
        c_ph = find_dates(["08/11/2026"], country="PH", today=ref_today)[0]
        assert c_ph.value.isoformat() == "2026-08-11"
        c_gb = find_dates(["08/11/2026"], country="GB", today=ref_today)[0]
        assert c_gb.value.isoformat() == "2026-11-08"
        c_swap = find_dates(["15/11/2026"], country="PH", today=ref_today)[0]
        assert c_swap.value.isoformat() == "2026-11-15"

    def test_textual_month_formats(self):
        from datetime import date
        ref_today = date(2026, 12, 31)
        assert find_dates(["Date: Nov 11, 2026"], today=ref_today)[0].value.isoformat() == "2026-11-11"
        assert find_dates(["Date: Nov 11, '26"], today=ref_today)[0].value.isoformat() == "2026-11-11"
        assert find_dates(["Date: 11-Nov-2026"], today=ref_today)[0].value.isoformat() == "2026-11-11"
        assert find_dates(["Date: Nov-11-2026"], today=ref_today)[0].value.isoformat() == "2026-11-11"
        assert find_dates(["Date: 11/Nov/2026"], today=ref_today)[0].value.isoformat() == "2026-11-11"
        assert find_dates(["Date: 11th of November, 2026"], today=ref_today)[0].value.isoformat() == "2026-11-11"

    def test_adjacent_punctuation_and_time_boundary(self):
        assert find_dates(["07/31/2026(Fri) 12:47:42"])[0].value.isoformat() == "2026-07-31"
        assert find_dates(["Str No: 4032 Date:08/06/2026 Time:15:39:47"])[0].value.isoformat() == "2026-08-06"
        assert find_dates(["2026-07-30T17:54:07"])[0].value.isoformat() == "2026-07-30"
        assert find_dates(["08 / 11 / 2026"])[0].value.isoformat() == "2026-08-11"

    def test_same_line_expiry_does_not_drop_transaction_date(self):
        lines = ["03/06/2016 05:25 PM EXPIRES 06/04/16"]
        candidates = find_dates(lines)
        valid = [c for c in candidates if c.score > 0]
        assert len(valid) >= 1
        assert valid[0].value.isoformat() == "2016-03-06"
        # 06/04/16 directly follows EXPIRES and must be penalized below zero
        expiry_candidates = [c for c in candidates if c.value.isoformat() == "2016-06-04"]
        assert len(expiry_candidates) == 1
        assert expiry_candidates[0].score < 0

    def test_scoped_ocr_character_recovery(self):
        assert find_dates(["Date: O8/11/2026"])[0].value.isoformat() == "2026-08-11"
        assert find_dates(["Date: 08/ll/2026"])[0].value.isoformat() == "2026-08-11"
        assert find_dates(["Date: 2O26-07-30"])[0].value.isoformat() == "2026-07-30"

    def test_previous_line_label_lookback(self):
        from datetime import date
        ref_today = date(2026, 12, 31)
        lines = ["Transaction Date:", "Nov 11, 2026"]
        candidates = find_dates(lines, today=ref_today)
        assert len(candidates) == 1
        assert candidates[0].value.isoformat() == "2026-11-11"
        assert candidates[0].score > 2.0

    def test_timezone_day_tolerance(self):
        from datetime import date
        today = date(2026, 11, 7)
        # Next-day PHT date relative to UTC must be accepted
        res = find_dates(["Date: 11/08/2026"], today=today)
        assert len(res) == 1
        assert res[0].value.isoformat() == "2026-11-08"
        # Dates beyond +1 day rejected as future dates
        future = find_dates(["Date: 11/15/2026"], today=today)
        assert future == []


class TestInvoiceNumber:
    def test_labelled_invoice_number(self):
        number, _ = find_invoice_number(["Invoice number: 340"])
        assert number == "340"

    def test_check_number_on_pos_receipt(self):
        number, _ = find_invoice_number(["Chk No:20052"])
        assert number == "20052"

    def test_printer_accreditation_number_is_not_a_receipt_number(self):
        # Receipt 8: 'No. 038MP20140000000045' would otherwise yield '038'.
        number, _ = find_invoice_number(
            ["DAU PRINTERS Printers Accreditation No. 038MP20140000000045"]
        )
        assert number is None

    def test_semicolon_separator_from_fast_path(self):
        # Fast-path OCR read 'Invoice number; 45065' (semicolon instead of colon).
        number, evidence = find_invoice_number(
            ["Mydin Wholesale Sdn. Bhd.", "21-10-2024", "Invoice number; 45065"]
        )
        assert number == "45065"
        assert "45065" in evidence

    def test_comma_separator_from_fast_path(self):
        # Fast-path OCR read 'No, 38326 ...' (comma instead of dot).
        number, _ = find_invoice_number(
            ["No, 38326 aoe aera : _Date: 07-05-2026", "SALES INVOICE"]
        )
        assert number == "38326"


class TestAddressCandidates:
    def test_extracts_address_candidate_from_lines(self):
        line_sets = [[
            ("Jollibee Food Corp", 95.0),
            ("Ayala Center, Makati City, Metro Manila, Philippines", 90.0),
            ("TIN: 000-111-222-000", 85.0),
        ]]
        candidates = find_address_candidates(line_sets)
        assert len(candidates) >= 1
        assert "Makati City" in candidates[0]

    def test_multi_line_us_address_assembly(self):
        line_sets = [[
            ("Jollibee Las Vegas", 98.0),
            ("3890 S Maryland Parkway Suite 137", 95.0),
            ("Las Vegas NV 89119", 94.0),
            ("(702) 893-3005", 90.0),
            ("5/24/2025, 11:47:32 AM", 92.0),
        ]]
        candidates = find_address_candidates(line_sets, vendor_name="Jollibee Las Vegas")
        assert len(candidates) >= 1
        assert "3890 S Maryland Parkway Suite 137, Las Vegas NV 89119" in candidates[0]

    def test_multi_line_ph_address_assembly(self):
        line_sets = [[
            ("YAMACHAN JAPANESE RESTAURANT", 98.0),
            ("VAT Reg. TIN 303-124-202-00000", 95.0),
            ("Unit 6-1 Makati Cinema Square, Fernando St.", 92.0),
            ("Pio Del Pilar, Makati City", 91.0),
            ("NCR, Fourth District, Philippines", 90.0),
            ("SOLD TO:", 85.0),
        ]]
        candidates = find_address_candidates(line_sets, vendor_name="Yamachan Japanese Restaurant")
        assert len(candidates) >= 1
        assert "Makati Cinema Square" in candidates[0]
        assert "Pio Del Pilar, Makati City" in candidates[0]

    def test_suite_and_zip_plus_four(self):
        line_sets = [[
            ("DOLLAR TREE", 99.0),
            ("Store# 156 (517) 993-0356", 90.0),
            ("1941 W Grand River Ave.", 94.0),
            ("Suite 101", 92.0),
            ("Okemos MI 56252-5601", 93.0),
        ]]
        candidates = find_address_candidates(line_sets, vendor_name="Dollar Tree")
        assert len(candidates) >= 1
        assert "1941 W Grand River Ave." in candidates[0]
        assert "Okemos MI 56252-5601" in candidates[0]

    def test_city_state_short_address(self):
        line_sets = [[
            ("Dos Amigos Restaurant", 95.0),
            ("New York, NY", 92.0),
            ("SALE", 88.0),
        ]]
        candidates = find_address_candidates(line_sets, vendor_name="Dos Amigos Restaurant")
        assert len(candidates) >= 1
        assert candidates[0] == "New York, NY"

    def test_customer_address_is_excluded(self):
        line_sets = [[
            ("YAMACHAN JAPANESE RESTAURANT", 98.0),
            ("VAT Reg. TIN 303-124-202-00000", 95.0),
            ("Unit 6-1 Makati Cinema Square, Fernando St., Pio Del Pilar, Makati City", 90.0),
            ("SOLD TO:", 88.0),
            ("Registered Name: Scientific Biotech Specialties, Inc.", 85.0),
            ("Business Address: 6023 Sacred Heart cor. Kamagong Sts.", 80.0),
            ("San Antonio, 1203 Makati City, Metro Manila, Philippines", 80.0),
        ]]
        candidates = find_address_candidates(
            line_sets,
            vendor_name="Yamachan Japanese Restaurant",
            customer_names=["Scientific Biotech Specialties, Inc."],
        )
        assert len(candidates) >= 1
        assert "Makati Cinema Square" in candidates[0]
        assert all("6023 Sacred Heart" not in c for c in candidates)

    def test_bir_printer_accreditation_is_excluded(self):
        line_sets = [[
            ("UNITED DAILY PRESS INC.", 98.0),
            ("812 Benavidez Street Barangay 295 Binondo Manila", 92.0),
            ("VAT Reg TIN: 008-666-750-000", 90.0),
            ("LILEDAU PRINTERS Printers Accreditation No. 038MP20140000000045", 85.0),
            ("58 Branches St. Sangandaan Q.C.", 80.0),
        ]]
        candidates = find_address_candidates(line_sets, vendor_name="United Daily Press Inc.")
        assert len(candidates) >= 1
        assert "812 Benavidez Street" in candidates[0]
        assert all("Sangandaan" not in c for c in candidates)

    def test_vendor_name_is_not_picked_as_address(self):
        line_sets = [[
            ("Jollibee Las Vegas", 98.0),
            ("3890 S Maryland Parkway Suite 137", 95.0),
            ("Las Vegas NV 89119", 94.0),
        ]]
        candidates = find_address_candidates(line_sets, vendor_name="Jollibee Las Vegas")
        assert "Jollibee Las Vegas" not in candidates
        assert any("3890 S Maryland Parkway" in c for c in candidates)

    def test_inline_phone_is_stripped_from_address(self):
        line_sets = [[
            ("Highpointe Medical Hub, 2/F Shaw Blvd. cor. M. Yulo, Brgy. Bagong Silang, Mandaluyong City Tel. No.: 02-8-2564681", 90.0),
        ]]
        candidates = find_address_candidates(line_sets)
        assert len(candidates) >= 1
        assert "Mandaluyong City" in candidates[0]
        assert "02-8-2564681" not in candidates[0]

