"""Regressions found by measuring against the synthetic corpus.

Every case here is a bug the nine-receipt sample could not surface. They are the
argument for having built the corpus at all.
"""

from __future__ import annotations

import random

import pytest
from PIL import Image

from app.core.extractors import find_vendor_candidates
from app.core.layout import LabeledAmount, LayoutScan
from app.core.locale import LocaleGuess
from app.core.ocr_engine import read_variant
from app.core.preprocessing import MIN_ORIENTATION_CONFIDENCE, correct_orientation
from app.core.reconcile import resolve_money
from app.core.vendor import rank_candidates
from app.eval.synthetic.render import render_receipt
from app.eval.synthetic.spec import build_spec


def _lines(pairs: list[str]) -> list[list[tuple[str, float]]]:
    return [[(text, 0.9) for text in pairs]]


class TestOrientationConfidence:
    """An upright receipt was reported as 180 degrees with confidence 0.12.

    Rotating on that reading destroyed every field on the page. Weak detections
    must be ignored: a wrongly upright page still reads, a wrongly inverted one
    reads as nothing.
    """

    def test_low_confidence_rotation_is_ignored(self, monkeypatch):
        import app.core.preprocessing as preprocessing

        monkeypatch.setattr(
            preprocessing.pytesseract, "image_to_osd",
            lambda *a, **k: {"rotate": 180, "orientation_conf": 0.12},
        )
        image = Image.new("L", (80, 40), color=255)
        assert correct_orientation(image).size == image.size

    def test_confident_rotation_is_applied(self, monkeypatch):
        import app.core.preprocessing as preprocessing

        monkeypatch.setattr(
            preprocessing.pytesseract, "image_to_osd",
            lambda *a, **k: {"rotate": 90,
                             "orientation_conf": MIN_ORIENTATION_CONFIDENCE + 1},
        )
        image = Image.new("L", (80, 40), color=255)
        # A 90 degree correction swaps the axes.
        assert correct_orientation(image).size == (40, 80)

    def test_missing_osd_data_is_not_fatal(self, monkeypatch):
        import app.core.preprocessing as preprocessing

        def boom(*_args, **_kwargs):
            raise RuntimeError("osd traineddata missing")

        monkeypatch.setattr(preprocessing.pytesseract, "image_to_osd", boom)
        image = Image.new("L", (80, 40), color=255)
        assert correct_orientation(image) is image


class TestReconciliationIsNotTautological:
    """Comparing a derived value against its own inputs always passes.

    Counting those as successful reconciliation reported confident agreement on
    receipts whose figures were never independently cross-checked.
    """

    @staticmethod
    def _scan(amounts: dict[str, str]) -> LayoutScan:
        """Build a layout scan directly, bypassing OCR geometry."""
        scan = LayoutScan()
        for field_name, token in amounts.items():
            scan.amounts[field_name] = [
                LabeledAmount(
                    field_name=field_name, label=field_name, raw_token=token,
                    line_text=f"{field_name} {token}", confidence=0.9,
                )
            ]
            scan.money_tokens.append(token)
        return scan

    def test_no_independent_figure_means_unreconciled(self):
        # Only a VAT-inclusive total is printed, so net and tax are both computed
        # from it. Nothing independent exists to check, and reconciliation must not
        # report success.
        result = resolve_money(
            self._scan({"total_sales": "14,185.00"}),
            LocaleGuess(country="PH", currency="PHP", score=9),
        )
        assert result.reconciled is False
        assert any("skipped" in note for note in result.reconciliation_notes)

    def test_two_separately_printed_figures_are_genuine_evidence(self):
        # total_sales and total_amount both appear on the page. They agree, and
        # because neither was derived from the other that agreement counts.
        result = resolve_money(
            self._scan({"total_sales": "14,185.00", "total_amount": "14,185.00"}),
            LocaleGuess(country="PH", currency="PHP", score=9),
        )
        assert result.reconciled is True

    def test_printed_tax_provides_a_real_check(self):
        result = resolve_money(
            self._scan({
                "net_sales": "12,665.18", "tax_amount": "1,519.82",
                "total_sales": "14,185.00", "total_amount": "14,185.00",
            }),
            LocaleGuess(country="PH", currency="PHP", score=9),
        )
        assert result.reconciled is True

    def test_no_tax_receipt_identity_catches_a_misread(self):
        # Tax-free slip where OCR read the subtotal and the total differently.
        # Without this identity there is nothing to check and the error passes.
        result = resolve_money(
            self._scan({"net_sales": "2,478.00", "total_amount": "2,470.00"}),
            LocaleGuess(country="BN", currency="BND", score=9),
        )
        assert result.reconciled is False
        assert any("no-tax receipt" in note for note in result.reconciliation_notes)

    def test_no_tax_receipt_that_agrees_reconciles(self):
        result = resolve_money(
            self._scan({"net_sales": "2,470.00", "total_amount": "2,470.00"}),
            LocaleGuess(country="BN", currency="BND", score=9),
        )
        assert result.reconciled is True


class TestVendorCandidateFiltering:
    def test_column_headings_are_not_vendor_names(self):
        # 'Description Amount' was returned as the vendor with confidence 0.994 -
        # a confident wrong answer, which is worse than abstaining.
        candidates = find_vendor_candidates(
            _lines(["Description Amount", "GRAB PHILIPPINES TRANSPORT"])
        )
        assert "Description Amount" not in candidates.lines
        assert "GRAB PHILIPPINES TRANSPORT" in candidates.lines

    def test_ocr_mangled_document_titles_are_excluded(self):
        # 'OFFICIAL RECEIPT' arrives as 'OFFICIAL RECEIpr'.
        candidates = find_vendor_candidates(
            _lines(["OFFICIAL RECEIpr", "United Daily Press Inc."])
        )
        assert "OFFICIAL RECEIpr" not in candidates.lines

    def test_a_word_from_the_heading_vocabulary_is_allowed_in_a_real_name(self):
        candidates = find_vendor_candidates(
            _lines(["Metro Hardware and Construction Supply"])
        )
        assert candidates.lines == ["Metro Hardware and Construction Supply"]


class TestAddressRanking:
    def test_floor_designation_loses_to_the_business_name(self):
        # '2/F Ever Gotesco Cc' contains no address *word*, so the substring list
        # missed it and it outranked the real header.
        ranked = rank_candidates(
            ["2/F Ever Gotesco Cc", "GRAB PHILIPPINES TRANSPORT"]
        )
        assert ranked[0][0] == "GRAB PHILIPPINES TRANSPORT"

    def test_corner_address_is_penalised(self):
        ranked = rank_candidates(
            ["6023 Sacred Heart cor. Kamagong Sts.", "Yamachan Japanese Restaurant"]
        )
        assert ranked[0][0] == "Yamachan Japanese Restaurant"


class TestDerivedValuesAreNotFabrications:
    def test_a_derived_amount_survives_verification(self):
        from app.core.verification import verify

        # A PH POS slip prints only the VAT-inclusive total and the VAT, so net
        # sales is obtained by division and appears nowhere in the text. Rejecting
        # it as ungrounded discarded a correct value.
        text = "Total Sales (VAT Inclusive) 14,185.00\nVAT 1,519.82"
        result = verify(
            {"net_sales": 12665.18, "total_sales": 14185.00},
            text,
            reconciled=True,
            locale_resolved=True,
            mean_word_confidence=0.9,
            derived_fields={"net_sales"},
        )
        assert result.fields["net_sales"] == 12665.18
        assert "net_sales" not in result.rejected

    def test_an_undeclared_invented_amount_is_still_rejected(self):
        from app.core.verification import verify

        result = verify(
            {"net_sales": 99999.99},
            "Total Sales (VAT Inclusive) 14,185.00",
            reconciled=True,
            locale_resolved=True,
            mean_word_confidence=0.9,
            derived_fields=set(),
        )
        assert result.fields["net_sales"] is None


class TestRenderedCorpusIsReadable:
    @pytest.mark.parametrize("template", ["ph_vat_or", "ph_pos", "us_pos"])
    def test_a_clean_rendering_yields_its_total(self, template):
        """Sanity check on the generator itself.

        If a clean render is not readable, an accuracy figure measured against the
        corpus says more about the renderer than about the pipeline.
        """
        spec = build_spec(0, random.Random(5), template=template)
        reading = read_variant(render_receipt(spec), psm=6)
        digits = "".join(ch for ch in reading.text if ch.isdigit())
        expected = f"{spec.total_amount:.2f}".replace(".", "")
        assert expected in digits or str(int(spec.total_amount)) in digits or expected[1:] in digits
