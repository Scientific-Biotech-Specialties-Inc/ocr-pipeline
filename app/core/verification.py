"""The verification gate.

Every extracted value passes through here before it reaches confidence scoring or
the callback. One choke point, so rejection is testable and auditable in a single
place.

Policy, decided deliberately: a value that fails verification is set to ``None``,
not merely marked low-confidence. A fabricated total with confidence 0.4 still
enters the financial system; a null routes the receipt to a human. Rejections are
recorded so the decision can be audited afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.grounding import is_grounded
from app.core.schema import DEFAULT_EXPENSE_CATEGORY, EXPENSE_CATEGORIES, MONEY_FIELDS

# Reasons SERMS reviewers care about. Emitted alongside the score so a human knows
# where to look instead of re-reading the whole receipt.
REASON_HANDWRITTEN = "handwriting_or_illegible"
REASON_NO_TOTAL = "total_amount_missing"
REASON_RECONCILE = "reconciliation_failed"
REASON_LOCALE = "locale_unresolved"
REASON_REJECTED = "ungrounded_values_rejected"
REASON_TAX_ID_AMBIGUOUS = "tax_id_ambiguous"
REASON_NON_LATIN = "non_latin_script"
REASON_SEGMENT_GAP = "segment_gap_suspected"
REASON_NEAR_RECONCILED = "near_reconciled"

# Confidence ceilings per condition, calibrated to the consumer's threshold.
# SERMS routes anything below 0.80 to 'flagged' for manual confirmation
# (OcrCallbackService: $isLowConfidence = $confidenceScore < 0.80), so every
# abstention condition must land below it. Without caps, a receipt yielding two
# grounded fields out of a possible fifteen would score a perfect grounding rate
# and sail past 0.80 on almost no information.
REASON_CAPS: dict[str, float] = {
    REASON_HANDWRITTEN: 0.45,
    REASON_NO_TOTAL: 0.50,
    REASON_RECONCILE: 0.60,
    REASON_REJECTED: 0.70,
    REASON_LOCALE: 0.70,
    REASON_TAX_ID_AMBIGUOUS: 0.70,
    REASON_NON_LATIN: 0.45,
    REASON_SEGMENT_GAP: 0.75,
    REASON_NEAR_RECONCILED: 0.79,
}

# Below this share of legible characters the page is treated as unreadable.
MIN_LEGIBLE_RATIO = 0.55
MIN_WORD_CONFIDENCE = 0.45
_NON_LATIN_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u0e00-\u0e7f\u0400-\u04ff]")


@dataclass
class Verification:
    fields: dict[str, Any] = field(default_factory=dict)
    rejected: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    grounded_count: int = 0
    populated_count: int = 0

    @property
    def grounding_pass_rate(self) -> float:
        """Share of emitted values that trace back to the page.

        Vacuously 1.0 when nothing was emitted, which is why the hard caps exist.
        """
        if self.populated_count == 0:
            return 1.0
        return self.grounded_count / self.populated_count

    @property
    def needs_manual_review(self) -> bool:
        return bool(self.reasons)

    @property
    def cap(self) -> float:
        return min((REASON_CAPS[r] for r in self.reasons if r in REASON_CAPS), default=1.0)


def verify(
    fields: dict[str, Any],
    ocr_text: str,
    *,
    reconciled: bool,
    locale_resolved: bool,
    mean_word_confidence: float = 0.0,
    tax_id_ambiguous: bool = False,
    derived_fields: set[str] | None = None,
    near_reconciled: bool = False,
    segment_gap: bool = False,
) -> Verification:
    """Reject ungrounded values and decide whether a human needs to look.

    ``derived_fields`` names values computed arithmetically rather than read off
    the page. A PH POS receipt prints only the VAT-inclusive total and the VAT, so
    net sales is obtained by division and legitimately appears nowhere in the text.
    Grounding does not apply to it - the arithmetic identities do, and those are
    checked separately. Note that a derived ``total_amount`` is still required to
    match a figure printed on the page, because inventing the amount paid is the
    one error with direct financial consequence.
    """
    result = Verification(fields=dict(fields))
    derived_fields = derived_fields or set()

    for name, value in list(result.fields.items()):
        if value is None:
            continue
        result.populated_count += 1
        if name in derived_fields or is_grounded(name, value, ocr_text):
            result.grounded_count += 1
            continue
        result.rejected[name] = value
        result.fields[name] = None
        result.populated_count -= 1

    # Coerce only a category that was actually proposed. Defaulting an unknown to
    # 'Others' would assert a classification we never made, and inflate the
    # grounding denominator with a value carrying no information.
    if result.fields.get("expense_category") is not None:
        result.fields["expense_category"] = coerce_category(
            result.fields["expense_category"]
        )

    _drop_unreconciled_money(result, reconciled)
    if not result.fields.get("discount_amount"):
        result.fields["discount_type"] = None
    _collect_reasons(
        result,
        ocr_text=ocr_text,
        reconciled=reconciled,
        locale_resolved=locale_resolved,
        mean_word_confidence=mean_word_confidence,
        tax_id_ambiguous=tax_id_ambiguous,
        near_reconciled=near_reconciled,
        segment_gap=segment_gap,
    )
    return result


def _drop_unreconciled_money(result: Verification, reconciled: bool) -> None:
    """Withhold amounts whose arithmetic cannot be confirmed.

    Receipt 3 is the case this exists for: OCR drops the leading digit of
    ``139.93`` and returns ``39.93``. The value is grounded - ``39.93`` really is
    on the page - so grounding alone cannot catch it. But it fails the VAT
    identity, and a tax figure that contradicts the sales figures is worse than no
    tax figure, because it would be claimed against input tax.

    Only applied when a total is also absent, so a fully-read receipt with one odd
    charge line is not stripped of everything.
    """
    if reconciled:
        return
    if result.fields.get("total_amount") is not None:
        return
    for name in MONEY_FIELDS:
        value = result.fields.get(name)
        if value is not None:
            result.rejected.setdefault(f"{name} (unreconciled)", value)
            result.fields[name] = None
            result.populated_count = max(0, result.populated_count - 1)
            result.grounded_count = max(0, result.grounded_count - 1)


def _collect_reasons(
    result: Verification,
    *,
    ocr_text: str,
    reconciled: bool,
    locale_resolved: bool,
    mean_word_confidence: float,
    tax_id_ambiguous: bool,
    near_reconciled: bool = False,
    segment_gap: bool = False,
) -> None:
    if result.rejected:
        result.reasons.append(REASON_REJECTED)
    if result.fields.get("total_amount") is None:
        result.reasons.append(REASON_NO_TOTAL)
    if not reconciled:
        if near_reconciled:
            result.reasons.append(REASON_NEAR_RECONCILED)
        else:
            result.reasons.append(REASON_RECONCILE)
    if segment_gap:
        result.reasons.append(REASON_SEGMENT_GAP)
    if not locale_resolved:
        result.reasons.append(REASON_LOCALE)
    if tax_id_ambiguous:
        result.reasons.append(REASON_TAX_ID_AMBIGUOUS)
    if _NON_LATIN_RE.search(ocr_text):
        result.reasons.append(REASON_NON_LATIN)
    if looks_illegible(ocr_text, mean_word_confidence):
        result.reasons.append(REASON_HANDWRITTEN)


def looks_illegible(ocr_text: str, mean_word_confidence: float) -> bool:
    """Heuristic detector for handwriting and failed reads.

    Handwritten forms come back as fragments of printed labels surrounded by
    noise, so the signal is a low ratio of well-formed words plus low per-word
    confidence. Cheap, and it only needs to be right about the extremes - the
    money-side gates already catch the rest.
    """
    text = ocr_text.strip()
    if not text:
        return True

    words = re.findall(r"[A-Za-z]{2,}", text)
    tokens = [token for token in re.split(r"\s+", text) if token]
    if not tokens:
        return True

    word_ratio = len(words) / len(tokens)
    if mean_word_confidence and mean_word_confidence < MIN_WORD_CONFIDENCE:
        return True
    return word_ratio < MIN_LEGIBLE_RATIO


def coerce_category(value: Any) -> str:
    """Map a category onto SERMS' closed list.

    SERMS does ``ExpenseCategory::firstOrCreate(['name' => ...])``, so any novel
    string permanently creates a category row. 'Meals & Entertainment' and
    'Office Supplies' - both produced by the previous prompt - are exactly the
    strings that polluted the taxonomy.
    """
    if not value:
        return DEFAULT_EXPENSE_CATEGORY
    text = str(value).strip()
    for category in EXPENSE_CATEGORIES:
        if text.casefold() == category.casefold():
            return category
    lowered = text.casefold()
    for category in EXPENSE_CATEGORIES:
        if category.casefold() in lowered or lowered in category.casefold():
            return category
    return DEFAULT_EXPENSE_CATEGORY
