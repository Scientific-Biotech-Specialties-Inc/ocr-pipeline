"""Grounding: does an extracted value actually appear in the OCR text?

This is the mechanical half of the anti-hallucination design. Prompt instructions
("return null if unsure") are advisory and a 1.5B model follows them unreliably.
Grounding is a check: a value is either supported by text the reader saw, or it is
not. That turns a 0% fabrication target into something measurable rather than
aspirational.

Comparison is deliberately lenient about formatting and strict about content.
``000-774-47 1-025`` grounds against ``000774471025`` because OCR mangles
separators, but no amount of leniency will ground a total that was never printed.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any

from rapidfuzz import fuzz

from app.core.numbers import digits_only, normalize_money
from app.core.schema import DERIVED_FIELDS, FIELD_KINDS, FieldKind

FUZZY_GROUNDING_THRESHOLD = 75.0

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().casefold()


def parse_date_value(raw: Any) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    text = str(raw).strip()
    for fmt in (
        "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y",
        "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y", "%d %b %y", "%d %b %Y",
        "%d-%b-%Y", "%d-%b-%y", "%b-%d-%Y", "%b-%d-%y",
        "%d/%b/%Y", "%d/%b/%y", "%b/%d/%Y", "%b/%d/%y",
    ):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        try:
            return date(*(int(group) for group in match.groups()))
        except ValueError:
            return None
    return None


def is_grounded(field_name: str, value: Any, ocr_text: str) -> bool:
    """Is there textual evidence in the OCR output for this value?"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return True  # abstention is always grounded
    if field_name in DERIVED_FIELDS:
        return True

    kind = FIELD_KINDS.get(field_name, FieldKind.EXACT)
    digit_stream = digits_only(ocr_text)

    if kind is FieldKind.MONEY:
        return _money_grounded(value, digit_stream)
    if kind is FieldKind.RATE:
        return True  # computed from other figures, never transcribed
    if kind is FieldKind.TAX_ID:
        target = digits_only(value)
        return bool(target) and target in digit_stream
    if kind is FieldKind.DATE:
        parsed = parse_date_value(value)
        return parsed is not None and _date_appears(parsed, ocr_text, digit_stream)
    if kind is FieldKind.FUZZY_TEXT:
        return (
            fuzz.partial_ratio(normalize_text(value), normalize_text(ocr_text))
            >= FUZZY_GROUNDING_THRESHOLD
        )
    return normalize_text(value) in normalize_text(ocr_text)


def _money_grounded(value: Any, digit_stream: str) -> bool:
    number = normalize_money(value)
    if number is None:
        return False
    candidates = {f"{number:.2f}"}
    if number == int(number):
        candidates.add(f"{int(number)}")
    return any(digits_only(c) in digit_stream for c in candidates if digits_only(c))


_OCR_CHAR_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
})


def _date_appears(value: date, ocr_text: str, digit_stream: str) -> bool:
    year4, year2 = f"{value.year:04d}", f"{value.year % 100:02d}"
    # Both padded and bare forms: receipts print '5/24/2025' as often as
    # '05-24-2025', and digit-stream matching is padding-sensitive.
    months = {f"{value.month:02d}", str(value.month)}
    days = {f"{value.day:02d}", str(value.day)}
    renderings = {
        f"{m}{d}{y}" for m in months for d in days for y in (year4, year2)
    }
    renderings |= {f"{y}{m}{d}" for m in months for d in days for y in (year4,)}
    renderings |= {f"{d}{m}{y}" for m in months for d in days for y in (year4, year2)}
    if any(rendering in digit_stream for rendering in renderings):
        return True

    # Check normalized digit stream for OCR-character recovery (e.g. O->0, l->1)
    normalized_stream = digits_only(ocr_text.translate(_OCR_CHAR_MAP))
    if any(rendering in normalized_stream for rendering in renderings):
        return True

    lowered = normalize_text(ocr_text)
    abbreviation = next(
        (abbr for abbr, number in _MONTHS.items() if number == value.month), None
    )
    if abbreviation and abbreviation in lowered:
        return str(value.day) in lowered or f"{value.day:02d}" in lowered
    return False
