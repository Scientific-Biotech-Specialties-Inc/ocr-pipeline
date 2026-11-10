"""Validation and deterministic rules for financial semantics.

The model may classify tax basis and currency context, but it never supplies money.
All accepted answers are bounded by this module and must cite OCR text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

TAX_BASES = frozenset({"inclusive", "exclusive", "unknown"})
UNKNOWN = "unknown"
# ISO-4217 alphabetic codes used by callers and common receipt locales. The
# callback schema deliberately has a smaller consumer allowlist.
ISO_4217_CODES = frozenset({
    "AED", "ARS", "AUD", "BHD", "BND", "BRL", "CAD", "CHF", "CNY", "COP",
    "CZK", "DKK", "EGP", "EUR", "GBP", "HKD", "IDR", "ILS", "INR", "JPY",
    "KRW", "KWD", "MYR", "MXN", "NOK", "NZD", "PHP", "PLN", "QAR", "RUB",
    "SAR", "SEK", "SGD", "THB", "TRY", "TWD", "USD", "VND", "ZAR",
})

@dataclass(frozen=True)
class FinancialSemantics:
    tax_basis: str = UNKNOWN
    confidence: float = 0.0
    evidence: tuple[str, ...] = field(default_factory=tuple)
    currency: str = UNKNOWN
    currency_confidence: float = 0.0
    currency_evidence: tuple[str, ...] = field(default_factory=tuple)
    source: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tax_basis": self.tax_basis,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "currency": self.currency,
            "currency_confidence": self.currency_confidence,
            "currency_evidence": list(self.currency_evidence),
            "source": self.source,
        }


def normalize_iso_currency(value: Any) -> str | None:
    if value is None:
        return None
    code = str(value).strip().upper()
    return code if code in ISO_4217_CODES else None


def _evidence_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def validate_financial_semantics(
    raw: Any, ocr_text: str = "", *, source: str = "gemini"
) -> FinancialSemantics | None:
    """Validate a provider response and reject guesses or unusable evidence."""
    if not isinstance(raw, dict):
        return None
    basis = str(raw.get("tax_basis", UNKNOWN)).strip().lower()
    if basis not in TAX_BASES:
        basis = UNKNOWN
    try:
        confidence = float(raw.get("tax_basis_confidence", raw.get("confidence", 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = _evidence_list(raw.get("tax_basis_evidence", raw.get("evidence")))
    try:
        currency_confidence = float(raw.get("currency_confidence", raw.get("confidence", 0.0)))
    except (TypeError, ValueError):
        currency_confidence = 0.0
    currency = normalize_iso_currency(raw.get("currency")) or UNKNOWN
    currency_evidence = _evidence_list(raw.get("currency_evidence", raw.get("evidence")))
    lowered = re.sub(r"\s+", " ", ocr_text.casefold())
    def grounded(items: tuple[str, ...]) -> bool:
        return bool(items) and all(
            re.sub(r"\s+", " ", item.casefold()).strip() in lowered
            for item in items
        )
    if basis != UNKNOWN and (confidence < 0.75 or not grounded(evidence)):
        basis, confidence, evidence = UNKNOWN, 0.0, ()
    if currency != UNKNOWN and (currency_confidence < 0.75 or not grounded(currency_evidence)):
        currency, currency_confidence, currency_evidence = UNKNOWN, 0.0, ()
    resolved_source = source if raw else "none"
    return FinancialSemantics(
        tax_basis=basis, confidence=max(0.0, min(confidence, 1.0)), evidence=evidence,
        currency=currency, currency_confidence=max(0.0, min(currency_confidence, 1.0)),
        currency_evidence=currency_evidence, source=resolved_source,
    )


def infer_tax_basis(text: str, country: str | None = None) -> tuple[str, list[str]]:
    """Infer only explicit wording; country is a fallback prior, never proof."""
    lowered = text.casefold()
    inclusive = re.search(r"vat\s*inclusive|tax\s*inclusive|inclusive\s*(?:of|vat)|incl\.?\s*vat", lowered)
    exclusive = re.search(r"vat\s*exclusive|tax\s*exclusive|exclusive\s*(?:of|vat)|net\s+of\s+vat|before\s+tax|plus\s+(?:vat|tax)|add\s*:\s*(?:vat|tax)", lowered)
    if inclusive:
        return "inclusive", [inclusive.group(0)]
    if exclusive:
        return "exclusive", [exclusive.group(0)]
    if country == "US" and re.search(r"\b(?:sales\s+)?tax\b", lowered):
        return "exclusive", ["US country fallback"]
    if country == "MY" and re.search(r"\b(?:sst|gst)\b", lowered):
        # Malaysian receipts add SST/GST on top of the subtotal (net + tax = total).
        return "exclusive", ["MY SST/GST fallback"]
    if country == "PH":
        return "inclusive", ["PH country fallback"]
    return UNKNOWN, []


def semantics_requested(text: str, currency_ambiguous: bool) -> bool:
    return currency_ambiguous or bool(re.search(r"\b(?:vat|tax|gst|sst|inclusive|exclusive)\b|[₱€£฿$]", text, re.I))
