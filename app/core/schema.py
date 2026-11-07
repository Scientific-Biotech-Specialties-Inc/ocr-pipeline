"""Canonical extraction schema.

Owned by ``app.core`` so the pipeline and the evaluation harness cannot drift
apart. The harness layers accuracy gates on top of these definitions rather than
restating them.
"""

from __future__ import annotations

from enum import Enum


class FieldKind(str, Enum):
    """How a field is compared and grounded. Money is not compared like text."""

    MONEY = "money"
    RATE = "rate"
    TAX_ID = "tax_id"
    DATE = "date"
    FUZZY_TEXT = "fuzzy_text"
    EXACT = "exact"


FIELD_KINDS: dict[str, FieldKind] = {
    "vendor_name": FieldKind.FUZZY_TEXT,
    "country": FieldKind.EXACT,
    "currency": FieldKind.EXACT,
    "vendor_tax_id": FieldKind.TAX_ID,
    "vendor_tax_id_type": FieldKind.EXACT,
    "transaction_date": FieldKind.DATE,
    "net_sales": FieldKind.MONEY,
    "tax_amount": FieldKind.MONEY,
    "tax_type": FieldKind.EXACT,
    "tax_rate": FieldKind.RATE,
    "total_sales": FieldKind.MONEY,
    "service_charge": FieldKind.MONEY,
    "discount_amount": FieldKind.MONEY,
    "discount_type": FieldKind.EXACT,
    "withholding_tax": FieldKind.MONEY,
    "total_amount": FieldKind.MONEY,
    "vat_classification": FieldKind.EXACT,
    "invoice_number": FieldKind.EXACT,
    "expense_category": FieldKind.EXACT,
    "location": FieldKind.FUZZY_TEXT,
}

MONEY_FIELDS: tuple[str, ...] = tuple(
    name for name, kind in FIELD_KINDS.items() if kind is FieldKind.MONEY
)

DISCOUNT_TYPES: tuple[str, ...] = (
    "none", "senior", "pwd", "student", "promo", "employee", "other",
)

# Fields inferred from context or computed arithmetically rather than read off the
# page. Asking whether they "appear in the OCR text" is meaningless, so they are
# exempt from grounding and from recoverability measurement.
DERIVED_FIELDS: frozenset[str] = frozenset({
    "country", "currency", "vendor_tax_id_type", "tax_type", "tax_rate",
    "vat_classification", "expense_category", "tax_basis", "financial_reconciliation_status",
    "needs_manual_review", "reported_total", "computed_total", "total_discrepancy",
    "discount_type",
})

# SERMS' canonical list, from ExpenseCategory::DEFAULT_NAMES. Anything outside it
# is coerced to 'Others': SERMS calls firstOrCreate() on the category name, so a
# novel string permanently creates a new category row.
EXPENSE_CATEGORIES: tuple[str, ...] = (
    "Meals", "Travel", "Supplies", "Accommodation", "Transportation", "Others",
)
DEFAULT_EXPENSE_CATEGORY = "Others"

# Fields the SERMS callback contract accepts. Anything else is stored internally
# only - Laravel's validated() silently drops undeclared keys.
CALLBACK_FIELDS: tuple[str, ...] = (
    "vendor_name", "transaction_date", "total_amount", "vat_amount", "tin",
    "invoice_number", "vat_classification", "currency", "expense_category", "items",
    "location", "tax_basis", "financial_reconciliation_status", "needs_manual_review",
    "discount_amount", "discount_type",
)

