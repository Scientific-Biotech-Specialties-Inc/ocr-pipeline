"""Money resolution and arithmetic reconciliation.

Money has internal structure a model cannot fake consistently, which makes it the
strongest hallucination detector available. A guessed total will not satisfy
``net + tax + charges = total``; a correctly read one will.

Reconciliation is locale-parameterised. PH VAT is a fixed 12% and inclusive, so
the identity is a hard assertion. US sales tax is exclusive and varies by state,
so only the weaker additive identity is checked and the rate is derived rather
than validated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.financial_semantics import TAX_BASES, UNKNOWN
from app.core.layout import LayoutScan
from app.core.locale import LocaleGuess
from app.core.numbers import normalize_money

PH_VAT_RATE = 0.12
TOLERANCE = 0.02
PLAUSIBLE_RATE_RANGE = (0.0, 0.30)

MONEY_FIELDS = (
    "net_sales", "tax_amount", "total_sales", "service_charge", "discount_amount",
    "withholding_tax", "total_amount",
)


@dataclass
class MoneyResult:
    values: dict[str, float | None] = field(default_factory=dict)
    derived: set[str] = field(default_factory=set)
    evidence: dict[str, str] = field(default_factory=dict)
    reconciled: bool = False
    reconciliation_notes: list[str] = field(default_factory=list)
    tax_rate: float | None = None
    tax_type: str | None = None
    vat_registered: bool = True
    derivations: set[str] = field(default_factory=set)
    tax_basis: str = UNKNOWN
    reported_total: float | None = None
    computed_total: float | None = None
    discrepancy: float | None = None
    financial_reconciliation_status: str = "unresolved"
    needs_manual_review: bool = False
    tax_breakdown: list[dict[str, object]] = field(default_factory=list)
    discount_breakdown: list[dict[str, object]] = field(default_factory=list)

    def get(self, name: str) -> float | None:
        return self.values.get(name)


def resolve_money(
    scan: LayoutScan,
    locale: LocaleGuess,
    vat_registered: bool = True,
    tax_basis: str | None = None,
) -> MoneyResult:
    """Resolve money with an explicit/validated tax basis and safe abstention."""
    country = locale.country
    result = MoneyResult()
    result.vat_registered = vat_registered
    result.tax_basis = tax_basis if tax_basis in TAX_BASES else (
        "inclusive" if country == "PH" else "exclusive" if country == "US" else UNKNOWN
    )

    for name in MONEY_FIELDS:
        candidates = scan.all(name)
        if name == "tax_amount" and candidates:
            parsed = [normalize_money(item.raw_token, country) for item in candidates]
            values = [abs(value) if item.label.startswith("less") else value for item, value in zip(candidates, parsed) if value is not None]
            if values:
                result.values[name] = round(sum(values), 2)
                result.tax_breakdown = [
                    {"amount": abs(value) if item.label.startswith("less") else value, "label": item.label, "evidence": item.evidence}
                    for item, value in zip(candidates, parsed) if value is not None
                ]
                result.evidence[name] = "; ".join(item.evidence for item in candidates)
                continue
        if name == "discount_amount" and candidates:
            parsed = [normalize_money(item.raw_token, country) for item in candidates]
            values = [abs(value) for value in parsed if value is not None and abs(value) > 0]
            if values:
                total_disc = round(sum(values), 2)
                result.values[name] = total_disc
                if len(values) > 1:
                    result.derived.add(name)
                result.discount_breakdown = [
                    {"amount": abs(value), "label": item.label, "line_text": item.line_text, "evidence": item.evidence}
                    for item, value in zip(candidates, parsed) if value is not None and abs(value) > 0
                ]
                result.evidence[name] = "; ".join(item.evidence for item in candidates)
                continue
        labeled = candidates[0] if candidates else None
        if labeled is None:
            result.values[name] = None
            continue
        value = normalize_money(labeled.raw_token, country)
        if value is not None and name in ("discount_amount", "withholding_tax"):
            value = abs(value)
        result.values[name] = value
        if value is not None:
            result.evidence[name] = labeled.evidence

    result.reported_total = result.get("total_amount")
    _fill_gaps(result, scan, country, vat_registered, result.tax_basis)
    _reconcile(result, country, result.tax_basis)
    if result.financial_reconciliation_status == "unresolved":
        result.financial_reconciliation_status = "reconciled" if result.reconciled else "unresolved"
    return result


def _fill_gaps(
    result: MoneyResult, scan: LayoutScan, country: str | None, vat_registered: bool,
    tax_basis: str,
) -> None:
    net = result.get("net_sales")
    tax = result.get("tax_amount")
    total_sales = result.get("total_sales")
    total = result.get("total_amount")
    service = result.get("service_charge") or 0.0
    discount = result.get("discount_amount") or 0.0
    is_ph = country == "PH"

    if is_ph and not vat_registered:
        if net is None and total_sales is not None:
            result.values["net_sales"] = total_sales
            result.derived.add("net_sales")
            result.derivations.add("net_equals_sales_non_vat")
            result.evidence["net_sales"] = f"derived: equals total sales {total_sales} (non-VAT, no tax)"
        is_ph = False

    # PH inclusive is the only country prior. It may still be overridden by
    # explicit wording or validated semantics supplied by the caller.
    if is_ph and tax_basis == "inclusive" and net is None and total_sales is not None:
        net = round(total_sales / (1 + PH_VAT_RATE), 2)
        result.values["net_sales"] = net
        result.derived.add("net_sales")
        result.derivations.add("net_from_sales")
        result.evidence["net_sales"] = f"derived: {total_sales} / 1.12"

    if is_ph and tax_basis == "inclusive" and tax is None and net is not None:
        tax = round(net * PH_VAT_RATE, 2)
        result.values["tax_amount"] = tax
        result.derived.add("tax_amount")
        result.derivations.add("tax_from_net")
        result.evidence["tax_amount"] = f"derived: {net} x 0.12"

    if tax_basis == "exclusive" and total_sales is None and net is not None and tax is not None:
        total_sales = round(net + tax, 2)
        result.values["total_sales"] = total_sales
        result.derived.add("total_sales")
        result.derivations.add("sales_from_net_tax")
        result.evidence["total_sales"] = f"derived: {net} + {tax}"

    if tax_basis == "inclusive" and total_sales is None and net is not None and tax is not None:
        total_sales = round(net + tax, 2)
        result.values["total_sales"] = total_sales
        result.derived.add("total_sales")
        result.derivations.add("sales_from_net_tax")
        result.evidence["total_sales"] = "derived: inclusive gross from printed net and tax"

    # Only an exclusive receipt with independently printed net/subtotal and tax
    # can authorize a computed callback total. Unknown semantics never substitute.
    grounded_net = net is not None and "net_sales" not in result.derived
    grounded_tax = tax is not None and "tax_amount" not in result.derived
    if tax_basis == "exclusive" and grounded_net and grounded_tax:
        computed = round(net - discount + tax + service, 2)
        result.computed_total = computed
        result.discrepancy = round((result.reported_total - computed), 2) if result.reported_total is not None else None
        if total is None:
            result.values["total_amount"] = computed
            result.derived.add("total_amount")
            result.derivations.add("total_from_net_tax_exclusive")
            result.evidence["total_amount"] = f"computed: net {net} - discount {discount} + tax {tax} + service {service}"
            result.financial_reconciliation_status = "computed"
        elif abs(total - computed) <= TOLERANCE:
            result.financial_reconciliation_status = "reported"
        else:
            reported = scan.first("total_amount")
            strong = bool(reported and reported.confidence >= 0.80 and re.search(
                r"total|amount due|grand|balance", reported.label, re.IGNORECASE
            ))
            result.needs_manual_review = True
            result.financial_reconciliation_status = "reported_conflict" if strong else "computed_conflict"
            if not strong:
                result.values["total_amount"] = computed
                result.derived.add("total_amount")
                result.derivations.add("total_from_net_tax_exclusive")
                result.evidence["total_amount"] = f"computed despite weak reported conflict: {computed}"
        return

    # Inclusive totals are already gross: never add tax to total_sales. A printed
    # sales total may be carried to callback only when a money token supports it.
    if total is None and tax_basis == "inclusive" and total_sales is not None:
        candidate = round(total_sales - discount + service, 2)
        if _token_present(candidate, scan, country):
            result.values["total_amount"] = candidate
            result.derived.add("total_amount")
            result.derivations.add("total_from_sales")
            result.evidence["total_amount"] = f"derived: inclusive total sales {total_sales} - discount {discount} + service {service}"
    # Unknown basis intentionally leaves total_amount untouched.


def _token_present(value: float, scan: LayoutScan, country: str | None) -> bool:
    target = round(value, 2)
    for token in scan.money_tokens:
        parsed = normalize_money(token, country)
        if parsed is not None and abs(parsed - target) <= TOLERANCE:
            return True
    return False


def _reconcile(result: MoneyResult, country: str | None, tax_basis: str) -> None:
    net = result.get("net_sales")
    tax = result.get("tax_amount")
    total_sales = result.get("total_sales")
    total = result.get("total_amount")
    service = result.get("service_charge") or 0.0
    discount = result.get("discount_amount") or 0.0
    checks: list[bool] = []
    derivations = result.derivations

    if tax is not None and net is not None and net > 0:
        rate = tax / net
        plausible = PLAUSIBLE_RATE_RANGE[0] <= rate <= PLAUSIBLE_RATE_RANGE[1]
        result.tax_rate = PH_VAT_RATE if country == "PH" and tax_basis == "inclusive" else round(rate, 4)
        if country == "PH" and tax_basis == "inclusive" and "tax_from_net" not in derivations:
            vat_ok = abs(net * PH_VAT_RATE - tax) <= max(TOLERANCE, net * 0.001)
            if not vat_ok and discount > 0 and abs((net - discount) * PH_VAT_RATE - tax) <= max(TOLERANCE, net * 0.001):
                vat_ok = True
            checks.append(vat_ok)
            result.reconciliation_notes.append(
                f"PH VAT identity net*0.12={net * PH_VAT_RATE:.2f} vs tax={tax:.2f}"
            )
        elif country != "PH" or tax_basis == "exclusive":
            checks.append(plausible)
            result.reconciliation_notes.append(
                f"derived rate {rate:.4f} -> {'plausible' if plausible else 'IMPLAUSIBLE'}"
            )

    if net is not None and tax is not None and total_sales is not None:
        if "sales_from_net_tax" in derivations or {"net_from_sales", "tax_from_net"} <= derivations:
            result.reconciliation_notes.append("net+tax identity skipped: figures derived from one another")
        else:
            ok = (
                abs(net + tax - total_sales) <= TOLERANCE
                or (discount > 0 and abs(net + tax - discount - total_sales) <= TOLERANCE)
            )
            checks.append(ok)
            result.reconciliation_notes.append(
                f"net+tax={net + tax:.2f} vs total_sales={total_sales:.2f} -> {'ok' if ok else 'FAIL'}"
            )

    total_is_circular = bool({"sales_from_total", "total_from_sales", "total_from_net_tax_exclusive"} & derivations)
    if total_sales is not None and total is not None:
        if total_is_circular:
            result.reconciliation_notes.append("total identity skipped: total and sales derived from one another")
        else:
            expected_net_total = total_sales - discount + service
            ok = abs(expected_net_total - total) <= TOLERANCE
            # Statutory PH discounts (Senior/PWD) deduct both 12% VAT and 20% discount from gross sales
            if not ok and discount > 0 and tax is not None and abs(total_sales - tax - discount + service - total) <= TOLERANCE:
                ok = True
            if not ok and discount > 0 and net is not None and abs(net - discount + service - total) <= TOLERANCE:
                ok = True
            if not ok and discount > 0 and abs(total_sales + service - total) <= TOLERANCE:
                # Some receipts report total_sales already net of discount
                ok = True
            checks.append(ok)
            result.reconciliation_notes.append(
                f"total_sales-discount+service={expected_net_total:.2f} vs total={total:.2f} -> {'ok' if ok else 'FAIL'}"
            )

    if tax is None and net is not None and total is not None and not service:
        if "net_sales" in result.derived:
            result.reconciliation_notes.append("no-tax identity skipped: net was derived from the total")
        else:
            ok = abs(net - discount - total) <= TOLERANCE or abs(net - total) <= TOLERANCE
            checks.append(ok)
            result.reconciliation_notes.append(
                f"no-tax receipt: net-discount={net - discount:.2f} vs total={total:.2f} -> {'ok' if ok else 'FAIL'}"
            )

    result.reconciled = bool(checks) and all(checks)
    if result.reconciled and result.financial_reconciliation_status == "unresolved":
        result.financial_reconciliation_status = "reconciled"
    elif not result.reconciled and result.financial_reconciliation_status == "unresolved":
        result.reconciliation_notes.append("no identity could be checked" if not checks else "identity failed")



def resolve_tax_type(
    country: str | None, vat_classification: str | None, tax_amount: float | None
) -> str | None:
    """Name the tax regime.

    For PH this follows the vendor's registration, so a VAT-registered receipt is
    typed ``VAT`` even when its amounts are illegible, and a non-VAT sales invoice
    is typed ``None``. Elsewhere there is no registration marker to read, so the
    type is only asserted when a tax figure was actually found.
    """
    if country == "PH":
        return "VAT" if vat_classification == "vat" else None
    if tax_amount is None:
        return None
    return "SALES_TAX" if country == "US" else "TAX"


def classify_tax(text: str, locale: LocaleGuess, tax_amount: float | None) -> str | None:
    """VAT classification, locale aware.

    Returns ``vat`` / ``non-vat`` for PH and ``None`` elsewhere - SERMS validates
    this field with ``in:vat,non-vat``, so any third value would 422 the whole
    callback. Non-PH semantics are carried by ``tax_type`` internally instead.
    """
    if locale.country != "PH":
        return None

    lowered = text.casefold()
    if "non-vat reg" in lowered or "non vat reg" in lowered:
        return "non-vat"
    if "not valid for claiming input tax" in lowered:
        return "non-vat"
    if "vat reg" in lowered:
        return "vat"
    if "vat-exempt sales" in lowered or "vat exempt sales" in lowered:
        # A printed breakdown label alone does not make a receipt VAT-exempt.
        return "vat" if tax_amount else "non-vat"
    if tax_amount:
        return "vat"
    return "non-vat"
