"""Line-item extraction.

Deterministic throughout: an item row is a line inside the item region carrying a
money token, and its quantity is the leading integer. No language model is
involved, because an item name is a transcription rather than a judgement - a model
asked to tidy 'ChickenJjoy' will eventually produce product names that were never
printed, and unlike a vendor name there is no candidate shortlist to constrain it
against.

Two rules carry most of the weight:

* A line with no money token is a DESCRIPTION of the item above it, not a new
  item. 'Family bundle--10pcs' followed by 'Chickenjoy bucket + 3 large sides' is
  one physical purchase and must produce one row.
* Items are emitted only when they add up. Receipts disagree about whether the
  printed figure is the line total or the unit price, so both readings are summed
  and whichever reconciles against the subtotal wins. If neither does, the items
  are withheld entirely - a partial list looks authoritative while being wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings
from app.core.layout import _best_label, _collect_money_tokens
from app.core.numbers import normalize_money
from app.core.ocr_engine import Word

# Lines that introduce the item region. The region may also begin at a printed
# rule, so a separator counts too.
REGION_START_PATTERNS = (
    r"^\s*details\b", r"nature\s+of\s+service", r"^\s*particulars\b",
    r"^\s*description\b", r"^\s*desc\b",
    r"^\s*qty\b", r"\bunit\s+cost\b", r"\bunit\s+price\b",
    r"^\s*item(?:s)?\s*(?:desc|name|price|amt|amount|\b|$)",
    r"^\s*item\s*#",
    r"^\*{4,}$", r"^-{4,}$", r"^={4,}$",
)
_REGION_START_RE = re.compile("|".join(REGION_START_PATTERNS), re.IGNORECASE)

# The item region ends at the first summary line. These are the money labels the
# layout scanner already recognises.
SUMMARY_FIELDS = frozenset({
    "total_amount", "total_sales", "net_sales", "tax_amount",
    "service_charge", "discount_amount", "withholding_tax",
})

# Footer, payment, and promo noise that can carry money tokens.
ITEM_NOISE_PATTERNS = (
    r"change\s+due", r"\bpayment\b", r"\bcash\b", r"\bcard\b", r"tender",
    r"authority\s+to\s+print", r"\bocn\b", r"booklets", r"valid\s+until",
    r"\batp\b", r"vat\s*reg", r"\btin\b", r"\bbalance\b", r"amount\s+in\s+words",
    r"tota[l!1i]\s+item", r"item\s+sold", r"re[sg]ular\s+price", r"card\s+savin[gces]*",
    r"card\s+saver", r"mfrcpn", r"cartwheel", r"you\s+saved",
    r"carry\s*-?\s*out", r"dine\s*-?\s*in", r"take\s*-?\s*out",
    r"^\s*amt\b", r"suggested\s+gratuity", r"grat(?:uity)?\s+\d+%",
    r"^\s*tip\b", r"tax[i]?\s*\+?\s*tip",
)
_ITEM_NOISE_RE = re.compile("|".join(ITEM_NOISE_PATTERNS), re.IGNORECASE)

# Weight breakdown lines like '2.51 lb @ $0.99 /lb'
_WEIGHT_BREAKDOWN_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:lb|kg|g|oz|lbs)\s*@\s*[$pP€£¥]?\d+(?:[.,]\d+)?\s*/\s*(?:lb|kg|g|oz|lbs)\b",
    re.IGNORECASE,
)

# Header metadata lines (e.g. store address, cashier, phone)
_HEADER_METADATA_RE = re.compile(
    r"\b(?:tel|phone|fax|mobile|cel|contact)\b"
    r"|\b(?:cashier|terminal|register|trans(?:action)?|order|table|check|chk)\s*[:#0-9]"
    r"|\b(?:str|store)\s*(?:no|#)?\s*[:0-9]"
    r"|\b(?:date|time)\s*:"
    r"|\b(?:min|sn|ptu|atp|pn|spm)\s*:"
    r"|\b(?:vat\s*reg|taxpayer|non\s*vat)\b"
    r"|\b(?:dine\s*in|take\s*out|drive\s*thru)\b",
    re.IGNORECASE,
)

_LEADING_QUANTITY_RE = re.compile(
    r"^\s*(\d{1,3})\s*(?:[xX×@*]|pc|pcs|pieces|qty)?\s+(?=\S)", re.IGNORECASE
)
_UPC_PREFIX_RE = re.compile(r"^\s*(?:\d{8,14}|[A-Za-z0-9]{2,6}\d{7,14})\s+")
_TRAILING_QUANTITY_RE = re.compile(
    r"(?:\s+[xX×@*]\s*(\d{1,3})|\s+(\d{1,3})\s*[xX×@*]?)\s*$", re.IGNORECASE
)
_TRAILING_TAX_FLAG_RE = re.compile(
    r"\s+(?:[VTEZSYN]|NV|VAT)\s*$", re.IGNORECASE
)
_INLINE_UNIT_PRICE_RE = re.compile(
    r"@\s*[pP$€£¥]?\d+(?:[,.]\d{2,3})?(?:\s*/\s*(?:unit|pc|pcs|ea|kg|lb|gal|l)\b)?"
    r"|[pP$€£¥]?\d+(?:[,.]\d{2,3})?\s*/\s*(?:unit|pc|pcs|ea|kg|lb|gal|l)\b",
    re.IGNORECASE,
)

MAX_NAME_LENGTH = 255
MAX_ITEMS = getattr(settings, "ocr_max_items", 250)

# Stand-in for a name OCR could not read. Faded thermal print yields rows whose
# price is perfectly legible and whose product name is noise ('ci cr } UM SDR \').
#
# Dropping such a row would be worse: on a three-item receipt one unreadable name
# would leave two rows summing to less than the subtotal, the sum check would fail,
# and ALL items would be withheld - including the two read correctly. Keeping the
# row preserves the arithmetic, and a placeholder states honestly that the name is
# unknown instead of presenting noise as a product.
UNREADABLE_NAME = "Unreadable item"


@dataclass
class ExtractedItem:
    name: str
    quantity: int
    price: float
    line_text: str = ""
    descriptions: list[str] = field(default_factory=list)
    price_is_line_total: bool = False

    def payload(self, price: float | None = None) -> dict[str, object]:
        return {
            "name": self.full_name,
            "quantity": self.quantity,
            "price": round(price if price is not None else self.price, 2),
        }

    @property
    def full_name(self) -> str:
        parts = [self.name, *self.descriptions]
        combined = " ".join(part for part in parts if part).strip()
        if not looks_like_item_name(combined):
            return UNREADABLE_NAME
        return combined[:MAX_NAME_LENGTH]

    @property
    def name_is_readable(self) -> bool:
        return self.full_name != UNREADABLE_NAME


@dataclass
class ItemScan:
    items: list[ExtractedItem] = field(default_factory=list)
    reconciled: bool = False
    price_basis: str = "none"
    """``line_total`` or ``unit_price`` - which reading of the printed figure
    reconciled. The emitted price is always the line total regardless, because
    SERMS' receipt_items table has a single price column."""
    total: float | None = None
    target: float | None = None
    notes: list[str] = field(default_factory=list)
    near_reconciled: bool = False

    @property
    def count(self) -> int:
        return len(self.items)

    def payload(self) -> list[dict[str, object]]:
        """Rows to transmit."""
        rows: list[dict[str, object]] = []
        for item in self.items:
            price = (
                item.price
                if self.price_basis in ("line_total", "none") or item.price_is_line_total
                else round(item.price * item.quantity, 2)
            )
            if price <= 0:
                # SERMS' PRS path validates item price as gt:0, so a non-positive
                # line is withheld from payload to avoid API rejection.
                continue
            rows.append(item.payload(price))
        return rows


def parse_items_candidates(lines: list[list[Word]]) -> list[list[ExtractedItem]]:
    """Return distinct candidate item lists from both marker-required and markerless passes."""
    cands: list[list[ExtractedItem]] = []
    seen_signatures: set[tuple] = set()

    for require_marker in (True, False):
        raw_items = _parse_region(lines, require_marker=require_marker)
        items = _strip_running_subtotal(raw_items)
        if items:
            sig = tuple((it.full_name, it.quantity, it.price) for it in items)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                cands.append(items)

    return cands or [[]]


def parse_items(lines: list[list[Word]]) -> list[ExtractedItem]:
    """Extract item rows from one OCR reading."""
    cands = parse_items_candidates(lines)
    if not cands or not cands[0]:
        return []
    if len(cands) == 1:
        return cands[0]

    marker_items, no_marker_items = cands[0], cands[1]
    # If the marker pass only caught a few late lines near the end while markerless caught a fuller list, prefer markerless
    if len(no_marker_items) > len(marker_items) and len(marker_items) <= 2:
        return no_marker_items
    return marker_items


def _strip_running_subtotal(items: list[ExtractedItem]) -> list[ExtractedItem]:
    """Drop a row whose amount equals the sum of the rows above it.

    Such a row is a subtotal, not a product. Receipt 11 prints its department
    subtotal as 'FOOD 64.90', which no label vocabulary would catch - but the
    arithmetic gives it away, because 64.90 is exactly the eight lines above it.

    Solving this by arithmetic rather than by adding more label spellings means it
    works for 'BEVERAGE', 'DEPT 1', 'SUB-TOTAL' and anything else a POS prints,
    including wordings OCR has mangled beyond recognition.
    """
    if len(items) < 3:
        return items  # too few rows for a sum to be meaningful

    running = 0.0
    for index, item in enumerate(items):
        if index >= 2 and abs(item.price - running) <= 0.02:
            return items[:index]
        running = round(running + item.price, 2)
    return items


def _is_header_or_metadata_line(text: str, seen_item: bool) -> bool:
    if not seen_item:
        if _HEADER_METADATA_RE.search(text):
            return True
        lowered = text.casefold()
        if any(term in lowered for term in ("st.", "street", "ave.", "avenue", "blvd", "rd.", "road", "city", "brgy")):
            return True
    return False


def _parse_region(lines: list[list[Word]], require_marker: bool) -> list[ExtractedItem]:
    tokens = _collect_money_tokens(lines)
    token_word_ids = {id(word) for token in tokens for word in token.words}

    items: list[ExtractedItem] = []
    started = not require_marker
    seen_item = False
    pending: ExtractedItem | None = None
    pending_qty_prefix: tuple[int, float] | None = None
    buffered_name: str | None = None

    def _flush_pending() -> None:
        nonlocal pending, seen_item
        if pending is not None:
            if pending.name and pending.name.strip():
                items.append(pending)
                seen_item = True
            pending = None

    def _flush_buffered_to_desc() -> None:
        nonlocal buffered_name
        if buffered_name is not None:
            if items and _plausible_description(buffered_name):
                items[-1].descriptions.append(buffered_name)
            buffered_name = None

    for line in lines:
        text = " ".join(word.text for word in line).strip()
        if not text:
            continue

        label = _best_label(text)
        if label and label[0] == "discount_amount":
            _flush_pending()
            _flush_buffered_to_desc()
            continue
        if label and label[0] in SUMMARY_FIELDS:
            is_summary = True
            if label[1] in ("total", "net of vat", "subtotal", "tax"):
                words_in_line = [w for w in text.split() if w.isalnum()]
                if len(words_in_line) >= 3 and not text.lower().startswith(("subtotal", "total", "tax", "vat")):
                    is_summary = False

            if is_summary:
                _flush_pending()
                _flush_buffered_to_desc()
                if seen_item:
                    break
                continue

        line_word_ids = {id(w) for w in line}
        line_tokens = [t for t in tokens if any(id(w) in line_word_ids for w in t.words)]
        money = [t.text for t in line_tokens]

        if not money and not re.search(r"^\s*\d+\s+qty\b", text, re.IGNORECASE) and _REGION_START_RE.search(text):
            started = True
            continue
        if not started:
            continue
        if _ITEM_NOISE_RE.search(text):
            continue

        if _WEIGHT_BREAKDOWN_RE.search(text):
            if items:
                items[-1].descriptions.append(text)
            continue

        if money:
            chosen_token = line_tokens[-1]
            words_left = [w for w in line if id(w) not in token_word_ids and w.left < chosen_token.left]
            words_right = [w for w in line if id(w) not in token_word_ids and w.left >= chosen_token.left + chosen_token.word.width]
            text_left = " ".join(w.text for w in words_left).strip()
            remainder = text_left if text_left else " ".join(w.text for w in words_right).strip()

            if pending is not None:
                if _is_quantity_line(remainder):
                    quantity = _quantity_of(remainder)
                    if quantity is not None:
                        pending.quantity = quantity
                    continue
                qty_price_match = _QUANTITY_LINE_WITH_PRICE_RE.match(text)
                if qty_price_match:
                    q_digits = qty_price_match.group(1).replace("I", "1").replace("l", "1")
                    q_val = max(1, int(q_digits))
                    u_val = float(qty_price_match.group(2).replace(",", "."))
                    pending.quantity = q_val
                    if abs(q_val * u_val - pending.price) <= 0.05:
                        pending.price_is_line_total = True
                    continue

            # Split-line grocery format where quantity precedes price: '2 x 14.75' or '2 @ 14.75'
            qty_price_match = _QUANTITY_LINE_WITH_PRICE_RE.match(text)
            if qty_price_match:
                q_digits = qty_price_match.group(1).replace("I", "1").replace("l", "1")
                q_val = max(1, int(q_digits))
                u_val = float(qty_price_match.group(2).replace(",", "."))
                pending_qty_prefix = (q_val, u_val)
                continue

            _flush_pending()
            item = _build_item(remainder, money[-1], line_text=text, all_money=money)
            if item is not None:
                if pending_qty_prefix is not None:
                    q_val, u_val = pending_qty_prefix
                    pending_qty_prefix = None
                    if abs(q_val * u_val - item.price) <= 0.05:
                        item.quantity = q_val
                        item.price_is_line_total = True
                    elif abs(u_val - item.price) <= 0.05:
                        item.quantity = q_val

                if not item.name or not looks_like_item_name(item.name) or item.name in ("V", "Y", "T", "E", "Z"):
                    if buffered_name is not None:
                        c_name = _clean_name(buffered_name)
                        if c_name:
                            item.name = c_name
                        buffered_name = None

                if looks_like_item_name(item.name):
                    _flush_buffered_to_desc()
                    items.append(item)
                    seen_item = True
                else:
                    pending = item
            continue

        if pending is not None:
            name = _clean_name(text)
            if name:
                pending.name = name
            _flush_pending()
            continue

        if _is_header_or_metadata_line(text, seen_item):
            continue

        if not seen_item:
            buffered_name = text
            continue

        _flush_buffered_to_desc()
        buffered_name = text

        if len(items) >= MAX_ITEMS:
            break

    _flush_pending()
    _flush_buffered_to_desc()
    return items


_QUANTITY_MARKER_RE = re.compile(r"^\s*[0-9Il]{1,3}\s*[xX×@*]\s*$")
_QUANTITY_DIGITS_RE = re.compile(r"^\s*(\d{1,3})\s*[xX×@*]\s*$")
_QUANTITY_LINE_WITH_PRICE_RE = re.compile(
    r"^\s*([0-9Il]{1,3})\s*[xX×@*]\s*(\d{1,3}(?:[,.]\d{2})?)\s*$"
)


def _is_quantity_line(remainder: str) -> bool:
    """Is this money-line remainder a bare '1 ×' or '1 @' quantity marker?"""
    return bool(_QUANTITY_MARKER_RE.match(remainder))


def _quantity_of(remainder: str) -> int | None:
    cleaned = remainder.replace("I", "1").replace("l", "1")
    match = _QUANTITY_DIGITS_RE.match(cleaned)
    if match:
        return max(1, int(match.group(1)))
    return 1


def _build_item(
    remainder: str,
    money_token: str,
    line_text: str = "",
    all_money: list[str] | None = None,
) -> ExtractedItem | None:
    """Build an item from a line with its money words already removed."""
    price = normalize_money(money_token)
    if price is None or price <= 0:
        return None

    quantity = 1
    inferred_qty: int | None = None
    if all_money and len(all_money) >= 2:
        prev_price = normalize_money(all_money[-2])
        if prev_price and prev_price > 0 and price >= prev_price:
            ratio = price / prev_price
            rounded_ratio = round(ratio)
            if 1 <= rounded_ratio <= 999 and abs(rounded_ratio * prev_price - price) <= 0.05:
                inferred_qty = int(rounded_ratio)

    stripped_rem = _TRAILING_TAX_FLAG_RE.sub("", remainder).strip()
    if stripped_rem and any(ch.isalpha() for ch in stripped_rem):
        remainder = stripped_rem

    unit_mult_match = re.search(
        r"^\s*(\d{1,3})\s*[xX×@*]\s*(?:[pP$€£¥]?\d+(?:[,.]\d{2,3})?(?:\s*/\s*\w+)?)\s*$",
        remainder,
    )
    if unit_mult_match:
        quantity = max(1, int(unit_mult_match.group(1)))
        remainder = ""
    else:
        remainder = _INLINE_UNIT_PRICE_RE.sub("", remainder).strip()

        match = _LEADING_QUANTITY_RE.match(remainder)
        if match:
            quantity = max(1, int(match.group(1)))
            remainder = remainder[match.end():]
        else:
            trailing = _TRAILING_QUANTITY_RE.search(remainder)
            if trailing:
                digits = trailing.group(1) or trailing.group(2)
                quantity = max(1, int(digits))
                remainder = remainder[: trailing.start()]
            elif inferred_qty is not None:
                quantity = inferred_qty

    price_is_line_total = bool(inferred_qty is not None and inferred_qty == quantity)

    name = _clean_name(remainder)
    if not name and inferred_qty is not None:
        name = "V"

    return ExtractedItem(
        name=name,
        quantity=quantity,
        price=price,
        line_text=line_text or remainder,
        price_is_line_total=price_is_line_total,
    )


def _clean_name(text: str) -> str:
    cleaned = re.sub(r"[|_]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-*\\/@")
    cleaned = _UPC_PREFIX_RE.sub("", cleaned).strip(" .,:;-*\\/@")
    cleaned = _INLINE_UNIT_PRICE_RE.sub("", cleaned).strip(" .,:;-*\\/@")
    stripped_flag = _TRAILING_TAX_FLAG_RE.sub("", cleaned).strip(" .,:;-*\\/@")
    if stripped_flag and any(ch.isalpha() for ch in stripped_flag):
        cleaned = stripped_flag
    if not any(ch.isalpha() for ch in cleaned):
        return ""
    return cleaned[:MAX_NAME_LENGTH]


_ITEM_NAME_BLACKLIST = frozenset({
    "AMT", "TAX", "SUB", "TOT", "QTY", "POS", "CHK", "TAB", "BAL", "FOR", "AND", "THE",
})


def looks_like_item_name(text: str) -> bool:
    """Is this a plausible product name?"""
    stripped = text.strip()
    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters < 3:
        return False
    if letters < len(stripped) * 0.35:
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z.'&/-]*", stripped)
    solid = [token for token in tokens if len(token) >= 3]
    if not solid:
        return False

    if len(solid) == 1 and len(solid[0]) == 3:
        if len(tokens) > 2 or solid[0].upper() in _ITEM_NAME_BLACKLIST:
            return False

    multi_odd = sum(
        1 for token in solid if len(re.findall(r"[a-z][A-Z]", token)) >= 2
    )
    return multi_odd == 0


def _plausible_description(text: str) -> bool:
    letters = sum(1 for ch in text if ch.isalpha())
    return letters >= 3 and letters >= len(text) * 0.35


def looks_like_store_code(name: str) -> bool:
    """Is this item name a grocery-POS store code rather than a product name?

    Codes like 'DRD PEV-CUT EP025G' or 'JNJ MRCHPSN / CHS24G' are the printed
    identity of a shelf item: uppercase fragments mixed with digits, dashes and
    slashes. They cannot be "corrected" into a real product name by any model,
    so they are excluded from LLM batch normalization - the call would cost
    ~80s on CPU per job and can only mangle them further.
    """
    if not name or not name.isupper():
        return False
    # A mixed alphanumeric token (EP025G, CHS24G, PUF69G, 5300G) marks the
    # store-code style. Real product names rarely carry such tokens; even where
    # they do ('CHICKEN 3PCS'), the LLM cannot improve on what was printed.
    return any(
        re.search(r"[A-Z0-9]*\d[A-Z0-9]*[A-Z][A-Z0-9]*", token)
        for token in name.split()
    )


def reconcile_items(
    items: list[ExtractedItem],
    targets: dict[str, float | None],
    tolerance_base: float = 0.02,
) -> ItemScan:
    """Decide whether the item rows add up, and on which price basis.

    ``targets`` holds the candidate subtotals - ``net_sales`` on a tax-exclusive
    receipt, ``total_sales`` on a VAT-inclusive one. Either is a legitimate match
    depending on the document, so both are tried.
    """
    scan = ItemScan(items=items)
    if not items:
        scan.notes.append("no item rows found")
        return scan

    tolerance = max(tolerance_base, 0.01 * len(items))
    line_total = round(sum(item.price for item in items), 2)
    unit_total = round(
        sum(
            item.price if item.price_is_line_total else item.price * item.quantity
            for item in items
        ),
        2,
    )

    candidates = {
        name: value for name, value in targets.items() if value is not None
    }
    if not candidates:
        scan.price_basis = "none"
        scan.total = line_total
        scan.notes.append("no subtotal available to check items against")
        if len(items) >= 3:
            scan.near_reconciled = True
            scan.notes.append(f"{len(items)} items emitted without total target -> needs manual review")
        return scan

    # Prefer the line-total reading: it needs no assumption about the receipt's
    # convention, and for single-quantity rows the two are identical anyway.
    for basis, total in (("line_total", line_total), ("unit_price", unit_total)):
        for name, target in candidates.items():
            if abs(total - target) <= tolerance:
                scan.reconciled = True
                scan.price_basis = basis
                scan.total = total
                scan.target = target
                scan.notes.append(
                    f"items reconcile on {basis}: {total:.2f} vs {name}={target:.2f}"
                )
                return scan

    # Near-reconciliation fallback (>= 3 items within 3% discrepancy)
    if len(items) >= 3:
        for basis, total in (("line_total", line_total), ("unit_price", unit_total)):
            for name, target in candidates.items():
                if target is not None and target > 0 and abs(total - target) / target <= 0.03:
                    scan.reconciled = False
                    scan.near_reconciled = True
                    scan.price_basis = basis
                    scan.total = total
                    scan.target = target
                    diff_pct = abs(total - target) / target * 100.0
                    scan.notes.append(
                        f"items near-reconciled on {basis} ({diff_pct:.1f}% discrepancy): "
                        f"{total:.2f} vs {name}={target:.2f} -> emitted with manual review flag"
                    )
                    return scan

    scan.price_basis = "none"
    scan.total = line_total
    scan.notes.append(
        f"items do not reconcile: line_total={line_total:.2f}, "
        f"unit_total={unit_total:.2f}, targets="
        + ", ".join(f"{k}={v:.2f}" for k, v in candidates.items())
        + " -> withheld"
    )
    return scan
