"""Geometric label-to-value pairing.

Money on a receipt is positional: a label sits on the left, its figure sits on the
same visual row to the right. Pairing them with word bounding boxes is reliable
and cheap, and it removes any need to ask a 1.5B model to transcribe digits -
which is where fabricated totals came from.

Pairing works on pixel coordinates rather than OCR line indices. Sparse-text page
segmentation fragments a single visual row into several reported "lines", so index
adjacency pairs a label with the *next* label's value. Vertical overlap survives
that fragmentation; line numbering does not.

Every value carries the label it was paired with and the raw token it came from,
so verification can trace it back to the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.ocr_engine import Word

# Amounts must carry a decimal point. Requiring one rejects the identifiers,
# quantities and OCR debris that bare-integer matching swept up: '0015' from a form
# grid, '306.' from a truncated total, '640' from noise.
#
# One decimal place is allowed as well as two. Plenty of POS systems print '$15.8'
# rather than '$15.80', and demanding two digits meant such a receipt appeared to
# contain no money at all - no items, no subtotal, no total.
MONEY_STRICT_RE = re.compile(
    r"^[^\d]{0,3}(\d{1,3}(?:[,.\s]\d{3})*[.,]\d{1,2})[^\d]{0,3}$"
)
MONEY_FRAGMENT_RE = re.compile(r"^[.,]\d{1,2}[^\d]{0,2}$")
INTEGER_PART_RE = re.compile(r"^[^\d]{0,3}\d{1,7}$")

# Longest match wins, so 'total amount due' beats 'total' and 'subtotal' is never
# mistaken for the grand total.
LABELS: dict[str, tuple[str, ...]] = {
    "total_amount": (
        "total amount due", "total amt due", "amount due", "amt due", "total due",
        "grand total", "balance due", "total payable", "net amount due", "balance",
        # Bare 'total' last: longest-match ordering means 'total sales' and
        # 'total amount due' still win, and the alphabetic-boundary check keeps
        # it from firing inside 'subtotal'. Without it a POS receipt printing a
        # plain 'Total: 3.70' yields no total at all.
        "total",
    ),
    "total_sales": (
        "total sales (vat inclusive)", "vatable sales (vat inclusive)",
        "total (vat inclusive)", "total sales", "gross amount", "gross sales",
    ),
    "net_sales": (
        "amount net of vat", "vatable sales", "net of vat", "vat exclusive",
        "subtotal", "sub total", "sub-total", "taxable amount",
    ),
    "tax_amount": (
        "less: vat", "value added tax", "vat amount", "12% vat", "vat 12%",
        "add: vat", "sales tax", "output vat", "vat", "gst", "sst", "tax",
    ),
    "service_charge": ("service charge", "svc charge", "svc chg", "service chg"),
    "discount_amount": (
        "less: senior citizen discount", "less: sc discount", "senior citizen discount",
        "senior citizen disc", "senior discount", "senior disc", "20% sc discount",
        "20% sc disc", "sc discount", "sc disc", "less: sc", "osca discount",
        "less: pwd discount", "pwd discount", "pwd disc", "20% pwd discount",
        "20% pwd disc", "less: pwd",
        "less: student discount", "student discount", "student disc",
        "less: employee discount", "employee discount", "employee disc", "staff discount",
        "less: promo discount", "promo discount", "promo disc", "voucher discount",
        "coupon discount", "member discount", "loyalty discount", "card savings",
        "card saver", "annivdiscount", "anniv discount",
        "less: discount", "less discount", "total discount", "discount", "disc",
    ),
    "withholding_tax": ("less: withholding tax", "withholding tax", "ewt"),
}

# Lines matching these never yield money values - print-shop and regulatory
# footers whose numbers look like amounts but are not.
NOISE_LINE_PATTERNS = (
    r"authority\s+to\s+print", r"\bocn\b", r"permit\s+no", r"accreditation",
    r"booklets", r"valid\s+until", r"date\s+of\s+atp", r"printers?\b",
    r"bkl?ts", r"\batp\b", r"vat\s*reg", r"reference\s*#", r"card\s+number",
)
_NOISE_RE = re.compile("|".join(NOISE_LINE_PATTERNS), re.IGNORECASE)


@dataclass
class MoneyToken:
    text: str
    word: Word
    words: list[Word] = field(default_factory=list)
    """Every word that formed this token.

    Usually one, but two when OCR splits an amount ('$29' + ',99'). Callers that
    need the rest of the line - item-name extraction - must exclude all of them,
    since the reassembled token string no longer matches the raw text.
    """

    def __post_init__(self) -> None:
        if not self.words:
            self.words = [self.word]

    @property
    def center_y(self) -> float:
        return self.word.center_y

    @property
    def left(self) -> int:
        return self.word.left


@dataclass
class LabeledAmount:
    field_name: str
    label: str
    raw_token: str
    line_text: str
    confidence: float
    pairing: str = "same_row"

    @property
    def evidence(self) -> str:
        return f"{self.label!r} -> {self.raw_token!r} [{self.pairing}] on {self.line_text!r}"


@dataclass
class LayoutScan:
    amounts: dict[str, list[LabeledAmount]] = field(default_factory=dict)
    money_tokens: list[str] = field(default_factory=list)

    @property
    def label_count(self) -> int:
        return len(self.amounts)

    def first(self, field_name: str) -> LabeledAmount | None:
        candidates = self.amounts.get(field_name) or []
        return candidates[0] if candidates else None

    def all(self, field_name: str) -> list[LabeledAmount]:
        return list(self.amounts.get(field_name) or [])


def scan_layout(lines: list[list[Word]]) -> LayoutScan:
    """Pair money labels with their figures using pixel geometry."""
    scan = LayoutScan()
    tokens = _collect_money_tokens(lines)
    scan.money_tokens = [token.text for token in tokens]

    for line in lines:
        line_text = " ".join(word.text for word in line)
        if _NOISE_RE.search(line_text):
            continue

        match = _best_label(line_text)
        if match is None:
            continue
        field_name, label = match

        anchor = line[0]
        line_height = max(anchor.height, 8)
        label_right = max(word.right for word in line)

        chosen, pairing = _pair_value(tokens, line, label_right, line_height)
        if chosen is None:
            continue

        scan.amounts.setdefault(field_name, []).append(
            LabeledAmount(
                field_name=field_name,
                label=label,
                raw_token=chosen.text,
                line_text=line_text,
                confidence=chosen.word.confidence,
                pairing=pairing,
            )
        )

    return scan


def _pair_value(
    tokens: list[MoneyToken], line: list[Word], label_right: int, line_height: int
) -> tuple[MoneyToken | None, str]:
    """Prefer a figure on the same visual row; fall back to the row below."""
    row_center = sum(word.center_y for word in line) / len(line)
    line_lefts = {id(word) for word in line}

    same_row = [
        token
        for token in tokens
        if abs(token.center_y - row_center) <= line_height * 0.6
        and (token.left >= label_right - line_height or id(token.word) in line_lefts)
    ]
    if same_row:
        return max(same_row, key=lambda t: t.left), "same_row"

    below = [
        token
        for token in tokens
        if 0 < token.center_y - row_center <= line_height * 2.5
    ]
    if below:
        return max(below, key=lambda t: t.left), "row_below"

    return None, "unpaired"


def _collect_money_tokens(lines: list[list[Word]]) -> list[MoneyToken]:
    """Find money tokens, stitching together ones OCR split across two words.

    Tesseract renders receipt 1's ``$29.99`` as the two words ``$29`` and ``,99``.
    Read separately they are junk; joined they are the line item price.
    """
    tokens: list[MoneyToken] = []
    for line in lines:
        index = 0
        while index < len(line):
            word = line[index]
            text = word.text.strip()

            if index + 1 < len(line) and MONEY_FRAGMENT_RE.match(line[index + 1].text.strip()):
                if INTEGER_PART_RE.match(text):
                    merged = f"{text}{line[index + 1].text.strip()}"
                    if _plausible_amount(merged):
                        tokens.append(MoneyToken(
                            text=merged, word=word, words=[word, line[index + 1]]
                        ))
                    index += 2
                    continue

            if MONEY_STRICT_RE.match(text) and _plausible_amount(text):
                tokens.append(MoneyToken(text=text, word=word, words=[word]))
            index += 1
    return tokens


def _plausible_amount(token: str) -> bool:
    # Two digits is the floor, so a single-decimal amount like '$1.8' still counts.
    digits = re.sub(r"\D", "", token)
    return 2 <= len(digits) <= 9


def _best_label(line_text: str) -> tuple[str, str] | None:
    """Longest keyword match on the line."""
    lowered = line_text.casefold()
    # Reject bare 'total' match if line looks like a product item line (e.g. 'COLGATE TOTAL WHTNING')
    is_item_product_line = bool(
        re.search(r"\b(colgate|tpst|sauce|chips|brush|shampoo|soap|pkg|oz|fl|ea|pcs|qty|price)\b", lowered)
    )
    best: tuple[str, str] | None = None
    for field_name, keywords in LABELS.items():
        for keyword in keywords:
            if keyword == "total" and is_item_product_line:
                continue
            if keyword == "disc":
                # Guard against product names like 'BRAKE DISC' or 'SANDING DISC'
                # Only recognize as discount when line has discount cues or explicit punctuation/minus
                has_disc_cue = bool(
                    re.search(
                        r"\b(less|%|sc|pwd|osca|snr|senior|student|promo|member|total)\b",
                        lowered,
                    )
                    or ":" in lowered
                    or "disc." in lowered
                    or "-" in lowered
                    or lowered.strip().startswith("disc")
                )
                if not has_disc_cue:
                    continue
            position = lowered.find(keyword)
            if position == -1:
                continue
            end = position + len(keyword)
            if end < len(lowered) and lowered[end].isalpha():
                continue
            if position > 0 and lowered[position - 1].isalpha():
                continue
            if best is None or len(keyword) > len(best[1]):
                best = (field_name, keyword)
    return best


_DISCOUNT_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("senior", re.compile(r"\b(senior\s*citizen|senior|osca|sc\s*disc(?:ount)?|less\s*:?\s*sc)\b", re.IGNORECASE)),
    ("pwd", re.compile(r"\b(pwd|person\s+with\s+disabilit\w*)\b", re.IGNORECASE)),
    ("student", re.compile(r"\b(student)\b", re.IGNORECASE)),
    ("employee", re.compile(r"\b(employee|staff|associate)\b", re.IGNORECASE)),
    ("promo", re.compile(r"\b(promo|promotion|voucher|coupon|member|loyalty|card\s+sav\w*|anniv\w*|bundle|mfrcpn|cartwheel)\b", re.IGNORECASE)),
)


def classify_discount_type(
    discount_amount: float | None,
    discount_lines: list[str] | None = None,
    full_text: str = "",
) -> str | None:
    """Classify the discount category from matched discount rows or full text.

    Returns one of: ``senior``, ``pwd``, ``student``, ``promo``, ``employee``,
    ``other``, or ``None`` when no positive discount is present.
    """
    if discount_amount is None or discount_amount <= 0:
        return None

    for line in (discount_lines or []):
        for dtype, pattern in _DISCOUNT_TYPE_PATTERNS:
            if pattern.search(line):
                return dtype

    if full_text:
        for dtype, pattern in _DISCOUNT_TYPE_PATTERNS:
            if pattern.search(full_text):
                return dtype

    return "other"

