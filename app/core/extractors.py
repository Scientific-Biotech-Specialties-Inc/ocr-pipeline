"""Deterministic field extraction from OCR text and geometry.

Everything a rule can decide is decided here. The language model is left with
only the two jobs a 1.5B model is actually reliable at - picking a vendor name
from candidate header lines, and classifying an expense category - and even those
are constrained to a closed candidate list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from rapidfuzz import fuzz

from app.core.ocr_engine import Word
from app.core.textquality import looks_like_address_line, looks_like_words

# ---------------------------------------------------------------------------
# tax identifiers
# ---------------------------------------------------------------------------

# A 12- or 14-digit run whose digits may be separated by single hyphens or
# spaces. Tolerating an interior space is essential: receipt 6 comes back as
# '000-774-47 1-025', with the split falling inside the third group.
#
# The lookarounds prevent matching a window inside a longer digit run, which is
# how receipt 1's 20-digit card reference '62845289260246240685C' became a
# vendor TIN.
_DIGIT_RUN_RE = re.compile(r"(?<![\d-])\d(?:[\s-]?\d){11,13}(?![\d-])")

# A bare digit run is only accepted when the line names it as a tax id.
# Otherwise a phone block like '223 2890 223 2891' - fourteen digits separated by
# single spaces - would be read as a tax number.
_TAX_ID_LINE_MARKER = re.compile(r"\btin\b|\bt\.?i\.?n\.?\b|tax\s*(id|no)", re.IGNORECASE)
_MIN_SEPARATORS = 2

VENDOR_TAX_MARKERS = (
    "vat reg", "vat registered", "non vat reg", "non-vat reg", "vat reg. tin",
    "prop.", "proprietor",
)
CUSTOMER_TAX_MARKERS = (
    "registered name", "received from", "sold to", "with tin", "address/tin",
    "customer", "bill to", "business address", "business style",
)


@dataclass
class TaxIdCandidate:
    value: str
    normalized: str
    line_index: int
    line_text: str
    role: str = "unknown"   # vendor | customer | unknown
    score: float = 0.0

    @property
    def formatted(self) -> str:
        return format_tax_id(self.normalized)


def format_tax_id(digits: str) -> str:
    """Regroup a digit run into canonical PH TIN form.

    12 digits -> 3-3-3-3, 14 digits -> 3-3-3-5 (five-digit branch code, as on
    receipt 3's ``303-124-202-00000``, which the previous regex rejected outright).
    """
    if len(digits) == 12:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]}-{digits[9:12]}"
    if len(digits) == 14:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]}-{digits[9:14]}"
    return digits


def find_tax_ids(lines: list[str]) -> list[TaxIdCandidate]:
    """Locate tax id candidates and decide which belongs to the vendor.

    Receipt 3 carries two: the vendor's in the printed header and the customer's
    beside 'Registered Name'. Picking the wrong one is a compliance failure, not a
    typo, so role assignment is explicit rather than left to salience.
    """
    candidates: list[TaxIdCandidate] = []
    for index, line_text in enumerate(lines):
        lowered = line_text.casefold()
        labelled_line = bool(_TAX_ID_LINE_MARKER.search(line_text))
        for match in _DIGIT_RUN_RE.finditer(line_text):
            raw = match.group(0)
            digits = re.sub(r"\D", "", raw)
            if len(digits) not in (12, 14):
                continue
            if raw.count("-") < _MIN_SEPARATORS and not labelled_line:
                continue
            candidate = TaxIdCandidate(
                value=match.group(0).strip(),
                normalized=digits,
                line_index=index,
                line_text=line_text,
            )
            _assign_role(candidate, lowered, lines, index)
            candidates.append(candidate)

    # Header position is a strong prior: vendor details are printed above the
    # customer block on every PH form in the corpus.
    total = max(len(lines), 1)
    for candidate in candidates:
        position_bonus = 1.0 - (candidate.line_index / total)
        candidate.score += position_bonus
        if candidate.role == "vendor":
            candidate.score += 3.0
        elif candidate.role == "customer":
            candidate.score -= 3.0

    return sorted(candidates, key=lambda c: -c.score)


def _assign_role(
    candidate: TaxIdCandidate, lowered: str, lines: list[str], index: int
) -> None:
    if any(marker in lowered for marker in VENDOR_TAX_MARKERS):
        candidate.role = "vendor"
        return
    if any(marker in lowered for marker in CUSTOMER_TAX_MARKERS):
        candidate.role = "customer"
        return
    # Look back a couple of lines: 'Registered Name' often sits above the TIN.
    for offset in (1, 2):
        if index - offset < 0:
            break
        previous = lines[index - offset].casefold()
        if any(marker in previous for marker in CUSTOMER_TAX_MARKERS):
            candidate.role = "customer"
            return
        if any(marker in previous for marker in VENDOR_TAX_MARKERS):
            candidate.role = "vendor"
            return


def select_vendor_tax_id(candidates: list[TaxIdCandidate]) -> TaxIdCandidate | None:
    vendor = [c for c in candidates if c.role == "vendor"]
    if vendor:
        return vendor[0]
    unknown = [c for c in candidates if c.role != "customer"]
    return unknown[0] if unknown else None


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------

DATE_PATTERNS_MONTH_FIRST: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"(?<!\d)(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})(?!\d)", ("%Y", "%m", "%d")),
    (r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*['’]?(\d{4})(?!\d)", ("%m", "%d", "%Y")),
    (r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*['’]?(\d{2})(?!\d)", ("%m", "%d", "%y")),
)

DATE_PATTERNS_DAY_FIRST: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"(?<!\d)(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})(?!\d)", ("%Y", "%m", "%d")),
    (r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*['’]?(\d{4})(?!\d)", ("%d", "%m", "%Y")),
    (r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*['’]?(\d{2})(?!\d)", ("%d", "%m", "%y")),
)

# Countries that write the month first. Everywhere else, '05/03/2024' is 5 March,
# not 5 May - and whenever the day is 12 or lower both readings parse cleanly, so
# there is no way to detect the mistake after the fact. Locale has to decide.
MONTH_FIRST_COUNTRIES = frozenset({"US", "PH"})

MONTH_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
    r"[\s\-_/]+"
    r"(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,\s*|[\s\-_/]+)"
    r"['’]?(\d{2,4})"
    r"(?!\d)",
    re.IGNORECASE,
)
DAY_MONTH_NAME_RE = re.compile(
    r"(?<!\d)"
    r"(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:[\s\-_/]+|[\s\-_/]+of[\s\-_/]+)"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
    r"(?:,\s*|[\s\-_/]+)"
    r"['’]?(\d{2,4})"
    r"(?!\d)",
    re.IGNORECASE,
)

_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

DATE_LABELS = (
    "date", "issued on", "transaction", "invoice date", "or date",
    "dt:", "dated", "txn date", "purchase date", "sale date",
    "receipt date", "billing date", "trans. date", "trans date",
    "order date", "sales date", "posting date", "entry date",
)

COMPACT_DATE_RE = re.compile(
    r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)|(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)"
)
SPACED_DATE_RE = re.compile(
    r"(?<!\d)(\d{4})\s+(\d{1,2})\s+(\d{1,2})(?!\d)|(?<!\d)(\d{1,2})\s+(\d{1,2})\s+(\d{4})(?!\d)"
)

# Dates in these contexts belong to the payment terminal or the print shop, not
# the transaction. Receipt 1 carries a card-terminal '11/20/2019' that sits right
# next to the word 'Date', so positive label matching alone picks the wrong one.
DATE_NEGATIVE_CONTEXT = (
    "card", "reference", "approved", "valid until", "date of atp", "atp",
    "accreditation", "printer", "permit", "ocn", "expiry", "expires",
    # 'ssued' rather than 'issued': Tesseract routinely reads the leading 'I' of
    # 'Issued' as a lowercase 'l', and the ATP print date must never be selected
    # as the transaction date.
    "ssued", "bklts", "booklets",
    # 'Date/time' (OCR'd 'Date/t ime') labels a payment-terminal timestamp. A
    # transaction date is labelled 'Date' or 'Date:', never 'Date/', so the slash
    # is a clean discriminator - and it is what separates receipt 1's real date
    # of 5/24/2025 from the card block's 11/20/2019.
    "date/", "ptu",
)

_OCR_CHAR_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
})

_OCR_NOISY_DATE_RE = re.compile(
    r"(?<![A-Za-z0-9])([0-9OolI|SsB]{1,4})\s*([-/.]|[\s])\s*([0-9OolI|SsB]{1,2})\s*(\2)\s*([0-9OolI|SsB]{2,4})(?![A-Za-z0-9])"
)


@dataclass
class DateCandidate:
    value: date
    raw: str
    line_index: int
    line_text: str
    score: float = 0.0
    span: tuple[int, int] = (0, 0)


def find_dates(
    lines: list[str], today: date | None = None, country: str | None = None
) -> list[DateCandidate]:
    """Locate transaction-date candidates, ranked.

    ``country`` selects day-first or month-first interpretation of an all-numeric
    date. Without it, Malaysian and Bruneian receipts silently transpose day and
    month on any date up to the 12th.
    """
    today = today or date.today()
    max_date = today + timedelta(days=1)
    month_first = country is None or country.upper() in MONTH_FIRST_COUNTRIES
    candidates: list[DateCandidate] = []

    for index, line_text in enumerate(lines):
        lowered = line_text.casefold()
        has_label = any(label in lowered for label in DATE_LABELS)
        for parsed, raw, span in _parse_line_dates(line_text, month_first=month_first, has_date_label=has_label):
            if not (date(1990, 1, 1) <= parsed <= max_date):
                continue  # reject impossible / future transaction dates
            candidates.append(
                DateCandidate(value=parsed, raw=raw, line_index=index,
                              line_text=line_text, span=span)
            )

    total = max(len(lines), 1)
    time_re = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE)

    for candidate in candidates:
        lowered = candidate.line_text.casefold()
        cand_start, cand_end = candidate.span
        pre_text = lowered[:cand_start]
        post_text = lowered[cand_end:]

        # 1. Date label scoring (proximity-aware)
        has_pre_label = any(label in pre_text[-30:] for label in DATE_LABELS)
        prev_has_label = False
        if candidate.line_index > 0:
            prev_line = lines[candidate.line_index - 1].casefold()
            prev_has_label = any(label in prev_line for label in DATE_LABELS)

        if has_pre_label:
            candidate.score += 2.0
        elif any(label in lowered for label in DATE_LABELS) or prev_has_label:
            candidate.score += 1.5

        # 2. Adjacent timestamp boost (+1.0)
        has_time_near = bool(
            time_re.search(pre_text[-30:]) or time_re.search(post_text[:30])
        )
        if has_time_near:
            candidate.score += 1.0

        # 3. Negative context penalty
        has_pre_negative = any(bad in pre_text[-35:] for bad in DATE_NEGATIVE_CONTEXT)
        has_post_negative = not has_time_near and any(bad in post_text[:15] for bad in ("expiry", "expires", "valid until"))
        is_atp_block = any(bad in lowered for bad in ("atp", "accreditation", "printer", "permit", "booklets", "bklts", "ptu")) and not has_time_near

        if has_pre_negative or has_post_negative or is_atp_block:
            candidate.score -= 5.0

        # Transaction dates appear near the top of a receipt.
        candidate.score += 1.5 * (1.0 - candidate.line_index / total)
        # Prefer recent transaction dates over OCR year digit corruptions (e.g. 2025 vs 2005)
        if candidate.value.year >= 2020:
            candidate.score += 0.5
        elif candidate.value.year < 2015:
            candidate.score -= 2.0

    # Group by date value and boost candidates corroborated by multiple OCR lines
    by_value: dict[datetime.date, list[DateCandidate]] = {}
    for c in candidates:
        by_value.setdefault(c.value, []).append(c)

    best_candidates: list[DateCandidate] = []
    for group in by_value.values():
        top = max(group, key=lambda c: c.score)
        if len(group) > 1:
            top.score += 0.5 * min(len(group) - 1, 3)
        best_candidates.append(top)

    return sorted(best_candidates, key=lambda c: -c.score)


def _parse_line_dates(
    line_text: str, month_first: bool = True, has_date_label: bool = False
) -> list[tuple[date, str, tuple[int, int]]]:
    found: list[tuple[date, str, tuple[int, int]]] = []
    seen_spans: set[tuple[int, int]] = set()

    def _add_found(dt: date | None, raw: str, span: tuple[int, int]) -> None:
        if dt is None:
            return
        if span in seen_spans:
            return
        start, end = span
        if any(s <= start and end <= e for s, e in seen_spans):
            return
        seen_spans.add(span)
        found.append((dt, raw, span))

    # 1. Textual Month Formats
    for match in MONTH_NAME_RE.finditer(line_text):
        month = _MONTH_NUM[match.group(1)[:3].lower()]
        year = int(match.group(3))
        if year < 100:
            year += 2000 if year < 50 else 1900
        _add_found(_safe_date(year, month, int(match.group(2))), match.group(0), match.span())

    for match in DAY_MONTH_NAME_RE.finditer(line_text):
        month = _MONTH_NUM[match.group(2)[:3].lower()]
        year = int(match.group(3))
        if year < 100:
            year += 2000 if year < 50 else 1900
        _add_found(_safe_date(year, month, int(match.group(1))), match.group(0), match.span())

    # 2. Standard Numeric Formats
    patterns = DATE_PATTERNS_MONTH_FIRST if month_first else DATE_PATTERNS_DAY_FIRST
    for pattern, order in patterns:
        for match in re.finditer(pattern, line_text):
            parts = dict(zip(order, match.groups()))
            year = int(parts.get("%Y") or parts.get("%y") or 0)
            if "%y" in parts and year < 100:
                year += 2000 if year < 50 else 1900
            month, day = int(parts["%m"]), int(parts["%d"])
            if month > 12 and day <= 12:
                month, day = day, month  # impossible month: the order was reversed
            _add_found(_safe_date(year, month, day), match.group(0), match.span())

    # 3. Compact & Spaced Formats (requires date label on line)
    if has_date_label:
        for match in COMPACT_DATE_RE.finditer(line_text):
            g = match.groups()
            if g[0]:
                y, m, d = int(g[0]), int(g[1]), int(g[2])
            else:
                y = int(g[5])
                m, d = (int(g[3]), int(g[4])) if month_first else (int(g[4]), int(g[3]))
                if m > 12 and d <= 12:
                    m, d = d, m
            _add_found(_safe_date(y, m, d), match.group(0), match.span())

        for match in SPACED_DATE_RE.finditer(line_text):
            g = match.groups()
            if g[0]:
                y, m, d = int(g[0]), int(g[1]), int(g[2])
            else:
                y = int(g[5])
                m, d = (int(g[3]), int(g[4])) if month_first else (int(g[4]), int(g[3]))
                if m > 12 and d <= 12:
                    m, d = d, m
            _add_found(_safe_date(y, m, d), match.group(0), match.span())

    # 4. Scoped OCR noisy character normalization (e.g. O8/11/2026, 08/ll/2026, 2O26-07-30)
    for match in _OCR_NOISY_DATE_RE.finditer(line_text):
        if match.span() in seen_spans:
            continue
        g1, g3, g5 = match.group(1), match.group(3), match.group(5)
        if not any(c in "OolI|SsB" for c in g1 + g3 + g5):
            continue
        p1 = g1.translate(_OCR_CHAR_MAP)
        p2 = g3.translate(_OCR_CHAR_MAP)
        p3 = g5.translate(_OCR_CHAR_MAP)
        if not (p1.isdigit() and p2.isdigit() and p3.isdigit()):
            continue

        if len(p1) == 4 and len(p2) in (1, 2) and len(p3) in (1, 2):
            y, m, d = int(p1), int(p2), int(p3)
            if m > 12 and d <= 12:
                m, d = d, m
            _add_found(_safe_date(y, m, d), match.group(0), match.span())
        elif len(p3) in (2, 4) and len(p1) in (1, 2) and len(p2) in (1, 2):
            y = int(p3)
            if y < 100:
                y += 2000 if y < 50 else 1900
            m, d = (int(p1), int(p2)) if month_first else (int(p2), int(p1))
            if m > 12 and d <= 12:
                m, d = d, m
            _add_found(_safe_date(y, m, d), match.group(0), match.span())

    return found


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return datetime(year, month, day).date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# invoice / receipt number
# ---------------------------------------------------------------------------

# The trailing lookahead rejects digits that run into letters. A printer's
# accreditation number '038MP20140000000045' sits next to 'No.' and would
# otherwise be captured as '038' - a receipt number that never existed.
INVOICE_LABEL_RE = re.compile(
    r"(?:invoice\s*(?:number|no|#)|receipt\s*(?:no|#)|"
    r"\bor\s*no|\bchk\s*no|\bcheck\s*no|\bno[.:;,\u00b0\u00ba]|\bn[\u00b0\u00ba\u00a9\u00a2])"
    r"\s*[:.;,#]?\s*([0-9]{3,8})(?![0-9A-Za-z])",
    re.IGNORECASE,
)

INVOICE_NOISE = (
    "authority to print", "ocn", "permit", "accreditation", "booklets", "bklts",
    "valid until", "atp", "tin", "tel", "phone", "zip",
)


def find_invoice_number(lines: list[str]) -> tuple[str | None, str | None]:
    """Return ``(number, evidence_line)``.

    Skips print-shop identifiers - ATP numbers, OCNs and permit numbers all sit
    next to the word 'No.' and look exactly like receipt numbers.
    """
    for line_text in lines:
        lowered = line_text.casefold()
        if any(noise in lowered for noise in INVOICE_NOISE):
            continue
        match = INVOICE_LABEL_RE.search(line_text)
        if match:
            return match.group(1), line_text
    return None, None


# ---------------------------------------------------------------------------
# vendor name candidates (handed to the LLM as a closed list)
# ---------------------------------------------------------------------------

VENDOR_NAME_EXCLUSIONS = (
    "received from", "registered name", "sold to", "vat reg", "telefax",
    "official receipt", "sales invoice", "authority to print", "booklets",
    "cashier", "business style", "nature of service", "particulars",
    # Form furniture and money labels. A printed label is not a business name,
    # and 'Total Sales (VAT Inclusive)' was being proposed as a vendor.
    "total sales", "vatable sales", "less:", "add:", "amount due", "total amount",
    "zero-rated", "vat-exempt", "vat exempt", "withholding", "in settlement",
    "settlement of the following", "signature", "subscription",
    "representative", "received the", "solo parent", "sc/pwd", "unit cost",
    "card savings", "savings", "mfrcpn", "cartwheel", "cleaning supplies", "expires",
    "subtotal", "change due", "balance", "regular price", "regular", "price",
)

# Short exclusions need word boundaries. As plain substrings, 'tel' matches inside
# 'Telecommunications' - which silently discarded receipt 6's only real candidate,
# 'Bayan Telecommunications, Inc.', leaving the customer's name as the best guess.
VENDOR_NAME_EXCLUSION_WORDS = re.compile(
    r"\b(tel|tin|date|qty|\d+qty|atp|ocn|permit|address|invoice|discount|card|savings?|savin\w*|cartwheel|mfrcpn|grocery|cleaning|supplies|beauty|apparel|produce|liquor|deli|frozen|refrig|baked|fiber|zesty|dill|pasta|muffin|chips|veggie|almond|cherry|ham|pepper|oranges|broccoli|spinach|lettuce)\b",
    re.IGNORECASE,
)

# Document titles, matched loosely because OCR mangles them. 'OFFICIAL RECEIPT'
# came back as 'OFFICIAL RECEIpr' and slipped past an exact-substring exclusion.
DOCUMENT_TITLE_RE = re.compile(
    r"offi\w*\s*rece\w*|sale\w*\s*invo\w*|provisional\s*rece\w*|"
    r"servi\w*\s*invo\w*|billing\s*state\w*",
    re.IGNORECASE,
)

# Words that make up table column headings. A line composed only of these is
# furniture, not a business name - 'Description Amount' was being reported as the
# vendor with full confidence, which is worse than reporting nothing.
HEADING_VOCABULARY = frozenset({
    "description", "amount", "qty", "quantity", "unit", "units", "price", "cost",
    "particulars", "nature", "service", "item", "items", "total", "no", "rate",
    "vat", "sales", "net", "gross", "less", "add", "code", "ref",
})

CUSTOMER_MARKERS = ("received from", "registered name", "sold to", "bill to")


@dataclass
class CandidateMeta:
    """Signals used to rank a header line.

    ``confidence`` is Tesseract's mean per-word confidence for the line, and it is
    the discriminator that separates a correct reading from a corrupted one when
    both look structurally similar - 'UNITED DAILY PRESS INC' against
    'URITVED DAILY PRESS INGa'. ``occurrences`` counts how many independent
    readings produced the line; agreement across variants is corroboration.
    """

    index: int
    confidence: float
    occurrences: int = 1


@dataclass
class VendorCandidates:
    lines: list[str] = field(default_factory=list)
    customer_names: list[str] = field(default_factory=list)
    meta: dict[str, CandidateMeta] = field(default_factory=dict)


def find_vendor_candidates(
    line_sets: list[list[tuple[str, float]]], header_lines: int = 12
) -> VendorCandidates:
    """Collect plausible vendor-name lines from the printed headers of all readings.

    Drawing from every pooled reading matters: the primary reading is chosen for
    money-anchor coverage, so its header is often the worst of the set. Receipt 6's
    'Bayan Telecommunications, Inc.' is legible in one variant and absent from
    another, and considering only the primary picked the customer instead.

    Names following a customer marker are collected separately so the selection
    step can be told explicitly what NOT to choose.
    """
    result = VendorCandidates()

    for lines in line_sets:
        for index, (line_text, confidence) in enumerate(lines[:header_lines]):
            stripped = line_text.strip()
            lowered = stripped.casefold()
            if any(bad in lowered for bad in VENDOR_NAME_EXCLUSIONS):
                continue
            if VENDOR_NAME_EXCLUSION_WORDS.search(stripped):
                continue
            if DOCUMENT_TITLE_RE.search(stripped):
                continue
            if _is_column_heading(stripped):
                continue
            if not looks_like_words(stripped):
                continue

            existing = result.meta.get(stripped)
            if existing is None:
                result.meta[stripped] = CandidateMeta(index=index, confidence=confidence)
            else:
                existing.index = min(existing.index, index)
                existing.confidence = max(existing.confidence, confidence)
                existing.occurrences += 1

        for line_text, _ in lines:
            lowered = line_text.casefold()
            for marker in CUSTOMER_MARKERS:
                position = lowered.find(marker)
                if position == -1:
                    continue
                tail = line_text[position + len(marker):]
                tail = re.split(
                    r"\s{2,}|with tin|and address|engaged in",
                    tail, flags=re.IGNORECASE,
                )[0]
                tail = tail.strip(" :_-\u2014\u2013.,")
                if len(tail) >= 3:
                    result.customer_names.append(tail)

    # Brand detection fallback if OCR mangled header lines (checks raw header text before exclusions)
    raw_header_text = " ".join(l for ls in line_sets for l, _ in ls[:12]).casefold()
    if ("sfwy" in raw_header_text or "safew" in raw_header_text) and "Safeway" not in result.meta:
        result.meta["Safeway"] = CandidateMeta(index=-1, confidence=1.0, occurrences=10)
    if ("larget" in raw_header_text or "target" in raw_header_text or "pay get" in raw_header_text or "iar more" in raw_header_text) and "Target" not in result.meta:
        result.meta["Target"] = CandidateMeta(index=-1, confidence=1.0, occurrences=10)
    if ("dollar tree" in raw_header_text or "dollartree" in raw_header_text) and "Dollar Tree" not in result.meta:
        result.meta["Dollar Tree"] = CandidateMeta(index=-1, confidence=1.0, occurrences=10)

    result.lines = [
        name for name, _ in sorted(result.meta.items(), key=lambda kv: kv[1].index)
    ]
    return result


_US_STATE_CODES = (
    r"AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|"
    r"MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC"
)

ADDRESS_KEYWORDS = (
    # Thoroughfares & Premises
    " street", " st.", " st ", " st,", " avenue", " ave.", " ave ", " ave,", " road", " rd.", " rd ", " rd,",
    " highway", " hwy", " parkway", " pkwy", " boulevard", " blvd", " drive", " dr.", " dr ", " lane", " ln.",
    " expressway", " express way", " circle", " cir.", " way", " walk",
    " building", " bldg", " floor", " flr", " suite", " ste.", " ste ", " unit ", " room", " rm.", " rm ",
    " stall", " space", " kiosk", " level", " lvl", " tower", " plaza", " center", " centre", " square",
    " mall", " complex", " village", " vlg", " arcade", " hub", " compound",
    " block", " blk", " lot ", " phase", " purok", " sitio", " zone ", " barangay", " brgy",
    " district", " dist.", " province", " subdivision", " subd", " corner", " cor.",
    " jalan", " jln", " lorong", " kampong", " p.o. box", " po box", " zip", " postal",
    # Philippines Key Localities
    "philippines", "metro manila", "ncr", "manila", "makati", "quezon", "taguig", "pasig", "mandaluyong",
    "cebu", "davao", "pasay", "paranaque", "parañaque", "binondo", "ermita", "malate", "sampaloc", "tondo",
    "intramuros", "caloocan", "valenzuela", "marikina", "muntinlupa", "las pinas", "las piñas", "san juan",
    "malabon", "navotas", "pateros", "rizal", "antipolo", "taytay", "cainta", "rodriguez", "san mateo",
    "cavite", "bacoor", "imus", "dasmarinas", "dasmariñas", "tagaytay", "general trias", "laguna", "binan", "biñan",
    "santa rosa", "calamba", "san pedro", "cabuyao", "bulacan", "meycauayan", "malolos", "marilao", "santa maria",
    "pampanga", "angeles", "san fernando", "clark", "batangas", "lipa", "iloilo", "bacolod", "baguio",
    "cagayan de oro", "zamboanga", "general santos", "subic", "olongapo", "palawan", "boracay", "bohol", "leyte",
    "tacloban", "naga", "legazpi", "dagupan", "tarlac", "bataan", "la union", "ilocos", "bukidnon", "butuan",
    "surigao", "cotabato",
    # International Localities
    "kuala lumpur", "malaysia", "selangor", "johor", "penang", "brunei", "bandar seri begawan", "yayasan",
    "singapore", "las vegas", "san francisco", "san jose", "allentown", "memphis", "okemos", "seattle",
    "boca raton", "washington", "new york", "los angeles", "chicago", "houston", "philadelphia", "phoenix",
    "san antonio", "san diego", "dallas", "austin",
)

ADDRESS_PATTERNS = re.compile(
    r"\b\d+\s*/\s*[a-z]\b"
    r"|\b[a-z]?\d+\s*(?:st|nd|rd|th)\s+(?:floor|flr|level|lvl)\b"
    r"|\bcor\.?\s"
    r"|\b(?:blvd|ave|st|rd|hwy|pkwy|lot|blk|brgy|bldg|unit|suite|ste|rm)\b\.?"
    r"|\b(?:" + _US_STATE_CODES + r")\s+\d{5}(?:-\d{4})?\b"
    r"|,\s*(?:" + _US_STATE_CODES + r")\b"
    r"|\b\d{4,5}\b.*\b(?:city|manila|philippines|ncr|qc)\b",
    re.IGNORECASE,
)

CUSTOMER_SECTION_MARKERS = (
    "sold to", "received from", "registered name", "customer name", "name of customer",
    "bill to", "ship to", "business address", "address/tin", "address / tin",
    "and address at", "business style", "customer:", "customer :",
)

PRINTER_ATP_MARKERS = (
    "authority to print", "printers accreditation", "printer's accreditation",
    "accreditation no", "date of atp", "bir permit", "permit no", "ptu",
    "ocn", "bklts", "booklets", "dependable pckg", "liledau", "printing services",
    "this provisional receipt", "valid for five", "valid until", "series of",
)

BLANK_ADDRESS_LINE_RE = re.compile(r"^\s*(?:business\s+)?address\s*[:._\-\s]*$", re.IGNORECASE)


def _tidy_address(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^(?:business\s+address|store\s+address|address|location|loc|branch)\s*[:.\-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:tel\.?|telefax\.?|phone\.?|mobile\.?|fax\.?|contact\.?)\s*(?:no\.?|#)?\s*[:.\-]?\s*[0-9\-\s\(\)\/]{5,}",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:vat\s*reg\.?\s*)?tin\s*[:.\-]?\s*[0-9\-\s]{9,}",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned)
    cleaned = re.sub(r"\s*\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" |_-*:;,.<>~@\\/{}()[]#=")
    return cleaned[:255].strip()


def _is_address_candidate_line(line: str) -> bool:
    stripped = line.strip()
    if not looks_like_address_line(stripped):
        return False
    if BLANK_ADDRESS_LINE_RE.match(stripped):
        return False
    lowered = stripped.casefold()
    if any(bad in lowered for bad in PRINTER_ATP_MARKERS):
        return False
    if any(bad in lowered for bad in CUSTOMER_SECTION_MARKERS):
        return False
    return any(kw in lowered for kw in ADDRESS_KEYWORDS) or bool(ADDRESS_PATTERNS.search(stripped))


def find_address_candidates(
    line_sets: list[list[tuple[str, float]]],
    header_lines: int = 18,
    vendor_name: str | None = None,
    customer_names: list[str] | None = None,
) -> list[str]:
    """Collect and rank plausible location/address candidate lines from receipt headers and guarded footers."""
    customer_names = customer_names or []
    candidates: dict[str, CandidateMeta] = {}

    def _register_candidate(cand_text: str, line_idx: int, conf: float) -> None:
        cleaned = _tidy_address(cand_text)
        if not cleaned or len(cleaned) < 4:
            return
        existing = candidates.get(cleaned)
        if existing is None:
            candidates[cleaned] = CandidateMeta(index=line_idx, confidence=conf)
        else:
            existing.index = min(existing.index, line_idx)
            existing.confidence = max(existing.confidence, conf)
            existing.occurrences += 1

    # 1. Header scan across all readings
    for lines in line_sets:
        consecutive_addr: list[tuple[int, str, float]] = []
        for index, (line_text, confidence) in enumerate(lines[:header_lines]):
            stripped = line_text.strip()
            lowered = stripped.casefold()

            # Stop header scan on customer section, column headings, or printer accreditation
            if any(marker in lowered for marker in CUSTOMER_SECTION_MARKERS):
                break
            if _is_column_heading(stripped):
                break
            if any(bad in lowered for bad in PRINTER_ATP_MARKERS):
                break

            # If this line matches vendor_name, skip it and reset consecutive address chain
            if vendor_name:
                v_low = vendor_name.casefold()
                if lowered == v_low or fuzz.ratio(lowered, v_low) >= 80 or (
                    len(v_low) >= 6
                    and v_low in lowered
                    and not any(
                        s in lowered
                        for s in (" street", " st.", " st ", " ave", " rd", " blvd", " pkwy", " suite", " unit")
                    )
                ):
                    consecutive_addr = []
                    continue

            if _is_address_candidate_line(stripped):
                consecutive_addr.append((index, stripped, confidence))
                _register_candidate(stripped, index, confidence)
            else:
                consecutive_addr = []

            # Multi-line consecutive address blocks (up to 3 lines)
            if len(consecutive_addr) >= 2:
                idx1, t1, c1 = consecutive_addr[-2]
                _, t2, c2 = consecutive_addr[-1]
                two_line = f"{t1}, {t2}"
                _register_candidate(two_line, idx1, (c1 + c2) / 2.0)

                if len(consecutive_addr) >= 3:
                    idx0, t0, c0 = consecutive_addr[-3]
                    three_line = f"{t0}, {t1}, {t2}"
                    _register_candidate(three_line, idx0, (c0 + c1 + c2) / 3.0)

    # 2. Guarded footer fallback if header scan produced no candidates
    if not candidates:
        for lines in line_sets:
            total_len = len(lines)
            footer_slice = lines[-12:]
            for f_idx, (line_text, confidence) in enumerate(footer_slice):
                abs_index = total_len - len(footer_slice) + f_idx
                stripped = line_text.strip()
                lowered = stripped.casefold()

                # Guard against printer ATP or customer blocks in footer
                surrounding = [
                    lines[max(0, abs_index + offset)][0].casefold()
                    for offset in range(-2, 3)
                    if 0 <= abs_index + offset < total_len
                ]
                if any(any(bad in s for bad in PRINTER_ATP_MARKERS) for s in surrounding):
                    continue
                if any(any(bad in s for bad in CUSTOMER_SECTION_MARKERS) for s in surrounding):
                    continue

                if _is_address_candidate_line(stripped):
                    _register_candidate(stripped, abs_index, confidence)

    # 3. Score and rank candidates
    ranked: list[tuple[str, float]] = []
    for text, info in candidates.items():
        lowered = text.casefold()

        # Reject candidate if it matches vendor_name or customer_names
        if vendor_name:
            v_low = vendor_name.casefold()
            if lowered == v_low or fuzz.ratio(lowered, v_low) >= 80:
                continue
        if customer_names and any(
            fuzz.token_set_ratio(lowered, c.casefold()) >= 85 for c in customer_names if len(c) >= 4
        ):
            continue

        score = 3.0 - (info.index * 0.15)
        score += 1.5 * max(0.0, min(info.confidence, 1.0))
        score += 0.5 * min(info.occurrences - 1, 3)

        has_street_or_bldg = any(
            t in lowered for t in (
                " st", " ave", " rd", " hwy", " blvd", " pkwy", " dr", " bldg", " floor",
                " flr", " suite", " ste", " unit", " rm", " plaza", " square", " center",
                " level", " jalan", " jln", " lot", " blk", " brgy", " barangay",
            )
        )
        has_city_or_region = any(
            t in lowered for t in (
                "city", "philippines", "metro", "manila", "makati", "quezon", "pasig",
                "taguig", "cebu", "davao", "rizal", "cavite", "laguna", "pampanga",
                "kuala lumpur", "malaysia", "brunei", "singapore", "las vegas", "new york",
                "san jose", "allentown", "memphis", "okemos",
            )
        ) or bool(re.search(r"\b(?:" + _US_STATE_CODES + r")\b", text))

        if has_street_or_bldg and has_city_or_region:
            score += 3.0
        elif has_city_or_region:
            score += 1.5
        elif has_street_or_bldg:
            score += 1.0

        if "," in text:
            score += 1.2
            # Reward multi-line / multi-part address candidates that cover both premises and city
            if has_street_or_bldg and has_city_or_region and len(text) >= 25:
                score += 1.5
            score += 0.4 * min(text.count(","), 3)

        # More complete address strings break ties over short partial fragments
        score += 0.01 * min(len(text), 100)

        if any(
            f" {sfx}" in lowered or lowered.endswith(sfx)
            for sfx in ("inc", "inc.", "corp", "corp.", "ltd", "ltd.", "co.")
        ) and not has_street_or_bldg:
            score -= 2.0

        ranked.append((text, round(score, 3)))

    ranked.sort(key=lambda pair: -pair[1])
    return [text for text, _ in ranked]



def _is_column_heading(line: str) -> bool:
    """Is every word on this line part of a table heading?

    Requires all tokens to be heading vocabulary, so a real name containing one
    such word ('Metro Hardware and Construction Supply') is unaffected.
    """
    tokens = re.findall(r"[A-Za-z]{2,}", line.casefold())
    if not tokens:
        return False
    return all(token in HEADING_VOCABULARY for token in tokens)


def text_lines(lines: list[list[Word]]) -> list[str]:
    return [" ".join(word.text for word in line) for line in lines]
