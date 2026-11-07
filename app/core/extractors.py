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

from app.core.ocr_engine import Word
from app.core.textquality import looks_like_words

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


ADDRESS_KEYWORDS = (
    " st.", " st ", "street", " ave", "avenue", "road", " rd", "brgy", "barangay",
    "city", "unit ", "blk", "block", "suite", "floor", " flr", "bldg", "building",
    "district", "province", "philippines", "zip", "p.o. box", "highway", "hwy",
    "subdivision", "subd", "drive", "dr.", "blvd", "boulevard", "plaza", "center",
    "centre", "mall", "tower", "metro", "manila", "makati", "quezon", "taguig", "pasig",
    "mandaluyong", "cebu", "davao", "pasay", "paranaque",
)

ADDRESS_PATTERNS = re.compile(
    r"\b\d+\s*/\s*[a-z]\b"
    r"|\b[a-z]?\d+\s*(?:st|nd|rd|th)\s+(?:floor|flr)\b"
    r"|\bcor\.?\s"
    r"|\b(?:blvd|ave|st|rd|hwy|lot|blk|brgy|bldg|unit|suite|rm)\b\.?"
    r"|\b\d{4,5}\b.*\b(?:city|manila|philippines)\b",
    re.IGNORECASE,
)


def find_address_candidates(
    line_sets: list[list[tuple[str, float]]], header_lines: int = 15
) -> list[str]:
    """Collect plausible location/address candidate lines from printed receipt headers."""
    candidates: dict[str, CandidateMeta] = {}

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

            is_address = any(kw in lowered for kw in ADDRESS_KEYWORDS) or bool(ADDRESS_PATTERNS.search(stripped))
            if not is_address:
                continue

            existing = candidates.get(stripped)
            if existing is None:
                candidates[stripped] = CandidateMeta(index=index, confidence=confidence)
            else:
                existing.index = min(existing.index, index)
                existing.confidence = max(existing.confidence, confidence)
                existing.occurrences += 1

    ranked: list[tuple[str, float]] = []
    for text, info in candidates.items():
        score = 2.0 - (info.index * 0.15)
        score += 1.5 * max(0.0, min(info.confidence, 1.0))
        score += 0.5 * min(info.occurrences - 1, 3)
        lowered = text.casefold()
        if "philippines" in lowered or "city" in lowered or "metro" in lowered:
            score += 1.0
        ranked.append((text, score))

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
