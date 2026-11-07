"""Scoring OCR output by whether it surfaced the fields we actually need.

Tesseract's mean word confidence is a poor selection signal: it is routinely
confident about noise. The near-illegible samples in the corpus score
respectably on confidence while containing nothing useful.

So candidate readings are ranked by how many domain anchors they expose - a tax
id, money next to a total label, a parseable date, receipt vocabulary - with mean
confidence demoted to a tiebreak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Tax ids: PH TINs use 3- or 5-digit branch codes; separators are frequently
# mangled by OCR, so whitespace is tolerated between groups.
TAX_ID_RE = re.compile(r"\b\d{3}\s*[-\s]\s*\d{3}\s*[-\s]\s*\d{3}\s*[-\s]\s*\d{3,5}\b")

MONEY_RE = re.compile(r"\d{1,3}(?:[,.\s]\d{3})*[,.]\d{2}\b|\b\d+[,.]\d{2}\b")

DATE_RE = re.compile(
    r"(?<!\d)\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*['’]?\d{2,4}(?!\d)"
    r"|(?<!\d)\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}(?!\d)"
    r"|(?<!\d)\d{1,2}(?:st|nd|rd|th)?(?:[\s\-_/]+|[\s\-_/]+of[\s\-_/]+)(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z.]*(?:,\s*|[\s\-_/]+)['’]?\d{2,4}(?!\d)"
    r"|(?<![A-Za-z0-9])(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z.]*[\s\-_/]+\d{1,2}(?:st|nd|rd|th)?(?:,\s*|[\s\-_/]+)['’]?\d{2,4}(?!\d)",
    re.IGNORECASE,
)

TOTAL_KEYWORDS = (
    "total", "amount due", "amt due", "grand total", "balance due", "subtotal", "balance",
)
TAX_KEYWORDS = ("vat", "tax", "sst", "gst", "vatable", "zero-rated", "vat-exempt")
DOC_KEYWORDS = (
    "receipt", "invoice", "official receipt", "sales invoice", "tin",
    "cashier", "change", "payment", "qty", "service charge",
)


@dataclass
class AnchorScore:
    total: float
    has_tax_id: bool = False
    money_near_total: int = 0
    money_tokens: int = 0
    has_date: bool = False
    keyword_hits: int = 0
    mean_confidence: float = 0.0
    matched: dict[str, list[str]] = field(default_factory=dict)


def score_text(text: str, mean_confidence: float = 0.0) -> AnchorScore:
    lowered = text.casefold()

    tax_ids = TAX_ID_RE.findall(text)
    money = MONEY_RE.findall(text)
    dates = DATE_RE.findall(text)

    near_total = _money_near_keywords(text, TOTAL_KEYWORDS + TAX_KEYWORDS)
    keyword_hits = sum(
        1 for word in TOTAL_KEYWORDS + TAX_KEYWORDS + DOC_KEYWORDS if word in lowered
    )

    total = (
        3.0 * (1 if tax_ids else 0)
        + 3.0 * min(near_total, 4) / 4.0 * 1.0
        + 2.0 * (1 if dates else 0)
        + 1.0 * min(keyword_hits, 10) / 10.0
        + 0.5 * max(0.0, min(mean_confidence, 1.0))
    )
    # A reading with no money at all is near-useless for a reimbursement claim.
    total += 0.5 * min(len(money), 6) / 6.0

    return AnchorScore(
        total=round(total, 4),
        has_tax_id=bool(tax_ids),
        money_near_total=near_total,
        money_tokens=len(money),
        has_date=bool(dates),
        keyword_hits=keyword_hits,
        mean_confidence=mean_confidence,
        matched={"tax_ids": tax_ids[:5], "dates": [str(d) for d in dates[:5]],
                 "money": money[:8]},
    )


def _money_near_keywords(text: str, keywords: tuple[str, ...], window: int = 60) -> int:
    """Count money tokens appearing close to a total/tax label.

    Proximity matters: a bare '3.70' could be anything, but '3.70' within a few
    characters of 'Total' is the figure we are after.
    """
    lowered = text.casefold()
    spans: list[tuple[int, int]] = []
    for keyword in keywords:
        start = 0
        while (index := lowered.find(keyword, start)) != -1:
            spans.append((index, index + len(keyword)))
            start = index + 1
    if not spans:
        return 0

    count = 0
    for match in MONEY_RE.finditer(text):
        position = match.start()
        if any(begin - window <= position <= end + window for begin, end in spans):
            count += 1
    return count
