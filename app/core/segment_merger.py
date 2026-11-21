"""Segment merging, ordering, and seam deduplication for long receipts.

Supports both:
1. Multi-part uploads: separate image files uploaded in file_urls.
2. Single tall images: automatically sliced into overlapping virtual strips.

Deduplication occurs at the text/line level after independent OCR passes,
preserving geometry, avoiding resolution distortion, and eliminating duplicate items.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field

from PIL import Image
from rapidfuzz import fuzz

from app.core.anchors import DATE_RE, MONEY_RE, TAX_ID_RE, score_text
from app.core.extraction import extract
from app.core.items import looks_like_store_code
from app.core.numbers import digits_only, normalize_money
from app.core.ocr_engine import OcrBundle, OcrReading, Word
from app.core.textquality import is_gobbly_vendor_name
from app.core.vendor import ENTITY_MARKERS

logger = logging.getLogger(__name__)

# Line patterns indicative of grocery/retail item lines
QTY_PRICE_RE = re.compile(r"\b\d{1,3}\s*[xX×]\s*\d+[,.]\d{2}\b")
PRICE_TAX_CODE_RE = re.compile(r"\b\d+[,.]\d{2}\s+[VvEeZz]\b")
BARCODE_RE = re.compile(r"\b\d{8,14}\b")
_BARE_QTY_OR_PRICE_RE = re.compile(
    r"^\s*(?:\d{1,3}\s*[xX×@*]\s*\d+(?:[,.]\d{2})?(?:\s+\d+[,.]\d{2})?|\d+[,.]\d{2})(?:\s+[VvEeZzTt])?\s*$"
)

# Header & Footer anchor keywords
HEADER_KEYWORDS = (
    "tin", "taxpayer", "invoice", "sales invoice", "official receipt",
    "pos", "cashier", "terminal", "ptu", "min", "bir", "store",
)
FOOTER_KEYWORDS = (
    "subtotal", "sub-total", "total amount", "total amt due", "amount due",
    "grand total", "cash", "tender", "change", "vatable sale", "vat (12%)",
    "vat exempt", "zero rated", "item purchased",
)

# Regulatory / tax-ID / printer-accreditation lines must always be ignored as
# grouping seams even when they contain 8+ digit permit/MIN/PTU numbers.
_REGULATORY_BOILERPLATE_RE = re.compile(
    r"\b(?:bir|tin|ptu|min|ocn|atp|accreditation|authority\s+to\s+print|"
    r"permit\s*no|serial\s*no|machine\s*id|telefax|tel\.?\s*no|phone)\b",
    re.IGNORECASE,
)

# Generic receipt header/footer/summary lines that recur across unrelated receipts.
_GENERIC_BOILERPLATE_RE = re.compile(
    r"\b(?:"
    r"thank\s+you|thanks|come\s+again|have\s+a\s+nice\s+day|please\s+retain"
    r"|official\s+receipt|sales\s+invoice|provisional\s+receipt"
    r"|collection\s+receipt|acknowledg(?:e)?ment\s+receipt|billing\s+statement"
    r"|vatable\s+sales?|12%\s*vat|vat[-\s]+exempt|zero[-\s]+rated|non[-\s]*vat|vat"
    r"|customer\s+copy|merchant\s+copy|cashier\s+copy|duplicate\s+copy|reprint"
    r"|this\s+serves\s+as|valid\s+for\s+five|valid\s+until"
    r"|pos|bir|tin|ptu|min|ocn|atp"
    r"|sub[-\s]?total|grand\s+total|total\s+amount(?:\s+due)?|total\s+amt\s+due"
    r"|amount\s+due|amt\s+due|total\s+sales|net\s+sales|total\s+due|balance\s+due"
    r"|cash|change|tender(?:ed)?|payment|credit\s+card|debit\s+card"
    r"|cashier|terminal|register|store\s*(?:#|no\.?|id|:|\d)|branch\s*(?:#|no\.?|id|:|\d)"
    r")\b"
    r"|\bvat\s*\(\s*12\s*%\s*\)"
    r"|^\s*total(?:\s*[:=]|\s+[\d$pP€£¥]|$)",
    re.IGNORECASE,
)

# Boundary markers indicating the start of receipt footer / regulatory blocks
_FOOTER_OR_ATP_START_RE = re.compile(
    r"\b(?:"
    r"sub[-\s]?total|grand\s+total|total\s+amount(?:\s+due)?|total\s+amt\s+due"
    r"|amount\s+due|amt\s+due|total\s+sales|vatable\s+sales?|vat\s+exempt"
    r"|zero[-\s]+rated|cash|change|tender(?:ed)?|authority\s+to\s+print"
    r"|date\s+of\s+atp|accreditation|ptu|bir\s+permit"
    r")\b"
    r"|\bvat\s*\(\s*12\s*%\s*\)",
    re.IGNORECASE,
)
_BARE_TOTAL_LINE_RE = re.compile(r"^\s*total\s*[:=]?\s*(?:[$pP€£¥]?\s*\d|$)", re.IGNORECASE)

ITEM_SEAM_OVERLAP_THRESHOLD = 1.0


@dataclass
class SegmentRoleScore:
    """Evaluation of a single segment's role in a long receipt."""

    header_score: float = 0.0
    footer_score: float = 0.0
    item_row_count: int = 0
    mean_confidence: float = 0.0

    @property
    def is_likely_middle(self) -> bool:
        """Middle segments have high item density but lack header/footer markers."""
        return self.item_row_count >= 5 and self.header_score < 2.0 and self.footer_score < 2.0


@dataclass
class SegmentAnalysis:
    """Deterministic structural and identity summary of a single OCR segment."""

    index: int
    vendor_name: str | None = None
    vendor_tax_id: str | None = None
    invoice_number: str | None = None
    transaction_date: str | None = None
    currency: str | None = None
    total_amount: float | None = None
    reconciled: bool = False
    has_header: bool = False
    has_footer: bool = False
    is_self_contained: bool = False
    role_score: SegmentRoleScore = field(default_factory=SegmentRoleScore)


@dataclass
class ReceiptSegmentGroup:
    """A group of segments belonging to a single receipt, ordered for merging."""

    segment_indices: list[int]
    segments: list[tuple[Image.Image, OcrReading]]
    analyses: list[SegmentAnalysis] = field(default_factory=list)
    is_self_contained: bool = False
    vendor_name: str | None = None
    total_amount: float | None = None
    reconciled: bool = False


def score_segment_density(text: str, mean_confidence: float = 0.0) -> SegmentRoleScore:
    """Analyze a single segment reading to determine its role and item density."""
    lowered = text.casefold()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Header indicators
    has_tin = bool(TAX_ID_RE.search(text))
    has_date = bool(DATE_RE.search(text))
    header_hits = sum(1 for kw in HEADER_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", lowered))
    header_score = (3.0 if has_tin else 0.0) + (2.0 if has_date else 0.0) + min(header_hits, 4) * 0.5

    # Footer indicators
    footer_hits = sum(1 for kw in FOOTER_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", lowered))
    footer_score = min(footer_hits, 6) * 1.0

    # Item row density
    item_count = 0
    for line in lines:
        if QTY_PRICE_RE.search(line) or PRICE_TAX_CODE_RE.search(line) or BARCODE_RE.search(line):
            item_count += 1

    return SegmentRoleScore(
        header_score=round(header_score, 2),
        footer_score=round(footer_score, 2),
        item_row_count=item_count,
        mean_confidence=mean_confidence,
    )


def order_segments(
    segments: list[tuple[Image.Image, OcrReading]],
) -> list[tuple[Image.Image, OcrReading]]:
    """Determine the optimal vertical reading order for user-uploaded segments.

    Uses header/footer anchors, item density, and overlap seam scores across
    candidate permutations. Falls back to original caller order if ambiguous.
    """
    if len(segments) <= 1:
        return segments

    # If already marked as virtual strips, preserve strict slicing order
    if all(getattr(img, "_is_virtual_strip", False) for img, _ in segments):
        return segments

    n = len(segments)
    scores = [score_segment_density(reading.text, reading.confidence) for _, reading in segments]

    # Pre-compute seam compatibility between all directed pairs (i -> j)
    pair_seam_scores: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair_seam_scores[(i, j)] = _evaluate_seam_overlap(
                segments[i][1].lines, segments[j][1].lines
            )

    best_perm: tuple[int, ...] | None = None
    best_score = -1e9
    caller_perm = tuple(range(n))
    caller_score = -1e9

    # Evaluate all candidate permutations (for n <= 6, n! <= 720)
    for perm in itertools.permutations(range(n)):
        first_idx = perm[0]
        last_idx = perm[-1]

        score = 0.0
        # Reward header at first segment
        score += 3.5 * scores[first_idx].header_score
        # Reward footer at last segment
        score += 3.5 * scores[last_idx].footer_score

        # Penalize footer markers appearing in early/middle segments
        for mid_idx in perm[:-1]:
            if scores[mid_idx].footer_score >= 2.0:
                score -= 3.0 * scores[mid_idx].footer_score

        # Penalize header markers appearing in late/middle segments
        for mid_idx in perm[1:]:
            if scores[mid_idx].header_score >= 3.0:
                score -= 3.0 * scores[mid_idx].header_score

        # Reward seam overlap transitions
        seam_total = 0.0
        for step in range(n - 1):
            seam_total += pair_seam_scores.get((perm[step], perm[step + 1]), 0.0)
        score += seam_total * 4.0

        # Baseline bonus for caller order: strong bias to prevent false shuffling
        if perm == caller_perm:
            score += 3.0
            caller_score = score

        if score > best_score:
            best_score = score
            best_perm = perm

    if best_perm is None or best_perm == caller_perm:
        return segments

    # Margin guard: require >= 3.0 score advantage AND clear rationale
    caller_seam_total = sum(
        pair_seam_scores.get((caller_perm[step], caller_perm[step + 1]), 0.0)
        for step in range(n - 1)
    )
    best_seam_total = sum(
        pair_seam_scores.get((best_perm[step], best_perm[step + 1]), 0.0)
        for step in range(n - 1)
    )
    margin = best_score - caller_score
    has_anchor_benefit = (
        scores[best_perm[0]].header_score >= 2.0 and scores[caller_perm[0]].header_score < 2.0
    ) or (
        scores[best_perm[-1]].footer_score >= 2.0 and scores[caller_perm[-1]].footer_score < 2.0
    )
    has_seam_benefit = (best_seam_total - caller_seam_total) >= 1.5

    if margin < 3.0 or not (has_anchor_benefit or has_seam_benefit):
        logger.info(
            "Auto-ordering candidate %s rejected by margin guard (margin=%.2f, caller_score=%.2f); keeping caller order",
            list(best_perm), margin, caller_score,
        )
        return segments

    ordered = [segments[idx] for idx in best_perm]
    logger.info("Auto-ordered segments from %s to %s (score=%.2f vs caller=%.2f)", list(caller_perm), list(best_perm), best_score, caller_score)
    return ordered


@dataclass
class LineFingerprint:
    words: list[Word]
    text: str
    barcodes: list[str] = field(default_factory=list)
    prices: list[float] = field(default_factory=list)
    mean_conf: float = 0.0

    @classmethod
    def from_line(cls, line: list[Word]) -> LineFingerprint:
        text = " ".join(w.text for w in line).strip()
        barcodes = BARCODE_RE.findall(text)
        prices: list[float] = []
        for token in text.split():
            clean = token.rstrip("VvEeZz")
            m = normalize_money(clean)
            if m is not None:
                prices.append(m)
        mean_conf = sum(w.confidence for w in line) / max(len(line), 1)
        return cls(words=line, text=text, barcodes=barcodes, prices=prices, mean_conf=mean_conf)


def _evaluate_seam_overlap(
    lines_prev: list[list[Word]], lines_next: list[list[Word]], max_window: int = 30
) -> float:
    """Quickly score overlap compatibility between tail of prev and head of next."""
    if not lines_prev or not lines_next:
        return 0.0

    tail = [LineFingerprint.from_line(ln) for ln in lines_prev[-max_window:]]
    head = [LineFingerprint.from_line(ln) for ln in lines_next[:max_window]]

    overlap_score = 0.0
    for t in tail:
        for h in head:
            # Tier 1: exact barcode match or 7+ digit common suffix/substring
            if t.barcodes and h.barcodes:
                if set(t.barcodes) & set(h.barcodes):
                    overlap_score += 2.0
                    continue
                if any(
                    len(b1) >= 7 and len(b2) >= 7 and (b1 in b2 or b2 in b1 or b1[-7:] == b2[-7:])
                    for b1 in t.barcodes for b2 in h.barcodes
                ):
                    overlap_score += 1.8
                    continue
            # Tier 2: identical prices + high text similarity
            if t.prices and h.prices and t.prices == h.prices:
                sim = fuzz.token_set_ratio(t.text, h.text)
                if sim >= 80:
                    overlap_score += 1.5
                elif sim >= 60:
                    overlap_score += 0.8
                continue
            # Tier 3: text similarity
            if fuzz.ratio(t.text, h.text) >= 85 and len(t.text) >= 8:
                overlap_score += 1.0

    return overlap_score


def merge_segment_readings(
    ordered_segments: list[tuple[Image.Image, OcrReading]],
) -> tuple[OcrReading, bool]:
    """Deduplicate overlapping lines across ordered segments into a single unified OcrReading.

    Returns (unified_reading, gap_suspected).
    """
    if not ordered_segments:
        raise ValueError("No segments to merge")

    if len(ordered_segments) == 1:
        reading = ordered_segments[0][1]
        if reading.anchors is None:
            reading.anchors = score_text(reading.text, reading.confidence)
        return reading, False

    unified_lines: list[list[Word]] = []
    cumulative_y_offset = 0
    gap_suspected = False

    for seg_idx, (img, reading) in enumerate(ordered_segments):
        curr_lines = [list(ln) for ln in reading.lines]

        if seg_idx == 0:
            unified_lines.extend(curr_lines)
            cumulative_y_offset += img.height
            continue

        prev_lines = unified_lines
        overlap_match = _find_best_overlap_alignment(prev_lines, curr_lines, max_window=30)

        if overlap_match is not None:
            prev_overlap_start, prev_overlap_end, next_overlap_start, next_overlap_end = overlap_match

            # Discard any truncated orphan lines at the cut boundaries
            # Keep previous lines up to prev_overlap_start
            clean_prev = prev_lines[:prev_overlap_start]

            # In the overlap zone, choose the line version with higher word confidence
            overlap_lines: list[list[Word]] = []
            overlap_prev = prev_lines[prev_overlap_start:prev_overlap_end + 1]
            overlap_next = curr_lines[next_overlap_start:next_overlap_end + 1]

            # Pick the higher confidence lines
            max_len = max(len(overlap_prev), len(overlap_next))
            for i in range(max_len):
                lp = overlap_prev[i] if i < len(overlap_prev) else None
                ln = overlap_next[i] if i < len(overlap_next) else None
                if lp is not None and ln is not None:
                    conf_p = sum(w.confidence for w in lp) / max(len(lp), 1)
                    conf_n = sum(w.confidence for w in ln) / max(len(ln), 1)
                    chosen = lp if conf_p >= conf_n else _shift_line_y(ln, cumulative_y_offset)
                    overlap_lines.append(chosen)
                elif lp is not None:
                    overlap_lines.append(lp)
                elif ln is not None:
                    overlap_lines.append(_shift_line_y(ln, cumulative_y_offset))

            # Non-overlapping tail from current segment
            tail_next = [
                _shift_line_y(ln, cumulative_y_offset)
                for ln in curr_lines[next_overlap_end + 1:]
            ]

            unified_lines = clean_prev + overlap_lines + tail_next
            logger.info(
                "Segment %d seam merged with %d overlapped lines (prev[%d:%d] <-> next[%d:%d])",
                seg_idx, len(overlap_lines), prev_overlap_start, prev_overlap_end,
                next_overlap_start, next_overlap_end,
            )
        else:
            # No overlap detected: user may have missed a gap between photos
            logger.warning("No seam overlap detected between segment %d and previous", seg_idx)
            gap_suspected = True
            shifted_next = [_shift_line_y(ln, cumulative_y_offset) for ln in curr_lines]
            unified_lines.extend(shifted_next)

        cumulative_y_offset += img.height

    # Reconstruct unified words and text
    all_words: list[Word] = [w for ln in unified_lines for w in ln]
    all_confidences = [w.confidence for w in all_words]
    mean_conf = (sum(all_confidences) / len(all_confidences)) if all_confidences else 0.0

    # Render unified text
    unified_text = "\n".join(" ".join(w.text for w in ln) for ln in unified_lines)

    # Pick the engine label from the segments
    primary_engine = ordered_segments[0][1].engine

    merged_reading = OcrReading(
        text=unified_text,
        words=all_words,
        confidence=mean_conf,
        variant="merged",
        psm=0,
        engine=primary_engine,
        lines=unified_lines,
    )
    merged_reading.anchors = score_text(unified_text, mean_conf)
    return merged_reading, gap_suspected


def _shift_line_y(line: list[Word], offset: int) -> list[Word]:
    """Return a new line with word top coordinates translated by offset."""
    return [
        Word(
            text=w.text,
            confidence=w.confidence,
            left=w.left,
            top=w.top + offset,
            width=w.width,
            height=w.height,
            block=w.block,
            par=w.par,
            line=w.line,
        )
        for w in line
    ]


def _find_best_overlap_alignment(
    lines_prev: list[list[Word]],
    lines_next: list[list[Word]],
    max_window: int = 30,
) -> tuple[int, int, int, int] | None:
    """Find the exact slice coordinates (prev_start, prev_end, next_start, next_end) of overlap."""
    if not lines_prev or not lines_next:
        return None

    tail_lines = lines_prev[-max_window:]
    head_lines = lines_next[:max_window]

    tail_fp = [LineFingerprint.from_line(ln) for ln in tail_lines]
    head_fp = [LineFingerprint.from_line(ln) for ln in head_lines]

    best_match: tuple[int, int, int, int] | None = None
    best_score = 0.0

    # Try matching blocks of length L >= 1
    for t_idx in range(len(tail_fp)):
        for h_idx in range(len(head_fp)):
            score = _match_fingerprints(tail_fp[t_idx], head_fp[h_idx])
            if score <= 0.0:
                continue

            # Extend the block forward
            curr_score = score
            match_len = 1
            while t_idx + match_len < len(tail_fp) and h_idx + match_len < len(head_fp):
                sub = _match_fingerprints(tail_fp[t_idx + match_len], head_fp[h_idx + match_len])
                if sub <= 0.0:
                    break
                curr_score += sub
                match_len += 1

            # Require a solid anchor: either multi-line match or a barcode match
            has_strong_anchor = (
                match_len >= 2
                or any(tail_fp[t_idx + k].barcodes for k in range(match_len))
                or curr_score >= 2.0
            )
            if has_strong_anchor and curr_score > best_score:
                best_score = curr_score
                actual_prev_start = len(lines_prev) - len(tail_lines) + t_idx
                actual_prev_end = actual_prev_start + match_len - 1
                actual_next_start = h_idx
                actual_next_end = h_idx + match_len - 1
                best_match = (
                    actual_prev_start, actual_prev_end,
                    actual_next_start, actual_next_end,
                )

    return best_match


def _match_fingerprints(
    f1: LineFingerprint, f2: LineFingerprint, *, ignore_boilerplate: bool = False
) -> float:
    """Score whether two lines represent the same printed receipt line."""
    if ignore_boilerplate and (
        _is_generic_boilerplate_line(f1.text) or _is_generic_boilerplate_line(f2.text)
    ):
        return 0.0

    # 1. Barcode match (exact or 7+ char common suffix/substring)
    if f1.barcodes and f2.barcodes:
        if set(f1.barcodes) & set(f2.barcodes):
            return 3.0
        if any(
            len(b1) >= 7 and len(b2) >= 7 and (b1 in b2 or b2 in b1 or b1[-7:] == b2[-7:])
            for b1 in f1.barcodes for b2 in f2.barcodes
        ):
            return 2.5

    # 2. Price match + text similarity
    if f1.prices and f2.prices and f1.prices == f2.prices:
        sim = fuzz.token_set_ratio(f1.text, f2.text)
        if sim >= 80:
            return 2.5
        if sim >= 60:
            return 1.5

    # 3. High text similarity without conflicting prices
    if not (f1.prices and f2.prices and f1.prices != f2.prices):
        sim = fuzz.ratio(f1.text, f2.text)
        if sim >= 85 and len(f1.text) >= 8:
            return 2.0

    return 0.0


def _is_generic_boilerplate_line(text: str) -> bool:
    """Return True for generic receipt boilerplate/headers/footers that should not link distinct receipts."""
    stripped = text.strip()
    if not stripped:
        return True
    if _REGULATORY_BOILERPLATE_RE.search(stripped):
        return True
    if BARCODE_RE.search(stripped) or QTY_PRICE_RE.search(stripped):
        return False
    return bool(_GENERIC_BOILERPLATE_RE.search(stripped))


def _evaluate_item_seam_overlap(
    lines_prev: list[list[Word]],
    lines_next: list[list[Word]],
    max_window: int = 30,
) -> float:
    """Score item/barcode seam overlap between tail of prev and head of next, ignoring boilerplate."""
    if not lines_prev or not lines_next:
        return 0.0

    tail = [
        fp
        for fp in (LineFingerprint.from_line(ln) for ln in lines_prev[-max_window:])
        if not _is_generic_boilerplate_line(fp.text)
    ]
    head = [
        fp
        for fp in (LineFingerprint.from_line(ln) for ln in lines_next[:max_window])
        if not _is_generic_boilerplate_line(fp.text)
    ]
    if not tail or not head:
        return 0.0

    overlap_score = 0.0
    for t in tail:
        best_for_t = 0.0
        for h in head:
            score = _match_fingerprints(t, h, ignore_boilerplate=True)
            if score > best_for_t:
                best_for_t = score
        overlap_score += best_for_t
    return overlap_score


def _header_region_lines(lines: list[str]) -> list[str]:
    """Return lines from top of segment up to the first item-body or footer/ATP boundary."""
    header_lines: list[str] = []
    item_rows = 0
    for line in lines:
        if _FOOTER_OR_ATP_START_RE.search(line) or _BARE_TOTAL_LINE_RE.match(line):
            break
        if QTY_PRICE_RE.search(line) or PRICE_TAX_CODE_RE.search(line):
            if not header_lines:
                break
            item_rows += 1
            if item_rows >= 3:
                break
        header_lines.append(line)
    return header_lines


def _validate_header_vendor(
    raw_vendor: str | None,
    header_lines: list[str],
    role: SegmentRoleScore,
    header_role: SegmentRoleScore,
) -> str | None:
    """Reject footer boilerplate, price lines, and split-line item names mistaken for vendors."""
    if not raw_vendor or not isinstance(raw_vendor, str):
        return None
    cleaned = raw_vendor.strip()
    if not cleaned or is_gobbly_vendor_name(cleaned) or looks_like_store_code(cleaned):
        return None
    if _is_generic_boilerplate_line(cleaned):
        return None
    if MONEY_RE.search(cleaned) or BARCODE_RE.search(cleaned) or QTY_PRICE_RE.search(cleaned):
        return None

    # Confirm the proposed vendor actually originates in the pre-footer header region
    # and is not a split-line item name followed immediately by a bare qty/price row.
    vendor_lower = cleaned.casefold()
    match_idx: int | None = None
    for idx, line in enumerate(header_lines):
        line_lower = line.casefold()
        if vendor_lower in line_lower or fuzz.ratio(vendor_lower, line_lower) >= 85:
            match_idx = idx
            break
    if match_idx is None:
        return None

    if match_idx + 1 < len(header_lines):
        next_line = header_lines[match_idx + 1].strip()
        if _BARE_QTY_OR_PRICE_RE.match(next_line):
            return None

    has_entity_marker = any(marker in vendor_lower for marker in ENTITY_MARKERS)
    if role.is_likely_middle and header_role.header_score < 1.0 and not has_entity_marker:
        return None
    return cleaned


def analyze_segment(index: int, image: Image.Image, reading: OcrReading) -> SegmentAnalysis:
    """Extract deterministic receipt identity and structural signals from one segment."""
    if reading.anchors is None:
        reading.anchors = score_text(reading.text, reading.confidence)

    role = score_segment_density(reading.text, reading.confidence)
    bundle = OcrBundle(primary=reading, supporting=[], all_readings=[reading], early_exit=True)
    extraction = extract(bundle)
    fields = extraction.fields

    raw_lines = [ln.strip() for ln in reading.text.splitlines() if ln.strip()]
    header_lines = _header_region_lines(raw_lines)
    header_role = (
        score_segment_density("\n".join(header_lines), reading.confidence)
        if header_lines
        else SegmentRoleScore()
    )

    total_amount = fields.get("total_amount")
    if total_amount is not None and role.footer_score == 0.0 and role.is_likely_middle:
        # Guard against a middle-segment item line containing the word 'total'
        total_evidence = extraction.evidence.get("total_amount", "")
        item_lines_after = 0
        seen_total_line = False
        for ln in raw_lines:
            if not seen_total_line and ln in total_evidence:
                seen_total_line = True
                continue
            if seen_total_line and (
                QTY_PRICE_RE.search(ln) or PRICE_TAX_CODE_RE.search(ln) or BARCODE_RE.search(ln)
            ):
                item_lines_after += 1
        if seen_total_line and item_lines_after >= 3:
            total_amount = None

    valid_vendor = _validate_header_vendor(
        fields.get("vendor_name"), header_lines, role, header_role
    )
    has_header = (header_role.header_score >= 2.0) or (valid_vendor is not None)
    vendor_name = valid_vendor if has_header else None

    raw_tin = fields.get("vendor_tax_id")
    norm_tin = digits_only(raw_tin) if raw_tin else ""
    vendor_tax_id = norm_tin or None

    reconciled = bool(
        extraction.reconciled or (extraction.item_scan and extraction.item_scan.reconciled)
    )
    has_footer = (total_amount is not None) or (role.footer_score >= 2.0)
    is_self_contained = bool(has_header and has_footer)

    return SegmentAnalysis(
        index=index,
        vendor_name=vendor_name,
        vendor_tax_id=vendor_tax_id,
        invoice_number=fields.get("invoice_number"),
        transaction_date=fields.get("transaction_date"),
        currency=fields.get("currency"),
        total_amount=total_amount,
        reconciled=reconciled,
        has_header=has_header,
        has_footer=has_footer,
        is_self_contained=is_self_contained,
        role_score=role,
    )


def _has_identity_conflict(a: SegmentAnalysis, b: SegmentAnalysis) -> bool:
    """Return True when two segments carry conflicting grounded receipt identities."""
    if a.vendor_tax_id and b.vendor_tax_id:
        tin_a = digits_only(a.vendor_tax_id)
        tin_b = digits_only(b.vendor_tax_id)
        if len(tin_a) >= 9 and len(tin_b) >= 9 and tin_a[:9] != tin_b[:9]:
            return True

    if a.invoice_number and b.invoice_number:
        inv_a = re.sub(r"[^A-Za-z0-9]", "", str(a.invoice_number)).upper()
        inv_b = re.sub(r"[^A-Za-z0-9]", "", str(b.invoice_number)).upper()
        if inv_a and inv_b and inv_a != inv_b:
            return True

    if a.transaction_date and b.transaction_date:
        if a.transaction_date.strip() != b.transaction_date.strip():
            return True

    if a.currency and b.currency and a.currency.strip().upper() != b.currency.strip().upper():
        return True

    if a.vendor_name and b.vendor_name:
        if fuzz.token_set_ratio(a.vendor_name.casefold(), b.vendor_name.casefold()) < 65:
            return True

    if (
        a.has_footer
        and b.has_footer
        and a.total_amount is not None
        and b.total_amount is not None
        and abs(a.total_amount - b.total_amount) > 0.01
    ):
        return True

    return False


@dataclass
class _SourceUnit:
    """Internal grouping unit representing one uploaded image (or its virtual strips)."""

    source_indices: list[int]
    segments: list[tuple[Image.Image, OcrReading]]
    analysis: SegmentAnalysis

    @property
    def head_lines(self) -> list[list[Word]]:
        return self.segments[0][1].lines if self.segments else []

    @property
    def tail_lines(self) -> list[list[Word]]:
        return self.segments[-1][1].lines if self.segments else []


def _partition_source_units(
    indexed_segments: list[tuple[int, Image.Image, OcrReading]],
) -> list[_SourceUnit]:
    """Keep virtual strips of a single tall image locked together as one unit."""
    all_unparented_strips = bool(indexed_segments) and all(
        getattr(img, "_is_virtual_strip", False)
        and getattr(img, "_parent_segment_index", None) is None
        for _, img, _ in indexed_segments
    )

    buckets: dict[tuple, list[tuple[int, Image.Image, OcrReading]]] = {}
    order: list[tuple] = []
    for pos, (idx, img, reading) in enumerate(indexed_segments):
        is_strip = bool(getattr(img, "_is_virtual_strip", False))
        parent_idx = getattr(img, "_parent_segment_index", None)
        if is_strip:
            if all_unparented_strips:
                key: tuple = ("single_tall", 0)
            elif parent_idx is not None:
                key = ("strip_parent", parent_idx)
            else:
                key = ("strip_idx", idx)
        else:
            key = ("seg", idx, pos)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append((idx, img, reading))

    units: list[_SourceUnit] = []
    for key in order:
        items = buckets[key]
        if any(getattr(img, "_is_virtual_strip", False) for _, img, _ in items):
            items.sort(key=lambda item: getattr(item[1], "_virtual_index", item[0]))
        source_indices = sorted(
            {
                int(getattr(img, "_parent_segment_index", idx)
                    if getattr(img, "_parent_segment_index", None) is not None
                    else idx)
                for idx, img, _ in items
            }
        )
        seg_pairs = [(img, reading) for _, img, reading in items]
        if len(seg_pairs) == 1:
            unit_reading = seg_pairs[0][1]
        else:
            unit_reading, _ = merge_segment_readings(seg_pairs)
        unit_analysis = analyze_segment(source_indices[0], seg_pairs[0][0], unit_reading)
        units.append(
            _SourceUnit(
                source_indices=source_indices,
                segments=seg_pairs,
                analysis=unit_analysis,
            )
        )
    return units


def _unit_seam_score(unit_a: _SourceUnit, unit_b: _SourceUnit) -> float:
    """Compute directional boundary item seam score between two source units."""
    forward = 0.0
    if not unit_a.analysis.has_footer and not unit_b.analysis.has_header:
        forward = _evaluate_item_seam_overlap(unit_a.tail_lines, unit_b.head_lines)

    backward = 0.0
    if not unit_b.analysis.has_footer and not unit_a.analysis.has_header:
        backward = _evaluate_item_seam_overlap(unit_b.tail_lines, unit_a.head_lines)

    return max(forward, backward)


def _group_aggregate_analysis(units: list[_SourceUnit]) -> SegmentAnalysis:
    """Build combined SegmentAnalysis for a group of units."""
    if len(units) == 1:
        return units[0].analysis
    ordered_segs = _order_units_segments(units)
    merged_reading, _ = merge_segment_readings(ordered_segs)
    return analyze_segment(units[0].source_indices[0], ordered_segs[0][0], merged_reading)


def _order_units_segments(units: list[_SourceUnit]) -> list[tuple[Image.Image, OcrReading]]:
    """Order units using order_segments() while keeping each unit's virtual strips contiguous."""
    if len(units) <= 1:
        return list(units[0].segments) if units else []

    if all(len(u.segments) == 1 for u in units):
        return order_segments([u.segments[0] for u in units])

    reps: list[tuple[Image.Image, OcrReading]] = []
    rep_to_unit: dict[int, _SourceUnit] = {}
    for unit in units:
        if len(unit.segments) == 1:
            img, reading = unit.segments[0]
            rep_img = Image.new("RGB", img.size) if getattr(img, "_is_virtual_strip", False) else img
            rep = (rep_img, reading)
        else:
            merged_reading, _ = merge_segment_readings(unit.segments)
            rep_img = Image.new("RGB", unit.segments[0][0].size)
            rep = (rep_img, merged_reading)
        reps.append(rep)
        rep_to_unit[id(rep[1])] = unit

    ordered_reps = order_segments(reps)
    ordered_segments: list[tuple[Image.Image, OcrReading]] = []
    for _, rep_reading in ordered_reps:
        ordered_segments.extend(rep_to_unit[id(rep_reading)].segments)
    return ordered_segments


def _can_merge_into_group(
    group_units: list[_SourceUnit],
    group_analysis: SegmentAnalysis,
    candidate: _SourceUnit,
) -> tuple[bool, float]:
    """Return (compatible, seam_score) for attaching candidate to an existing group."""
    cand_analysis = candidate.analysis

    if _has_identity_conflict(group_analysis, cand_analysis):
        return False, 0.0
    if any(_has_identity_conflict(u.analysis, cand_analysis) for u in group_units):
        return False, 0.0

    # Two self-contained receipts are distinct documents (including intra-batch duplicates).
    if group_analysis.is_self_contained and cand_analysis.is_self_contained:
        return False, 0.0

    best_seam = max(_unit_seam_score(u, candidate) for u in group_units)
    has_seam = best_seam >= ITEM_SEAM_OVERLAP_THRESHOLD

    if group_analysis.has_header and cand_analysis.has_header and not has_seam:
        return False, 0.0
    if group_analysis.has_footer and cand_analysis.has_footer and not has_seam:
        return False, 0.0

    if group_analysis.is_self_contained and not has_seam:
        return False, 0.0
    if cand_analysis.is_self_contained and not has_seam:
        return False, 0.0

    if has_seam:
        return True, best_seam

    # Complementary parts without identity conflict
    if group_analysis.has_header and not group_analysis.has_footer and not cand_analysis.has_header:
        return True, 0.0
    if not group_analysis.has_header and group_analysis.has_footer and not cand_analysis.has_footer:
        return True, 0.0
    if not group_analysis.has_header and not group_analysis.has_footer and not cand_analysis.is_self_contained:
        return True, 0.0

    return False, 0.0


def group_segments_into_receipts(
    indexed_segments: list[tuple[int, Image.Image, OcrReading]],
) -> list[ReceiptSegmentGroup]:
    """Group uploaded segments into distinct receipts while stitching connected parts."""
    if not indexed_segments:
        return []

    units = _partition_source_units(indexed_segments)
    if len(units) == 1:
        unit = units[0]
        return [
            ReceiptSegmentGroup(
                segment_indices=list(unit.source_indices),
                segments=list(unit.segments),
                analyses=[unit.analysis],
                is_self_contained=unit.analysis.is_self_contained,
                vendor_name=unit.analysis.vendor_name,
                total_amount=unit.analysis.total_amount,
                reconciled=unit.analysis.reconciled,
            )
        ]

    groups_units: list[list[_SourceUnit]] = []
    groups_analyses: list[SegmentAnalysis] = []

    for unit in units:
        best_group_idx: int | None = None
        best_seam = -1.0

        # Prefer highest seam overlap; on tie without seam, prefer most recent compatible incomplete group
        for g_idx in range(len(groups_units) - 1, -1, -1):
            ok, seam = _can_merge_into_group(groups_units[g_idx], groups_analyses[g_idx], unit)
            if not ok:
                continue
            if seam > best_seam:
                best_seam = seam
                best_group_idx = g_idx

        if best_group_idx is not None:
            groups_units[best_group_idx].append(unit)
            groups_analyses[best_group_idx] = _group_aggregate_analysis(groups_units[best_group_idx])
        else:
            groups_units.append([unit])
            groups_analyses.append(unit.analysis)

    result: list[ReceiptSegmentGroup] = []
    for g_units, g_analysis in zip(groups_units, groups_analyses):
        ordered_segs = _order_units_segments(g_units)
        seg_indices = sorted({idx for u in g_units for idx in u.source_indices})
        result.append(
            ReceiptSegmentGroup(
                segment_indices=seg_indices,
                segments=ordered_segs,
                analyses=[u.analysis for u in g_units],
                is_self_contained=g_analysis.is_self_contained,
                vendor_name=g_analysis.vendor_name,
                total_amount=g_analysis.total_amount,
                reconciled=g_analysis.reconciled,
            )
        )
    return result


def split_tall_image_into_strips(
    image: Image.Image,
    target_height: int = 1400,
    overlap_pct: float = 0.15,
    max_strips: int = 4,
) -> list[Image.Image]:
    """Slice a single very tall receipt image into overlapping horizontal virtual strips.

    Only applies to tall receipts (height >= 1800 and height / width >= 2.8).
    """
    w, h = image.size
    if h < 1800 or h < 2.8 * w:
        return [image]

    strip_h = min(target_height, int(2.4 * w))
    overlap_px = int(strip_h * overlap_pct)
    step = strip_h - overlap_px
    if step <= 0:
        return [image]

    strips: list[Image.Image] = []
    y = 0
    strip_idx = 0

    while y < h and strip_idx < max_strips:
        bottom = min(y + strip_h, h)
        if strip_idx == max_strips - 1:
            bottom = h  # Last strip reaches all the way to bottom
        crop = image.crop((0, y, w, bottom))
        # Tag with metadata for merger
        crop._is_virtual_strip = True
        crop._virtual_y_offset = y
        crop._virtual_index = strip_idx
        strips.append(crop)
        strip_idx += 1

        if bottom >= h:
            break
        y += step

    logger.info("Auto-tiled single tall image (%dx%d px) into %d virtual strips", w, h, len(strips))
    return strips
