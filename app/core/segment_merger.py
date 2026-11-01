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

from app.core.anchors import DATE_RE, TAX_ID_RE, score_text
from app.core.numbers import normalize_money
from app.core.ocr_engine import OcrReading, Word

logger = logging.getLogger(__name__)

# Line patterns indicative of grocery/retail item lines
QTY_PRICE_RE = re.compile(r"\b\d{1,3}\s*[xX×]\s*\d+[,.]\d{2}\b")
PRICE_TAX_CODE_RE = re.compile(r"\b\d+[,.]\d{2}\s+[VvEeZz]\b")
BARCODE_RE = re.compile(r"\b\d{8,14}\b")

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


def _match_fingerprints(f1: LineFingerprint, f2: LineFingerprint) -> float:
    """Score whether two lines represent the same printed receipt line."""
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
