"""Receipt OCR candidate generation and selection.

The OCR stage runs PaddleOCR as the primary deep-learning pass on the upright image.
If the fast-path eligibility check fails or downstream extraction does not reconcile,
the pipeline escalates to a single-pass OpenCV + Tesseract fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pytesseract
from PIL import Image

from app.core.anchors import AnchorScore, score_text
from app.core.preprocessing import (
    correct_orientation,
    prepare_region_crop,
    preprocess_image,
)

logger = logging.getLogger(__name__)

# Paddle is optional at import time so the service can still start and use the
# Tesseract fallback when an image has not yet been rebuilt with PaddleOCR.
try:
    from paddleocr import PaddleOCR
except Exception as exc:  # pragma: no cover - depends on the runtime image # noqa: BLE001
    PaddleOCR = None  # type: ignore[assignment]
    logger.warning("PaddleOCR unavailable; using Tesseract fallback: %s", exc)

paddle_ocr = None
_paddle_init_attempted = False
_paddle_lock = threading.RLock()

# Global Thread Pool Executor for CPU-bound OCR and preprocessing tasks.
# Capped to avoid resource starvation under concurrent celery jobs.
MAX_OCR_WORKERS = max(1, min(os.cpu_count() or 4, 6))
ocr_executor = ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS, thread_name_prefix="ocr")


def _get_paddle_ocr():
    """Initialize Paddle lazily so a missing model cannot break module import."""
    global _paddle_init_attempted, paddle_ocr
    if paddle_ocr is not None:
        return paddle_ocr
    with _paddle_lock:
        if paddle_ocr is not None:
            return paddle_ocr
        if _paddle_init_attempted:
            return None
        _paddle_init_attempted = True
        if PaddleOCR is None:
            return None
        try:
            try:
                paddle_ocr = PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                )
            except (TypeError, ValueError):
                # PaddleOCR 2.x does not know the 3.x pipeline options.
                paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
        except Exception as exc:  # pragma: no cover - depends on runtime/model # noqa: BLE001
            logger.warning("PaddleOCR initialization failed; using Tesseract fallback: %s", exc)
        return paddle_ocr


# Fast-path eligibility and fallback settings.
# Clean receipts pass PaddleOCR and clear FAST_PATH_ANCHOR_SCORE on the native
# upright image (zero Tesseract passes). Receipts that fail this gate escalate
# to a single-pass Tesseract reading over the OpenCV-preprocessed image at FAST_PATH_PSM.
FAST_PATH_PSM = 6
FAST_PATH_ANCHOR_SCORE = 6.0
FAST_PATH_MAX_EDGE = 1600


@dataclass
class Word:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    block: int
    par: int
    line: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2

    @property
    def line_key(self) -> tuple[int, int, int]:
        return (self.block, self.par, self.line)


@dataclass
class OcrReading:
    text: str
    words: list[Word]
    confidence: float
    variant: str
    psm: int
    engine: Literal["paddle", "tesseract"] = "tesseract"
    anchors: AnchorScore | None = None
    lines: list[list[Word]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.anchors.total if self.anchors else 0.0

    @property
    def label(self) -> str:
        """Stable, engine-aware label for logs and audit metadata."""
        return f"{self.engine}/{self.variant}/psm{self.psm}"


def read_variant(image: Image.Image, psm: int, lang: str = "eng") -> OcrReading:
    """Run one OCR pass and keep both text and word geometry."""
    config = f"--psm {psm} --oem 1 --dpi 300"
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    words: list[Word] = []
    confidences: list[float] = []
    for index, token in enumerate(data["text"]):
        if not token.strip():
            continue
        confidence = float(data["conf"][index])
        words.append(
            Word(
                text=token,
                confidence=max(confidence, 0.0) / 100.0,
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
                block=int(data["block_num"][index]),
                par=int(data["par_num"][index]),
                line=int(data["line_num"][index]),
            )
        )
        if confidence >= 0:
            confidences.append(confidence)

    mean = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    text = _render_text(words)
    reading = OcrReading(
        text=text, words=words, confidence=mean, variant="?", psm=psm,
        engine="tesseract", lines=group_lines(words),
    )
    reading.anchors = score_text(text, mean)
    return reading


_VALUE_COLUMN_RE = re.compile(
    r"\d{1,3}(?:[,.\s]\d{3})*[.,]\d{1,2}\b"  # money
    r"|\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b"  # date
    r"|\b\d{1,3}\s*[xX×@*]\b"  # qty
    r"|\b[pP$€£¥]?\d+[.,]\d{2}\b"  # currency amount
    r"|\b\d+%\b"  # percentage
    r"|^\s*[VTEZSYNXA0-9]\s*$",  # tax code or single flag
    re.IGNORECASE,
)


def group_lines(words: Sequence[Word]) -> list[list[Word]]:
    """Group words into visual lines, ordered left to right.

    PaddleOCR often assigns separate line IDs to left-aligned text and right-aligned
    prices on the same visual horizontal row. This merges horizontally split fragments
    sharing vertical overlap into unified visual lines while preserving two-column
    form separation.
    """
    buckets: dict[tuple[int, int, int], list[Word]] = {}
    for word in words:
        buckets.setdefault(word.line_key, []).append(word)
    raw_lines = [sorted(group, key=lambda w: w.left) for group in buckets.values()]
    sorted_lines = sorted(raw_lines, key=lambda group: (group[0].top, group[0].left))

    if len(sorted_lines) <= 1:
        return sorted_lines

    merged_lines: list[list[Word]] = []
    for group in sorted_lines:
        if not merged_lines:
            merged_lines.append(group)
            continue

        prev = merged_lines[-1]
        prev_center_y = sum(w.center_y for w in prev) / len(prev)
        prev_h = max(w.height for w in prev)
        prev_right = max(w.right for w in prev)

        curr_center_y = sum(w.center_y for w in group) / len(group)
        curr_h = max(w.height for w in group)
        curr_left = min(w.left for w in group)

        max_h = max(prev_h, curr_h, 8)
        min_h = min(prev_h, curr_h, 8)
        y_overlap = abs(prev_center_y - curr_center_y) <= 0.55 * max_h
        ordered_horiz = curr_left >= prev_right - (0.5 * min_h)

        if y_overlap and ordered_horiz:
            gap = curr_left - prev_right
            curr_text = " ".join(w.text for w in group)
            # Merge if gap is small (adjacent words in sentence), or if right group is a value/amount column
            if gap <= 3.5 * max_h or _VALUE_COLUMN_RE.search(curr_text):
                merged_lines[-1] = sorted(prev + group, key=lambda w: w.left)
                continue

        merged_lines.append(group)

    return merged_lines


def _render_text(words: Sequence[Word]) -> str:
    """Rebuild text with newlines at line boundaries.

    Line structure is load-bearing: label/value pairing and the proximity checks
    in anchor scoring both rely on it, and the original space-joined output
    destroyed it.
    """
    return "\n".join(
        " ".join(word.text for word in line) for line in group_lines(words)
    )


@dataclass
class OcrBundle:
    """The primary reading plus its close runners-up.

    Measurement showed the top variants score within 0.01 of each other while
    recovering *different* fields, so committing to a single winner discards
    information. Downstream stages use ``primary`` for word geometry and
    ``combined_text`` for candidate discovery and grounding. Agreement across
    independent readings also serves as corroboration: two variants producing the
    same figure is real evidence, where one model's self-reported confidence is
    not.
    """

    primary: OcrReading
    supporting: list[OcrReading] = field(default_factory=list)
    all_readings: list[OcrReading] = field(default_factory=list)
    early_exit: bool = False

    @property
    def combined_text(self) -> str:
        seen: set[str] = set()
        blocks: list[str] = []
        for reading in [self.primary, *self.supporting]:
            if reading.text not in seen:
                seen.add(reading.text)
                blocks.append(reading.text)
        return "\n".join(blocks)

    @property
    def confidence(self) -> float:
        return self.primary.confidence

    def agreement(self, token: str) -> int:
        """How many independent readings contain this token."""
        needle = token.strip()
        if not needle:
            return 0
        return sum(1 for r in self.all_readings if needle in r.text)


def rank_candidates(candidates: Sequence[OcrReading]) -> list[OcrReading]:
    """Rank OCR candidates without adding another OCR or extraction pass.

    Anchor coverage is the primary signal. Confidence and text length only break
    ties because engine confidence values are not directly calibrated across
    PaddleOCR and Tesseract.
    """
    return sorted(
        (candidate for candidate in candidates if candidate.text.strip()),
        key=lambda candidate: (
            -candidate.score,
            -candidate.confidence,
            -len(candidate.text),
        ),
    )


def select_primary(candidates: Sequence[OcrReading]) -> OcrReading:
    """Select the strongest available OCR result from either engine."""
    ranked = rank_candidates(candidates)
    if not ranked:
        raise RuntimeError("All OCR candidates were empty")
    return ranked[0]


def _box_bounds(box) -> tuple[int, int, int, int]:
    """Normalize Paddle's polygon or ``[left, top, right, bottom]`` box."""
    points = np.asarray(box)
    if points.ndim == 1 and len(points) == 4:
        left, top, right, bottom = points
    else:
        points = points.reshape(-1, 2)
        left = points[:, 0].min()
        top = points[:, 1].min()
        right = points[:, 0].max()
        bottom = points[:, 1].max()
    return int(left), int(top), int(right), int(bottom)


def _append_paddle_line(
    words: list[Word],
    text: str,
    confidence: float,
    box,
    line_number: int,
    word_texts: Sequence[str] | None = None,
    word_boxes=None,
) -> None:
    """Append one Paddle line, using native word boxes where available."""
    text = str(text).strip()
    if not text:
        return
    confidence = max(0.0, min(float(confidence), 1.0))

    if word_texts and word_boxes is not None and len(word_texts) == len(word_boxes):
        for token, token_box in zip(word_texts, word_boxes):
            left, top, right, bottom = _box_bounds(token_box)
            words.append(Word(
                text=str(token),
                confidence=confidence,
                left=left,
                top=top,
                width=max(1, right - left),
                height=max(1, bottom - top),
                block=0,
                par=0,
                line=line_number,
            ))
        return

    left, top, right, bottom = _box_bounds(box)
    line_width = max(1, right - left)
    for match in re.finditer(r"\S+", text):
        token_left = left + round(line_width * match.start() / len(text))
        token_right = left + round(line_width * match.end() / len(text))
        words.append(Word(
            text=match.group(),
            confidence=confidence,
            left=token_left,
            top=top,
            width=max(1, token_right - token_left),
            height=max(1, bottom - top),
            block=0,
            par=0,
            line=line_number,
        ))


def read_paddle(image: Image.Image) -> OcrReading:
    """Run PaddleOCR across its legacy 2.x and pipeline 3.x APIs."""
    cached = getattr(image, "_paddle_reading", None)
    if cached is not None:
        return cached

    engine = _get_paddle_ocr()
    if engine is None:
        raise RuntimeError("PaddleOCR is not initialized")

    upright = correct_orientation(image)
    img_array = np.array(upright.convert("RGB"))
    words: list[Word] = []
    confidences: list[float] = []

    # Tall/narrow receipts (aspect >= 2.0) get a smaller detector resize limit:
    # the default 960px side produces a ~3820px internal image and Paddle spends
    # a minute per call on it. 512 was measured to cut that time by ~30% with
    # identical line/character output on long grocery receipts, so the reduced
    # detector context does not lose lines.
    width, height = upright.size
    det_limit = 512 if height >= 2.0 * width else None

    with _paddle_lock:
        if hasattr(engine, "predict"):
            # PaddleOCR 3.x returns OCRResult mappings. Disabling optional document
            # stages avoids the CPU PIR/oneDNN path that is not supported by all
            # Paddle runtime builds and is unnecessary for receipt crops.
            kwargs = {
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "return_word_box": True,
            }
            if det_limit is not None:
                kwargs["text_det_limit_side_len"] = det_limit
            result = engine.predict(img_array, **kwargs)
            first = result[0] if result else None
            if first is not None and hasattr(first, "get"):
                texts = first.get("rec_texts", [])
                scores = first.get("rec_scores", [])
                boxes = first.get("rec_boxes", [])
                word_texts = first.get("text_word", [])
                word_boxes = first.get("text_word_boxes", [])
                for idx, text in enumerate(texts):
                    if idx >= len(boxes):
                        continue
                    score = scores[idx] if idx < len(scores) else 0.0
                    line_words = word_texts[idx] if idx < len(word_texts) else None
                    line_boxes = word_boxes[idx] if idx < len(word_boxes) else None
                    _append_paddle_line(
                        words, text, score, boxes[idx], idx, line_words, line_boxes
                    )
                    confidences.append(max(0.0, min(float(score), 1.0)))
        else:
            # PaddleOCR 2.x returns ``[[[polygon, (text, confidence)], ...]]``.
            img_bgr = img_array[:, :, ::-1]
            result = engine.ocr(img_bgr, cls=True)
            for idx, line in enumerate(result[0] if result and result[0] else []):
                try:
                    box, (text, confidence) = line
                except (TypeError, ValueError):
                    logger.warning("PaddleOCR returned an unsupported line shape")
                    continue
                _append_paddle_line(words, text, confidence, box, idx)
                confidences.append(max(0.0, min(float(confidence), 1.0)))

    if not words:
        # A successful model call with no detections is not successful OCR. Let
        # read_pooled run the complete Tesseract search instead.
        raise RuntimeError("PaddleOCR returned no text")

    mean = sum(confidences) / len(confidences)
    text_content = _render_text(words)
    reading = OcrReading(
        text=text_content,
        words=words,
        confidence=mean,
        variant="source",
        psm=0,
        engine="paddle",
        lines=group_lines(words),
    )
    reading.anchors = score_text(text_content, mean)
    try:
        image._paddle_reading = reading
        if upright is not image:
            upright._paddle_reading = reading
    except AttributeError:
        pass
    return reading


def _supporting_readings(
    candidates: Sequence[OcrReading],
    primary: OcrReading,
    pool_size: int,
) -> list[OcrReading]:
    """Choose strong, diverse readings excluding the selected primary.

    Readings that score far below the primary are excluded outright: on
    tall/narrow receipts Tesseract degrades to near-zero anchor scores while
    Paddle reads the slip cleanly, and concatenating that garbage into
    ``combined_text`` pollutes the LLM prompt (wrong vendor, wrong country)
    and the item/region scanners. A floor of half the primary's score keeps
    genuinely independent second opinions while dropping engine failure.
    """
    supporting: list[OcrReading] = []
    seen_variants: set[str] = {primary.variant}
    seen_text: set[str] = {primary.text}
    ranked = rank_candidates(candidates)
    target = max(0, pool_size - 1)
    floor = primary.score / 2.0

    # Prefer different preprocessing variants/engines first, then fill remaining
    # slots with another candidate. Identical OCR text is pooled only once.
    for distinct_variants_only in (True, False):
        for reading in ranked:
            if len(supporting) >= target:
                return supporting
            if reading is primary or reading.text in seen_text:
                continue
            if reading.score < floor:
                continue
            if distinct_variants_only and reading.variant in seen_variants:
                continue
            supporting.append(reading)
            seen_variants.add(reading.variant)
            seen_text.add(reading.text)
    return supporting


def read_fast(image: Image.Image, lang: str = "eng") -> OcrReading:
    """Single-pass OpenCV + Tesseract reading for OCR fallback.

    Runs best-effort OpenCV preprocessing and extracts text using Tesseract at FAST_PATH_PSM.
    """
    cache = getattr(image, "_ocr_readings_cache", None)
    if cache is not None and ("clean", FAST_PATH_PSM, lang) in cache:
        return cache[("clean", FAST_PATH_PSM, lang)]

    prepared = preprocess_image(image)
    reading = read_variant(prepared, psm=FAST_PATH_PSM, lang=lang)
    reading.variant = "clean"
    if hasattr(image, "__dict__"):
        try:
            if cache is None:
                cache = {}
                image._ocr_readings_cache = cache
            cache[("clean", FAST_PATH_PSM, lang)] = reading
        except AttributeError:
            pass
    return reading


def read_segment_fast(image: Image.Image, lang: str = "eng") -> OcrReading:
    """Run PaddleOCR on a segment/strip, falling back to Tesseract only if Paddle fails or yields < 2 lines."""
    try:
        reading = read_paddle(image)
        if len(reading.lines) >= 2:
            return reading
    except Exception as exc:  # noqa: BLE001
        logger.warning("PaddleOCR failed on segment; falling back to Tesseract: %s", exc)
    return read_fast(image, lang=lang)


def read_region_recovery(
    image: Image.Image,
    y_start_ratio: float,
    y_end_ratio: float,
    lang: str = "eng",
) -> list[OcrReading]:
    """Targeted high-contrast crop recovery for missing header (TIN/vendor) or footer (total/subtotal)."""
    crop, y_offset = prepare_region_crop(image, y_start_ratio, y_end_ratio)
    candidates: list[OcrReading] = []

    # 1. PaddleOCR on contrast-enhanced crop (crop has small aspect ratio, no 512 resize penalty)
    try:
        r_paddle = read_paddle(crop)
        shifted_words = [
            Word(
                text=w.text,
                confidence=w.confidence,
                left=w.left,
                top=w.top + y_offset,
                width=w.width,
                height=w.height,
                block=w.block,
                par=w.par,
                line=w.line,
            )
            for w in r_paddle.words
        ]
        r_paddle_shifted = OcrReading(
            text=r_paddle.text,
            words=shifted_words,
            confidence=r_paddle.confidence,
            variant="region_paddle",
            psm=0,
            engine="paddle",
            lines=group_lines(shifted_words),
        )
        r_paddle_shifted.anchors = r_paddle.anchors
        candidates.append(r_paddle_shifted)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Paddle recovery on region [%.2f, %.2f] failed: %s", y_start_ratio, y_end_ratio, exc)

    # 2. Tesseract single-pass at PSM 6 on the CLAHE crop
    try:
        r_tess = read_variant(crop, psm=6, lang=lang)
        if r_tess.words:
            shifted_words_tess = [
                Word(
                    text=w.text,
                    confidence=w.confidence,
                    left=w.left,
                    top=w.top + y_offset,
                    width=w.width,
                    height=w.height,
                    block=w.block,
                    par=w.par,
                    line=w.line,
                )
                for w in r_tess.words
            ]
            r_tess_shifted = OcrReading(
                text=r_tess.text,
                words=shifted_words_tess,
                confidence=r_tess.confidence,
                variant="region_tess",
                psm=6,
                engine="tesseract",
                lines=group_lines(shifted_words_tess),
            )
            r_tess_shifted.anchors = r_tess.anchors
            candidates.append(r_tess_shifted)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tesseract recovery on region [%.2f, %.2f] failed: %s", y_start_ratio, y_end_ratio, exc)

    return candidates



_ITEM_ROW_DENSITY_RE = re.compile(
    r"\b\d{1,3}\s*[xX×]\s*\d+[,.]\d{2}\b|\b\d+[,.]\d{2}\s+[VvEeZz]\b|\b\d{8,14}\b"
)


def _fast_path_eligible(reading: OcrReading) -> bool:
    """Anchor score or item density decides eligibility; extraction confirms afterwards."""
    if reading.score >= FAST_PATH_ANCHOR_SCORE:
        return True
    # Middle segments of long receipts lack headers/totals but have dense item rows
    if reading.anchors and reading.anchors.money_tokens >= 4 and reading.confidence >= 0.70:
        lines = reading.text.splitlines()
        item_matches = sum(1 for ln in lines if _ITEM_ROW_DENSITY_RE.search(ln))
        if item_matches >= 5:
            return True
    return False


async def read_pooled(
    image: Image.Image,
    lang: str = "eng",
    pool_size: int = 2,
    fast_path: bool = True,
    **kwargs: Any,
) -> OcrBundle:
    """Run PaddleOCR primary pass, escalating to single-pass Tesseract when needed.

    Fast path: Run PaddleOCR on the upright image. When it succeeds and clears
    ``_fast_path_eligible``, the bundle is returned immediately with
    ``early_exit=True`` (zero Tesseract passes).

    Escalation: If fast_path=False, or if PaddleOCR fails or scores below the
    fast-path threshold, run a single-pass Tesseract reading over the
    OpenCV-preprocessed image. The readings are combined and the stronger
    candidate becomes primary.
    """
    loop = asyncio.get_running_loop()

    paddle_reading: OcrReading | None = None
    try:
        paddle_reading = await loop.run_in_executor(ocr_executor, read_paddle, image)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PaddleOCR failed; continuing to Tesseract fallback: %s", exc)
        paddle_reading = None

    if fast_path:
        if paddle_reading is not None and _fast_path_eligible(paddle_reading):
            logger.info(
                "Fast path: PaddleOCR score=%.3f conf=%.3f",
                paddle_reading.score, paddle_reading.confidence,
            )
            return OcrBundle(
                primary=paddle_reading,
                supporting=[],
                all_readings=[paddle_reading],
                early_exit=True,
            )
        if paddle_reading is not None:
            logger.info(
                "PaddleOCR score %.3f below %.2f; escalating to Tesseract fallback",
                paddle_reading.score, FAST_PATH_ANCHOR_SCORE,
            )

    candidates: list[OcrReading] = []
    if paddle_reading is not None:
        candidates.append(paddle_reading)

    try:
        tesseract_reading = await loop.run_in_executor(ocr_executor, read_fast, image, lang)
        candidates.append(tesseract_reading)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tesseract fallback failed: %s", exc)

    if not candidates:
        raise RuntimeError("PaddleOCR and Tesseract produced no OCR candidates")

    primary = select_primary(candidates)
    supporting = _supporting_readings(candidates, primary, pool_size)
    runner_up = next(
        (candidate for candidate in rank_candidates(candidates) if candidate is not primary),
        None,
    )
    margin = primary.score - runner_up.score if runner_up else 0.0
    logger.info(
        "OCR selected engine=%s variant=%s psm=%s score=%.3f conf=%.3f margin=%.3f",
        primary.engine,
        primary.variant,
        primary.psm,
        primary.score,
        primary.confidence,
        margin,
    )
    return OcrBundle(
        primary=primary,
        supporting=supporting,
        all_readings=candidates,
        early_exit=False,
    )


def extract_text(image: Image.Image, lang: str = "eng") -> tuple[str, float]:
    """Backward-compatible single-pass helper: returns ``(text, confidence)``."""
    reading = read_variant(image, psm=6, lang=lang)
    return reading.text, reading.confidence
