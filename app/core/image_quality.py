"""Pre-OCR image quality evaluation.

Evaluates blur, brightness, and resolution using OpenCV heuristics before
running the resource-heavy OCR pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.core.preprocessing import correct_orientation, normalize_illumination, to_gray
from app.core.textquality import is_gobbly_vendor_name

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Structured output from check_image_quality()."""

    passed: bool
    rejection_code: str | None = None       # "blurry" | "too_dark" | "faded" | "too_small" | "not_a_receipt" | None
    rejection_reason: str | None = None     # Human-readable sentence for frontend
    blur_score: float = 0.0                    # Laplacian variance (higher = sharper)
    brightness: float = 0.0                    # Mean pixel value 0-255
    resolution: tuple[int, int] = (0, 0)       # (width, height) after orientation correction
    contrast_score: float = 0.0                # 99th percentile shadow-normalized ink depth (0-255)


NOT_A_RECEIPT_REASON = (
    "This image does not appear to be a receipt (e.g., a photo, selfie, or non-receipt document). "
    "Please upload a clear photo of an official receipt or invoice."
)

BLURRY_REASON = (
    "The uploaded image is too blurry to read the receipt details accurately. "
    "Please upload a clearer, focused photo of the receipt."
)

FADED_REASON = (
    "The receipt text is too faded or washed out to read accurately. "
    "Please upload a clearer photo with better contrast or request a clearer copy of the receipt."
)

UNREADABLE_REASON = (
    "The receipt is damaged, incomplete, or too unreadable to extract critical details accurately. "
    "Please upload a clearer photo of the receipt."
)

_CRITICAL_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("total_amount", "Total Amount"),
    ("transaction_date", "Transaction Date"),
    ("vendor_name", "Merchant Name"),
)


def build_unreadable_reason(
    fields: dict,
    *,
    is_faded: bool = False,
    is_illegible: bool = False,
    is_gobbly: bool = False,
    rejected_fields: Iterable[str] | None = None,
) -> str:
    """Build a specific rejection reason naming missing critical fields and/or garbled OCR."""
    data = fields or {}
    rejected_set = set(rejected_fields or ())
    missing: list[str] = []
    for key, label in _CRITICAL_FIELD_LABELS:
        val = data.get(key)
        is_missing = val is None or (isinstance(val, str) and not val.strip())
        is_rejected = key in rejected_set or any(
            entry.startswith(f"{key} ") for entry in rejected_set
        )
        is_gobbly_vendor = (
            key == "vendor_name" and not is_missing and is_gobbly_vendor_name(val)
        )
        if is_missing or is_rejected or is_gobbly_vendor:
            missing.append(label)

    missing_text = ", ".join(missing)
    if is_faded:
        if missing:
            return (
                "The receipt text is too faded or washed out to read critical details "
                f"({missing_text}). Please upload a clearer photo with better contrast or "
                "request a clearer copy of the receipt."
            )
        return FADED_REASON

    garbled = is_illegible or is_gobbly
    if missing and garbled:
        return (
            "The receipt is damaged, incomplete, or too unreadable to read critical details "
            f"({missing_text}) due to garbled or illegible OCR text. "
            "Please upload a clearer photo of the receipt."
        )
    if missing:
        return (
            "The receipt is damaged, incomplete, or too unreadable to read critical details "
            f"({missing_text}). Please upload a clearer photo of the receipt."
        )
    if garbled:
        return (
            "The receipt is damaged, incomplete, or too unreadable because the extracted OCR text "
            "is garbled or largely illegible. Please upload a clearer photo of the receipt."
        )
    return UNREADABLE_REASON


def select_quality_failure(
    results: Iterable[QualityCheckResult | None],
) -> QualityCheckResult | None:
    """Select the highest-priority failing quality result (`faded` first, then first failure)."""
    failing = [res for res in results if res is not None and not res.passed]
    if not failing:
        return None
    for res in failing:
        if res.rejection_code == "faded":
            return res
    return failing[0]

_face_cascade = None


def _get_face_cascade() -> object | None:
    global _face_cascade
    if _face_cascade is None:
        if not hasattr(cv2, "CascadeClassifier"):
            return None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load face cascade: %s", e)
    return _face_cascade


def _detect_prominent_face(gray: np.ndarray, max_face_ratio: float = 0.04) -> bool:
    """Detect if a prominent human face exists in the image (selfie / portrait)."""
    cascade = _get_face_cascade()
    if cascade is None or cascade.empty():
        return False
    h, w = gray.shape[:2]
    total_area = float(h * w)
    if total_area <= 0:
        return False

    scale = 1.0
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        small_gray = gray

    faces = cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    for (x, y, fw, fh) in faces:
        orig_fw = fw / scale
        orig_fh = fh / scale
        face_area = orig_fw * orig_fh
        if (face_area / total_area) >= max_face_ratio:
            return True
    return False


def _check_non_receipt_heuristics(
    upright: Image.Image,
    gray: np.ndarray,
    sat_thresh: float = 110.0,
    min_density: float = 0.015,
) -> bool:
    """Return True if image visually behaves as a non-receipt scene, selfie, or photo."""
    # 1. Check for prominent human face / selfie
    if _detect_prominent_face(gray, settings.image_quality_max_face_area_ratio):
        logger.info("Visual check detected prominent face (selfie/portrait)")
        return True

    # 2. Check saturation vs horizontal text edge density
    if upright.mode not in ("RGB", "RGBA"):
        return False

    rgb = np.array(upright.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_bin = (np.abs(grad_x) > 30).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    connected = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, kernel)
    density = float(np.mean(connected))

    if sat_mean > sat_thresh and density < min_density:
        logger.info(
            "Visual check flagged colorful non-document scene: sat=%.1f > %.1f, density=%.4f < %.4f",
            sat_mean, sat_thresh, density, min_density
        )
        return True

    return False


def _three_zone_slices(height: int, width: int) -> list[tuple[slice, slice]]:
    """Return 3 equal slices along the longer image axis."""
    if height >= width:
        z_len = height // 3
        return [
            (slice(0, z_len), slice(None)),
            (slice(z_len, 2 * z_len), slice(None)),
            (slice(2 * z_len, height), slice(None)),
        ]
    z_len = width // 3
    return [
        (slice(None), slice(0, z_len)),
        (slice(None), slice(z_len, 2 * z_len)),
        (slice(None), slice(2 * z_len, width)),
    ]


def _compute_ink_contrast_stats(
    gray: np.ndarray,
    image: Image.Image | None = None,
) -> tuple[float, float, float, list[tuple[float, float]], bool]:
    """Measure shadow-normalized paper-masked ink depth and visible stroke ratio.

    Normalizes uneven illumination first and restricts percentiles to the bright
    paper region (eroding border halos) so dark table backgrounds do not masquerade
    as legible text ink on a faded receipt. Also reports whether the bright region
    is highly saturated (e.g., colorful sky/scenery rather than receipt paper).
    """
    h, w = gray.shape[:2]
    max_side = max(h, w)
    if max_side > 1600:
        scale = 1600.0 / max_side
        work = cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = gray

    flat, bg = normalize_illumination(work)
    ink = 255.0 - flat.astype(np.float32)

    thresh = float(np.percentile(bg, 85)) * 0.78
    mask = (bg >= thresh).astype(np.uint8)
    eroded = cv2.erode(mask, np.ones((11, 11), np.uint8))

    if float(np.mean(eroded)) >= 0.05:
        active_mask = eroded > 0
        selected = ink[active_mask]
    elif float(np.mean(mask)) >= 0.05:
        active_mask = mask > 0
        selected = ink[active_mask]
    else:
        active_mask = np.ones_like(ink, dtype=bool)
        selected = ink.ravel()

    is_colorful_region = False
    if image is not None and image.mode in ("RGB", "RGBA"):
        rgb = np.array(image.convert("RGB"))
        if max_side > 1600:
            rgb = cv2.resize(
                rgb,
                (work.shape[1], work.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        region_sat = float(np.mean(hsv[:, :, 1][active_mask]))
        is_colorful_region = region_sat > settings.image_quality_max_saturation_mean

    p99_ink = float(np.percentile(selected, 99))
    p95_ink = float(np.percentile(selected, 95))
    ink_ratio = float(np.mean(selected >= 40.0))

    zh, zw = ink.shape[:2]
    zone_stats: list[tuple[float, float]] = []
    if min(zh, zw) >= 150:
        for row_sl, col_sl in _three_zone_slices(zh, zw):
            z_ink = ink[row_sl, col_sl]
            z_mask = active_mask[row_sl, col_sl]
            if z_ink.size == 0:
                continue
            z_sel = z_ink[z_mask] if float(np.mean(z_mask)) >= 0.05 else z_ink.ravel()
            zone_stats.append((float(np.percentile(z_sel, 99)), float(np.mean(z_sel >= 40.0))))

    return p99_ink, p95_ink, ink_ratio, zone_stats, is_colorful_region


def _is_faded_metrics(
    p99_ink: float,
    p95_ink: float,
    ink_ratio: float,
    zone_stats: list[tuple[float, float]],
    *,
    is_colorful_region: bool = False,
) -> bool:
    """Return True when whole-image ink contrast is severely faded/washed out."""
    if is_colorful_region:
        return False
    if p99_ink < settings.image_quality_faded_max_ink_depth:
        return True
    if (
        p95_ink < settings.image_quality_faded_p95_ink_depth
        and ink_ratio < settings.image_quality_faded_min_ink_ratio
    ):
        return True
    # Catch regionally faded thermal receipts with weak body contrast and at least one
    # severely faded zone (e.g., washed-out totals while a header/logo remains darker).
    if p95_ink < 45.0 and any(
        z_p99 < settings.image_quality_faded_zone_ink_depth and z_ratio < 0.02
        for z_p99, z_ratio in zone_stats
    ):
        return True
    # Only fall back to multi-zone fading when global peak contrast is already weak,
    # avoiding false positives on colorful scenes or sparse crisp receipts with blank margins.
    if p99_ink >= 90.0:
        return False
    faded_zones = sum(
        1
        for z_p99, z_ratio in zone_stats
        if z_p99 < settings.image_quality_faded_max_ink_depth
        or z_ratio < settings.image_quality_faded_min_ink_ratio
    )
    return faded_zones >= 2


def is_faded_image(image: Image.Image) -> bool:
    """Return True if an image fails the pre-OCR whole-image faded check."""
    gray = to_gray(image)
    if float(np.mean(gray)) < settings.image_quality_brightness_floor:
        return False
    p99_ink, p95_ink, ink_ratio, zone_stats, is_colorful = _compute_ink_contrast_stats(
        gray, image=image
    )
    return _is_faded_metrics(
        p99_ink, p95_ink, ink_ratio, zone_stats, is_colorful_region=is_colorful
    )


def has_faded_ink_signal(image: Image.Image) -> bool:
    """Return True if the image or any vertical third shows washed-out/faded ink.

    Used post-OCR (Tier 2) after CLAHE recovery so partially faded receipts that fail
    to yield critical fields or legible text can be rejected as ``faded``.
    """
    gray = to_gray(image)
    if float(np.mean(gray)) < settings.image_quality_brightness_floor:
        return False
    p99_ink, p95_ink, ink_ratio, zone_stats, is_colorful = _compute_ink_contrast_stats(
        gray, image=image
    )
    if is_colorful:
        return False
    if _is_faded_metrics(
        p99_ink, p95_ink, ink_ratio, zone_stats, is_colorful_region=is_colorful
    ):
        return True
    if p99_ink < settings.image_quality_faded_zone_ink_depth and ink_ratio < 0.02:
        return True
    return any(
        z_p99 < settings.image_quality_faded_zone_ink_depth and z_ratio < 0.02
        for z_p99, z_ratio in zone_stats
    )


def check_image_quality(image: Image.Image) -> QualityCheckResult:
    """Run size, brightness, faded-ink, blur, and non-receipt checks on a PIL receipt image.

    Checks are performed in priority order:
    size -> too_dark -> faded -> blurry -> not_a_receipt.

    Faded ink is evaluated before blur so bright washed-out thermal receipts with weak
    edges are classified as ``faded`` rather than ``blurry``.
    """
    raw_w, raw_h = image.size
    short_side, long_side = min(raw_w, raw_h), max(raw_w, raw_h)
    min_short = settings.image_quality_min_short_side
    min_long = settings.image_quality_min_long_side
    if short_side < min_short or long_side < min_long:
        reason = (
            f"Image dimensions ({raw_w}x{raw_h}px) are too small for accurate OCR. "
            f"Minimum required is {min_short}px on the short side and "
            f"{min_long}px on the long side."
        )
        logger.info("Quality check failed [too_small]: %s", reason)
        return QualityCheckResult(
            passed=False,
            rejection_code="too_small",
            rejection_reason=reason,
            resolution=(raw_w, raw_h),
        )

    gray = to_gray(image)

    # 2. Brightness / Darkness check via mean pixel intensity (rotation-invariant)
    brightness = float(np.mean(gray))
    brightness_floor = settings.image_quality_brightness_floor
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < brightness_floor:
        reason = (
            "Image is too dark or underexposed for clear OCR reading. "
            "Please capture under better lighting conditions."
        )
        logger.info("Quality check failed [too_dark]: mean %.2f < floor %.2f", brightness, brightness_floor)
        return QualityCheckResult(
            passed=False,
            rejection_code="too_dark",
            rejection_reason=reason,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            resolution=(raw_w, raw_h),
        )

    # 3. Faded ink check (evaluated before blur so faint thermal paper isn't mislabeled blurry)
    p99_ink, p95_ink, ink_ratio, zone_stats, is_colorful = _compute_ink_contrast_stats(
        gray, image=image
    )
    if _is_faded_metrics(
        p99_ink, p95_ink, ink_ratio, zone_stats, is_colorful_region=is_colorful
    ):
        logger.info(
            "Quality check failed [faded]: p99_ink=%.1f, p95_ink=%.1f, ink_ratio=%.4f",
            p99_ink, p95_ink, ink_ratio,
        )
        return QualityCheckResult(
            passed=False,
            rejection_code="faded",
            rejection_reason=FADED_REASON,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            resolution=(raw_w, raw_h),
            contrast_score=round(p99_ink, 2),
        )

    # 4. Blur check via Laplacian variance (rotation-invariant)
    blur_thresh = settings.image_quality_blur_threshold

    # Zone-aware blur check for tall receipts: camera tilt can cause one end to be out-of-focus
    is_tall = max(raw_w, raw_h) >= 2.0 * min(raw_w, raw_h)
    min_zone_score = blur_score
    if is_tall and min(raw_w, raw_h) >= 150:
        zones = [gray[row_sl, col_sl] for row_sl, col_sl in _three_zone_slices(gray.shape[0], gray.shape[1])]
        zone_scores = [float(cv2.Laplacian(z, cv2.CV_64F).var()) for z in zones if z.size > 0]
        if zone_scores:
            min_zone_score = min(zone_scores)

    if blur_score < blur_thresh or (is_tall and min_zone_score < blur_thresh * 0.55):
        reason = BLURRY_REASON
        effective_score = min(blur_score, min_zone_score)
        logger.info(
            "Quality check failed [blurry]: score %.2f (min_zone=%.2f) < threshold %.2f",
            blur_score, min_zone_score, blur_thresh,
        )
        return QualityCheckResult(
            passed=False,
            rejection_code="blurry",
            rejection_reason=reason,
            blur_score=round(effective_score, 2),
            brightness=round(brightness, 2),
            resolution=(raw_w, raw_h),
            contrast_score=round(p99_ink, 2),
        )

    # 5. Orientation and non-receipt visual & selfie check
    upright = correct_orientation(image)
    width, height = upright.size
    resolution = (width, height)
    upright_gray = to_gray(upright) if upright is not image else gray

    if settings.image_quality_check_not_receipt and _check_non_receipt_heuristics(
        upright,
        upright_gray,
        sat_thresh=settings.image_quality_max_saturation_mean,
        min_density=settings.image_quality_min_text_edge_density,
    ):
        logger.info("Quality check failed [not_a_receipt]: %s", NOT_A_RECEIPT_REASON)
        return QualityCheckResult(
            passed=False,
            rejection_code="not_a_receipt",
            rejection_reason=NOT_A_RECEIPT_REASON,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            resolution=resolution,
            contrast_score=round(p99_ink, 2),
        )

    return QualityCheckResult(
        passed=True,
        blur_score=round(blur_score, 2),
        brightness=round(brightness, 2),
        resolution=resolution,
        contrast_score=round(p99_ink, 2),
    )
