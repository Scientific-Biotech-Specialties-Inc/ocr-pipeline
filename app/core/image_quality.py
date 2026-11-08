"""Pre-OCR image quality evaluation.

Evaluates blur, brightness, and resolution using OpenCV heuristics before
running the resource-heavy OCR pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.core.preprocessing import correct_orientation, to_gray

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Structured output from check_image_quality()."""

    passed: bool
    rejection_code: str | None = None       # "blurry" | "too_dark" | "too_small" | None
    rejection_reason: str | None = None     # Human-readable sentence for frontend
    blur_score: float = 0.0                    # Laplacian variance (higher = sharper)
    brightness: float = 0.0                    # Mean pixel value 0-255
    resolution: tuple[int, int] = (0, 0)       # (width, height) after orientation correction


NOT_A_RECEIPT_REASON = (
    "This image does not appear to be a receipt (e.g., a photo, selfie, or non-receipt document). "
    "Please upload a clear photo of an official receipt or invoice, or continue manually if this is a valid expense document."
)

BLURRY_REASON = (
    "The uploaded image is too blurry to read the receipt details accurately. "
    "Please upload a clearer, focused photo or continue to enter the details manually."
)

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


def check_image_quality(image: Image.Image) -> QualityCheckResult:
    """Run size, blur, and brightness checks on a PIL receipt image.

    Checks are performed in priority order: size -> blur -> brightness -> non-receipt.

    Size, blur, and brightness are rotation-invariant and evaluated first before
    running the slower orientation check.
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

    # 2. Blur check via Laplacian variance (rotation-invariant)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_thresh = settings.image_quality_blur_threshold

    # Zone-aware blur check for tall receipts: camera tilt can cause one end to be out-of-focus
    is_tall = max(raw_w, raw_h) >= 2.0 * min(raw_w, raw_h)
    min_zone_score = blur_score
    if is_tall and min(raw_w, raw_h) >= 150:
        if raw_h >= raw_w:
            z_len = gray.shape[0] // 3
            zones = [gray[0:z_len, :], gray[z_len:2*z_len, :], gray[2*z_len:, :]]
        else:
            z_len = gray.shape[1] // 3
            zones = [gray[:, 0:z_len], gray[:, z_len:2*z_len], gray[:, 2*z_len:]]
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
            resolution=(raw_w, raw_h),
        )

    # 3. Brightness / Darkness check via mean pixel intensity (rotation-invariant)
    brightness = float(np.mean(gray))
    brightness_floor = settings.image_quality_brightness_floor
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

    # 4. Orientation and non-receipt visual & selfie check
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
        )

    return QualityCheckResult(
        passed=True,
        blur_score=round(blur_score, 2),
        brightness=round(brightness, 2),
        resolution=resolution,
    )
