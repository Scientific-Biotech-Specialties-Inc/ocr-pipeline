"""Image preparation for OCR.

Provides single best-effort OpenCV image normalization for receipt OCR:
1. orientation      - must run first; every later stage assumes upright text
2. perspective      - guarded; declines to warp when no confident document quad
3. shadow removal   - divide by a blurred background estimate ("flatbed" look)
4. upscale          - ensures text reaches sufficient x-height
5. denoise          - bilateral, which preserves thin thermal strokes
6. deskew           - small residual rotation left over after warping
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

TARGET_LONG_EDGE = 1600
MAX_UPSCALE = 3.0
MAX_DESKEW_DEGREES = 15.0
MIN_QUAD_AREA_RATIO = 0.50

# Tesseract's orientation confidence is unbounded; genuine rotations typically
# score well above 1, while noise sits near zero. Measured false positives on
# receipts came in around 0.1-0.2.
MIN_ORIENTATION_CONFIDENCE = 2.0


# ---------------------------------------------------------------------------
# individual stages
# ---------------------------------------------------------------------------

def to_gray(image: Image.Image) -> np.ndarray:
    array = np.array(image.convert("RGB"))[:, :, ::-1]
    return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)


def correct_orientation(image: Image.Image) -> Image.Image:
    """Rotate to upright using Tesseract's orientation detection.

    Only acts on a *confident* detection. Tesseract's OSD reports a rotation
    regardless of how weak the evidence is, and on receipts - short lines, sparse
    text, large blank areas - it guesses badly. An upright synthetic slip was
    reported as 180 degrees with ``orientation_conf`` 0.12 (and its script
    identified as Arabic), and rotating on that reading turned a perfectly
    readable receipt upside down, taking every field with it.

    Ignoring rotation when unsure is the safe failure: a wrongly upright page
    still reads, whereas a wrongly inverted one reads as nothing.

    Requires the ``osd`` traineddata; if unavailable the image is returned
    unchanged rather than failing the job.
    """
    cached = getattr(image, "_ocr_upright", None)
    if cached is not None:
        return cached

    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotate = int(osd.get("rotate", 0)) % 360
        confidence = float(osd.get("orientation_conf", 0.0))
    except Exception as exc:  # pragma: no cover - depends on tessdata presence # noqa: BLE001
        logger.debug("Orientation detection unavailable: %s", exc)
        try:
            image._ocr_upright = image
        except AttributeError:
            pass
        return image

    if not rotate or confidence < MIN_ORIENTATION_CONFIDENCE:
        if rotate and confidence < MIN_ORIENTATION_CONFIDENCE:
            logger.debug(
                "Ignoring %s deg orientation: confidence %.2f below %.2f",
                rotate, confidence, MIN_ORIENTATION_CONFIDENCE,
            )
        try:
            image._ocr_upright = image
        except AttributeError:
            pass
        return image

    # PIL rotates counter-clockwise; OSD reports the clockwise correction.
    upright = image.rotate(-rotate, expand=True, fillcolor="white")
    try:
        image._ocr_upright = upright
        upright._ocr_upright = upright
    except AttributeError:
        pass
    return upright


def correct_perspective(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Flatten a photographed receipt to a head-on view.

    Declines when no convincing document quadrilateral is found. Several samples
    are cropped fragments with no visible border, and warping on a false
    detection destroys the image - a slightly tilted receipt reads better than a
    confidently mangled one.
    """
    height, width = gray.shape[:2]
    scale = 800.0 / max(height, width)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else gray.copy()

    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray, False

    small_area = small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        perimeter = cv2.arcLength(contour, True)
        quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        if cv2.contourArea(quad) < MIN_QUAD_AREA_RATIO * small_area:
            continue
        points = quad.reshape(4, 2).astype(np.float32) / (scale if scale < 1 else 1.0)
        return _warp(gray, _order_corners(points)), True

    return gray, False


def _order_corners(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)]   # top-left
    ordered[2] = points[np.argmax(total)]   # bottom-right
    diff = np.diff(points, axis=1).ravel()
    ordered[1] = points[np.argmin(diff)]    # top-right
    ordered[3] = points[np.argmax(diff)]    # bottom-left
    return ordered


def _warp(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return gray
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def remove_shadow(gray: np.ndarray, kernel: int = 41) -> np.ndarray:
    """Normalise uneven illumination by dividing out a blurred background.

    Division rather than subtraction is what produces the flat white page;
    subtraction leaves grey blotches where the shadow was.
    """
    kernel = kernel if kernel % 2 else kernel + 1
    background = cv2.medianBlur(gray, kernel)
    normalized = cv2.divide(gray, background, scale=255)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def upscale(gray: np.ndarray) -> np.ndarray:
    """Scale up small images so character height reaches Tesseract's sweet spot."""
    height, width = gray.shape[:2]

    # For tall/narrow receipts (aspect ratio height/width >= 2.0), scale up based on width
    # so text characters reach sufficient x-height for accurate Tesseract reading.
    if height > 0 and width > 0 and (height / width) >= 2.0 and width < 750:
        factor = min(750.0 / width, MAX_UPSCALE)
        return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    longest = max(height, width)
    if longest >= TARGET_LONG_EDGE:
        return gray
    factor = min(TARGET_LONG_EDGE / longest, MAX_UPSCALE)
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def slice_tall_image(image: Image.Image, max_aspect_ratio: float = 2.2) -> list[Image.Image]:
    """Slice a tall receipt into overlapping vertical regions if aspect ratio exceeds threshold."""
    width, height = image.size
    if width <= 0 or height <= 0:
        return [image]

    aspect = height / width
    if aspect <= max_aspect_ratio:
        return [image]

    num_slices = 2 if aspect <= 3.5 else 3
    slice_height = int(height / (num_slices - 0.2))  # Overlapping slices
    overlap = int(slice_height * 0.2)

    slices: list[Image.Image] = []
    for i in range(num_slices):
        start_y = max(0, i * (slice_height - overlap))
        end_y = min(height, start_y + slice_height)
        sub_img = image.crop((0, start_y, width, end_y))
        slices.append(sub_img)

    return slices


def denoise(gray: np.ndarray) -> np.ndarray:
    """Bilateral filter: smooths noise while keeping stroke edges intact."""
    return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Correct small residual rotation using the dominant text angle."""
    inverted = cv2.bitwise_not(gray)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 50:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.3 or abs(angle) > MAX_DESKEW_DEGREES:
        return gray

    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Single best-effort rendering for OCR fallback.

    Returns the shadow-flattened, denoised, deskewed grayscale image
    designed for Tesseract single-pass extraction.
    """
    cached = getattr(image, "_ocr_preprocessed", None)
    if cached is not None:
        return cached

    upright = correct_orientation(image)
    gray = to_gray(upright)
    warped, did_warp = correct_perspective(gray)
    base = upscale(warped if did_warp else gray)
    flattened = remove_shadow(base)
    cleaned = denoise(flattened)
    result = Image.fromarray(cleaned)

    try:
        image._ocr_preprocessed = result
        image._ocr_flattened = flattened
        image._ocr_cleaned = cleaned
        if upright is not image:
            upright._ocr_preprocessed = result
            upright._ocr_flattened = flattened
            upright._ocr_cleaned = cleaned
    except AttributeError:
        pass

    return result


def prepare_region_crop(
    image: Image.Image,
    y_start_ratio: float,
    y_end_ratio: float,
) -> tuple[Image.Image, int]:
    """Crop a vertical region of an upright receipt image and enhance contrast for faded text.

    Used for targeted recovery of headers (vendor/TIN) or footers (total/subtotal).
    Returns (preprocessed_crop_image, y_offset_px).
    """
    upright = correct_orientation(image)
    w, h = upright.size
    y_start = max(0, int(h * y_start_ratio))
    y_end = min(h, int(h * y_end_ratio))
    if y_end <= y_start:
        y_end = min(h, y_start + 100)

    cropped = upright.crop((0, y_start, w, y_end))
    gray = to_gray(cropped)
    crop_h, crop_w = gray.shape[:2]
    if crop_w < 800 and crop_h > 0:
        factor = min(800.0 / crop_w, 2.0)
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    flattened = remove_shadow(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(flattened)
    cleaned = denoise(enhanced)
    return Image.fromarray(cleaned), y_start


