"""Unit tests for pre-OCR image quality checks."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.image_quality import check_image_quality

RECEIPTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "receipts"


def create_test_image(width: int = 400, height: int = 600, color: int = 200) -> Image.Image:
    """Helper to create a solid gray image with high-contrast text pattern."""
    arr = np.full((height, width), color, dtype=np.uint8)
    # Add high-contrast lines to simulate sharp text for Laplacian test
    for y in range(50, height - 50, 40):
        arr[y : y + 5, 50 : width - 50] = 0
        arr[y + 10 : y + 15, 50 : width - 50] = 255
    return Image.fromarray(arr)


def test_sharp_image_passes():
    img = create_test_image(400, 600)
    result = check_image_quality(img)
    assert result.passed is True
    assert result.rejection_code is None
    assert result.blur_score > 80.0


def test_tiny_image_rejected():
    img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    result = check_image_quality(img)
    assert result.passed is False
    assert result.rejection_code == "too_small"
    assert "too small" in result.rejection_reason.lower()


def test_narrow_tall_receipt_passes_size_gate():
    """Thermal scans are often ~225px wide; fixed min-width=300 used to reject them."""
    img = create_test_image(225, 571)
    result = check_image_quality(img)
    assert result.passed is True
    assert result.rejection_code is None


def test_landscape_receipt_passes_size_gate():
    """Size uses short/long sides, so a 580x356 landscape crop is not rejected."""
    img = create_test_image(580, 356)
    result = check_image_quality(img)
    assert result.passed is True


def test_dark_image_rejected():
    # Large enough dimensions, solid dark gray (mean < 40) with no text
    arr = np.full((600, 400), 20, dtype=np.uint8)
    img = Image.fromarray(arr)
    result = check_image_quality(img)
    assert result.passed is False
    # Size passes, but blur or darkness fails
    assert result.rejection_code in ("blurry", "too_dark")


def test_blurry_image_rejected():
    # Large uniform gray image with no sharp edges (Laplacian variance near 0)
    arr = np.full((600, 400), 180, dtype=np.uint8)
    img = Image.fromarray(arr)
    result = check_image_quality(img)
    assert result.passed is False
    assert result.rejection_code == "blurry"
    assert "too blurry to read the receipt details accurately" in result.rejection_reason


def test_colorful_non_document_rejected_as_not_a_receipt():
    # Sharp, bright, colorful photo (e.g., sky and grass scene with sharp horizon and sun)
    # but zero horizontal text line structures
    arr = np.zeros((600, 400, 3), dtype=np.uint8)
    arr[:300, :, :] = [100, 180, 240]  # Sky
    arr[300:, :, :] = [30, 160, 40]    # Grass
    cv2.line(arr, (0, 300), (400, 300), (0, 0, 0), 5)
    cv2.circle(arr, (200, 150), 50, (255, 255, 0), -1)
    img = Image.fromarray(arr, mode="RGB")
    result = check_image_quality(img)
    assert result.passed is False
    assert result.rejection_code == "not_a_receipt"
    assert "does not appear to be a receipt" in result.rejection_reason


def test_receipt_11_passes_quality_gate():
    path = RECEIPTS_DIR / "receipt 11.jpg"
    if not path.exists():
        return
    result = check_image_quality(Image.open(path))
    assert result.passed is True, result.rejection_reason


def test_receipt_15_orange_table_passes_quality_gate():
    path = RECEIPTS_DIR / "receipt 15.jpeg"
    if not path.exists():
        return
    result = check_image_quality(Image.open(path))
    assert result.passed is True, result.rejection_reason
