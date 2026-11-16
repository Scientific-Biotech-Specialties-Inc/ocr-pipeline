"""Unit tests for pre-OCR image quality checks."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.image_quality import check_image_quality, has_faded_ink_signal, is_faded_image

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
    # Sharp text pattern degraded with heavy Gaussian blur: still has ink contrast, fails blur
    sharp_arr = np.array(create_test_image(400, 600))
    blurred_arr = cv2.GaussianBlur(sharp_arr, (25, 25), 0)
    img = Image.fromarray(blurred_arr)
    result = check_image_quality(img)
    assert result.passed is False
    assert result.rejection_code == "blurry"
    assert "too blurry to read the receipt details accurately" in result.rejection_reason


def test_faded_washed_out_receipt_rejected_as_faded():
    # Bright thermal paper with extremely faint text (~8% ink contrast)
    arr = np.full((600, 400), 235, dtype=np.uint8)
    for y in range(50, 550, 40):
        arr[y : y + 5, 50:350] = 215
    img = Image.fromarray(arr)
    result = check_image_quality(img)
    assert result.passed is False
    assert result.rejection_code == "faded"
    assert "too faded or washed out" in result.rejection_reason.lower()
    assert result.contrast_score < 55.0


def test_faded_receipt_on_dark_table_rejected_as_faded():
    # Dark textured desk border around a faded white thermal receipt must not mask faded ink
    rng = np.random.default_rng(7)
    arr = rng.integers(30, 80, size=(700, 400), dtype=np.uint8)
    paper = np.full((500, 220), 220, dtype=np.uint8)
    for y in range(40, 460, 25):
        paper[y : y + 3, 20:200] = 205
    arr[100:600, 90:310] = paper
    img = Image.fromarray(arr)
    result = check_image_quality(img)
    assert result.passed is False
    assert result.rejection_code == "faded"


def test_faded_white_receipt_on_saturated_colored_table_rejected_as_faded():
    # Bright orange desk border must not mark white faded receipt paper as a colorful non-receipt
    arr = np.zeros((700, 400, 3), dtype=np.uint8)
    arr[:, :] = [230, 110, 20]  # Saturated orange desk
    paper = np.full((500, 220, 3), 235, dtype=np.uint8)
    for y in range(40, 460, 25):
        paper[y : y + 3, 20:200] = [215, 215, 215]
    arr[100:600, 90:310] = paper
    img = Image.fromarray(arr, mode="RGB")
    result = check_image_quality(img)
    assert result.passed is False
    assert result.rejection_code == "faded"


def test_partially_faded_receipt_passes_tier1_but_triggers_tier2_signal():
    # Top 2/3 sharp ink, bottom 1/3 sharp-edged but faint/faded strokes (allow OCR + CLAHE recovery)
    img = create_test_image(400, 900, color=225)
    arr = np.array(img)
    arr[600:, :] = 225
    for y in range(620, 870, 35):
        arr[y : y + 4, 40:360] = 195
    partial_img = Image.fromarray(arr)
    res = check_image_quality(partial_img)
    assert res.passed is True
    assert is_faded_image(partial_img) is False
    assert has_faded_ink_signal(partial_img) is True


KNOWN_FADED_SAMPLES = {"receipt 40.jpg"}


def test_all_sample_receipts_pass_faded_check():
    if not RECEIPTS_DIR.exists():
        return
    for path in sorted(RECEIPTS_DIR.glob("*.*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = Image.open(path)
        assert is_faded_image(img) is False, f"{path.name} falsely flagged by is_faded_image"
        if path.name in KNOWN_FADED_SAMPLES:
            assert has_faded_ink_signal(img) is True, f"{path.name} expected to trigger faded signal"
        else:
            assert has_faded_ink_signal(img) is False, f"{path.name} falsely flagged by has_faded_ink_signal"


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
