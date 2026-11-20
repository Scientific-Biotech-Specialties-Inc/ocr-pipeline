"""Unit tests for pre-OCR image quality checks."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.core.image_quality import (
    BLURRY_REASON,
    FADED_REASON,
    NOT_A_RECEIPT_REASON,
    QUALITY_FAILED_REASON,
    UNREADABLE_REASON,
    QualityCheckResult,
    _detect_prominent_face,
    build_unreadable_reason,
    check_image_quality,
    has_faded_ink_signal,
    is_faded_image,
    select_quality_failure,
)
from app.core.pipeline import load_images
from app.core.preprocessing import apply_exif_orientation

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
    assert "continue" not in result.rejection_reason.lower()
    assert "manually" not in result.rejection_reason.lower()


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
        if path.name in KNOWN_FADED_SAMPLES:
            assert is_faded_image(img) is True, f"{path.name} expected to fail Tier 1 is_faded_image"
            assert has_faded_ink_signal(img) is True, f"{path.name} expected to trigger faded signal"
            res = check_image_quality(img)
            assert res.passed is False, f"{path.name} expected to fail check_image_quality"
            assert res.rejection_code == "faded", f"{path.name} expected rejection_code='faded', got {res.rejection_code}"
        else:
            assert is_faded_image(img) is False, f"{path.name} falsely flagged by is_faded_image"
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
    assert "continue" not in result.rejection_reason.lower()
    assert "manually" not in result.rejection_reason.lower()


def test_rejection_reasons_do_not_offer_manual_override():
    for reason in (
        BLURRY_REASON,
        NOT_A_RECEIPT_REASON,
        FADED_REASON,
        UNREADABLE_REASON,
        QUALITY_FAILED_REASON,
    ):
        assert "continue" not in reason.lower()
        assert "manually" not in reason.lower()


def test_build_unreadable_reason_variants():
    reason_faded_all = build_unreadable_reason({}, is_faded=True)
    assert "too faded or washed out" in reason_faded_all.lower()
    assert "Total Amount, Transaction Date, Merchant Name" in reason_faded_all

    reason_faded_partial = build_unreadable_reason(
        {"vendor_name": "ACME", "transaction_date": "2026-03-01", "total_amount": None},
        is_faded=True,
    )
    assert "Total Amount" in reason_faded_partial
    assert "Transaction Date" not in reason_faded_partial
    assert "Merchant Name" not in reason_faded_partial

    reason_faded_none_missing = build_unreadable_reason(
        {"vendor_name": "ACME", "transaction_date": "2026-03-01", "total_amount": 100.0},
        is_faded=True,
    )
    assert reason_faded_none_missing == FADED_REASON

    reason_unreadable_both = build_unreadable_reason(
        {"vendor_name": "ACME", "total_amount": None},
        is_faded=False,
        is_illegible=True,
    )
    assert "damaged, incomplete, or too unreadable" in reason_unreadable_both.lower()
    assert "Total Amount, Transaction Date" in reason_unreadable_both
    assert "Merchant Name" not in reason_unreadable_both
    assert "illegible" in reason_unreadable_both.lower()

    reason_unreadable_illegible_only = build_unreadable_reason(
        {"vendor_name": "ACME", "transaction_date": "2026-03-01", "total_amount": 100.0},
        is_faded=False,
        is_illegible=True,
    )
    assert "damaged, incomplete, or too unreadable" in reason_unreadable_illegible_only.lower()
    assert "illegible" in reason_unreadable_illegible_only.lower()

    reason_unreadable_fallback = build_unreadable_reason(
        {"vendor_name": "ACME", "transaction_date": "2026-03-01", "total_amount": 100.0},
        is_faded=False,
        is_illegible=False,
    )
    assert reason_unreadable_fallback == UNREADABLE_REASON


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


def test_undersized_colorful_non_receipt_prioritizes_not_a_receipt():
    arr = np.zeros((180, 160, 3), dtype=np.uint8)
    arr[:90, :, :] = [100, 180, 240]
    arr[90:, :, :] = [30, 160, 40]
    cv2.line(arr, (0, 90), (160, 90), (0, 0, 0), 3)
    cv2.circle(arr, (80, 45), 20, (255, 255, 0), -1)
    img = Image.fromarray(arr, mode="RGB")

    result = check_image_quality(img)
    assert result.passed is False
    assert result.rejection_code == "not_a_receipt"
    assert "too_small" in result.detected_issues
    assert "not_a_receipt" in result.detected_issues


def test_tiny_solid_black_records_detected_issues_and_picks_too_small():
    img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    result = check_image_quality(img)
    assert result.passed is False
    assert result.rejection_code == "too_small"
    assert {"too_small", "too_dark", "blurry"}.issubset(result.detected_issues)
    assert "not_a_receipt" not in result.detected_issues


def test_dark_and_faded_suppress_blurry_while_recording_detected_issues():
    dark_img = Image.fromarray(np.full((600, 400), 20, dtype=np.uint8))
    dark_res = check_image_quality(dark_img)
    assert dark_res.passed is False
    assert dark_res.rejection_code == "too_dark"
    assert "too_dark" in dark_res.detected_issues
    assert "blurry" in dark_res.detected_issues

    faded_arr = np.full((600, 400), 235, dtype=np.uint8)
    for y in range(50, 550, 40):
        faded_arr[y : y + 5, 50:350] = 215
    faded_blur_img = Image.fromarray(cv2.GaussianBlur(faded_arr, (15, 15), 0))
    faded_res = check_image_quality(faded_blur_img)
    assert faded_res.passed is False
    assert faded_res.rejection_code == "faded"
    assert "faded" in faded_res.detected_issues
    assert "blurry" in faded_res.detected_issues


def test_all_four_gate_groups_failed_returns_quality_failed():
    # Undersized (160x180), dark (mean=14), blurry (lap=0), colorful non-receipt (sat=219, dens=0)
    dark_colorful = Image.fromarray(np.full((180, 160, 3), [35, 5, 5], dtype=np.uint8), mode="RGB")
    res_dark = check_image_quality(dark_colorful)
    assert res_dark.passed is False
    assert res_dark.rejection_code == "quality_failed"
    assert res_dark.rejection_reason == QUALITY_FAILED_REASON
    assert {"too_small", "too_dark", "blurry", "not_a_receipt"}.issubset(res_dark.detected_issues)

    # Also verify faded variant when all 4 gate groups fail
    faded_small = np.full((180, 160), 235, dtype=np.uint8)
    for y in range(20, 160, 30):
        faded_small[y : y + 4, 20:140] = 220
    faded_small_img = Image.fromarray(cv2.GaussianBlur(faded_small, (11, 11), 0))
    with patch("app.core.image_quality._detect_prominent_face", return_value=True):
        res_faded = check_image_quality(faded_small_img)
    assert res_faded.passed is False
    assert res_faded.rejection_code == "quality_failed"
    assert res_faded.rejection_reason == QUALITY_FAILED_REASON
    assert {"too_small", "faded", "blurry", "not_a_receipt"}.issubset(res_faded.detected_issues)

    # 100x100 solid-black does not trigger not_a_receipt, so it still returns too_small
    black_100 = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    res_black = check_image_quality(black_100)
    assert res_black.rejection_code == "too_small"
    assert "not_a_receipt" not in res_black.detected_issues


def _make_exif_orientation_6_jpeg_bytes() -> bytes:
    upright = create_test_image(400, 600).convert("RGB")
    stored = upright.rotate(90, expand=True)
    assert stored.size == (600, 400)
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = BytesIO()
    stored.save(buf, format="JPEG", exif=exif.tobytes(), quality=95)
    return buf.getvalue()


def test_exif_orientation_6_normalized_in_quality_and_load_images():
    jpeg_bytes = _make_exif_orientation_6_jpeg_bytes()

    raw_img = Image.open(BytesIO(jpeg_bytes))
    assert raw_img.size == (600, 400)
    normalized = apply_exif_orientation(raw_img)
    assert normalized.size == (400, 600)
    assert apply_exif_orientation(normalized).size == (400, 600)

    # Malformed or missing EXIF returns the image safely without raising
    plain_img = create_test_image(240, 480)
    assert apply_exif_orientation(plain_img).size == (240, 480)
    corrupt_img = create_test_image(240, 480)
    corrupt_img.info["exif"] = b"corrupt-exif-header"
    assert apply_exif_orientation(corrupt_img).size == (240, 480)

    quality_res = check_image_quality(Image.open(BytesIO(jpeg_bytes)))
    assert quality_res.passed is True
    assert quality_res.resolution == (400, 600)

    loaded = load_images(jpeg_bytes, "https://example.test/rotated.jpg")
    assert len(loaded) == 1
    assert loaded[0].size == (400, 600)


def test_detect_prominent_face_checks_profiles_and_small_image_guard():
    # Guard when min(h, w) < 60
    assert _detect_prominent_face(np.zeros((59, 120), dtype=np.uint8)) is False
    assert _detect_prominent_face(np.zeros((120, 59), dtype=np.uint8)) is False

    # Asymmetric grayscale image so unflipped and flipped arrays differ
    gray = np.zeros((200, 200), dtype=np.uint8)
    gray[:, :100] = 40
    gray[:, 100:] = 210
    flipped_gray = cv2.flip(gray, 1)

    class _NoFrontalCascade:
        def empty(self) -> bool:
            return False

        def detectMultiScale(self, *_args, **_kwargs):
            return ()

    class _RecordingProfileCascade:
        def __init__(self, detect_on_flipped: bool, box: tuple[int, int, int, int]):
            self.detect_on_flipped = detect_on_flipped
            self.box = box
            self.seen: list[np.ndarray] = []
            self.min_neighbors_calls: list[int | None] = []

        def empty(self) -> bool:
            return False

        def detectMultiScale(self, img_gray: np.ndarray, *args, **kwargs):
            self.seen.append(img_gray.copy())
            mn = kwargs.get("minNeighbors", args[1] if len(args) >= 2 else None)
            self.min_neighbors_calls.append(mn)
            is_flipped = np.array_equal(img_gray, flipped_gray)
            if is_flipped == self.detect_on_flipped:
                return [self.box]
            return ()

    # Right profile detected on flipped array (60x60 / 200x200 = 0.09 >= 0.06)
    right_profile = _RecordingProfileCascade(detect_on_flipped=True, box=(10, 10, 60, 60))
    with (
        patch("app.core.image_quality._get_face_cascade", return_value=_NoFrontalCascade()),
        patch("app.core.image_quality._get_profile_face_cascade", return_value=right_profile),
    ):
        assert _detect_prominent_face(gray, max_face_ratio=0.04) is True
    assert len(right_profile.seen) == 2
    assert right_profile.min_neighbors_calls == [9, 9]
    assert np.array_equal(right_profile.seen[0], gray)
    assert np.array_equal(right_profile.seen[1], flipped_gray)

    # Left profile detected on unflipped array (60x60 / 200x200 = 0.09 >= 0.06)
    left_profile = _RecordingProfileCascade(detect_on_flipped=False, box=(10, 10, 60, 60))
    with (
        patch("app.core.image_quality._get_face_cascade", return_value=_NoFrontalCascade()),
        patch("app.core.image_quality._get_profile_face_cascade", return_value=left_profile),
    ):
        assert _detect_prominent_face(gray, max_face_ratio=0.04) is True
    assert len(left_profile.seen) == 1
    assert left_profile.min_neighbors_calls == [9]
    assert np.array_equal(left_profile.seen[0], gray)

    # Profile box at ~0.046 ratio (43x43 / 200x200 = 0.046225) passes frontal 0.04
    # threshold if frontal, but must NOT trigger profile detection (< 0.06).
    below_profile_thresh = _RecordingProfileCascade(detect_on_flipped=True, box=(10, 10, 43, 43))
    with (
        patch("app.core.image_quality._get_face_cascade", return_value=_NoFrontalCascade()),
        patch("app.core.image_quality._get_profile_face_cascade", return_value=below_profile_thresh),
    ):
        assert _detect_prominent_face(gray, max_face_ratio=0.04) is False
    assert len(below_profile_thresh.seen) == 2
    assert below_profile_thresh.min_neighbors_calls == [9, 9]

    # End-to-end check_image_quality: ~0.046 profile box on sharp 400x600 receipt passes,
    # while >= 0.06 (120x120 / 400x600 = 0.06) triggers not_a_receipt.
    receipt_img = create_test_image(400, 600)

    class _FixedProfileCascade:
        def __init__(self, box: tuple[int, int, int, int]):
            self.box = box

        def empty(self) -> bool:
            return False

        def detectMultiScale(self, *_args, **_kwargs):
            return [self.box]

    with (
        patch("app.core.image_quality._get_face_cascade", return_value=_NoFrontalCascade()),
        patch("app.core.image_quality._get_profile_face_cascade", return_value=_FixedProfileCascade((10, 10, 105, 105))),
    ):
        res_below = check_image_quality(receipt_img)
    assert res_below.passed is True
    assert res_below.rejection_code is None

    with (
        patch("app.core.image_quality._get_face_cascade", return_value=_NoFrontalCascade()),
        patch("app.core.image_quality._get_profile_face_cascade", return_value=_FixedProfileCascade((10, 10, 120, 120))),
    ):
        res_above = check_image_quality(receipt_img)
    assert res_above.passed is False
    assert res_above.rejection_code == "not_a_receipt"


def test_select_quality_failure_priority():
    passing = QualityCheckResult(passed=True)
    blurry = QualityCheckResult(passed=False, rejection_code="blurry", rejection_reason=BLURRY_REASON)
    too_dark = QualityCheckResult(passed=False, rejection_code="too_dark", rejection_reason="dark")
    faded = QualityCheckResult(passed=False, rejection_code="faded", rejection_reason=FADED_REASON)
    quality_failed = QualityCheckResult(
        passed=False, rejection_code="quality_failed", rejection_reason=QUALITY_FAILED_REASON
    )
    not_a_receipt = QualityCheckResult(
        passed=False, rejection_code="not_a_receipt", rejection_reason=NOT_A_RECEIPT_REASON
    )

    assert select_quality_failure([passing, blurry, faded, quality_failed, not_a_receipt]) is not_a_receipt
    assert select_quality_failure([passing, blurry, faded, quality_failed]) is quality_failed
    assert select_quality_failure([passing, blurry, faded]) is faded
    assert select_quality_failure([passing, too_dark, blurry]) is too_dark
    assert select_quality_failure([passing, None]) is None


def test_non_faded_sample_receipts_do_not_trigger_not_a_receipt():
    if not RECEIPTS_DIR.exists():
        return
    for path in sorted(RECEIPTS_DIR.glob("*.*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if path.name in KNOWN_FADED_SAMPLES:
            continue
        res = check_image_quality(Image.open(path))
        assert res.rejection_code != "not_a_receipt", f"{path.name} falsely rejected as not_a_receipt"
        assert "not_a_receipt" not in res.detected_issues, (
            f"{path.name} falsely flagged not_a_receipt in detected_issues: {res.detected_issues}"
        )
