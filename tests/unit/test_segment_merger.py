"""Unit tests for segment role scoring, auto-ordering, seam deduplication, and tall image strip tiling."""
from PIL import Image

from app.core.ocr_engine import OcrReading, Word
from app.core.segment_merger import (
    merge_segment_readings,
    order_segments,
    score_segment_density,
    split_tall_image_into_strips,
)


def _make_word(text: str, left: int = 10, top: int = 10, width: int = 50, height: int = 15, conf: float = 0.9) -> Word:
    return Word(
        text=text,
        confidence=conf,
        left=left,
        top=top,
        width=width,
        height=height,
        block=1,
        par=1,
        line=1,
    )


def _make_reading(lines_of_text: list[str], start_top: int = 10, line_height: int = 20, mean_conf: float = 0.9) -> OcrReading:
    words: list[Word] = []
    lines: list[list[Word]] = []
    current_y = start_top

    for line_idx, line_str in enumerate(lines_of_text):
        tokens = line_str.split()
        line_words: list[Word] = []
        current_x = 10
        for token in tokens:
            w = Word(
                text=token,
                confidence=mean_conf,
                left=current_x,
                top=current_y,
                width=max(len(token) * 8, 10),
                height=line_height,
                block=1,
                par=1,
                line=line_idx + 1,
            )
            line_words.append(w)
            words.append(w)
            current_x += w.width + 5
        lines.append(line_words)
        current_y += line_height + 5

    text = "\n".join(" ".join(w.text for w in ln) for ln in lines)
    return OcrReading(
        text=text,
        words=words,
        confidence=mean_conf,
        variant="test",
        psm=6,
        engine="tesseract",
        lines=lines,
    )


def test_score_segment_density_roles():
    header_text = "Puregold Price Club, Inc\nTIN: 201-277-095-00040\nTrans Date: 2026-03-30\nCashier: REYNA"
    h_score = score_segment_density(header_text, 0.95)
    assert h_score.header_score >= 5.0
    assert h_score.footer_score == 0.0
    assert not h_score.is_likely_middle

    body_text = (
        "1 x 88.50 88.50 V\n"
        "4800011812047 SUPEHO G/BHON 500G\n"
        "1 x 46.25 46.25 V\n"
        "4800024588963 DM CC S/S P/P900G\n"
        "2 x 14.75 29.50 V\n"
        "4800361415347 NSCFORGTP52/40GX10\n"
    )
    b_score = score_segment_density(body_text, 0.90)
    assert b_score.header_score == 0.0
    assert b_score.footer_score == 0.0
    assert b_score.item_row_count >= 5
    assert b_score.is_likely_middle

    footer_text = "Subtotal 6,213.70\nTotal Amt Due 6,193.70\nCASH 6,500.00\nCHANGE 306.30\nVATable Sale 5,530.09"
    f_score = score_segment_density(footer_text, 0.92)
    assert f_score.footer_score >= 4.0
    assert not f_score.is_likely_middle


def test_order_segments_reversed_input():
    img_h = Image.new("RGB", (450, 600))
    img_m = Image.new("RGB", (450, 600))
    img_f = Image.new("RGB", (450, 600))

    r_header = _make_reading([
        "Puregold Price Club, Inc",
        "TIN: 201-277-095-00040",
        "Trans Date: 2026-03-30",
        "1 x 88.50 88.50 V",
        "4800011812047 SUPEHO G/BHON 500G",
    ])

    r_middle = _make_reading([
        "4800011812047 SUPEHO G/BHON 500G",
        "1 x 46.25 46.25 V",
        "4800024588963 DM CC S/S P/P900G",
        "2 x 14.75 29.50 V",
        "4800361415347 NSCFORGTP52/40GX10",
    ])

    r_footer = _make_reading([
        "2 x 14.75 29.50 V",
        "4800361415347 NSCFORGTP52/40GX10",
        "Subtotal 6,213.70",
        "Total Amt Due 6,193.70",
        "CASH 6,500.00",
        "VATable Sale 5,530.09",
    ])

    # Given reversed order [footer, middle, header]
    input_segments = [(img_f, r_footer), (img_m, r_middle), (img_h, r_header)]
    ordered = order_segments(input_segments)

    # Should be auto-ordered to [header, middle, footer]
    assert ordered[0][1] == r_header
    assert ordered[1][1] == r_middle
    assert ordered[2][1] == r_footer


def test_merge_segment_readings_deduplicates_overlap():
    img1 = Image.new("RGB", (450, 400))
    img2 = Image.new("RGB", (450, 400))

    # Segment 1 ends with barcode 4800011812047 and 1 x 46.25
    r1 = _make_reading([
        "Puregold Price Club",
        "TIN: 201-277-095-00040",
        "1 x 88.50 88.50 V",
        "4800011812047 SUPEHO G/BHON 500G",
        "1 x 46.25 46.25 V",
        "4800024588963 DM CC S/S P/P900G",
    ])

    # Segment 2 starts with the same overlap + continues with new items
    r2 = _make_reading([
        "4800011812047 SUPEHO G/BHON 500G",
        "1 x 46.25 46.25 V",
        "4800024588963 DM CC S/S P/P900G",
        "2 x 14.75 29.50 V",
        "Subtotal 178.75",
    ])

    merged_reading, gap_suspected = merge_segment_readings([(img1, r1), (img2, r2)])
    assert gap_suspected is False

    merged_lines = [ln for ln in merged_reading.text.splitlines() if ln.strip()]
    # Overlapping lines should only appear once
    supeho_count = sum(1 for ln in merged_lines if "SUPEHO" in ln)
    dm_count = sum(1 for ln in merged_lines if "DM CC" in ln)
    assert supeho_count == 1
    assert dm_count == 1

    # End item and total are preserved
    assert any("29.50 V" in ln for ln in merged_lines)
    assert any("Subtotal" in ln for ln in merged_lines)


def test_merge_segment_readings_translates_y_coordinates():
    img1 = Image.new("RGB", (450, 500))
    img2 = Image.new("RGB", (450, 500))

    r1 = _make_reading(["Line One", "Line Two"], start_top=20)
    r2 = _make_reading(["Line Three", "Line Four"], start_top=30)

    # When merged without overlap, r2 lines should be shifted by img1.height
    merged_reading, gap_suspected = merge_segment_readings([(img1, r1), (img2, r2)])
    assert gap_suspected is True

    # Check words from Line Three
    words_three = [w for w in merged_reading.words if w.text == "Three"]
    assert len(words_three) == 1
    # Original top was 30; shifted by 500 -> top >= 530
    assert words_three[0].top >= 530


def test_split_tall_image_into_strips():
    # Normal image: not sliced
    normal_img = Image.new("RGB", (600, 800))
    strips = split_tall_image_into_strips(normal_img)
    assert len(strips) == 1
    assert strips[0] is normal_img

    # Very tall receipt (450 x 2700, aspect ratio 6.0)
    tall_img = Image.new("RGB", (450, 2700))
    strips = split_tall_image_into_strips(tall_img, target_height=1200, overlap_pct=0.15)
    assert len(strips) >= 2
    for s in strips:
        assert getattr(s, "_is_virtual_strip", False) is True
        assert hasattr(s, "_virtual_y_offset")


def test_merge_segment_readings_populates_anchors():
    img1 = Image.new("RGB", (450, 400))
    img2 = Image.new("RGB", (450, 400))

    r1 = _make_reading([
        "Store Name",
        "1 x 100.00 100.00",
        "1 x 50.00 50.00",
    ])
    r2 = _make_reading([
        "1 x 50.00 50.00",
        "1 x 25.00 25.00",
        "Total 175.00",
    ])

    merged_reading, _ = merge_segment_readings([(img1, r1), (img2, r2)])
    assert merged_reading.anchors is not None
    assert merged_reading.anchors.money_tokens >= 3
    assert merged_reading.score > 0.0


def test_order_segments_margin_guard_preserves_caller_order():
    """When segments have ambiguous middle items without clear header/footer or strong seam, keep caller order."""
    img1 = Image.new("RGB", (450, 400))
    img2 = Image.new("RGB", (450, 400))
    img3 = Image.new("RGB", (450, 400))

    r1 = _make_reading(["Item 1 10.00", "Item 2 20.00"])
    r2 = _make_reading(["Item 3 30.00", "Item 4 40.00"])
    r3 = _make_reading(["Item 5 50.00", "Item 6 60.00"])

    caller_segments = [(img1, r1), (img2, r2), (img3, r3)]
    ordered = order_segments(caller_segments)

    # Margin guard ensures caller order is not arbitrarily scrambled
    assert [r for _, r in ordered] == [r1, r2, r3]


def test_merge_segment_readings_partial_barcode_overlap():
    img1 = Image.new("RGB", (450, 400))
    img2 = Image.new("RGB", (450, 400))

    # Segment 1 has barcode 4800011812047, segment 2 has 11812047 (common 7-digit suffix)
    r1 = _make_reading([
        "4800011812047 SUPEHO G/BHON 500G",
        "1 x 46.25 46.25 V",
    ])
    r2 = _make_reading([
        "11812047 SUPEHO G/BHON 500G",
        "1 x 46.25 46.25 V",
        "4800024588963 DM CC S/S P/P900G",
    ])

    merged_reading, gap = merge_segment_readings([(img1, r1), (img2, r2)])
    assert gap is False
    merged_lines = [ln for ln in merged_reading.text.splitlines() if ln.strip()]
    supeho_count = sum(1 for ln in merged_lines if "SUPEHO" in ln)
    assert supeho_count == 1

