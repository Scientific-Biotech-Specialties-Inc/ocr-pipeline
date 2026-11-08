"""Unit tests for multi-segment image stitcher."""

import pytest
from PIL import Image

from app.core.image_stitcher import stitch_segments


def test_stitch_two_equal_width_images():
    img1 = Image.new("RGB", (400, 300), color=(255, 0, 0))
    img2 = Image.new("RGB", (400, 500), color=(0, 255, 0))

    result = stitch_segments([img1, img2])
    assert result.width == 400
    assert result.height == 800


def test_stitch_varying_width_images():
    img1 = Image.new("RGB", (300, 200), color=(255, 0, 0))
    img2 = Image.new("RGB", (500, 400), color=(0, 255, 0))

    result = stitch_segments([img1, img2])
    assert result.width == 500
    assert result.height == 600


def test_stitch_too_few_segments_raises_error():
    img1 = Image.new("RGB", (300, 200))
    with pytest.raises(ValueError, match="At least 2 image segments are required"):
        stitch_segments([img1])


def test_stitch_exceeding_max_segments_raises_error():
    imgs = [Image.new("RGB", (300, 200)) for _ in range(7)]
    with pytest.raises(ValueError, match="Exceeded maximum segment count"):
        stitch_segments(imgs)
