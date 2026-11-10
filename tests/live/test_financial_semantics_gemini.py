"""Opt-in Gemini financial-semantics checks against the receipt corpus.

Run with RUN_LIVE_LLM=1 and a configured GEMINI_API_KEY.
"""

from pathlib import Path
import os

import pytest
from PIL import Image

from app.core.ocr_engine import read_pooled
from app.llm.gemini_provider import GeminiProvider

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(os.getenv("RUN_LIVE_LLM") != "1", reason="live LLM tests are opt-in"),
]

RECEIPTS = Path(__file__).parents[2] / "docs" / "receipts"
NAMES = [
    "receipt 4.jpg", "receipt 5.jpg", "receipt 9.jpg", "receipt 10.png",
    "receipt 11.jpg", "receipt 12.png", "receipt 13.jpg", "receipt 14.jpg",
    "receipt 15.jpeg", "receipt 16.png", "receipt 17.jpg", "receipt 18.jpg",
    "receipt 19.jpg", "receipt 19.png", "receipt 20.png", "receipt 21.jpg",
    "receipt 22.jpg",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", NAMES)
async def test_financial_semantics_is_bounded_for_corpus_receipt(name):
    image = RECEIPTS / name
    if not image.exists():
        pytest.skip(f"fixture unavailable: {name}")
    bundle = await read_pooled(Image.open(image), lang="eng")
    raw = await GeminiProvider().analyze_financial_semantics(bundle.combined_text)
    if raw is None:
        pytest.skip("Gemini unavailable")
    assert set(raw) >= {"tax_basis", "confidence", "evidence", "currency"}
    assert raw["tax_basis"] in {"inclusive", "exclusive", "unknown"}
    assert isinstance(raw["evidence"], (list, str))
    assert not any(key in raw for key in ("total_amount", "tax_amount", "net_sales"))

