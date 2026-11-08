import pytest
from pathlib import Path
from PIL import Image
from app.core.ocr_engine import read_pooled
from app.core.extraction import extract

RECEIPTS = Path(__file__).parents[2] / "docs" / "receipts"


@pytest.fixture(scope="session")
def bundles():
    """Cache bundles to avoid re-running OCR during regression tests.

    Forces the full pool: these tests assert extraction correctness against the
    multi-pass OCR bundle, which is the semantics they were written for.
    """
    cache = {}
    async def _get(name):
        if name not in cache:
            path = RECEIPTS / name
            if not path.exists():
                pytest.skip(f"fixture unavailable: {name}")
            cache[name] = await read_pooled(Image.open(path), lang="eng",
                                            fast_path=False)
        return cache[name]
    return _get

@pytest.mark.asyncio
async def test_receipt_21_exclusive_tax_computed_total(bundles):
    bundle = await bundles("receipt 21.jpg")
    extraction = extract(bundle, caller_country="US")
    
    assert extraction.fields["tax_basis"] == "exclusive"
    assert extraction.fields["net_sales"] == 29.47
    assert extraction.fields["tax_amount"] == 2.50
    assert extraction.fields["computed_total"] == 31.97
    assert extraction.fields["total_amount"] == 31.97
    assert extraction.fields["financial_reconciliation_status"] == "reported"

@pytest.mark.parametrize("name", [
    "receipt 4.jpg",
    "receipt 5.jpg",
    "receipt 9.jpg",
    "receipt 10.png",
    "receipt 11.jpg",
    "receipt 12.png",
    "receipt 13.jpg",
    "receipt 14.jpg",
    "receipt 15.jpeg",
    "receipt 16.png",
    "receipt 17.jpg",
    "receipt 18.jpg",
    "receipt 19.jpg",
    "receipt 19.png",
    "receipt 20.png",
    "receipt 22.jpg",
])
@pytest.mark.asyncio
async def test_other_receipts_smoke(bundles, name):
    bundle = await bundles(name)
    extraction = extract(bundle)
    assert extraction.fields is not None
    assert "total_amount" in extraction.fields
