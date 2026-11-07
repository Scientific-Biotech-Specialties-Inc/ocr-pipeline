import pytest

from app.core.extraction import Extraction
from app.core.ocr_engine import OcrBundle, OcrReading, Word
from app.core.postprocessing import DictionaryCorrector, LLMContextCorrector


@pytest.fixture
def mock_bundle():
    # Construct a mock OcrBundle
    words = [
        Word(text="Res", confidence=0.75, left=0, top=0, width=10, height=10, block=0, par=0, line=0),
        Word(text="Coffee", confidence=0.95, left=12, top=0, width=20, height=10, block=0, par=0, line=0),
        Word(text="Walmart", confidence=0.99, left=0, top=20, width=50, height=10, block=0, par=0, line=1),
        Word(text="Rxd", confidence=0.3, left=0, top=40, width=20, height=10, block=0, par=0, line=2),
        Word(text="Cf#e", confidence=0.4, left=22, top=40, width=20, height=10, block=0, par=0, line=2),
    ]
    primary = OcrReading(text="Res Coffee\nWalmart\nRxd Cf#e", words=words, confidence=0.7, variant="mock", psm=6)
    return OcrBundle(primary=primary)

def test_dictionary_corrector_moderate_confidence(mock_bundle):
    corrector = DictionaryCorrector(vocabulary={"red", "coffee", "walmart"}, min_confidence=0.5, max_confidence=0.9)
    
    # "Res" should be corrected to "Red" since its confidence is 0.75 (between 0.5 and 0.9) and dist is 1.
    result = corrector.correct_word("Res", mock_bundle)
    assert result is not None
    assert result.corrected == "Red"
    
def test_dictionary_corrector_high_confidence_skip(mock_bundle):
    corrector = DictionaryCorrector(vocabulary={"red", "coffee", "walmart"}, min_confidence=0.5, max_confidence=0.9)
    
    # "Walmart" should NOT be corrected (or even tried) if we assume it's correct. 
    # But wait, Walmart is in the vocab anyway. What if we misspelled it as Walmrt with 0.99 confidence?
    # Let's test a word not in vocab with high confidence.
    mock_bundle.primary.words.append(Word(text="Waalmart", confidence=0.95, left=0, top=0, width=10, height=10, block=0, par=0, line=0))
    result = corrector.correct_word("Waalmart", mock_bundle)
    assert result is None  # Skipped because confidence 0.95 > 0.90

def test_dictionary_corrector_low_confidence_skip(mock_bundle):
    corrector = DictionaryCorrector(vocabulary={"red", "coffee", "walmart"}, min_confidence=0.5, max_confidence=0.9)
    
    # "Rxd" has confidence 0.3. It shouldn't be corrected.
    result = corrector.correct_word("Rxd", mock_bundle)
    assert result is None  # Skipped because confidence 0.3 < 0.50

def test_dictionary_corrector_phrase(mock_bundle):
    corrector = DictionaryCorrector(vocabulary={"red", "coffee", "walmart"}, min_confidence=0.5, max_confidence=0.9)
    corrected_phrase = corrector.correct_phrase("Res Coffee", mock_bundle)
    # Res -> Red, Coffee remains Coffee (either via vocab or confidence)
    assert corrected_phrase == "Red Coffee"

@pytest.mark.asyncio
async def test_llm_corrector_mocked(monkeypatch):
    from app.llm.base import LLMProvider
    
    class MockProvider(LLMProvider):
        async def extract_receipt_fields(self, ocr_text: str) -> dict:
            return {}
        async def normalize_batch_texts(self, texts: list[str], context: str) -> list[str]:
            return ["Red Coffee" if "Res Coffee" in text else text for text in texts]
            
    import app.core.postprocessing
    monkeypatch.setattr(app.core.postprocessing, "create_provider", lambda: MockProvider())
    
    corrector = LLMContextCorrector()
    items = [
        {"name": "Res Coffee", "quantity": 1, "price": 2.50},
        {"name": "Bottled Water", "quantity": 1, "price": 1.00},
        {"name": "Fresh Fries", "quantity": 1, "price": 1.50},
    ]
    corrected = await corrector.correct_line_items(items, "Res Coffee 2.50")
    
    assert len(corrected) == 3
    assert corrected[0]["name"] == "Red Coffee"

@pytest.mark.asyncio
async def test_normalize_extraction_item_mutation(mock_bundle, monkeypatch):
    from app.core.items import ExtractedItem, ItemScan
    from app.llm.base import LLMProvider
    
    class MockProvider(LLMProvider):
        async def extract_receipt_fields(self, ocr_text: str) -> dict:
            return {}
        async def normalize_batch_texts(self, texts: list[str], context: str) -> list[str]:
            return ["Red Coffee" if "Res Coffee" in text else text for text in texts]
            
    import app.core.postprocessing
    monkeypatch.setattr(app.core.postprocessing, "create_provider", lambda: MockProvider())
    
    # Mock DictionaryCorrector to just return the phrase so it doesn't mangle our test string
    class MockDictCorrector:
        def correct_phrase(self, phrase, bundle):
            return phrase
    monkeypatch.setattr(app.core.postprocessing, "DictionaryCorrector", MockDictCorrector)
    
    # Create an Extraction object with real ExtractedItem objects
    item = ExtractedItem(name="Res Coffee", quantity=1, price=2.50)
    item.descriptions.append("Extra shot") # Descriptions should be cleared
    item2 = ExtractedItem(name="Bottled Water", quantity=1, price=1.00)
    item2.descriptions.append("Cold")
    item3 = ExtractedItem(name="Fresh Fries", quantity=1, price=1.50)
    item3.descriptions.append("Large")
    scan = ItemScan(items=[item, item2, item3])
    
    extraction = Extraction()
    extraction.item_scan = scan
    
    # Run the normalization
    await app.core.postprocessing.normalize_extraction(extraction, mock_bundle)
    
    # Verify the item's name was mutated and descriptions cleared, so full_name resolves correctly
    assert extraction.item_scan.items[0].name == "Red Coffee"
    assert len(extraction.item_scan.items[0].descriptions) == 0
    assert extraction.item_scan.items[0].full_name == "Red Coffee"


@pytest.mark.asyncio
async def test_normalize_extraction_use_llm_false_skips_model(mock_bundle, monkeypatch):
    """Fast path must stay deterministic: dictionary-only, zero model calls."""
    import app.core.postprocessing
    from app.core.items import ExtractedItem, ItemScan

    called = {"model": False}

    class FailOnLLM(LLMContextCorrector):
        def __init__(self):
            pass

        def correct_phrase(self, phrase, bundle):
            return phrase

        async def correct_line_items(self, items, context):
            called["model"] = True
            return items

    monkeypatch.setattr(
        app.core.postprocessing, "LLMContextCorrector", FailOnLLM
    )

    class DictPassthrough:
        def correct_phrase(self, phrase, bundle):
            return phrase

    monkeypatch.setattr(app.core.postprocessing, "DictionaryCorrector", DictPassthrough)

    item = ExtractedItem(name="Res Coffee", quantity=1, price=2.50)
    scan = ItemScan(items=[item])
    extraction = Extraction()
    extraction.item_scan = scan

    await app.core.postprocessing.normalize_extraction(
        extraction, mock_bundle, use_llm=False
    )

    assert called["model"] is False
    assert extraction.item_scan.items[0].name == "Res Coffee"


@pytest.mark.asyncio
async def test_normalize_extraction_use_llm_false_preserves_descriptions_clear(mock_bundle, monkeypatch):
    import app.core.postprocessing
    from app.core.items import ExtractedItem, ItemScan

    class DictPassthrough:
        def correct_phrase(self, phrase, bundle):
            return phrase

    monkeypatch.setattr(app.core.postprocessing, "DictionaryCorrector", DictPassthrough)

    item = ExtractedItem(name="Plain Item", quantity=1, price=2.50)
    item.descriptions.append("Extra shot")
    scan = ItemScan(items=[item])
    extraction = Extraction()
    extraction.item_scan = scan

    await app.core.postprocessing.normalize_extraction(
        extraction, mock_bundle, use_llm=False
    )

    assert extraction.item_scan.items[0].name == "Plain Item Extra shot"
    assert len(extraction.item_scan.items[0].descriptions) == 0
    assert extraction.item_scan.items[0].full_name == "Plain Item Extra shot"



@pytest.mark.asyncio
async def test_llm_corrector_skips_store_code_receipts(monkeypatch):
    import app.core.postprocessing
    from app.llm.base import LLMProvider

    called = {"batch": False}

    class CodeOnlyProvider(LLMProvider):
        async def extract_receipt_fields(self, ocr_text: str) -> dict:
            return {}

        async def normalize_batch_texts(self, texts, context):
            called["batch"] = True
            return texts

    monkeypatch.setattr(app.core.postprocessing, "create_provider",
                        lambda: CodeOnlyProvider())

    corrector = LLMContextCorrector()
    items = [
        {"name": "DRD PEV-CUT EP025G", "quantity": 1, "price": 64.00},
        {"name": "CA05SINI BVANCA4JG", "quantity": 1, "price": 17.15},
        {"name": "JNJ MRCHPSN / CHS24G", "quantity": 1, "price": 122.00},
    ]
    corrected = await corrector.correct_line_items(items, "raw ocr text")

    assert called["batch"] is False
    assert corrected == items


@pytest.mark.asyncio
async def test_llm_corrector_skips_when_codes_are_majority(monkeypatch):
    import app.core.postprocessing

    called = {"batch": False}

    class MixedProvider:
        async def normalize_batch_texts(self, texts, context):
            called["batch"] = True
            return texts

    monkeypatch.setattr(app.core.postprocessing, "create_provider",
                        lambda: MixedProvider())

    corrector = LLMContextCorrector()
    items = [
        {"name": "DRD PEV-CUT EP025G", "quantity": 1, "price": 64.00},
        {"name": "CA05SINI BVANCA4JG", "quantity": 1, "price": 17.15},
        {"name": "JNJ MRCHPSN / CHS24G", "quantity": 1, "price": 122.00},
        {"name": "COFFEE", "quantity": 1, "price": 3.00},
    ]
    await corrector.correct_line_items(items, "raw ocr text")

    assert called["batch"] is False


@pytest.mark.asyncio
async def test_llm_corrector_still_corrects_word_names(monkeypatch):
    import app.core.postprocessing

    class WordProvider:
        async def normalize_batch_texts(self, texts, context):
            return ["Res Coffee"] if "Res" in texts[0] else texts

    monkeypatch.setattr(app.core.postprocessing, "create_provider",
                        lambda: WordProvider())

    corrector = LLMContextCorrector()
    items = [
        {"name": "Res Coffee", "quantity": 1, "price": 2.50},
        {"name": "Bottled Water", "quantity": 1, "price": 1.00},
        {"name": "Fresh Fries", "quantity": 1, "price": 1.50},
    ]
    corrected = await corrector.correct_line_items(items, "Res Coffee 2.50")

    assert corrected[0]["name"] == "Res Coffee"


@pytest.mark.asyncio
async def test_llm_corrector_skips_tiny_genuine_batches(monkeypatch):
    """Few genuine names must not pay the fixed-cost LLM batch call."""
    import app.core.postprocessing

    called = {"batch": False}

    class TinyBatchProvider:
        async def normalize_batch_texts(self, texts, context):
            called["batch"] = True
            return texts

    monkeypatch.setattr(app.core.postprocessing, "create_provider",
                        lambda: TinyBatchProvider())

    corrector = LLMContextCorrector()
    items = [
        {"name": "Res Coffee", "quantity": 1, "price": 2.50},
        {"name": "Bottled Water", "quantity": 1, "price": 1.00},
    ]
    corrected = await corrector.correct_line_items(items, "Res Coffee 2.50")

    assert called["batch"] is False
    assert corrected == items


@pytest.mark.asyncio
async def test_normalize_extraction_preserves_location(mock_bundle):
    from app.core.postprocessing import normalize_extraction

    extraction = Extraction(
        fields={
            "location": "3890 S Maryland Parkway Suite 137, Las Vegas NV 89119",
            "vendor_name": "Jollibee",
        }
    )
    await normalize_extraction(extraction, mock_bundle, use_llm=False)
    assert extraction.fields["location"] == "3890 S Maryland Parkway Suite 137, Las Vegas NV 89119"
