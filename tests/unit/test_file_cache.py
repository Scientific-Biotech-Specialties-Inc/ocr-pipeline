import pytest

from app.core.file_cache import (
    cache_downloaded_bytes,
    compute_content_hash,
    get_cached_ocr_result,
    pop_cached_bytes,
    set_cached_ocr_result,
)


def test_compute_content_hash():
    b1 = b"image_data_part_1"
    b2 = b"image_data_part_2"
    h1 = compute_content_hash([b1, b2], country="PH", currency="PHP", location="Manila")
    h2 = compute_content_hash([b1, b2], country="PH", currency="PHP", location="Manila")
    h3 = compute_content_hash([b1, b2], country="US", currency="USD")

    assert h1 == h2
    assert len(h1) == 64
    assert h1 != h3

    h_gemini = compute_content_hash(
        [b1, b2], country="PH", currency="PHP", location="Manila", llm_provider="gemini"
    )
    assert h1 == h_gemini

    h_other = compute_content_hash(
        [b1, b2], country="PH", currency="PHP", location="Manila", llm_provider="other_provider"
    )
    assert h1 != h_other


@pytest.mark.asyncio
async def test_cache_and_pop_bytes(monkeypatch):
    store = {}

    class FakeRedis:
        async def set(self, key, value, ex=None):
            store[key] = value

        async def getdel(self, key):
            return store.pop(key, None)

    fake_client = FakeRedis()
    monkeypatch.setattr("app.core.file_cache.get_redis_client", lambda: fake_client)

    url = "https://example.com/receipt.jpg"
    data = b"\xff\xd8\xffreceipt_bytes"

    await cache_downloaded_bytes(url, data, ttl_seconds=300)
    assert len(store) == 1

    popped = await pop_cached_bytes(url)
    assert popped == data
    assert len(store) == 0

    # Second pop returns None
    assert await pop_cached_bytes(url) is None


@pytest.mark.asyncio
async def test_get_and_set_cached_ocr_result(monkeypatch):
    store = {}

    class FakeRedis:
        async def set(self, key, value, ex=None):
            store[key] = value

        async def get(self, key):
            return store.get(key)

    fake_client = FakeRedis()
    monkeypatch.setattr("app.core.file_cache.get_redis_client", lambda: fake_client)

    content_hash = "abc123hash"
    payload = {
        "combined_text": "Sample Receipt Text",
        "confidence": 0.95,
        "callback_fields": {"vendor_name": "Test Store", "total_amount": 100.0},
    }

    await set_cached_ocr_result(content_hash, payload, ttl_seconds=600)
    cached = await get_cached_ocr_result(content_hash)

    assert cached is not None
    assert cached["combined_text"] == "Sample Receipt Text"
    assert cached["confidence"] == 0.95
    assert cached["callback_fields"]["total_amount"] == 100.0


@pytest.mark.asyncio
async def test_redis_failure_fails_soft(monkeypatch):
    class BrokenRedis:
        async def set(self, *args, **kwargs):
            raise ConnectionError("Redis is down")

        async def get(self, *args, **kwargs):
            raise ConnectionError("Redis is down")

        async def getdel(self, *args, **kwargs):
            raise ConnectionError("Redis is down")

    monkeypatch.setattr("app.core.file_cache.get_redis_client", lambda: BrokenRedis())

    # None of these should raise
    await cache_downloaded_bytes("https://example.com/r.jpg", b"bytes")
    assert await pop_cached_bytes("https://example.com/r.jpg") is None
    assert await get_cached_ocr_result("nonexistent") is None
    await set_cached_ocr_result("hash", {"key": "value"})
