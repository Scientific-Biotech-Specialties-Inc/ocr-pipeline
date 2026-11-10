import json
import pytest
import httpx

from app.config import settings
from app.llm.factory import create_provider
from app.llm.gemini_provider import GeminiProvider


@pytest.mark.asyncio
async def test_factory_creates_gemini_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    provider = create_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.model == settings.gemini_model


@pytest.mark.asyncio
async def test_gemini_provider_fails_soft_when_key_empty(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    provider = GeminiProvider()
    result = await provider._generate("test prompt", 64)
    assert result is None


@pytest.mark.asyncio
async def test_gemini_provider_generate_success(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-fake-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.0-flash")
    provider = GeminiProvider()

    captured_request = {}

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps({"vendor_name": "Jollibee", "location": "Makati"})
                                }
                            ]
                        }
                    }
                ]
            }

    async def fake_post(self, url, headers=None, json=None, timeout=None):
        captured_request["url"] = url
        captured_request["headers"] = headers
        captured_request["json"] = json
        captured_request["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await provider._generate("Test prompt", 64)

    assert result == {"vendor_name": "Jollibee", "location": "Makati"}
    assert captured_request["headers"]["x-goog-api-key"] == "test-fake-key"
    assert "models/gemini-2.0-flash:generateContent" in captured_request["url"]
    assert captured_request["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert captured_request["json"]["contents"][0]["parts"][0]["text"] == "Test prompt"


@pytest.mark.asyncio
async def test_gemini_provider_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-fake-key")
    provider = GeminiProvider()

    calls = 0

    class DummyResponse429:
        status_code = 429

        def raise_for_status(self):
            request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("Rate limited", request=request, response=response)

    class DummyResponse200:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '{"subtotal": 99.50}'}]
                        }
                    }
                ]
            }

    async def fake_post(self, url, headers=None, json=None, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return DummyResponse429()
        return DummyResponse200()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    # Speed up test sleep
    import asyncio
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda sec: real_sleep(0.001))

    result = await provider._generate("Test prompt", 64)
    assert calls == 2
    assert result == {"subtotal": 99.50}


@pytest.mark.asyncio
async def test_gemini_provider_fails_soft_on_invalid_json(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-fake-key")
    provider = GeminiProvider()

    class DummyBadJsonResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "not valid json at all"}]
                        }
                    }
                ]
            }

    async def fake_post(*args, **kwargs):
        return DummyBadJsonResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await provider._generate("Test prompt", 64)
    assert result is None


@pytest.mark.asyncio
async def test_gemini_analyze_assists_integration(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-fake-key")
    provider = GeminiProvider()

    async def fake_generate(prompt, token_limit, timeout=None):
        return {
            "vendor_name": "JOLLIBEE INC",
            "location": "Makati City",
            "tax_basis": "inclusive",
            "confidence": 0.95,
            "evidence": ["VAT Inclusive"],
            "currency": "PHP",
            "currency_confidence": 0.95,
            "subtotal": 150.00,
        }

    monkeypatch.setattr(provider, "_generate", fake_generate)

    assists = await provider.analyze_assists(
        "Header Jollibee Makati\nTotal 150.00 VAT Inclusive",
        vendor_candidates=["JOLLIBEE INC", "OTHER STORE"],
        location_candidates=["Makati City", "Quezon City"],
        want_semantics=True,
        want_subtotal=True,
    )

    assert assists is not None
    assert assists["vendor_name"] == "JOLLIBEE INC"
    assert assists["location"] == "Makati City"
    assert assists["tax_basis"] == "inclusive"
    assert assists["subtotal"] == 150.00
