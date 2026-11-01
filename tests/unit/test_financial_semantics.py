"""Offline financial-semantics and safe arithmetic coverage."""

from app.api.schemas.ocr import OcrCallbackPayload, OcrProcessRequest
from app.core.financial_semantics import (
    normalize_iso_currency,
    validate_financial_semantics,
)
from app.core.layout import LabeledAmount, LayoutScan
from app.core.locale import LocaleGuess, resolve_locale
from app.core.reconcile import resolve_money


def _scan(*entries: tuple[str, str, str, float]) -> LayoutScan:
    scan = LayoutScan()
    for field, token, label, confidence in entries:
        scan.amounts.setdefault(field, []).append(LabeledAmount(
            field_name=field, label=label, raw_token=token,
            line_text=f"{label} {token}", confidence=confidence,
        ))
        scan.money_tokens.append(token)
    return scan


def test_semantics_requires_grounded_high_confidence_evidence():
    raw = {"tax_basis": "exclusive", "confidence": 0.8, "evidence": ["plus tax"],
           "currency": "USD", "currency_confidence": 0.8, "currency_evidence": ["USD"]}
    assert validate_financial_semantics(raw, "Subtotal 100 plus tax USD") is not None
    assert validate_financial_semantics(raw, "Subtotal 100") .tax_basis == "unknown"


def test_semantics_malformed_and_currency_are_safe():
    assert validate_financial_semantics(None, "text") is None
    assert normalize_iso_currency("usd") == "USD"
    assert normalize_iso_currency("XYZ") is None


def test_exclusive_computed_total_and_unknown_abstention():
    locale = LocaleGuess(country="US", currency="USD", score=8)
    exclusive = resolve_money(
        _scan(("net_sales", "100.00", "subtotal", .95), ("tax_amount", "12.00", "tax", .95)),
        locale, tax_basis="exclusive",
    )
    assert exclusive.get("total_amount") == 112.0
    assert exclusive.computed_total == 112.0
    unknown = resolve_money(
        _scan(("net_sales", "100.00", "subtotal", .95), ("tax_amount", "12.00", "tax", .95)),
        LocaleGuess(country="XX", currency=None, score=0), tax_basis="unknown",
    )
    assert unknown.get("total_amount") is None


def test_multiple_tax_lines_aggregate_without_losing_breakdown():
    result = resolve_money(
        _scan(("net_sales", "100.00", "subtotal", .9), ("tax_amount", "5.00", "VAT", .9),
              ("tax_amount", "2.00", "sales tax", .9)),
        LocaleGuess(country="US", currency="USD", score=8), tax_basis="exclusive",
    )
    assert result.get("tax_amount") == 7.0
    assert len(result.tax_breakdown) == 2


def test_request_context_and_callback_fields_are_optional_and_normalized():
    request = OcrProcessRequest(receipt_id=1, callback_url="https://example.test", country="us",
                                currency="usd", location=" Las Vegas ")
    assert (request.country, request.currency, request.location) == ("US", "USD", "Las Vegas")
    payload = OcrCallbackPayload(receipt_id=1, tax_basis="exclusive",
                                 financial_reconciliation_status="computed",
                                 needs_manual_review=True)
    assert payload.model_dump()["tax_basis"] == "exclusive"


def test_currency_precedence_caller_then_receipt_then_country_then_ocr():
    assert resolve_locale("TOTAL EUR 10.00", caller_currency="USD", caller_country="US").currency == "USD"
    assert resolve_locale("TOTAL EUR 10.00", caller_country="US").currency == "EUR"
    assert resolve_locale("TOTAL $10.00", caller_country="GB").currency == "GBP"
    assert resolve_locale("TOTAL $10.00", llm_currency="CAD").currency == "CAD"


import pytest


def test_my_sst_receipt_infers_exclusive_basis():
    # 'SST 6%' after a subtotal means tax is added on top (net + tax = total),
    # which lets reconcile derive total_sales for Malaysian receipts.
    from app.core.financial_semantics import infer_tax_basis
    basis, evidence = infer_tax_basis(
        "Subtotal 17.00\nSST 6% 1.02\nTOTAL 18.02", country="MY"
    )
    assert basis == "exclusive"
    assert evidence == ["MY SST/GST fallback"]


@pytest.mark.asyncio
async def test_ollama_financial_semantics_is_dedicated_and_fail_soft(monkeypatch):
    from app.llm.ollama_provider import OllamaProvider
    provider = OllamaProvider()
    seen = {}

    async def fake_generate(prompt, token_limit, timeout=120.0):
        seen["prompt"] = prompt
        return {"tax_basis": "inclusive", "confidence": .9, "evidence": ["VAT Inclusive"],
                "currency": "PHP", "currency_confidence": .9, "currency_evidence": ["PHP"]}

    monkeypatch.setattr(provider, "_generate", fake_generate)
    result = await provider.analyze_financial_semantics("Total Sales (VAT Inclusive) PHP")
    assert result["tax_basis"] == "inclusive"
    assert "amount" in seen["prompt"].lower()

    async def failed(*args, **kwargs):
        return None
    monkeypatch.setattr(provider, "_generate", failed)
    assert await provider.analyze_financial_semantics("VAT") is None


@pytest.mark.asyncio
async def test_ollama_verify_subtotal_is_fail_soft(monkeypatch):
    from app.llm.ollama_provider import OllamaProvider
    provider = OllamaProvider()
    
    async def fake_generate(prompt, token_limit, timeout=120.0):
        return {"subtotal": 12.34}
        
    monkeypatch.setattr(provider, "_generate", fake_generate)
    assert await provider.verify_subtotal("Subtotal: 12.34") == 12.34
    
    async def failed(*args, **kwargs):
        return None
    monkeypatch.setattr(provider, "_generate", failed)
    assert await provider.verify_subtotal("Subtotal: 12.34") is None


def test_bound_context_short_text_unchanged():
    from app.llm.ollama_provider import (
        _bound_context,
    )
    short = "Vendor\nTotal 123.45"
    assert _bound_context(short) == short


def test_bound_context_keeps_header_and_tail():
    from app.llm.ollama_provider import _bound_context
    body = "line\n" * 3000
    text = f"VENDOR HEADER {body} FOOTER 99.95"
    bounded = _bound_context(text)
    assert "VENDOR HEADER" in bounded
    assert "FOOTER 99.95" in bounded
    assert len(bounded) < len(text)
    assert _bound_context("") == ""


def test_qwen3_think_block_is_stripped_before_json_parsing():
    from app.llm.ollama_provider import OllamaProvider
    raw = (
        "<think>\nThe user wants financial semantics in JSON.\n</think>\n"
        '{"tax_basis": "inclusive", "confidence": 0.92}'
    )
    parsed = OllamaProvider._parse_json(raw)
    assert parsed["tax_basis"] == "inclusive"
    assert parsed["confidence"] == 0.92


def test_semantics_evidence_grounding_normalizes_whitespace():
    raw = {
        "tax_basis": "inclusive",
        "confidence": 0.9,
        "evidence": ["VAT INCLUSIVE"],
        "currency": "PHP",
        "currency_confidence": 0.9,
        "currency_evidence": ["PHP"],
    }
    sem = validate_financial_semantics(
        raw, "Total Sales (VAT   INCLUSIVE)   PHP 250.00"
    )
    assert sem is not None
    assert sem.tax_basis == "inclusive"
    assert sem.currency == "PHP"


@pytest.mark.asyncio
async def test_ollama_keep_alive_int_conversion(monkeypatch):
    import httpx

    from app.config import settings
    from app.llm.ollama_provider import OllamaProvider

    provider = OllamaProvider()
    captured_payload = {}

    class DummyResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": '{"test": 1}'}

    async def fake_post(self, url, json=None, timeout=120.0):
        captured_payload.update(json)
        return DummyResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(settings, "ollama_keep_alive", "-1")

    res = await provider._generate("test prompt", 64)
    assert res == {"test": 1}
    assert captured_payload.get("keep_alive") == -1
    assert isinstance(captured_payload.get("keep_alive"), int)


