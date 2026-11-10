from __future__ import annotations

from app.config import settings
from app.llm.base import LLMProvider


def create_provider() -> LLMProvider:
    provider_name = settings.llm_provider.lower().strip()

    if provider_name in ("gemini", ""):
        from app.llm.gemini_provider import GeminiProvider

        return GeminiProvider()

    raise ValueError(f"Unsupported LLM provider: {provider_name}")
