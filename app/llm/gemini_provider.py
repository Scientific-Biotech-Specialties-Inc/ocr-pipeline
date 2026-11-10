from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google AI Studio Gemini provider (e.g. gemini-2.0-flash).

    Uses direct REST API via httpx. Returns parsed JSON or None (fail-soft).
    """

    def __init__(self) -> None:
        self.api_key = settings.gemini_api_key.strip()
        self.model = settings.gemini_model.strip() or "gemini-2.0-flash"
        self.base_url = settings.gemini_base_url.rstrip("/")
        self.assist_timeout = settings.gemini_timeout
        self.default_timeout = settings.gemini_timeout

    async def _generate(
        self, prompt: str, token_limit: int, timeout: float | None = None
    ) -> dict[str, Any] | None:
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not configured; skipping Gemini LLM call")
            return None

        effective_timeout = timeout if timeout is not None else self.default_timeout
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 1.0,
                "maxOutputTokens": token_limit,
                "responseMimeType": "application/json",
            },
        }

        async with httpx.AsyncClient() as client:
            for attempt in (1, 2):
                try:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=effective_timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    candidates = data.get("candidates") or []
                    if not candidates:
                        logger.warning("Gemini returned no candidates (attempt %s)", attempt)
                        if attempt == 2:
                            return None
                        continue

                    parts = candidates[0].get("content", {}).get("parts") or []
                    if not parts or "text" not in parts[0]:
                        logger.warning("Gemini candidate missing text parts (attempt %s)", attempt)
                        if attempt == 2:
                            return None
                        continue

                    raw_text = parts[0]["text"]
                    return self._parse_json(raw_text)

                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "Gemini returned unparseable JSON (attempt %s): %s",
                        attempt,
                        exc,
                    )
                    if attempt == 2:
                        return None
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    logger.warning(
                        "Gemini API HTTP error %d (attempt %s)",
                        status_code,
                        attempt,
                    )
                    if attempt == 1 and status_code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(1.0)
                    else:
                        return None
                except httpx.HTTPError as exc:
                    logger.warning(
                        "Gemini API network error (attempt %s): %s",
                        attempt,
                        type(exc).__name__,
                    )
                    if attempt == 1:
                        await asyncio.sleep(1.0)
                    else:
                        return None

        return None

