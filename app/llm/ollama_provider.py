import json
import logging
import re
from typing import Any

import httpx

from app.config import settings
from app.llm.base import LLMProvider
from app.llm.prompts import (
    ASSIST_LOCATION_QUESTION,
    ASSIST_SEMANTICS_QUESTION,
    ASSIST_SUBTOTAL_QUESTION,
    ASSIST_VENDOR_QUESTION,
    ASSISTS_PROMPT,
    BATCH_TEXT_NORMALIZATION_PROMPT,
    CATEGORY_TIEBREAK_PROMPT,
    FINANCIAL_SEMANTICS_PROMPT,
    ITEM_ANALYSIS_PROMPT,
    LOCATION_SELECTION_PROMPT,
    RECEIPT_EXTRACTION_PROMPT,
    VENDOR_SELECTION_PROMPT,
)

logger = logging.getLogger(__name__)

# Selection answers are a few tokens. Capping output keeps CPU inference short and
# stops a small model from rambling past valid JSON.
SELECTION_TOKEN_LIMIT = 64
EXTRACTION_TOKEN_LIMIT = 768

# The model only answers value-suffice questions: vendor/location names sit in the
# receipt header, totals and tax wording in the footer. Feeding a long run-on body
# just taxes CPU prefill and stalls. Bound context to the header + a fixed tail so
# the model sees the relevant regions without the full itemized middle.
CONTEXT_HEAD_CHARS = 2000
CONTEXT_TAIL_CHARS = 2000

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _bound_context(ocr_text: str) -> str:
    """Keep the receipt header and the totals region, drop the long middle body."""
    if not ocr_text:
        return ocr_text
    if len(ocr_text) <= CONTEXT_HEAD_CHARS + CONTEXT_TAIL_CHARS:
        return ocr_text
    return (
        ocr_text[:CONTEXT_HEAD_CHARS]
        + "\n[... truncated middle ...]\n"
        + ocr_text[-CONTEXT_TAIL_CHARS:]
    )


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    async def _generate(
        self, prompt: str, token_limit: int, timeout: float = 120.0
    ) -> dict[str, Any] | None:
        """Call Ollama and parse JSON, with one repair attempt.

        A single stray token used to fail the whole job: the old implementation let
        ``json.loads`` raise straight through into a failed callback, losing an
        otherwise good extraction.
        """
        keep_alive = settings.ollama_keep_alive
        if isinstance(keep_alive, str):
            clean_ka = keep_alive.strip()
            if clean_ka.lstrip("-").isdigit():
                keep_alive = int(clean_ka)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": keep_alive,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": token_limit,
                "num_ctx": settings.ollama_num_ctx,
            },
        }

        async with httpx.AsyncClient() as client:
            for attempt in (1, 2):
                try:
                    response = await client.post(
                        f"{self.base_url}/api/generate", json=payload, timeout=timeout
                    )
                    response.raise_for_status()
                    raw = response.json().get("response", "")
                    return self._parse_json(raw)
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Ollama returned unparseable JSON (attempt %s): %s",
                                   attempt, exc)
                    if attempt == 2:
                        return None
                except httpx.HTTPError as exc:
                    logger.warning("Ollama request failed (attempt %s): %s", attempt, exc)
                    if attempt == 2:
                        return None
        return None

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = _THINK_BLOCK_RE.sub("", (raw or "")).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Salvage the first JSON object if the model wrapped it in prose.
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise
            parsed = json.loads(text[start:end + 1])
        if not isinstance(parsed, dict):
            raise TypeError(f"Expected a JSON object, got {type(parsed).__name__}")
        return parsed

    async def extract_receipt_fields(self, ocr_text: str) -> dict:
        result = await self._generate(
            RECEIPT_EXTRACTION_PROMPT.format(ocr_text=_bound_context(ocr_text)),
            token_limit=EXTRACTION_TOKEN_LIMIT,
        )
        if result is None:
            raise RuntimeError("Ollama extraction failed after retry")
        return result

    async def select_vendor_name(
        self, candidates: list[str], excluded: list[str] | None = None
    ) -> str | None:
        if not candidates:
            return None
        prompt = VENDOR_SELECTION_PROMPT.format(
            candidates="\n".join(f"- {line}" for line in candidates),
            excluded="\n".join(f"- {line}" for line in (excluded or [])) or "- (none)",
        )
        result = await self._generate(prompt, token_limit=SELECTION_TOKEN_LIMIT, timeout=60.0)
        if not result:
            return None
        choice = result.get("vendor_name")
        return str(choice).strip() if choice else None

    async def select_location(self, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        prompt = LOCATION_SELECTION_PROMPT.format(
            candidates="\n".join(f"- {line}" for line in candidates)
        )
        result = await self._generate(prompt, token_limit=SELECTION_TOKEN_LIMIT, timeout=60.0)
        if not result:
            return None
        choice = result.get("location")
        return str(choice).strip() if choice else None

    async def choose_category(self, details: str, options: tuple[str, str]) -> str | None:
        prompt = CATEGORY_TIEBREAK_PROMPT.format(
            option_a=options[0], option_b=options[1], details=details[:1200]
        )
        result = await self._generate(prompt, token_limit=SELECTION_TOKEN_LIMIT, timeout=60.0)
        if not result:
            return None
        choice = result.get("category")
        if not choice:
            return None
        picked = str(choice).strip()
        # Reject anything outside the two options offered.
        return picked if picked in options else None

    async def analyze_line_items(self, lines: list[str]) -> list[dict] | None:
        if not lines:
            return None
        prompt = ITEM_ANALYSIS_PROMPT.format(
            lines="\n".join(f"- {line}" for line in lines[:50])
        )
        result = await self._generate(prompt, token_limit=EXTRACTION_TOKEN_LIMIT, timeout=60.0)
        if not result or "items" not in result:
            return None
        items = result.get("items")
        return items if isinstance(items, list) else None

    async def analyze_financial_semantics(self, ocr_text: str) -> dict | None:
        """Return a bounded semantics answer; amounts are prohibited by prompt."""
        if not ocr_text.strip():
            return None
        return await self._generate(
            FINANCIAL_SEMANTICS_PROMPT.format(ocr_text=_bound_context(ocr_text)),
            token_limit=SELECTION_TOKEN_LIMIT,
            timeout=60.0,
        )

    async def verify_subtotal(self, ocr_text: str) -> float | None:
        if not ocr_text.strip():
            return None
        from app.llm.prompts import SUBTOTAL_VERIFICATION_PROMPT
        result = await self._generate(
            SUBTOTAL_VERIFICATION_PROMPT.format(ocr_text=_bound_context(ocr_text)),
            token_limit=SELECTION_TOKEN_LIMIT,
            timeout=60.0,
        )
        if not result:
            return None
        choice = result.get("subtotal")
        try:
            return float(choice) if choice is not None else None
        except (ValueError, TypeError):
            return None

    async def analyze_assists(
        self,
        ocr_text: str,
        *,
        vendor_candidates: list[str] | None = None,
        excluded_vendors: list[str] | None = None,
        location_candidates: list[str] | None = None,
        want_semantics: bool = False,
        want_subtotal: bool = False,
    ) -> dict | None:
        """Consolidated bounded assist call: vendor, location, semantics, subtotal.

        One Ollama generation replaces up to four separate calls, which matters
        on CPU where each generation costs tens of seconds and concurrent calls
        serialize on the model queue. Every field is still validated by the
        caller against its closed list before it is used, so a wrong answer stays
        bounded and a hallucinated one is rejected exactly as before.
        """
        questions: list[str] = []
        shape: list[str] = []
        if vendor_candidates:
            questions.append(
                ASSIST_VENDOR_QUESTION.format(
                    candidates="\n".join(f"- {line}" for line in vendor_candidates),
                    excluded="\n".join(f"- {line}" for line in (excluded_vendors or []))
                    or "- (none)",
                )
            )
            shape.append('"vendor_name": "<exact candidate or null>"')
        if location_candidates:
            questions.append(
                ASSIST_LOCATION_QUESTION.format(
                    candidates="\n".join(f"- {line}" for line in location_candidates)
                )
            )
            shape.append('"location": "<exact candidate or null>"')
        if want_semantics:
            questions.append(ASSIST_SEMANTICS_QUESTION)
            shape.extend([
                '"tax_basis": "inclusive|exclusive|unknown"',
                '"confidence": 0.0',
                '"evidence": ["short exact snippets"]',
                '"currency": "ISO-4217|unknown"',
                '"currency_confidence": 0.0',
            ])
        if want_subtotal:
            questions.append(ASSIST_SUBTOTAL_QUESTION)
            shape.append('"subtotal": 12.34 or null')

        if not questions:
            return None

        prompt = ASSISTS_PROMPT.format(
            questions="\n\n".join(questions),
            shape=",\n".join(shape),
            ocr_text=_bound_context(ocr_text),
        )
        return await self._generate(
            prompt,
            token_limit=SELECTION_TOKEN_LIMIT + 128,
            timeout=settings.ollama_assist_timeout,
        )

    async def normalize_batch_texts(self, texts: list[str], context: str) -> list[str]:
        if not texts:
            return []
        prompt = BATCH_TEXT_NORMALIZATION_PROMPT.format(
            texts=json.dumps(texts), context=context[:2000]
        )
        result = await self._generate(prompt, token_limit=EXTRACTION_TOKEN_LIMIT, timeout=90.0)
        
        # Fallback to original if parsing fails or result is invalid
        if not result or "corrected_texts" not in result:
            return texts
            
        corrected = result.get("corrected_texts", [])
        
        # Validation: output count must equal input count
        if not isinstance(corrected, list) or len(corrected) != len(texts):
            logger.warning("Batch normalization returned %d items, expected %d. Falling back.", len(corrected) if isinstance(corrected, list) else 0, len(texts))
            return texts
            
        return [str(c).strip() if c else t for c, t in zip(corrected, texts)]
