from __future__ import annotations

import json
import logging
import re
from abc import ABC
from typing import Any

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
    SUBTOTAL_VERIFICATION_PROMPT,
    VENDOR_SELECTION_PROMPT,
)

logger = logging.getLogger(__name__)

SELECTION_TOKEN_LIMIT = 64
EXTRACTION_TOKEN_LIMIT = 768

CONTEXT_HEAD_CHARS = 2000
CONTEXT_TAIL_CHARS = 2000

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def bound_context(ocr_text: str) -> str:
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


class LLMProvider(ABC):
    """Provider interface for the narrowed model role.

    Subclasses implement ``_generate`` for transport, while common prompt
    formatting and parsing logic is shared here.
    """

    default_timeout: float = 60.0
    assist_timeout: float = 20.0

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = _THINK_BLOCK_RE.sub("", (raw or "")).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Salvage the first JSON object if the model wrapped it in prose.
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise
            parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise TypeError(f"Expected a JSON object, got {type(parsed).__name__}")
        return parsed

    async def _generate(
        self, prompt: str, token_limit: int, timeout: float | None = None
    ) -> dict[str, Any] | None:
        """Transport-specific generation method implemented by providers."""
        return None

    async def extract_receipt_fields(self, ocr_text: str) -> dict:
        """Legacy single-shot extraction from raw OCR text."""
        result = await self._generate(
            RECEIPT_EXTRACTION_PROMPT.format(ocr_text=bound_context(ocr_text)),
            token_limit=EXTRACTION_TOKEN_LIMIT,
            timeout=self.default_timeout,
        )
        if result is None:
            raise RuntimeError(f"{self.__class__.__name__} extraction failed after retry")
        return result

    async def select_vendor_name(
        self, candidates: list[str], excluded: list[str] | None = None
    ) -> str | None:
        """Pick the issuing business from a closed candidate list."""
        if not candidates:
            return None
        prompt = VENDOR_SELECTION_PROMPT.format(
            candidates="\n".join(f"- {line}" for line in candidates),
            excluded="\n".join(f"- {line}" for line in (excluded or [])) or "- (none)",
        )
        result = await self._generate(
            prompt, token_limit=SELECTION_TOKEN_LIMIT, timeout=self.default_timeout
        )
        if not result:
            return None
        choice = result.get("vendor_name")
        return str(choice).strip() if choice else None

    async def select_location(self, candidates: list[str]) -> str | None:
        """Pick the business location/address from a closed candidate list."""
        if not candidates:
            return None
        prompt = LOCATION_SELECTION_PROMPT.format(
            candidates="\n".join(f"- {line}" for line in candidates)
        )
        result = await self._generate(
            prompt, token_limit=SELECTION_TOKEN_LIMIT, timeout=self.default_timeout
        )
        if not result:
            return None
        choice = result.get("location")
        return str(choice).strip() if choice else None

    async def choose_category(self, details: str, options: tuple[str, str]) -> str | None:
        """Break a tie between exactly two candidate categories."""
        prompt = CATEGORY_TIEBREAK_PROMPT.format(
            option_a=options[0], option_b=options[1], details=details[:1200]
        )
        result = await self._generate(
            prompt, token_limit=SELECTION_TOKEN_LIMIT, timeout=self.default_timeout
        )
        if not result:
            return None
        choice = result.get("category")
        if not choice:
            return None
        picked = str(choice).strip()
        return picked if picked in options else None

    async def analyze_line_items(self, lines: list[str]) -> list[dict] | None:
        """Analyze line-item structures (quantity, name, price) from receipt lines."""
        if not lines:
            return None
        prompt = ITEM_ANALYSIS_PROMPT.format(
            lines="\n".join(f"- {line}" for line in lines[:50])
        )
        result = await self._generate(
            prompt, token_limit=EXTRACTION_TOKEN_LIMIT, timeout=self.default_timeout
        )
        if not result or "items" not in result:
            return None
        items = result.get("items")
        return items if isinstance(items, list) else None

    async def analyze_financial_semantics(self, ocr_text: str) -> dict | None:
        """Classify tax basis/currency context without extracting any amounts."""
        if not ocr_text.strip():
            return None
        return await self._generate(
            FINANCIAL_SEMANTICS_PROMPT.format(ocr_text=bound_context(ocr_text)),
            token_limit=SELECTION_TOKEN_LIMIT,
            timeout=self.default_timeout,
        )

    async def verify_subtotal(self, ocr_text: str) -> float | None:
        """Extract the explicitly printed subtotal (net sales before tax) from the text."""
        if not ocr_text.strip():
            return None
        result = await self._generate(
            SUBTOTAL_VERIFICATION_PROMPT.format(ocr_text=bound_context(ocr_text)),
            token_limit=SELECTION_TOKEN_LIMIT,
            timeout=self.default_timeout,
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
        """One consolidated bounded call answering the requested assists."""
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
            ocr_text=bound_context(ocr_text),
        )
        return await self._generate(
            prompt,
            token_limit=SELECTION_TOKEN_LIMIT + 128,
            timeout=self.assist_timeout,
        )

    async def normalize_batch_texts(self, texts: list[str], context: str) -> list[str]:
        """Correct spelling mistakes in batch line item texts using receipt context."""
        if not texts:
            return []
        prompt = BATCH_TEXT_NORMALIZATION_PROMPT.format(
            texts=json.dumps(texts), context=context[:2000]
        )
        result = await self._generate(prompt, token_limit=EXTRACTION_TOKEN_LIMIT, timeout=90.0)

        if not result or "corrected_texts" not in result:
            return texts

        corrected = result.get("corrected_texts", [])
        if not isinstance(corrected, list) or len(corrected) != len(texts):
            logger.warning(
                "Batch normalization returned %d items, expected %d. Falling back.",
                len(corrected) if isinstance(corrected, list) else 0,
                len(texts),
            )
            return texts

        return [str(c).strip() if c else t for c, t in zip(corrected, texts)]

    async def normalize_text(self, text: str, context: str) -> str:
        """Correct spelling mistakes in a specific text snippet using full receipt context."""
        return text
