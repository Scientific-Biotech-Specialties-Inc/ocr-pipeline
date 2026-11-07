"""Post-processing normalization for extracted receipt data.

Corrects OCR spelling mistakes using a hybrid approach:
1. Dictionary-based fuzzy matching for known terms (if confidence is moderate).
2. LLM-based context-aware correction for complex item lines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rapidfuzz import distance, process

from app.core.dictionaries import ALL_DICTIONARY_TERMS
from app.core.extraction import Extraction
from app.core.items import looks_like_store_code
from app.core.ocr_engine import OcrBundle
from app.llm.factory import create_provider

logger = logging.getLogger(__name__)


# The LLM batch-normalization call is fixed-cost (~20-80s on CPU) no matter how
# many names it is handed, so it is only worth paying when there is a meaningful
# batch of genuine product names to fix. Fewer than this many correctable names
# (e.g. most restaurant receipts) returns early instead of spending the call.
MIN_CORRECTABLE_NAMES = 3


@dataclass
class NormalizationResult:
    original: str
    corrected: str
    method: str


class DictionaryCorrector:
    """Fuzzy matches tokens against known domain dictionaries."""

    def __init__(
        self,
        vocabulary: set[str] = ALL_DICTIONARY_TERMS,
        min_confidence: float = 0.5,
        max_confidence: float = 0.9,
    ) -> None:
        self.vocabulary = {term.lower() for term in vocabulary}
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

    def _get_word_confidence(self, text: str, bundle: OcrBundle) -> float | None:
        """Estimate the OCR confidence of a specific token from the primary reading.
        
        Since spatial geometry is complex, we do a best-effort exact match against
        the primary reading's Word objects.
        """
        target = text.lower()
        matches = [w for w in bundle.primary.words if w.text.lower() == target]
        if not matches:
            return None
        # Return the lowest confidence if multiple matches, to be conservative
        return min(w.confidence for w in matches)

    def _should_correct(self, confidence: float | None) -> bool:
        """Decide if a word's confidence falls in the correction window."""
        if confidence is None:
            # If we can't find it in the bundle, assume it's moderate
            return True
        return self.min_confidence <= confidence <= self.max_confidence

    def correct_word(self, word: str, bundle: OcrBundle) -> NormalizationResult | None:
        """Attempt to correct a single word."""
        clean_word = re.sub(r"[^\w\s]", "", word).lower()
        if not clean_word or len(clean_word) < 3:
            return None

        # If it's already in the dictionary, it's correct
        if clean_word in self.vocabulary:
            return None

        confidence = self._get_word_confidence(word, bundle)
        if not self._should_correct(confidence):
            return None

        # Max allowed edits: 2 for >= 5 chars, 1 for < 5 chars
        max_edits = 2 if len(clean_word) >= 5 else 1

        match = process.extractOne(
            clean_word,
            self.vocabulary,
            scorer=distance.Levenshtein.distance,
            score_cutoff=max_edits,
        )

        if match:
            best_match_str, edit_dist, _ = match
            # Preserve original casing
            if word.isupper():
                corrected = best_match_str.upper()
            elif word.istitle():
                corrected = best_match_str.title()
            else:
                corrected = best_match_str

            # Re-apply stripped punctuation (naive approach)
            prefix = word[:word.lower().find(clean_word)] if clean_word in word.lower() else ""
            suffix = word[word.lower().find(clean_word) + len(clean_word):] if clean_word in word.lower() else ""
            
            # Fallback if naive sub-string search fails due to internal punctuation
            if not prefix and not suffix and clean_word != word.lower():
                corrected_full = corrected
            else:
                corrected_full = f"{prefix}{corrected}{suffix}"

            logger.debug(
                "Dictionary corrected %r -> %r (edits=%s, conf=%s)",
                word, corrected_full, edit_dist, confidence
            )
            return NormalizationResult(original=word, corrected=corrected_full, method="dictionary")

        return None

    def correct_phrase(self, phrase: str, bundle: OcrBundle) -> str:
        """Tokenize and correct each word in a phrase."""
        if not phrase:
            return phrase
        
        words = phrase.split()
        corrected_words = []
        for word in words:
            result = self.correct_word(word, bundle)
            if result:
                corrected_words.append(result.corrected)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)


class LLMContextCorrector:
    """Uses the LLM to fix complex fields based on surrounding context."""

    async def correct_line_items(self, items: list[dict], raw_text: str) -> list[dict]:
        if not items:
            return items

        provider = create_provider()
        
        # We only process if the provider has normalize_batch_texts implemented
        if not hasattr(provider, "normalize_batch_texts"):
            return items

        # Extract non-empty names
        names = [item.get("name", "") for item in items]
        valid_indices = [i for i, name in enumerate(names) if name]

        if not valid_indices:
            return items

        # Grocery-POS receipts print store codes ('DRD PEV-CUT EP025G') as item
        # names. No model can "correct" a code into a real product name, and
        # sending 40 of them through the LLM costs ~80s per job on CPU for zero
        # gain - so code-style names skip the batch normalization call entirely.
        correctable_indices = [
            i for i in valid_indices if not looks_like_store_code(names[i])
        ]

        # The batch call costs ~20-80s on CPU regardless of how many names it is
        # sent, so a receipt whose items are mostly store codes skips it
        # entirely: the few non-code names are usually the same codes with OCR
        # noise, and the model has no product dictionary to recover them with.
        if not correctable_indices or len(correctable_indices) < len(valid_indices) / 2:
            return items

        # Fixed-cost LLM call: it pays the same ~20-80s whether it is asked to
        # fix one name or forty. A single or pair of genuine names (e.g. most
        # restaurant receipts) is never worth that, and the model has no product
        # dictionary to improve on an already-printed name - so require a
        # meaningful batch before spending the CPU.
        if len(correctable_indices) < MIN_CORRECTABLE_NAMES:
            logger.debug(
                "Skipping LLM batch normalization: %d correctable names "
                "below minimum %d", len(correctable_indices), MIN_CORRECTABLE_NAMES,
            )
            return items

        texts_to_correct = [names[i] for i in correctable_indices]

        try:
            # Batch LLM call for complex name normalization
            corrected_texts = await provider.normalize_batch_texts(texts_to_correct, raw_text)
            
            # Since validation ensures length match, we can safely zip
            if len(corrected_texts) == len(texts_to_correct):
                for idx, c_name in zip(correctable_indices, corrected_texts):
                    orig_name = names[idx]
                    if c_name and c_name.lower() != orig_name.lower():
                        logger.debug("LLM batch corrected item name %r -> %r", orig_name, c_name)
                        items[idx]["name"] = c_name
        except Exception as exc:
            logger.warning("LLM batch correction failed: %s", exc)
            
        return items


async def normalize_extraction(
    extraction: Extraction, bundle: OcrBundle, use_llm: bool = True
) -> None:
    """In-place mutation of the extraction object to apply normalization rules.

    ``use_llm=False`` keeps the deterministic dictionary corrections while
    skipping the Ollama batch call - used by the single-pass fast path where a
    round trip to the model would cost more than the correction it could make.
    """
    dict_corrector = DictionaryCorrector()
    llm_corrector = LLMContextCorrector()
    
    # 1. Dictionary corrections on string fields
    if extraction.vendor_choice and extraction.vendor_choice.name:
        corrected_vendor = dict_corrector.correct_phrase(extraction.vendor_choice.name, bundle)
        extraction.fields["vendor_name"] = corrected_vendor
        extraction.vendor_choice.name = corrected_vendor

    # 2. Hybrid corrections on line items
    if extraction.item_scan and extraction.item_scan.items:
        items_payload = extraction.item_scan.payload()
        
        # Apply dictionary first - store codes are printed identities and must
        # stay verbatim; the dictionary would only mangle them (PEV-CUT -> PEANUT).
        for item_dict in items_payload:
            name = item_dict.get("name")
            if name and not looks_like_store_code(name):
                item_dict["name"] = dict_corrector.correct_phrase(name, bundle)
                
        if not use_llm:
            for original_item, corrected_dict in zip(
                extraction.item_scan.items, items_payload
            ):
                corrected_name = corrected_dict.get("name")
                if corrected_name:
                    original_item.name = corrected_name
                    original_item.descriptions.clear()
            return

        # Apply LLM correction
        corrected_payload = await llm_corrector.correct_line_items(items_payload, bundle.combined_text)
        
        # Update original item objects
        for original_item, corrected_dict in zip(extraction.item_scan.items, corrected_payload):
            corrected_name = corrected_dict.get("name")
            if corrected_name:
                original_item.name = corrected_name
                original_item.descriptions.clear()
