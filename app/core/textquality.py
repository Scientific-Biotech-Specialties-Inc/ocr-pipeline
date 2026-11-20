"""Shared text-quality heuristics.

Used by both candidate collection and candidate ranking so a junk line is
rejected wherever it enters, rather than only at the first gate.
"""

from __future__ import annotations

import re
from typing import Any

MIN_LENGTH = 5
MIN_ALPHA_RATIO = 0.6
MAX_MIXED_CASE_RATIO = 0.34

_PUNCT_ONLY_RE = re.compile(r"^[\s/\\:;,.!?@#$%^&*()\-_=+\[\]{}|~`'\"<>]+$")
_NUMERIC_OR_QTY_RE = re.compile(
    r"^[$€£¥pP]?[-+]?\d[\d,./:%-]*[xX×*%-]*$|^[*Xx#]+\d*$"
)
_ALPHA_WORD_RE = re.compile(r"[A-Za-z][A-Za-z.'&-]*")
_KNOWN_UNIT_OR_TIME_RE = re.compile(
    r"^\d+(?:st|nd|rd|th|pcs|pc|ml|kg|lb|oz|mg|g|l|am|pm)$",
    re.IGNORECASE,
)
_KNOWN_TWO_CHAR_LINES = frozenset({
    "OR", "SI", "DR", "PH", "US", "UK", "DC", "NV", "CA", "TX", "NY", "ID",
    "NO", "OK", "IN", "ON", "TO", "BY", "OF", "PM", "AM", "AT", "AS", "IS",
    "IT", "WE", "GO", "UP", "MY", "SG", "BN",
})
_POS_TAX_FLAG_LINES = frozenset({"T", "F", "N", "FC", "FT", "FN", "TX", "NT"})
_VALID_TWO_CHAR_VENDOR_TOKENS = frozenset({
    "of", "in", "at", "by", "to", "on", "co", "us", "dc", "ph", "la",
    "el", "de", "mr", "ms", "dr", "st", "jr", "sr", "go", "up", "no",
})


def looks_like_words(line: str) -> bool:
    """Reject OCR noise while keeping legitimately messy business names.

    Handwritten pages produce lines like 'fel tolnall Ante latalentl A rer = OI'
    and 'PS eee Z ie Qlva'. Requiring a couple of word-shaped tokens plus
    mostly-alphabetic content filters those without discarding real names such as
    'Ent rek (B) Sdn.Bhd.'.
    """
    stripped = line.strip()
    if len(stripped) < MIN_LENGTH:
        return False

    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters < MIN_LENGTH or letters < len(stripped) * MIN_ALPHA_RATIO:
        return False

    tokens = _ALPHA_WORD_RE.findall(stripped)
    solid = [token for token in tokens if len(token) >= 3]
    if len(solid) < 2:
        # A single word of 4+ letters (e.g. 'Target', 'Safeway', 'Subway') can be a business name,
        # but a scatter of short fragments or trailing 1-char lowercase debris ('Swag t') is noise.
        return (
            len(solid) == 1
            and len(solid[0]) >= 4
            and len(tokens) <= 2
            and not any(re.fullmatch(r"[a-z]", tok) for tok in tokens)
        )

    # Noise alternates case mid-word; real headers are caps or title case.
    odd = sum(1 for token in solid if re.search(r"[a-z][A-Z]", token))
    return odd <= len(solid) * MAX_MIXED_CASE_RATIO


def _has_weird_case_flip(token: str) -> bool:
    clean = token.strip(".,:;!?()\"'")
    if not re.search(r"[a-z][A-Z]", clean):
        return False
    if re.match(r"^(?:Mc|Mac|i[A-Z]|e[A-Z])[A-Za-z.'-]*$", clean):
        return False
    # Two-part PascalCase retail abbreviations (e.g. 'MfrCpn', 'PayPal', 'ShopRite') are valid.
    if re.fullmatch(r"[A-Z][a-z]{2,}[A-Z][a-z]{2,}", clean):
        return False
    return True


def is_gobbly_vendor_name(vendor_name: str | None) -> bool:
    """Return True when a proposed merchant name is missing, truncated, or OCR garbage."""
    if not vendor_name or not isinstance(vendor_name, str):
        return True
    stripped = vendor_name.strip()
    if len(stripped) < 3:
        return True
    if re.search(r"[{}\[\]~|\\<>]", stripped):
        return True
    if re.match(r"^['\"=+*!;,.:/-]", stripped) or re.search(r"[=+*;,:~\"']$", stripped):
        return True
    normalized = stripped.replace("\u2019", "'")
    raw_tokens = [tok for tok in normalized.split() if tok]
    if len(raw_tokens) == 1:
        letters = sum(ch.isalpha() for ch in normalized)
        if letters < 3 or letters < len(normalized) * MIN_ALPHA_RATIO:
            return True
        if not _ALPHA_WORD_RE.fullmatch(normalized):
            return True
    elif not looks_like_words(normalized):
        return True
    # Dangling isolated 1-char lowercase prefix/suffix (e.g. 'Swag t', 'Wet Pets a', 'f ...')
    if re.match(r"^[a-z]\s+", normalized) or re.search(r"(?<![&/])\s+[a-z]$", normalized):
        return True
    # Bare uppercase single-char trailing fragment when not paired with '&' (e.g. 'SF BRINE SHRIMP E')
    if re.search(r"(?<![&/])\s+[A-Z]$", normalized) and len(raw_tokens) >= 3:
        return True
    if any(_has_weird_case_flip(tok) for tok in raw_tokens):
        return True
    for tok in raw_tokens:
        clean = tok.strip(".,:;!?()\"'")
        if not clean or _KNOWN_UNIT_OR_TIME_RE.match(clean):
            continue
        if re.fullmatch(r"[a-z]+\d+[a-z]*|[OoIl]\d{1,3}", clean):
            return True
    alpha_tokens = [tok for tok in _ALPHA_WORD_RE.findall(normalized) if tok != "&"]
    one_char = [tok for tok in alpha_tokens if len(tok) == 1]
    if len(one_char) >= 2:
        return True
    if len(alpha_tokens) >= 4:
        short_noise = [
            tok
            for tok in alpha_tokens
            if len(tok) == 1 or (len(tok) == 2 and tok.islower() and tok not in _VALID_TWO_CHAR_VENDOR_TOKENS)
        ]
        if len(short_noise) / len(alpha_tokens) >= 0.5:
            return True
    return False


def ocr_text_quality_stats(ocr_text: str) -> dict[str, Any]:
    """Compute non-numeric word-shape and garbled-OCR indicators for a full OCR reading."""
    text = ocr_text.replace("\u2019", "'").strip()
    if not text:
        return {
            "text_tokens": [],
            "words": [],
            "word_ratio": 0.0,
            "gobbly_hits": 0,
            "gobbly_ratio": 0.0,
            "line_count": 0,
            "noisy_sym_count": 0,
            "noisy_line_count": 0,
            "case_flip_count": 0,
        }

    raw_tokens = [tok for tok in re.split(r"\s+", text) if tok]
    text_tokens: list[str] = []
    for tok in raw_tokens:
        core = tok.strip("()")
        if not core or _PUNCT_ONLY_RE.match(tok) or _PUNCT_ONLY_RE.match(core):
            continue
        if _NUMERIC_OR_QTY_RE.match(core):
            continue
        text_tokens.append(tok)

    words = [
        tok
        for tok in text_tokens
        if _ALPHA_WORD_RE.fullmatch(tok) and sum(ch.isalpha() for ch in tok) >= 2
    ]
    word_ratio = (len(words) / len(text_tokens)) if text_tokens else 0.0

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    dang_1c = sum(
        1
        for ln in lines
        if re.search(r"(?<![&/.'\d])\s+[A-Za-z]$", ln)
        and not re.search(r"(?:\d|:\s*)\s*[A-Za-z]$", ln)
    )
    mangled_alnum = [
        tok
        for tok in text_tokens
        if len(tok) <= 6
        and not _KNOWN_UNIT_OR_TIME_RE.match(tok.strip(".,:;!?()\"'"))
        and re.fullmatch(r"[a-z]+\d+[a-z]*|[OoIl]\d{1,3}", tok.strip(".,:;!?()\"'"))
    ]
    noisy_sym = [
        tok
        for tok in text_tokens
        if re.search(r"[{}\[\]~|\\<>]", tok)
        or re.match(r"^[=+\-*~^][A-Za-z]|[A-Za-z][=+\-*~^]$", tok)
    ]
    noisy_lines = [
        ln
        for ln in lines
        if re.match(
            r"^(?:[=:~|*+<>\"'\s\d]{1,6}[A-Za-z]{1,3}|[A-Za-z0-9]{1,2}\s*[=:~|*+<>]+(?:\s*[A-Za-z0-9]{1,3})?)$",
            ln,
        )
    ]
    case_flips = [tok for tok in text_tokens if _has_weird_case_flip(tok)]
    short_nonword_lines = [
        ln
        for ln in lines
        if re.fullmatch(r"[A-Za-z]{1,2}", ln)
        and ln.upper() not in _KNOWN_TWO_CHAR_LINES
        and ln not in _POS_TAX_FLAG_LINES
    ]

    gobbly_hits = (
        dang_1c
        + len(mangled_alnum)
        + len(noisy_sym)
        + len(noisy_lines)
        + len(case_flips)
        + len(short_nonword_lines)
    )
    gobbly_ratio = gobbly_hits / max(len(lines), 1)
    return {
        "text_tokens": text_tokens,
        "words": words,
        "word_ratio": word_ratio,
        "gobbly_hits": gobbly_hits,
        "gobbly_ratio": gobbly_ratio,
        "line_count": len(lines),
        "noisy_sym_count": len(noisy_sym),
        "noisy_line_count": len(noisy_lines),
        "case_flip_count": len(case_flips),
    }


def looks_gobbly_ocr(ocr_text: str, mean_word_confidence: float = 1.0) -> bool:
    """Detect truncated, garbled, or symbol-polluted OCR output (even at high engine confidence)."""
    stats = ocr_text_quality_stats(ocr_text)
    if not stats["text_tokens"]:
        return True
    if stats["noisy_sym_count"] >= 2 or stats["noisy_line_count"] >= 2:
        return True
    if stats["case_flip_count"] >= 2:
        return True
    if stats["gobbly_hits"] >= 3 and stats["gobbly_ratio"] >= 0.12:
        return True
    if stats["line_count"] <= 4 and stats["gobbly_hits"] >= 2:
        return True
    return False


def looks_like_address_line(line: str) -> bool:
    """Check if a line looks like a plausible address line.

    Unlike business names which are primarily alphabetic, address lines frequently
    contain street numbers, suite/unit numbers, postal/ZIP codes, and abbreviations.
    Requires minimum alphanumeric content and at least some alphabetic tokens,
    rejecting pure numeric/price lines, punctuation scatter, and random OCR noise.
    """
    stripped = line.strip()
    if len(stripped) < 4:
        return False

    letters = sum(1 for ch in stripped if ch.isalpha())
    digits = sum(1 for ch in stripped if ch.isdigit())
    alnum = letters + digits

    if letters < 2 or (alnum / len(stripped)) < 0.50:
        return False

    # Reject lines that are almost entirely digits with only 1-2 stray letters
    if letters < 3 and digits > 6:
        return False

    tokens = re.findall(r"[A-Za-z0-9]+", stripped)
    if not tokens:
        return False

    if len(tokens) == 1 and not tokens[0].isalpha():
        return False

    alpha_tokens = [t for t in tokens if t.isalpha() and len(t) >= 3]
    if alpha_tokens:
        odd = sum(1 for t in alpha_tokens if re.search(r"[a-z][A-Z]", t))
        if odd > len(alpha_tokens) * MAX_MIXED_CASE_RATIO:
            return False

    return True
