"""Redis caching helpers for receipt bytes and 90-day extraction results.

Provides:
- Short-lived byte handoff (ocr:file_bytes:<url_sha256>, TTL 10m) between FastAPI
  and Celery worker to eliminate redundant network downloads.
- 90-day file-hash OCR result cache (ocr:file_hash:<content_sha256>, TTL 90d)
  to bypass OCR and LLM processing on re-uploaded identical receipts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

_redis_client: aioredis.Redis | None = None
_redis_warning_logged = False


def get_redis_client() -> aioredis.Redis:
    """Return shared async Redis client for caching."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=False,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
    return _redis_client


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


async def cache_downloaded_bytes(
    url: str,
    data: bytes,
    ttl_seconds: int | None = None,
) -> None:
    """Store downloaded file bytes in Redis for worker handoff."""
    global _redis_warning_logged
    if not url or not data:
        return
    ttl = ttl_seconds if ttl_seconds is not None else settings.file_cache_byte_ttl
    key = f"ocr:file_bytes:{_hash_url(url)}"
    try:
        client = get_redis_client()
        await client.set(key, data, ex=ttl)
        logger.debug("Cached %d bytes in Redis for url %s (ttl=%ss)", len(data), url, ttl)
    except Exception as exc:  # noqa: BLE001
        if not _redis_warning_logged:
            logger.warning("Redis byte cache unavailable, skipping handoff: %s", exc)
            _redis_warning_logged = True


async def pop_cached_bytes(url: str) -> bytes | None:
    """Retrieve and delete cached file bytes for url from Redis."""
    global _redis_warning_logged
    if not url:
        return None
    key = f"ocr:file_bytes:{_hash_url(url)}"
    try:
        client = get_redis_client()
        # Redis GETDEL (available in Redis 6.2+) or pipeline GET + DEL
        try:
            data = await client.getdel(key)
        except Exception:  # noqa: BLE001
            pipe = client.pipeline()
            pipe.get(key)
            pipe.delete(key)
            results = await pipe.execute()
            data = results[0]
        if data:
            logger.debug("Cache hit: retrieved %d bytes from Redis for url %s", len(data), url)
            return bytes(data) if not isinstance(data, bytes) else data
    except Exception as exc:  # noqa: BLE001
        if not _redis_warning_logged:
            logger.warning("Redis byte cache pop failed, falling back to download: %s", exc)
            _redis_warning_logged = True
    return None


def compute_content_hash(
    file_bytes_list: list[bytes],
    *,
    country: str | None = None,
    currency: str | None = None,
    location: str | None = None,
    llm_provider: str | None = None,
) -> str:
    """Compute deterministic SHA-256 hash for file content, locale overrides, and LLM provider."""
    hasher = hashlib.sha256()
    for b in file_bytes_list:
        hasher.update(b)
    provider_str = (llm_provider if llm_provider is not None else settings.llm_provider).lower().strip()
    meta = f"|{country or ''}|{currency or ''}|{location or ''}|{provider_str}"
    hasher.update(meta.encode("utf-8"))
    return hasher.hexdigest()


async def get_cached_ocr_result(content_hash: str) -> dict[str, Any] | None:
    """Retrieve 90-day cached OCR extraction result by content hash."""
    global _redis_warning_logged
    if not content_hash:
        return None
    key = f"ocr:file_hash:{content_hash}"
    try:
        client = get_redis_client()
        data = await client.get(key)
        if data:
            decoded = data.decode("utf-8") if isinstance(data, bytes) else str(data)
            logger.info("90-day OCR result cache hit for hash %s", content_hash)
            return json.loads(decoded)
    except Exception as exc:  # noqa: BLE001
        if not _redis_warning_logged:
            logger.warning("Redis result cache read failed: %s", exc)
            _redis_warning_logged = True
    return None


async def set_cached_ocr_result(
    content_hash: str,
    payload: dict[str, Any],
    ttl_seconds: int | None = None,
) -> None:
    """Store 90-day cached OCR extraction result by content hash."""
    global _redis_warning_logged
    if not content_hash or not payload:
        return
    ttl = ttl_seconds if ttl_seconds is not None else settings.file_cache_result_ttl
    key = f"ocr:file_hash:{content_hash}"
    try:
        client = get_redis_client()
        serialized = json.dumps(payload, default=str)
        await client.set(key, serialized.encode("utf-8"), ex=ttl)
        logger.debug("Saved 90-day OCR result cache for hash %s (ttl=%ss)", content_hash, ttl)
    except Exception as exc:  # noqa: BLE001
        if not _redis_warning_logged:
            logger.warning("Redis result cache write failed: %s", exc)
            _redis_warning_logged = True
