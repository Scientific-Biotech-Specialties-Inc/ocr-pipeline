"""Receipt processing pipeline.

Stage order is load-bearing:

    download -> page selection -> pooled OCR (variants x PSMs)
      -> locale detection            (must precede number parsing)
      -> deterministic baseline extraction (money by geometry, TIN roles, dates)
      -> optional financial-semantics assist (tax basis/currency only)
      -> optional model selection assists (vendor/category)
      -> verification gate           (null anything ungrounded)
      -> confidence with hard caps   (calibrated to the consumer's 0.80 threshold)
      -> validated callback

The language model performs only bounded semantics and closed-list selections;
amounts remain deterministic and every optional assist fails soft.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from io import BytesIO

import httpx
from PIL import Image

from app.api.schemas.ocr import build_callback_payload
from app.config import settings
from app.core.anchors import score_text
from app.core.callback import send_callback
from app.core.confidence import ConfidenceBreakdown, compute_confidence
from app.core.extraction import Extraction, extract, fast_path_sufficient
from app.core.file_cache import (
    compute_content_hash,
    get_cached_ocr_result,
    pop_cached_bytes,
    set_cached_ocr_result,
)
from app.core.financial_semantics import (
    FinancialSemantics,
    infer_tax_basis,
    semantics_requested,
    validate_financial_semantics,
)
from app.core.image_quality import NOT_A_RECEIPT_REASON
from app.core.ocr_engine import (
    OcrBundle,
    OcrReading,
    ocr_executor,
    rank_candidates,
    read_pooled,
    read_region_recovery,
    read_segment_fast,
)
from app.core.pdf_handler import pdf_to_images
from app.core.postprocessing import normalize_extraction
from app.core.segment_merger import (
    merge_segment_readings,
    order_segments,
    split_tall_image_into_strips,
)
from app.core.verification import Verification, verify
from app.db.models import ReceiptEmbedding
from app.db.mongodb import MongoDBClient
from app.db.postgres import AsyncSessionLocal
from app.embeddings.generator import EmbeddingGenerator
from app.embeddings.similarity import find_similar_receipts
from app.llm.factory import create_provider

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 30.0
MAX_PDF_PAGES = 5


def compact_text_for_llm(text: str, head_lines: int = 30, tail_lines: int = 35) -> str:
    """Slice middle item lines from very long OCR text to speed up LLM assists."""
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return text
    omitted = len(lines) - head_lines - tail_lines
    return "\n".join(
        lines[:head_lines]
        + [f"... [{omitted} middle item lines omitted for speed] ..."]
        + lines[-tail_lines:]
    )


def compact_text_for_embedding(text: str, head_lines: int = 15, tail_lines: int = 15) -> str:
    """Focus embedding on header metadata and footer totals rather than getting truncated after 256 tokens."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= head_lines + tail_lines:
        return text
    return "\n".join(lines[:head_lines] + lines[-tail_lines:])


async def download_file(url: str) -> bytes:
    cached = await pop_cached_bytes(url)
    if cached is not None:
        return cached
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        return response.content


def load_images(file_bytes: bytes, file_url: str) -> list[Image.Image]:
    if file_url.lower().split("?")[0].endswith(".pdf"):
        return pdf_to_images(file_bytes)[:MAX_PDF_PAGES]
    return [Image.open(BytesIO(file_bytes))]


def is_likely_receipt(bundle: OcrBundle, extraction: Extraction) -> bool:
    """Evaluate whether the OCR output exhibits signals of a genuine receipt."""
    anchors = bundle.primary.anchors
    if anchors is None:
        anchors = score_text(bundle.combined_text, bundle.confidence)
    has_money = bool((anchors and anchors.money_tokens > 0) or extraction.fields.get("total_amount") is not None)
    has_keywords = bool(anchors and anchors.keyword_hits > 0)
    has_tax_id = bool((anchors and anchors.has_tax_id) or extraction.fields.get("vendor_tax_id") is not None)
    has_date = bool((anchors and anchors.has_date) or extraction.fields.get("transaction_date") is not None)
    has_items = bool(extraction.item_scan and extraction.item_scan.count >= 2)

    # If it has money, receipt keywords, a tax id, or parsed items, it qualifies as a receipt
    if has_money or has_keywords or has_tax_id or has_items:
        return True

    # If it has a date AND a vendor name with at least some words, it might be a simple slip
    return bool(has_date and extraction.fields.get("vendor_name"))


async def process_receipt(
    job_id: str,
    receipt_id: int,
    file_url: str,
    callback_url: str,
    source_service: str,
    force_process: bool = False,
    file_urls: list[str] | None = None,
    country: str | None = None,
    currency: str | None = None,
    location: str | None = None,
) -> None:
    logger.info("Starting OCR pipeline for job %s (force_process=%s)", job_id, force_process)

    try:
        await MongoDBClient.update_job_status(job_id, "processing")

        raw_bytes_list: list[bytes] = []
        is_multipart = bool(file_urls and len(file_urls) >= 2)
        segment_images: list[Image.Image] = []

        if is_multipart:
            download_tasks = [download_file(url) for url in file_urls]  # type: ignore[union-attr]
            raw_bytes_list = list(await asyncio.gather(*download_tasks))
            for b, url in zip(raw_bytes_list, file_urls):  # type: ignore[union-attr]
                imgs = load_images(b, url)
                if imgs:
                    segment_images.append(imgs[0])
            if len(segment_images) < 2:
                raise ValueError("Failed to load at least 2 valid image segments")

            # Sub-tile tall segments in multi-part uploads if virtual tiling is enabled
            if settings.ocr_virtual_tiling:
                expanded_segments: list[Image.Image] = []
                for s_idx, s_img in enumerate(segment_images):
                    strips = split_tall_image_into_strips(s_img, max_strips=3)
                    if len(strips) > 1 and len(expanded_segments) + len(strips) <= 10:
                        for strip in strips:
                            strip._parent_segment_index = s_idx
                        expanded_segments.extend(strips)
                    else:
                        expanded_segments.append(s_img)
                segment_images = expanded_segments

            images = segment_images
        else:
            primary_url = file_url or (file_urls[0] if file_urls else "")
            file_bytes = await download_file(primary_url)
            raw_bytes_list.append(file_bytes)
            images = load_images(file_bytes, primary_url)
            if not images:
                raise ValueError("No images extracted from file")
            # Auto-tiling for single tall images
            if settings.ocr_virtual_tiling and len(images) == 1:
                strips = split_tall_image_into_strips(images[0])
                if len(strips) > 1:
                    segment_images = strips
                    images = strips
                    is_multipart = True

        content_hash = compute_content_hash(
            raw_bytes_list, country=country, currency=currency, location=location
        )

        if not force_process:
            cached_ocr = await get_cached_ocr_result(content_hash)
            if cached_ocr is not None:
                logger.info("Bypassing OCR pipeline via 90-day hash cache hit (job=%s)", job_id)
                combined_text = cached_ocr.get("combined_text", "")
                t0_embed = time.perf_counter()
                embedding = None
                if combined_text.strip():
                    try:
                        embed_text = compact_text_for_embedding(combined_text)
                        embedding = await asyncio.to_thread(
                            EmbeddingGenerator.generate, embed_text
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Embedding generation failed for cached receipt %s: %s", receipt_id, exc)
                await _store_embedding(receipt_id, source_service, combined_text, embedding)
                t_embed = time.perf_counter() - t0_embed

                t0_dup = time.perf_counter()
                is_dup, dup_score, matches = await _evaluate_duplicate(
                    receipt_id, source_service, combined_text, embedding
                )
                t_dup = time.perf_counter() - t0_dup

                job_doc = dict(cached_ocr["job_document"])
                job_doc["quality_override"] = force_process
                job_doc["duplicate_check"] = {
                    "is_duplicate": is_dup,
                    "similarity_score": round(dup_score, 4) if dup_score is not None else None,
                    "matches_count": len(matches),
                }
                await MongoDBClient.update_job_status(job_id, "completed", job_doc)

                cb_payload = build_callback_payload(
                    receipt_id=receipt_id,
                    fields=cached_ocr["callback_fields"],
                    confidence=cached_ocr["confidence"],
                    status="rejected" if is_dup else "completed",
                    items=cached_ocr.get("items", []),
                    tax_basis=cached_ocr.get("tax_basis"),
                    financial_reconciliation_status=cached_ocr.get("financial_reconciliation_status"),
                    needs_manual_review=cached_ocr.get("needs_manual_review"),
                    is_duplicate=is_dup,
                    duplicate_similarity=dup_score,
                    rejection_code="duplicate" if is_dup else None,
                    rejection_reason="Duplicate receipt detected based on semantic similarity." if is_dup else None,
                )
                if not await send_callback(callback_url, cb_payload.model_dump()):
                    raise RuntimeError(f"Callback failed to {callback_url}")
                logger.info("Job %s completed via 90-day cache hit (t_embed=%.3fs, t_dup=%.3fs)", job_id, t_embed, t_dup)
                return

        t0_ocr = time.perf_counter()
        if is_multipart:
            bundle, extraction, metrics, gap_suspected = await _read_and_extract_segments(
                images, caller_country=country, caller_currency=currency, caller_location=location
            )
        else:
            bundle, extraction, metrics = await _read_and_extract(
                images, caller_country=country, caller_currency=currency, caller_location=location
            )
            gap_suspected = False
        t_ocr_total = time.perf_counter() - t0_ocr

        if not force_process and not is_likely_receipt(bundle, extraction):
            logger.warning(
                "Job %s rejected: OCR output lacks receipt signals (money, keywords, tax id)",
                job_id,
            )
            await MongoDBClient.update_job_status(
                job_id,
                "rejected",
                {
                    "status": "rejected",
                    "rejection_code": "not_a_receipt",
                    "rejection_reason": NOT_A_RECEIPT_REASON,
                    "error": NOT_A_RECEIPT_REASON,
                },
            )
            payload = build_callback_payload(
                receipt_id=receipt_id,
                fields=dict(extraction.fields),
                confidence=0.0,
                status="rejected",
                rejection_code="not_a_receipt",
                rejection_reason=NOT_A_RECEIPT_REASON,
            )
            await send_callback(callback_url, payload.model_dump())
            return

        near_reconciled = bool(extraction.item_scan and extraction.item_scan.near_reconciled)
        verification = verify(
            extraction.as_dict(),
            bundle.combined_text,
            reconciled=extraction.reconciled,
            locale_resolved=bool(extraction.locale and extraction.locale.resolved),
            mean_word_confidence=bundle.confidence,
            tax_id_ambiguous=_tax_id_ambiguous(extraction),
            derived_fields=(
                set(extraction.money.derived) if extraction.money else set()
            ) | {"tax_basis", "financial_reconciliation_status", "needs_manual_review",
                 "reported_total", "computed_total", "total_discrepancy", "discount_type"}
            | ({"location"} if location else set()),
            near_reconciled=near_reconciled,
            segment_gap=gap_suspected,
        )
        breakdown = compute_confidence(
            verification,
            anchor_score=bundle.primary.score,
            mean_word_confidence=bundle.confidence,
            reconciled=extraction.reconciled,
            locale_certainty=extraction.locale.certainty if extraction.locale else 0.0,
        )

        t0_embed = time.perf_counter()
        embedding = None
        if bundle.combined_text.strip():
            try:
                embed_text = compact_text_for_embedding(bundle.combined_text)
                embedding = await asyncio.to_thread(
                    EmbeddingGenerator.generate, embed_text
                )
            except Exception as exc:
                logger.warning(
                    "Embedding generation failed for receipt %s: %s", receipt_id, exc
                )
        await _store_embedding(receipt_id, source_service, bundle.combined_text, embedding)
        t_embed = time.perf_counter() - t0_embed

        t0_dup = time.perf_counter()
        is_dup, dup_score, matches = await _evaluate_duplicate(
            receipt_id, source_service, bundle.combined_text, embedding
        )
        t_dup = time.perf_counter() - t0_dup

        await MongoDBClient.update_job_status(
            job_id, "completed",
            _job_document(
                bundle, extraction, verification, breakdown,
                force_process=force_process,
                is_duplicate=is_dup,
                duplicate_similarity=dup_score,
                matches_count=len(matches),
            ),
        )

        callback_fields = dict(verification.fields)
        has_callback_semantics = bool(
            extraction.financial_semantics
            and extraction.financial_semantics.source in {"qwen", "ollama", "gemini", "llm"}
        )
        if not has_callback_semantics:
            for key in ("tax_basis", "financial_reconciliation_status"):
                callback_fields.pop(key, None)

        needs_review = bool(
            verification.needs_manual_review or extraction.fields.get("needs_manual_review", False)
        )
        fin_recon_status = extraction.fields.get("financial_reconciliation_status")
        tax_basis_val = extraction.fields.get("tax_basis") if has_callback_semantics else None

        payload = build_callback_payload(
            receipt_id=receipt_id,
            fields=callback_fields,
            confidence=breakdown.score,
            status="rejected" if is_dup else "completed",
            items=extraction.item_scan.payload() if extraction.item_scan else [],
            tax_basis=tax_basis_val,
            financial_reconciliation_status=(
                fin_recon_status if has_callback_semantics else None
            ),
            needs_manual_review=needs_review if needs_review else None,
            is_duplicate=is_dup,
            duplicate_similarity=dup_score,
            rejection_code="duplicate" if is_dup else None,
            rejection_reason="Duplicate receipt detected based on semantic similarity." if is_dup else None,
        )

        # Cache valid extraction result in Redis for 90 days
        try:
            cache_data = {
                "job_document": _job_document(
                    bundle, extraction, verification, breakdown,
                    force_process=force_process,
                    is_duplicate=False,
                    duplicate_similarity=None,
                    matches_count=0,
                ),
                "combined_text": bundle.combined_text,
                "confidence": breakdown.score,
                "callback_fields": callback_fields,
                "items": extraction.item_scan.payload() if extraction.item_scan else [],
                "tax_basis": tax_basis_val,
                "financial_reconciliation_status": (
                    fin_recon_status if has_callback_semantics else None
                ),
                "needs_manual_review": needs_review if needs_review else None,
            }
            await set_cached_ocr_result(content_hash, cache_data)
        except Exception as cache_exc:  # noqa: BLE001
            logger.warning("Failed to save 90-day OCR result cache: %s", cache_exc)

        logger.info(
            "Job %s complete: engine=%s variant=%s psm=%s score=%.3f reasons=%s. "
            "Timing [Total OCR: %.3fs (Engine: %.3fs, LLM: %.3fs), Embed: %.3fs, Dup: %.3fs]",
            job_id, bundle.primary.engine, bundle.primary.variant, bundle.primary.psm,
            breakdown.score, verification.reasons or "none",
            t_ocr_total, metrics.get("engine", 0.0), metrics.get("llm", 0.0), t_embed, t_dup
        )

        if not await send_callback(callback_url, payload.model_dump()):
            raise RuntimeError(f"Callback failed to {callback_url}")

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        await MongoDBClient.update_job_status(job_id, "failed", {"error": str(exc)})
        await send_callback(
            callback_url,
            {"receipt_id": receipt_id, "status": "failed", "error": str(exc)},
        )
        raise


async def _read_and_extract(
    images: list[Image.Image],
    *,
    caller_country: str | None = None,
    caller_currency: str | None = None,
    caller_location: str | None = None,
) -> tuple[OcrBundle, Extraction, dict]:
    """Read the strongest page and extract from it.

    Multi-page PDFs are scored per page and the best-reading page wins. The
    previous implementation used ``images[0]`` and silently discarded the rest,
    which loses the receipt whenever page one is a cover sheet.
    """
    best: tuple[OcrBundle, Extraction] | None = None
    metrics = {"engine": 0.0, "llm": 0.0}

    # Bounded concurrency: allow processing up to MAX_PDF_PAGES concurrently
    semaphore = asyncio.Semaphore(MAX_PDF_PAGES)

    async def _process_page(index: int, image: Image.Image) -> tuple[OcrBundle, Extraction, float, float]:
        async with semaphore:
            t0_engine = time.perf_counter()
            bundle = await read_pooled(image, lang="eng")
            t_engine = time.perf_counter() - t0_engine

            if bundle.early_exit:
                baseline = extract(
                    bundle,
                    caller_country=caller_country,
                    caller_currency=caller_currency,
                    caller_location=caller_location,
                )
                if fast_path_sufficient(baseline) and baseline.fields.get("transaction_date") is not None:
                    # Check if location or vendor is missing; try cheap header crop recovery before shipping
                    needs_header_rec = (
                        (baseline.fields.get("location") is None and caller_location is None)
                        or baseline.fields.get("vendor_name") is None
                    )
                    if needs_header_rec:
                        try:
                            loop = asyncio.get_running_loop()
                            rec_reading = await loop.run_in_executor(
                                ocr_executor, read_region_recovery, image, 0.0, 0.30, "eng"
                            )
                            if rec_reading:
                                if isinstance(rec_reading, list):
                                    bundle.supporting.extend(rec_reading)
                                    bundle.all_readings.extend(rec_reading)
                                else:
                                    bundle.supporting.append(rec_reading)
                                    bundle.all_readings.append(rec_reading)
                                baseline = extract(
                                    bundle,
                                    caller_country=caller_country,
                                    caller_currency=caller_currency,
                                    caller_location=caller_location,
                                )
                        except Exception as rec_exc:  # noqa: BLE001
                            logger.debug("Header region recovery skipped: %s", rec_exc)

                    await normalize_extraction(baseline, bundle, use_llm=False)
                    return bundle, baseline, t_engine, 0.0

                # Single pass could not reconcile the receipt or date missing: escalate to the
                # full pool so hard receipts keep the exact pooled accuracy.
                logger.info(
                    "Page %s fast path insufficient (reconciled=%s, total=%s, date=%s); "
                    "escalating to pooled OCR",
                    index, baseline.reconciled,
                    baseline.fields.get("total_amount") is not None,
                    baseline.fields.get("transaction_date") is not None,
                )
                t0_engine = time.perf_counter()
                bundle = await read_pooled(image, lang="eng", fast_path=False)
                t_engine += time.perf_counter() - t0_engine

            t0_llm = time.perf_counter()
            extraction = await _extract_with_assists(
                bundle,
                caller_country=caller_country,
                caller_currency=caller_currency,
                caller_location=caller_location,
            )
            t_llm = time.perf_counter() - t0_llm
            return bundle, extraction, t_engine, t_llm

    tasks = [_process_page(i, img) for i, img in enumerate(images)]
    
    # Use return_exceptions=True so one failed page doesn't crash the whole receipt
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for index, res in enumerate(results):
        if isinstance(res, Exception):
            logger.warning("Page %s extraction failed: %s", index, res)
            continue
            
        bundle, extraction, engine_time, llm_time = res
        metrics["engine"] += engine_time
        metrics["llm"] += llm_time
        
        rank = (
            1 if extraction.reconciled else 0,
            1 if extraction.fields.get("total_amount") is not None else 0,
            bundle.primary.score,
        )
        if best is None or rank > _rank_of(best):
            best = (bundle, extraction)
        logger.debug("Page %s scored %s", index, rank)

    if best is None:
        raise RuntimeError("All pages failed to extract")
    return best[0], best[1], metrics


def _rank_of(pair: tuple[OcrBundle, Extraction]) -> tuple[int, int, float]:
    bundle, extraction = pair
    return (
        1 if extraction.reconciled else 0,
        1 if extraction.fields.get("total_amount") is not None else 0,
        bundle.primary.score,
    )


async def _read_and_extract_segments(
    segment_images: list[Image.Image],
    *,
    caller_country: str | None = None,
    caller_currency: str | None = None,
    caller_location: str | None = None,
) -> tuple[OcrBundle, Extraction, dict, bool]:
    """Process multiple segments (multi-part uploads or virtual strips).

    Runs fast OCR concurrently per segment, auto-orders them, removes seam duplicates,
    and constructs a unified OcrBundle before running extraction.
    """
    metrics = {"engine": 0.0, "llm": 0.0}

    # 1. Fast Pass across all segments
    t0_engine = time.perf_counter()
    loop = asyncio.get_running_loop()
    valid_pairs: list[tuple[Image.Image, OcrReading]] = []
    for idx, img in enumerate(segment_images):
        try:
            res = await loop.run_in_executor(ocr_executor, read_segment_fast, img, "eng")
            valid_pairs.append((img, res))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Segment %s fast OCR failed: %s", idx, exc)

    metrics["engine"] += time.perf_counter() - t0_engine

    if not valid_pairs:
        raise RuntimeError("All segments failed fast OCR")

    # 2. Auto-order segments
    ordered_pairs = order_segments(valid_pairs)

    # 3. Text-level seam merge
    merged_reading, gap_suspected = merge_segment_readings(ordered_pairs)
    merged_bundle = OcrBundle(
        primary=merged_reading,
        supporting=[],
        all_readings=[merged_reading],
        early_exit=True,
    )

    # 4. Fast deterministic extraction over merged reading
    embedder = _embedder()
    baseline = extract(
        merged_bundle,
        caller_country=caller_country,
        caller_currency=caller_currency,
        caller_location=caller_location,
        embedder=embedder,
    )

    is_sufficient = (
        (baseline.reconciled or (baseline.item_scan and baseline.item_scan.near_reconciled))
        and baseline.fields.get("total_amount") is not None
        and baseline.fields.get("transaction_date") is not None
    )

    if is_sufficient:
        await normalize_extraction(baseline, merged_bundle, use_llm=False)
        return merged_bundle, baseline, metrics, gap_suspected

    # 5. Targeted region recovery: contrast-enhanced crop OCR for missing header/footer
    needs_total_recovery = baseline.fields.get("total_amount") is None or not baseline.reconciled
    needs_vendor_recovery = (
        baseline.fields.get("vendor_name") is None and baseline.fields.get("vendor_tax_id") is None
    )
    needs_date_recovery = baseline.fields.get("transaction_date") is None
    needs_location_recovery = baseline.fields.get("location") is None and caller_location is None
    if (needs_total_recovery or needs_vendor_recovery or needs_date_recovery or needs_location_recovery) and ordered_pairs:
        logger.info(
            "Multi-segment fast path: targeted recovery triggered (needs_total=%s, needs_vendor=%s, needs_date=%s, needs_location=%s)",
            needs_total_recovery, needs_vendor_recovery, needs_date_recovery, needs_location_recovery,
        )
        t0_rec = time.perf_counter()
        recovery_tasks = []
        if (needs_total_recovery or needs_date_recovery) and ordered_pairs:
            # Bottom 35% of the last segment (where total/subtotal/footer-date resides)
            last_img, _ = ordered_pairs[-1]
            recovery_tasks.append(
                loop.run_in_executor(ocr_executor, read_region_recovery, last_img, 0.65, 1.0, "eng")
            )
        if (needs_vendor_recovery or needs_date_recovery or needs_location_recovery) and ordered_pairs:
            # Top 30% of the first segment (where vendor/TIN/header-date/location resides)
            first_img, _ = ordered_pairs[0]
            recovery_tasks.append(
                loop.run_in_executor(ocr_executor, read_region_recovery, first_img, 0.0, 0.30, "eng")
            )

        recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
        metrics["engine"] += time.perf_counter() - t0_rec

        recovered_readings: list[OcrReading] = []
        for res in recovery_results:
            if isinstance(res, Exception):
                logger.warning("Region recovery failed: %s", res)
            elif isinstance(res, list):
                recovered_readings.extend(res)
            elif isinstance(res, OcrReading):
                recovered_readings.append(res)

        if recovered_readings:
            merged_bundle.supporting.extend(recovered_readings)
            merged_bundle.all_readings.extend(recovered_readings)
            merged_bundle.early_exit = False
            logger.info("Added %d recovered reading(s) to merged bundle", len(recovered_readings))

    # 6. Extract with assists on merged bundle
    t0_llm = time.perf_counter()
    extraction = await _extract_with_assists(
        merged_bundle,
        caller_country=caller_country,
        caller_currency=caller_currency,
        caller_location=caller_location,
    )
    metrics["llm"] += time.perf_counter() - t0_llm

    return merged_bundle, extraction, metrics, gap_suspected


async def _extract_with_assists(
    bundle: OcrBundle,
    *,
    caller_country: str | None = None,
    caller_currency: str | None = None,
    caller_location: str | None = None,
) -> Extraction:
    """Deterministic extraction, then one bounded model assist call.

    Runs the deterministic pass first so shortlists exist to constrain the model
    with. All model asks - vendor, location, semantics, subtotal, and category -
    are answered through Gemini 2.0 Flash. Any failure leaves the deterministic
    result in place.
    """
    embedder = _embedder()
    baseline = extract(
        bundle,
        caller_country=caller_country,
        caller_currency=caller_currency,
        caller_location=caller_location,
        embedder=embedder,
    )

    vendor_candidates = baseline.vendor_choice.shortlist if baseline.vendor_choice else []
    if len(vendor_candidates) < 2:
        vendor_candidates = []

    location_candidates = baseline.address_candidates
    if len(location_candidates) < 2:
        location_candidates = []

    money = baseline.money
    want_subtotal = False
    if money and money.get("net_sales") is not None:
        net_sales = money.get("net_sales")
        total = money.get("total_amount")
        is_impossible = total is not None and net_sales > total + 0.05
        reconciled_on_net = (
            baseline.item_scan and baseline.item_scan.reconciled
            and baseline.item_scan.target == net_sales
        )
        want_subtotal = is_impossible or (
            baseline.item_scan and baseline.item_scan.count > 0
            and not reconciled_on_net
        )

    # Smart gating: if currency is already resolved and the baseline is reconciled
    # or explicit tax wording is present, extract() ignores LLM semantics anyway.
    want_semantics = False
    currency_known = baseline.fields.get("currency") is not None
    if not currency_known:
        want_semantics = semantics_requested(bundle.combined_text, currency_ambiguous=True)
    elif not baseline.reconciled:
        inferred_basis, basis_evidence = infer_tax_basis(
            bundle.combined_text, baseline.fields.get("country")
        )
        is_fallback = basis_evidence in (
            ["PH country fallback"], ["US country fallback"], ["MY SST/GST fallback"],
        )
        if inferred_basis == "unknown" or is_fallback:
            want_semantics = semantics_requested(bundle.combined_text, currency_ambiguous=False)

    semantics: FinancialSemantics | None = None
    vendor_choice: str | None = None
    location_choice: str | None = None
    corrected_subtotal: float | None = None

    if vendor_candidates or location_candidates or want_semantics or want_subtotal:
        raw = await _ask_model_for_assists(
            bundle.combined_text,
            vendor_candidates=vendor_candidates,
            excluded_vendors=baseline.vendor_candidates.customer_names,
            location_candidates=location_candidates,
            want_semantics=want_semantics,
            want_subtotal=want_subtotal,
        )
        if raw:
            if want_semantics:
                sem = validate_financial_semantics(
                    raw, bundle.combined_text, source=settings.llm_provider.lower()
                )
                arithmetic = _grounded_exclusive_basis(baseline, sem)
                semantics = arithmetic if arithmetic is not None else sem
            if vendor_candidates:
                vendor_choice = _closed_list_pick(raw.get("vendor_name"), vendor_candidates)
            if location_candidates:
                location_choice = _closed_list_pick(raw.get("location"), location_candidates)
            if want_subtotal:
                corrected_subtotal = _float_or_none(raw.get("subtotal"))

    category_tiebreaker_fn = None
    if (
        baseline.category_choice
        and baseline.category_choice.runner_up
        and baseline.category_choice.similarity is not None
        and baseline.category_choice.similarity < 0.30
    ):
        try:
            provider = create_provider()
            picked_cat = await provider.choose_category(
                bundle.combined_text[:1200],
                (baseline.category_choice.category, baseline.category_choice.runner_up),
            )
            if picked_cat:
                def category_tiebreaker_fn(_haystack: str, _options: tuple[str, ...]) -> str:
                    return picked_cat
        except Exception as exc:  # noqa: BLE001
            logger.debug("Category tiebreak assist skipped: %s", exc)

    if (vendor_choice is None and location_choice is None
            and semantics is None and corrected_subtotal is None
            and category_tiebreaker_fn is None):
        await normalize_extraction(baseline, bundle, use_llm=settings.enable_llm_item_normalization)
        return baseline

    final_extraction = await asyncio.to_thread(
        functools.partial(
            extract,
            bundle,
            llm_vendor_choice=vendor_choice,
            llm_location_choice=location_choice,
            financial_semantics=semantics,
            caller_country=caller_country,
            caller_currency=caller_currency,
            caller_location=caller_location,
            embedder=embedder,
            category_tiebreaker=category_tiebreaker_fn,
            llm_subtotal=corrected_subtotal,
        )
    )
    
    await normalize_extraction(final_extraction, bundle, use_llm=settings.enable_llm_item_normalization)
    return final_extraction


def _closed_list_pick(value: object, candidates: list[str]) -> str | None:
    """Reject any answer outside the offered closed list, allowing fuzzy alignment."""
    if not value or not isinstance(value, str):
        return None
    picked = value.strip()
    if picked in candidates:
        return picked
    from app.core.vendor import _closest_candidate
    return _closest_candidate(picked, candidates)


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _grounded_exclusive_basis(
    baseline: Extraction, current: FinancialSemantics | None = None
) -> FinancialSemantics | None:
    """Use independent printed arithmetic to outrank an LLM opinion."""
    money = baseline.money
    if money is None:
        return None
    explicit, evidence = infer_tax_basis(
        "\n".join(baseline.lines), baseline.locale.country if baseline.locale else None
    )
    if evidence and evidence != ["PH country fallback"] and explicit == "inclusive":
        return None
    net, tax, total = money.get("net_sales"), money.get("tax_amount"), money.get("total_amount")
    if net is None or tax is None or total is None:
        return None
    if any(name in money.derived for name in ("net_sales", "tax_amount", "total_amount")):
        return None
    if abs(net + tax + (money.get("service_charge") or 0.0) - total) > 0.02:
        return None
    snippets = tuple(value for key, value in money.evidence.items() if key in {"net_sales", "tax_amount", "total_amount"})
    return FinancialSemantics(
        tax_basis="exclusive", confidence=1.0, evidence=snippets,
        currency=current.currency if current else "unknown",
        currency_confidence=current.currency_confidence if current else 0.0,
        currency_evidence=current.currency_evidence if current else (), source="arithmetic",
    )
async def _ask_model_for_assists(
    ocr_text: str,
    *,
    vendor_candidates: list[str],
    excluded_vendors: list[str] | None = None,
    location_candidates: list[str],
    want_semantics: bool,
    want_subtotal: bool,
) -> dict | None:
    """One bounded model call answering all requested assists.

    Returns the raw JSON object; field-level validation happens in the caller.
    """
    try:
        provider = create_provider()
        compact_text = compact_text_for_llm(ocr_text)
        return await provider.analyze_assists(
            compact_text,
            vendor_candidates=vendor_candidates or None,
            excluded_vendors=excluded_vendors,
            location_candidates=location_candidates or None,
            want_semantics=want_semantics,
            want_subtotal=want_subtotal,
        )
    except Exception as exc:
        logger.warning("Model assists unavailable, keeping deterministic result: %s", exc)
        return None



def _embedder():
    """Return an embedding callable, or None when the model is unavailable."""
    try:
        model = EmbeddingGenerator.get_model()
    except Exception as exc:
        logger.debug("Embeddings unavailable for categorisation: %s", exc)
        return None

    def encode(texts):
        return [vector.tolist() for vector in model.encode(list(texts))]

    return encode


def _tax_id_ambiguous(extraction: Extraction) -> bool:
    """Several tax ids and none carrying an explicit vendor marker."""
    candidates = extraction.tax_id_candidates
    if any(candidate.role == "vendor" for candidate in candidates):
        return False
    return len([c for c in candidates if c.role == "unknown"]) > 1


async def _store_embedding(
    receipt_id: int, source_service: str, text: str, embedding: list[float] | None = None
) -> None:
    """Persist a text embedding for duplicate detection.

    Deliberately non-fatal: duplicate detection is a separate concern, and losing
    an embedding should not cost a successfully extracted receipt.
    """
    if not text.strip():
        return
    try:
        if embedding is None:
            embed_text = compact_text_for_embedding(text)
            embedding = await asyncio.to_thread(EmbeddingGenerator.generate, embed_text)
        async with AsyncSessionLocal() as session:
            session.add(
                ReceiptEmbedding(
                    receipt_id=receipt_id,
                    source_service=source_service,
                    embedding=embedding,
                    receipt_text=text,
                )
            )
            await session.commit()
    except Exception as exc:
        logger.warning("Embedding storage failed for receipt %s: %s", receipt_id, exc)


async def _evaluate_duplicate(
    receipt_id: int, source_service: str, text: str, embedding: list[float] | None = None
) -> tuple[bool, float | None, list]:
    """Check pgvector for duplicate receipts within the configured lookback window.

    Deliberately non-fatal: if vector query fails, defaults to (False, None, []).
    """
    if not text.strip():
        return False, None, []
    try:
        if embedding is None:
            embed_text = compact_text_for_embedding(text)
            embedding = await asyncio.to_thread(EmbeddingGenerator.generate, embed_text)
        async with AsyncSessionLocal() as session:
            matches = await find_similar_receipts(
                session=session,
                embedding=embedding,
                source_service=source_service,
                threshold=settings.duplicate_similarity_threshold,
                days_window=settings.duplicate_days_window,
                exclude_receipt_id=receipt_id,
            )
        if matches:
            top_score = matches[0].similarity_score
            return True, top_score, matches
        return False, None, []
    except Exception as exc:
        logger.warning("Duplicate evaluation failed for receipt %s: %s", receipt_id, exc)
        return False, None, []


def _job_document(
    bundle: OcrBundle,
    extraction: Extraction,
    verification: Verification,
    breakdown: ConfidenceBreakdown,
    force_process: bool = False,
    is_duplicate: bool = False,
    duplicate_similarity: float | None = None,
    matches_count: int = 0,
) -> dict:
    """The audit record.

    Holds the full internal model, not just the subset SERMS accepts. Laravel's
    ``validated()`` discards undeclared keys, so net_sales, service_charge,
    country, currency, review reasons and rejected values are only useful if we
    keep them - and they are what makes a score explainable after the fact.
    """
    ranked_candidates = rank_candidates(bundle.all_readings)
    runner_up = next(
        (candidate for candidate in ranked_candidates if candidate is not bundle.primary),
        None,
    )
    selection_margin = (
        bundle.primary.score - runner_up.score if runner_up is not None else 0.0
    )
    candidate_summary = [
        {
            "engine": candidate.engine,
            "variant": candidate.variant,
            "psm": candidate.psm,
            "anchor_score": round(candidate.score, 4),
            "confidence": round(candidate.confidence, 4),
        }
        for candidate in ranked_candidates[:5]
    ]

    return {
        "quality_override": force_process,
        "raw_ocr_text": bundle.combined_text,
        # Kept as an engine-neutral field. The legacy field is populated only
        # when Tesseract actually won, so it is not mislabeled for Paddle reads.
        "ocr_confidence": round(bundle.confidence, 4),
        "tesseract_confidence": (
            round(bundle.confidence, 4)
            if bundle.primary.engine == "tesseract" else None
        ),
        "composite_confidence_score": breakdown.score,
        "confidence_breakdown": breakdown.as_dict(),
        "duplicate_check": {
            "is_duplicate": is_duplicate,
            "similarity_score": round(duplicate_similarity, 4) if duplicate_similarity is not None else None,
            "matches_count": matches_count,
        },
        "extracted_data": verification.fields,
        "financial_semantics": (
            extraction.financial_semantics.as_dict()
            if extraction.financial_semantics else None
        ),
        "ocr_selection": {
            "engine": bundle.primary.engine,
            "variant": bundle.primary.variant,
            "psm": bundle.primary.psm,
            "anchor_score": round(bundle.primary.score, 4),
            "confidence": round(bundle.primary.confidence, 4),
            "selection_margin": round(selection_margin, 4),
            "pooled_variants": [
                f"{reading.engine}/{reading.variant}/psm{reading.psm}"
                for reading in bundle.supporting
            ],
            "candidates_evaluated": len(bundle.all_readings),
            "candidate_summary": candidate_summary,
        },
        "verification": {
            "needs_manual_review": verification.needs_manual_review,
            "review_reasons": verification.reasons,
            "rejected_fields": {k: str(v) for k, v in verification.rejected.items()},
            "grounding_pass_rate": round(verification.grounding_pass_rate, 4),
        },
        "evidence": extraction.evidence,
        "items": {
            "rows_found": extraction.item_scan.count if extraction.item_scan else 0,
            "reconciled": bool(extraction.item_scan and extraction.item_scan.reconciled),
            "price_basis": extraction.item_scan.price_basis if extraction.item_scan
            else "none",
            "notes": extraction.item_scan.notes if extraction.item_scan else [],
            # Rows found but withheld are kept here: useful for auditing why a
            # receipt shipped without items, and never transmitted.
            "candidates": [
                {"name": item.full_name, "quantity": item.quantity,
                 "price": item.price, "line": item.line_text}
                for item in (extraction.item_scan.items if extraction.item_scan else [])
            ],
        },
        "reconciliation": {
            "reconciled": extraction.reconciled,
            "tax_basis": extraction.money.tax_basis if extraction.money else "unknown",
            "financial_reconciliation_status": extraction.money.financial_reconciliation_status if extraction.money else "unresolved",
            "reported_total": extraction.money.reported_total if extraction.money else None,
            "computed_total": extraction.money.computed_total if extraction.money else None,
            "discrepancy": extraction.money.discrepancy if extraction.money else None,
            "needs_manual_review": extraction.money.needs_manual_review if extraction.money else False,
            "tax_breakdown": extraction.money.tax_breakdown if extraction.money else [],
            "notes": extraction.money.reconciliation_notes if extraction.money else [],
            "derived_fields": sorted(extraction.money.derived) if extraction.money else [],
        },
        "locale": {
            "country": extraction.fields.get("country"),
            "currency": extraction.fields.get("currency"),
            "evidence": extraction.locale.evidence if extraction.locale else [],
        },
    }
