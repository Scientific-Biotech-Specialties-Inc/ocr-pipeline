import asyncio
import logging
import uuid
from io import BytesIO

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from app.api.schemas.ocr import (
    OcrProcessRequest,
    OcrProcessResponse,
    QualityRejectionResponse,
    build_callback_payload,
)
from app.config import settings
from app.core.callback import send_callback
from app.core.file_cache import cache_downloaded_bytes
from app.core.image_quality import check_image_quality
from app.db.mongodb import MongoDBClient
from app.dependencies import verify_api_key
from app.tasks.process_receipt import process_receipt_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["OCR"])


async def _download_file(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        res = await client.get(url, timeout=30.0)
        res.raise_for_status()
        return res.content


@router.post("/process", response_model=OcrProcessResponse, status_code=202)
async def process_receipt(
    request: OcrProcessRequest,
    source_service: str = Depends(verify_api_key),
):
    request.source_service = source_service

    urls = request.file_urls if (request.file_urls and len(request.file_urls) > 0) else ([request.file_url] if request.file_url else [])
    if not urls:
        raise HTTPException(status_code=400, detail="Either file_url or file_urls must be provided.")
    if len(urls) > settings.image_quality_max_segments:
        raise HTTPException(
            status_code=400,
            detail=f"Exceeded maximum segment count of {settings.image_quality_max_segments} (got {len(urls)}).",
        )

    # Synchronous pre-OCR quality check.
    # Rejection is immediate (HTTP 422) — we also dispatch a callback if callback_url is present
    # so async consumers receive the rejection event.
    # When force_process=True, non-faded warnings are bypassed, while "faded" remains enforced.
    try:
        async def _check_one_segment(idx: int, url: str):
            if url.lower().split("?")[0].endswith(".pdf"):
                # Quality check on PDF is skipped in FastAPI; handled by worker
                return idx, url, None, None
            file_bytes = await _download_file(url)
            pil_img = Image.open(BytesIO(file_bytes))
            res = await asyncio.to_thread(check_image_quality, pil_img)
            return idx, url, file_bytes, res

        check_tasks = [_check_one_segment(i, u) for i, u in enumerate(urls)]
        segment_results = await asyncio.gather(*check_tasks)
        segment_results.sort(key=lambda x: x[0])

        failing_item = next(
            (item for item in segment_results if item[3] is not None and item[3].rejection_code == "faded"),
            None,
        )
        if failing_item is None and not request.force_process:
            failing_item = next(
                (item for item in segment_results if item[3] is not None and not item[3].passed),
                None,
            )

        if failing_item is not None:
            idx, _url, _file_bytes, result = failing_item
            segment_idx = idx if len(urls) > 1 else None
            reason = result.rejection_reason or "Image quality check failed."
            code = result.rejection_code or "quality_failed"
            job_id = str(uuid.uuid4())
            try:
                await MongoDBClient.create_job({
                    "job_id": job_id,
                    "receipt_id": request.receipt_id,
                    "file_url": request.file_url or (
                        request.file_urls[0] if request.file_urls else ""
                    ),
                    "file_urls": request.file_urls or [],
                    "callback_url": request.callback_url,
                    "source_service": request.source_service,
                    "force_process": request.force_process,
                    "country": request.country,
                    "currency": request.currency,
                    "location": request.location,
                    "status": "rejected",
                    "error": reason,
                    "rejection_code": code,
                    "quality_check": {
                        "rejection_code": code,
                        "rejection_reason": reason,
                        "blur_score": result.blur_score,
                        "brightness": result.brightness,
                        "resolution": list(result.resolution),
                        "segment_index": segment_idx,
                    },
                })
            except Exception:
                logger.exception(
                    "Failed to persist quality rejection for receipt_id=%s",
                    request.receipt_id,
                )

            if request.callback_url:
                cb_payload = build_callback_payload(
                    receipt_id=request.receipt_id,
                    status="rejected",
                    error=reason,
                    rejection_code=code,
                    rejection_reason=reason,
                )
                try:
                    await send_callback(request.callback_url, cb_payload.model_dump())
                except Exception:
                    logger.exception(
                        "Failed to send quality rejection callback for receipt_id=%s",
                        request.receipt_id,
                    )

            response_payload = QualityRejectionResponse(
                status="rejected",
                message=reason,
                rejection_code=code,
                rejection_reason=reason,
                blur_score=result.blur_score,
                brightness=result.brightness,
                resolution=result.resolution,
                segment_index=segment_idx,
                job_id=job_id,
                receipt_id=request.receipt_id,
            )
            logger.warning(
                "Receipt quality check rejected receipt_id=%s (code=%s, segment=%s, job=%s)",
                request.receipt_id, code, segment_idx, job_id,
            )
            return JSONResponse(
                status_code=422,
                content=response_payload.model_dump(),
            )

        # Store downloaded bytes in Redis for worker handoff (avoids second network download)
        for _idx, url, file_bytes, _result in segment_results:
            if file_bytes is not None:
                await cache_downloaded_bytes(url, file_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Pre-OCR quality check skipped due to error: %s", exc)
        # If image download fails during quality check, let Celery task handle download error or fail fast

    job_id = str(uuid.uuid4())

    job_data = {
        "job_id": job_id,
        "receipt_id": request.receipt_id,
        "file_url": request.file_url or (request.file_urls[0] if request.file_urls else ""),
        "file_urls": request.file_urls or [],
        "callback_url": request.callback_url,
        "source_service": request.source_service,
        "force_process": request.force_process,
        "country": request.country,
        "currency": request.currency,
        "location": request.location,
        "status": "queued",
    }
    await MongoDBClient.create_job(job_data)

    process_receipt_task.delay(
        job_id=job_id,
        receipt_id=request.receipt_id,
        file_url=request.file_url or "",
        file_urls=request.file_urls or [],
        callback_url=request.callback_url,
        source_service=request.source_service,
        force_process=request.force_process,
        country=request.country,
        currency=request.currency,
        location=request.location,
    )

    return OcrProcessResponse(
        job_id=job_id,
        status="queued",
        message="Receipt queued for OCR processing.",
    )
