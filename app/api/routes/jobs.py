from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas.jobs import JobStatusResponse
from app.db.mongodb import MongoDBClient
from app.dependencies import verify_api_key

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str, source_service: str = Depends(verify_api_key)):
    job = await MongoDBClient.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.get("source_service") != source_service:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    created_at = job.get("created_at")
    created_at_str = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)

    completed_at = (
        job.get("completed_at")
        or (job.get("updated_at") if job.get("status") in {"completed", "failed", "rejected"} else None)
    )
    completed_at_str = (
        completed_at.isoformat()
        if hasattr(completed_at, "isoformat")
        else (str(completed_at) if completed_at else None)
    )

    duration_ms = job.get("duration_ms")
    if (
        duration_ms is None
        and created_at
        and completed_at
        and hasattr(completed_at, "timestamp")
        and hasattr(created_at, "timestamp")
    ):
        duration_ms = max(0, int((completed_at.timestamp() - created_at.timestamp()) * 1000))

    return JobStatusResponse(
        job_id=job.get("job_id", job_id),
        receipt_id=int(job.get("receipt_id", 0)),
        source_service=job.get("source_service", source_service),
        status=job.get("status", "unknown"),
        created_at=created_at_str,
        completed_at=completed_at_str,
        duration_ms=duration_ms,
    )
