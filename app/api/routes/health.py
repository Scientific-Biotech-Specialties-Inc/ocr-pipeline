from fastapi import APIRouter
from app.api.schemas.health import HealthResponse

router = APIRouter(prefix="/api", tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    # TODO: Add actual checks for Gemini, Redis, MongoDB, PostgreSQL
    return HealthResponse(
        status="healthy",
        dependencies={
            "gemini": "healthy",
            "redis": "healthy",
            "mongodb": "healthy",
            "postgresql": "healthy"
        },
        version="1.0.0"
    )
