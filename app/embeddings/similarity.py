from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import ReceiptEmbedding
from datetime import datetime, timedelta, timezone
from app.api.schemas.duplicate import DuplicateMatch

async def find_similar_receipts(
    session: AsyncSession,
    embedding: list[float],
    source_service: str,
    threshold: float,
    days_window: int,
    exclude_receipt_id: int | None = None,
) -> list[DuplicateMatch]:
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_window)
    
    # pgvector provides vector_cosine_ops. The operator <=> calculates cosine distance.
    # cosine similarity = 1 - cosine distance
    # We want cosine distance <= 1 - threshold
    distance_threshold = 1.0 - threshold
    
    stmt = (
        select(
            ReceiptEmbedding.receipt_id,
            ReceiptEmbedding.source_service,
            ReceiptEmbedding.created_at,
            ReceiptEmbedding.embedding.cosine_distance(embedding).label('distance')
        )
        .where(ReceiptEmbedding.source_service == source_service)
        .where(ReceiptEmbedding.created_at >= cutoff_date)
        .where(ReceiptEmbedding.embedding.cosine_distance(embedding) <= distance_threshold)
    )

    if exclude_receipt_id is not None:
        stmt = stmt.where(ReceiptEmbedding.receipt_id != exclude_receipt_id)

    stmt = stmt.order_by(text('distance ASC')).limit(10)
    
    result = await session.execute(stmt)
    rows = result.all()
    
    matches = []
    for row in rows:
        similarity_score = 1.0 - float(row.distance)
        matches.append(DuplicateMatch(
            receipt_id=row.receipt_id,
            source_service=row.source_service,
            similarity_score=similarity_score,
            processed_at=row.created_at.isoformat()
        ))
        
    return matches
