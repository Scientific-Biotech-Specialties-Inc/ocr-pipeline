from typing import List, Optional
from pydantic import BaseModel

class DuplicateCheckRequest(BaseModel):
    receipt_text: str
    source_service: Optional[str] = None
    threshold: float = 0.85
    days_window: int = 90

class DuplicateMatch(BaseModel):
    receipt_id: int
    source_service: str
    similarity_score: float
    processed_at: str

class DuplicateCheckResponse(BaseModel):
    is_duplicate: bool
    matches: List[DuplicateMatch]
