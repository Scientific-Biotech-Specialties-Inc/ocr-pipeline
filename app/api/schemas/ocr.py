from typing import Any, Optional, List

from pydantic import BaseModel, Field, field_validator

from app.core.financial_semantics import normalize_iso_currency
from app.core.verification import coerce_category

# ISO 4217 codes the pipeline and SERMS both understand.
# Mirrors ``COUNTRY_CURRENCY`` in ``app.core.locale`` so both sides stay in sync.
# Any currency outside this set is coerced to None rather than 422-ing the callback.
SUPPORTED_CURRENCIES: frozenset[str] = frozenset({
    "PHP", "USD", "BND", "MYR", "SGD", "JPY", "HKD", "THB", "AUD", "GBP", "EUR",
})


class OcrProcessRequest(BaseModel):
    receipt_id: int
    file_url: Optional[str] = None
    file_urls: Optional[List[str]] = None
    callback_url: str
    source_service: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    location: Optional[str] = None
    force_process: bool = False

    @field_validator("country", mode="before")
    @classmethod
    def normalize_country(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip().upper()
        return text if len(text) == 2 and text.isalpha() else None

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_request_currency(cls, value: Any) -> Any:
        return normalize_iso_currency(value)

    @field_validator("location", mode="before")
    @classmethod
    def normalize_location(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class OcrProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str


class QualityRejectionResponse(BaseModel):
    """Synchronous pre-OCR rejection so the client can toast immediately.

    Returned as HTTP 422 *before* a Celery job is queued. Consumers should
    surface ``message`` / ``rejection_reason`` as a toast or modal — there is
    no async callback for this path.
    """

    status: str = "rejected"
    message: str = "Image quality check failed."
    rejection_code: str
    rejection_reason: str
    blur_score: float = 0.0
    brightness: float = 0.0
    resolution: tuple[int, int] = (0, 0)
    segment_index: Optional[int] = None
    job_id: Optional[str] = None
    receipt_id: Optional[int] = None


class ReceiptItem(BaseModel):
    """A line item, shaped to survive SERMS' validation.

    SERMS requires ``items.*.quantity`` to be an integer of at least 1 and rejects
    the ENTIRE callback otherwise. A weighed item (0.5 kg) or an OCR'd zero would
    therefore lose a correctly extracted receipt over one line, so quantity is
    clamped here instead.
    """

    name: str = Field(max_length=255)
    quantity: int = Field(ge=1)
    price: float = Field(ge=0)

    @field_validator("quantity", mode="before")
    @classmethod
    def clamp_quantity(cls, value: Any) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 1
        return max(1, round(number))

    @field_validator("name", mode="before")
    @classmethod
    def truncate_name(cls, value: Any) -> str:
        return str(value or "item")[:255]


class OcrCallbackPayload(BaseModel):
    """The outbound contract, mirroring SERMS' ``OcrCallbackRequest`` rules.

    Enforced before sending. Previously the pipeline posted a raw dict straight
    from the language model, so there was no type boundary at all between a 1.5B
    model and the financial system.
    """

    receipt_id: int
    vendor_name: Optional[str] = Field(default=None, max_length=255)
    transaction_date: Optional[str] = None
    total_amount: Optional[float] = Field(default=None, ge=0)
    vat_amount: Optional[float] = Field(default=None, ge=0)
    tin: Optional[str] = Field(default=None, max_length=255)
    invoice_number: Optional[str] = Field(default=None, max_length=255)
    vat_classification: Optional[str] = None
    currency: Optional[str] = Field(default=None, max_length=3)
    expense_category: Optional[str] = None
    location: Optional[str] = Field(default=None, max_length=255)
    ocr_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_duplicate: bool = False
    duplicate_similarity: Optional[float] = None
    items: List[ReceiptItem] = []
    status: Optional[str] = None
    error: Optional[str] = None
    rejection_code: Optional[str] = None
    rejection_reason: Optional[str] = None
    tax_basis: Optional[str] = None
    financial_reconciliation_status: Optional[str] = None
    needs_manual_review: Optional[bool] = None
    subtotal: Optional[float] = Field(default=None, ge=0)
    discount_amount: Optional[float] = Field(default=None, ge=0)
    discount_type: Optional[str] = None

    @field_validator("discount_type", mode="before")
    @classmethod
    def restrict_discount_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized if normalized in ("none", "senior", "pwd", "student", "promo", "employee", "other") else None

    @field_validator("vat_classification", mode="before")
    @classmethod
    def restrict_vat_classification(cls, value: Optional[str]) -> Optional[str]:
        """SERMS validates this with ``in:vat,non-vat``.

        Any third value - including a semantically correct 'not_applicable' for a
        foreign receipt - would 422 the whole callback, so anything else becomes
        null. Non-PH tax semantics are carried internally by ``tax_type`` instead.
        """
        return value if value in ("vat", "non-vat") else None

    @field_validator("currency", mode="before")
    @classmethod
    def restrict_currency(cls, value: Optional[str]) -> Optional[str]:
        """Coerce unrecognised currency codes to None.

        A code outside SUPPORTED_CURRENCIES would be stored as an uncontrolled
        string in SERMS and silently ignored rather than 422-ing the whole callback.
        OCR detection already limits output to the known set, so this is a safety
        net for edge cases rather than an expected path.
        """
        if value is None:
            return None
        upper = value.strip().upper()
        return upper if upper in SUPPORTED_CURRENCIES else None

    @field_validator("tax_basis", mode="before")
    @classmethod
    def restrict_tax_basis(cls, value: Optional[str]) -> Optional[str]:
        return value if value in ("inclusive", "exclusive", "unknown") else None

    @field_validator("financial_reconciliation_status", mode="before")
    @classmethod
    def restrict_reconciliation_status(cls, value: Optional[str]) -> Optional[str]:
        allowed = {"unresolved", "reconciled", "computed", "reported", "reported_conflict", "computed_conflict"}
        return value if value in allowed else None

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        # New fields are optional and omitted when unset, preserving legacy
        # callback payload shape while exposing them when the pipeline has a value.
        for key in (
            "tax_basis", "financial_reconciliation_status", "needs_manual_review",
            "subtotal", "discount_amount", "discount_type",
        ):
            if data.get(key) is None:
                data.pop(key, None)
        return data

    @field_validator("expense_category", mode="before")
    @classmethod
    def restrict_category(cls, value: Optional[str]) -> Optional[str]:
        return coerce_category(value) if value else None

    @field_validator("vendor_name", "tin", "invoice_number", "location", mode="before")
    @classmethod
    def blank_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


def build_callback_payload(
    receipt_id: int,
    fields: dict[str, Any] | None = None,
    confidence: float = 0.0,
    status: str = "completed",
    items: list[dict[str, Any]] | None = None,
    error: str | None = None,
    rejection_code: str | None = None,
    rejection_reason: str | None = None,
    is_duplicate: bool = False,
    duplicate_similarity: float | None = None,
    tax_basis: str | None = None,
    financial_reconciliation_status: str | None = None,
    needs_manual_review: bool | None = None,
    subtotal: float | None = None,
    discount_amount: float | None = None,
    discount_type: str | None = None,
) -> OcrCallbackPayload:
    """Map the internal field set onto the SERMS contract.

    Only the keys SERMS declares are sent. Laravel's ``validated()`` silently drops
    undeclared keys, so the wider internal model (net_sales, service_charge,
    country, currency, review reasons) is persisted on our side rather than
    transmitted and discarded.
    """
    fields_dict = fields or {}
    resolved_subtotal = subtotal if subtotal is not None else (
        fields_dict.get("subtotal") if fields_dict.get("subtotal") is not None else fields_dict.get("net_sales")
    )
    resolved_discount_amount = discount_amount if discount_amount is not None else fields_dict.get("discount_amount")
    resolved_discount_type = discount_type if discount_type is not None else fields_dict.get("discount_type")
    if resolved_discount_amount is not None and resolved_discount_amount > 0 and not resolved_discount_type:
        resolved_discount_type = "other"
    return OcrCallbackPayload(
        receipt_id=receipt_id,
        vendor_name=fields_dict.get("vendor_name"),
        transaction_date=fields_dict.get("transaction_date"),
        total_amount=fields_dict.get("total_amount"),
        subtotal=resolved_subtotal,
        vat_amount=fields_dict.get("tax_amount"),
        discount_amount=resolved_discount_amount,
        discount_type=resolved_discount_type,
        tin=fields_dict.get("vendor_tax_id"),
        invoice_number=fields_dict.get("invoice_number"),
        vat_classification=fields_dict.get("vat_classification"),
        currency=fields_dict.get("currency"),
        expense_category=fields_dict.get("expense_category"),
        location=fields_dict.get("location"),
        ocr_confidence_score=max(0.0, min(float(confidence), 1.0)),
        is_duplicate=is_duplicate,
        duplicate_similarity=round(duplicate_similarity, 4) if duplicate_similarity is not None else None,
        items=_valid_items(items or []),
        status=status,
        error=error,
        rejection_code=rejection_code,
        rejection_reason=rejection_reason or error,
        tax_basis=fields_dict.get("tax_basis", tax_basis),
        financial_reconciliation_status=fields_dict.get(
            "financial_reconciliation_status", financial_reconciliation_status
        ),
        needs_manual_review=fields_dict.get("needs_manual_review", needs_manual_review),
    )



def _valid_items(raw_items: list[dict[str, Any]]) -> list[ReceiptItem]:
    """Keep the line items that validate; drop the ones that do not.

    Losing an unparseable line beats losing the receipt to a 422.
    """
    valid: list[ReceiptItem] = []
    for entry in raw_items:
        try:
            valid.append(ReceiptItem(**entry))
        except Exception:
            continue
    return valid


__all__ = [
    "OcrProcessRequest", "OcrProcessResponse", "QualityRejectionResponse", "ReceiptItem",
    "OcrCallbackPayload", "build_callback_payload", "EXPENSE_CATEGORIES",
    "SUPPORTED_CURRENCIES",
]
