# API Contract

## Authentication
All endpoints (except health and root) require a Bearer token matching the respective consumer's API key.
```
Authorization: Bearer <API_KEY>
```
Missing or invalid API keys return `403 Forbidden` (`Forbidden: Missing or invalid API key.`).

## Rate Limiting
All HTTP responses carry standard rate-limiting headers:
- `X-RateLimit-Limit`: Maximum requests permitted within the current window.
- `X-RateLimit-Remaining`: Requests remaining in the current window.
- `X-RateLimit-Reset`: Seconds remaining until the window resets.

When limits are exceeded, the API returns **`429 Too Many Requests`** along with a `Retry-After: <seconds>` header:
```json
{
  "detail": "Rate limit exceeded. Try again in 42 seconds.",
  "error_code": "rate_limit_exceeded",
  "retry_after": 42
}
```

| Route Group | Scope | Default Quota |
|---|---|---|
| `POST /api/ocr/process` | Authenticated Service (`serms`/`prs`) | 30 requests / minute |
| `POST /api/duplicate-check` | Authenticated Service (`serms`/`prs`) | 60 requests / minute |
| `GET /api/jobs/*`, `GET /api/metrics` | Authenticated Service (`serms`/`prs`) | 120 requests / minute |
| `GET /api/health`, `GET /` | Client IP | 60 requests / minute |
| Unauthenticated / Failed Auth Attempts | Client IP | 15 attempts / minute |

## Endpoints

### POST `/api/ocr/process`
Submits a receipt for async processing. Accepts either a single `file_url` or a multi-part segment list `file_urls` (up to 6 segments for long/tall receipts).

**Request (single image):**
```json
{
  "receipt_id": 42,
  "file_url": "https://...",
  "callback_url": "https://...",
  "source_service": "serms"
}
```

**Request (multi-part long receipt):**
```json
{
  "receipt_id": 42,
  "file_urls": [
    "https://.../segment_01.jpg",
    "https://.../segment_02.jpg",
    "https://.../segment_03.jpg"
  ],
  "callback_url": "https://...",
  "source_service": "serms"
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Receipt queued for OCR processing."
}
```

**Response (422 Unprocessable Entity) — pre-OCR quality rejection:**

Returned *synchronously* when the image fails blur / brightness / size checks
and `force_process` is not set. In multi-segment requests, segments are checked
in parallel and `segment_index` identifies the failing image. **No Celery job is queued and no callback is
sent.** The client must surface `message` / `rejection_reason` as a toast or
modal. A MongoDB audit row is still written with `status: "rejected"`.

```json
{
  "status": "rejected",
  "message": "Image dimensions (100x100px) are too small for accurate OCR. ...",
  "rejection_code": "too_small",
  "rejection_reason": "Image dimensions (100x100px) are too small for accurate OCR. ...",
  "blur_score": 0.0,
  "brightness": 0.0,
  "resolution": [100, 100],
  "segment_index": null,
  "job_id": "uuid",
  "receipt_id": 42
}
```

`rejection_code` values: `too_small` | `blurry` | `too_dark` | `not_a_receipt` | `quality_failed`.

To bypass the gate (e.g. user confirms a retake is impossible), resubmit with
`"force_process": true`.

### POST Callback (Async)
Sent to the provided `callback_url`.
```json
{
  "receipt_id": 42,
  "ocr_confidence_score": 0.85,
  "status": "completed",
  "vendor_name": "...",
  "transaction_date": "...",
  "total_amount": 100.0,
  "subtotal": 90.0,
  "vat_amount": 10.0,
  "discount_amount": 0.0,
  "discount_type": "none",
  "tin": "...",
  "invoice_number": "...",
  "vat_classification": "vat",
  "expense_category": "Meals",
  "is_duplicate": false,
  "duplicate_similarity": null,
  "items": [
    {
      "name": "Item",
      "quantity": 1,
      "price": 100.0
    }
  ]
}
```
> **Note on `discount_type` & `discount_amount`:** `discount_type` is one of `none | senior | pwd | student | promo | employee | other` when a discount is detected (omitted when no discount is present). `discount_amount` is the positive total discount deducted from the gross receipt items/sales.
> **Note on `items[].price`:** `price` is always emitted as the **line total** (extended amount for that item row, already accounting for `quantity`). Consumers must sum `item.price` directly rather than multiplying `item.quantity * item.price` again.

### POST `/api/duplicate-check`
Checks for duplicate receipts.
```json
{
  "receipt_text": "...",
  "source_service": "serms",
  "threshold": 0.85,
  "days_window": 90
}
```

### DELETE `/api/duplicate/embeddings/{receipt_id}`
Purges the stored `pgvector` embedding for a receipt upon deletion or cancellation. Returns `204 No Content` on success.
```http
DELETE /api/duplicate/embeddings/42
Authorization: Bearer <API_KEY>
```

## Financial context and callback additions
`POST /api/ocr/process` accepts optional flat context fields. They are normalized before queueing and propagated unchanged through the job:
```json
{"country":"US","currency":"USD","location":"Las Vegas"}
```
Caller currency/country/location take priority over OCR context. Currency is validated internally as ISO-4217; the outbound callback retains the existing consumer allowlist and omits unsupported codes rather than rejecting the receipt.

Callbacks may additionally include `tax_basis` (`inclusive|exclusive|unknown`), `financial_reconciliation_status`, and `needs_manual_review`. These are optional and legacy fields remain unchanged. Detailed semantics, evidence, reported/computed totals, discrepancy, tax-line breakdown, and review reasons are retained in the Mongo job document.

For long receipts (>= 20 items), if item sums near-reconcile within 3% of subtotal/total, extracted items are preserved in `items`, `needs_manual_review` is set to `true`, `ocr_confidence_score` is capped at `0.79`, and the review reason is recorded as `near_reconciled`. If multi-segment ordering or overlap detection encounters an unexpected discontinuity, `needs_manual_review` is flagged with `segment_gap` (confidence capped at `0.75`).

