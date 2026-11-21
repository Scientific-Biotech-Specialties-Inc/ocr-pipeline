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

**Request (multi-part / Multi Receipt batch):**
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
> **Note on Multi-Receipt Auto-Grouping:** When `file_urls` contains multiple images (`len >= 2`), the worker automatically distinguishes whether the uploads are segments of **one long receipt** (stitched seamlessly as before) or **multiple distinct receipts / mixed batches** (grouped via deterministic identity signals, header/footer completeness, and boilerplate-hardened item/barcode seam overlap).

**Response (202 Accepted):**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Receipt queued for OCR processing."
}
```

**Response (422 Unprocessable Entity) — pre-OCR quality rejection:**

Returned *synchronously* when the EXIF-normalized image fails size / brightness / faded / blur / non-receipt checks.
On a single image, all applicable checks are evaluated: if all four pre-OCR gate groups fail (`too_small`, `too_dark` or `faded`, `blurry`, and `not_a_receipt`), the response reports `rejection_code: "quality_failed"` with a general quality/verification rejection message; otherwise a single code is chosen using the suppression and priority hierarchy `not_a_receipt > too_small > too_dark > faded > blurry` (`not_a_receipt` suppresses quality flags, and `too_dark` / `faded` suppress `blurry`).
In multi-segment requests (`len(file_urls) > 1`), segments are checked in parallel (`not_a_receipt` is prioritized first, then `quality_failed`, then `faded`, then the first failing segment). Synchronous `422` is returned when **all** non-PDF segments fail `check_image_quality`; when at least one segment passes, the API caches the downloaded bytes and returns `202 Accepted` so the worker can determine whether passing segments form complete separate receipts (isolating only the failing segment) or an incomplete fragment of a long receipt (rejecting the job).
All pre-OCR quality rejections (`too_small`, `blurry`, `too_dark`, `faded`, `not_a_receipt`, `quality_failed`) are **strictly non-overridable** regardless of `force_process`.
**No Celery job is queued and no callback is sent** unless `callback_url` is supplied (in which case a rejection callback is also dispatched). The client must surface `message` / `rejection_reason` as a toast or
modal. A MongoDB audit row is still written with `status: "rejected"`; raw triggered flags (`quality_check.detected_issues`) are persisted internally in MongoDB `quality_check` (both on FastAPI `422` rejections and worker pre-OCR rejections), while the HTTP `422` response and webhook callback payload schemas remain unchanged with a single `rejection_code`.

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

`rejection_code` values: `too_small` | `blurry` | `too_dark` | `faded` | `not_a_receipt` | `unreadable` | `quality_failed` | `duplicate`.

Quality, non-receipt, `unreadable`, and `duplicate` rejections are strictly non-overridable (`force_process` does not bypass any pre-OCR or post-OCR rejection gate).

### POST Callback (Async)
Sent to the provided `callback_url`. When post-OCR or post-verification evaluation rejects a receipt (`faded`, `not_a_receipt`, `unreadable`) or detects a duplicate (`duplicate`), `status` is `"rejected"` with `rejection_code` and a specific `rejection_reason` (for `faded` and `unreadable`, naming which critical fields failed to read: `Total Amount`, `Transaction Date`, `Merchant Name`, and/or garbled or illegible OCR text).
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

### Multi-Receipt Split Callback Fields (`segment_indices`, `split_receipts`)
When `file_urls` resolves to **one** receipt (single image or multi-segment long receipt), `segment_indices` and `split_receipts` remain `None` and are **omitted** from the outbound callback JSON so existing single-receipt contracts are unchanged.

When `file_urls` resolves to **multiple** receipt groups (`> 1`):
- **Primary payload (Group 0):** carries `segment_indices` (0-based indices into `file_urls` belonging to Group 0) alongside all standard receipt fields (`status`, `vendor_name`, `total_amount`, `items`, `rejection_code`, `is_duplicate`, etc.).
- **`split_receipts` (Groups `1..G-1`):** optional array of `SplitReceiptPayload` objects, each mirroring the standard receipt callback fields (with `receipt_id: null` before upstream creation) plus `segment_indices: int[]`.
- **Isolated rejections & intra-batch duplicates:** Each group is evaluated independently through pre-OCR quality, post-OCR/post-verify rejection gates, pgvector duplicate lookup, and intra-batch cosine similarity (`>= DUPLICATE_SIMILARITY_THRESHOLD`). Failing or duplicate groups are marked `"rejected"` individually without discarding valid sibling groups.
- **Child embedding persistence:** If the callback receiver returns JSON containing `data.split_receipt_ids` (ordered 1-to-1 with `split_receipts`), the worker persists `pgvector` embeddings for Group 0 (when `completed`) and each `completed` split child receipt ID.

