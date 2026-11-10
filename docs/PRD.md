# Product Requirements Document (PRD): OCR Pipeline for SERMS

## 1. App Overview & Objectives
The OCR Pipeline is a specialized microservice designed to ingest documents (images and PDFs) from backend systems, perform high-accuracy Optical Character Recognition (OCR), and securely process and store the metadata. The core objective is to provide a highly scalable, asynchronous processing engine capable of handling thousands of concurrent documents without blocking upstream applications.

### Success Metrics & KPIs
- **High Accuracy:** Accurate text extraction even on messy or degraded printed text.
- **Low Latency & High Throughput:** Fast processing speed achieved via distributed queues.
- **Scalability:** Ability to handle high volumes of documents concurrently.

## 2. Target Audience
The primary consumers of this pipeline are internal microservices, specifically the main SERMS backend, which offload heavy document processing and OCR tasks to this API.

## 3. Key Features & Functional Requirements
- **TASK-2 [Document Ingestion] (Must-Have):** API endpoints to securely receive image and PDF files from authenticated internal services.
  - *Constraints:* Supports JPEG, PNG, and PDF formats only. Maximum file size is strictly limited to 2MB.
- **TASK-3 [Asynchronous Processing] (Must-Have):** Offload document processing to background worker queues using Celery and Redis to prevent API blocking.
- **TASK-4 [OCR Extraction] (Must-Have):** Leverage PyTesseract and OpenCV to pre-process images (grayscale, deskew, denoising) and extract text accurately.
- **TASK-5 [Semantic Embeddings] (Should-Have):** Use `sentence-transformers` (specifically `all-MiniLM-L6-v2`) to create semantic embeddings of the extracted text.
- **TASK-6 [Duplicate Detection] (Must-Have):** Store embeddings in PostgreSQL via `pgvector` and use cosine similarity to detect duplicate documents within a 90-day window.
- **TASK-7 [Callback Webhooks] (Must-Have):** Automatically trigger HTTP callbacks/webhooks to notify upstream services (like SERMS) when a document finishes processing or fails.

## 4. Key User Flows
1. **Upstream Service Submission:** 
   - Upstream service (e.g., SERMS backend) submits a document payload (containing file data and webhook callback URL).
   - The API validates the file format/size and instantly responds with a `202 Accepted` status and a unique `job_id`.
2. **Background Processing & OCR:** 
   - A Celery worker pulls the task from the Redis queue.
   - OpenCV cleans the image, PyTesseract extracts raw text, and `sentence-transformers` generates a semantic vector embedding.
3. **Storage & Verification:**
   - Raw text and processing metadata are saved to MongoDB.
   - The vector embedding is saved to PostgreSQL (`pgvector`).
   - The worker compares the new embedding against existing documents in PostgreSQL within a 90-day window to evaluate similarity.
4. **Webhook Callback:**
   - If similarity is above the threshold (e.g., 0.85), the job is flagged as a duplicate.
   - The pipeline triggers the webhook callback, POSTing the OCR results, metadata, and duplicate flag back to the SERMS backend.

## 5. Non-Goals
- **Direct User Authentication:** The pipeline does not manage user accounts, sessions, or logins. Authentication is strictly service-to-service via pre-shared API keys.
- **Direct Payment or Disbursement:** The pipeline does not integrate with banking APIs or process financial disbursements. It is solely an ingestion and OCR processing engine.
- **Synchronous Document Processing:** The pipeline does not process documents synchronously in-request. All processing is strictly offloaded to background workers.
- **Manual Verification UI:** This microservice does not provide a UI for manual review. Confidence flagging is passed back via webhook for the consumer service to handle.

---

## Technical Appendix (For Context)

### Technical Stack
- **API Layer:** FastAPI (Python 3.12+)
- **Task Queue:** Celery, backed by Redis
- **Databases:** PostgreSQL (with `pgvector` for embeddings), MongoDB (for metadata)
- **AI/ML Tools:** PaddleOCR & PyTesseract (OCR), OpenCV (Image processing), `sentence-transformers` (Embeddings), Google AI Studio Gemini (LLM structured extraction & semantics)
- **Deployment:** Docker & Docker Compose

### Conceptual Data Model
- **PostgreSQL (pgvector):** `document_embeddings` table containing `id`, `job_id`, `embedding` (vector), `created_at`
- **MongoDB:** `documents` collection containing `job_id`, `raw_text`, `processing_status`, `webhook_url`, `duplicate_flag`, `metadata`

### Security Considerations
- **API Authentication:** Incoming API requests and outgoing webhook callbacks are authenticated using pre-shared API keys (`SERMS_API_KEY`, `CALLBACK_API_KEY`).
- **Network Security:** Containers communicate over a private bridge network (`ocr_network`).
