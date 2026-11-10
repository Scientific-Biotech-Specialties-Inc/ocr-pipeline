"""Pytest fixtures for End-to-End (E2E) testing.

Provides a self-contained in-memory HTTP server that:
1. Serves sample receipt images (GET /receipts/<filename>)
2. Captures outbound webhook callbacks (POST /webhook/callback)
3. Stubs Gemini LLM extraction (POST /models/{model}:generateContent) with deterministic JSON responses
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.config import settings

import socket
import urllib.parse
from app.db.mongodb import MongoDBClient
from app.tasks.process_receipt import process_receipt_task

RECEIPTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "receipts"


def _is_service_reachable(url: str, default_port: int) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or default_port
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture(autouse=True)
def fallback_when_services_offline(monkeypatch: pytest.MonkeyPatch):
    """Provide in-memory MongoDB/Redis fallback only when backing services are offline."""
    if not _is_service_reachable(settings.mongodb_url, 27017):
        jobs_db: dict[str, dict] = {}

        async def fake_create_job(job_data: dict) -> str:
            job_id = job_data.get("job_id", "test-job-id")
            jobs_db[job_id] = dict(job_data)
            return job_id

        async def fake_get_job(job_id: str) -> dict | None:
            return jobs_db.get(job_id)

        async def fake_update_job_status(job_id: str, status: str, extra_fields: dict | None = None) -> None:
            if job_id in jobs_db:
                jobs_db[job_id]["status"] = status
                if extra_fields:
                    jobs_db[job_id].update(extra_fields)

        monkeypatch.setattr(MongoDBClient, "create_job", fake_create_job)
        monkeypatch.setattr(MongoDBClient, "get_job", fake_get_job)
        monkeypatch.setattr(MongoDBClient, "update_job_status", fake_update_job_status)

    if not _is_service_reachable(settings.redis_url, 6379):
        monkeypatch.setattr(process_receipt_task, "delay", lambda *args, **kwargs: None)

    if not _is_service_reachable(settings.postgres_url, 5432):
        from app.embeddings.generator import EmbeddingGenerator
        monkeypatch.setattr(EmbeddingGenerator, "generate", lambda text: [0.05] * 384)

        async def fake_find_similar(*args, **kwargs):
            return []

        monkeypatch.setattr("app.api.routes.duplicate.find_similar_receipts", fake_find_similar)


class FixtureRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for fixture images, webhook callbacks, and mock Gemini API."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence default stderr logging during test runs
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/receipts/"):
            filename = self.path[len("/receipts/") :].split("?")[0]
            if filename == "tiny_invalid.png":
                img = Image.new("RGB", (50, 50), color=(0, 0, 0))
                bio = BytesIO()
                img.save(bio, format="PNG")
                content = bio.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            file_path = RECEIPTS_DIR / urllib_unquote(filename)
            if file_path.exists() and file_path.is_file():
                content = file_path.read_bytes()
                content_type = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Receipt not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def do_POST(self) -> None:  # noqa: N802
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""

        # Webhook callback receiver
        if self.path.startswith("/webhook/callback"):
            parsed_payload = {}
            if body:
                try:
                    parsed_payload = json.loads(body.decode("utf-8"))
                except Exception:
                    parsed_payload = {"raw": body.decode("utf-8", errors="replace")}

            self.server.received_callbacks.append({
                "headers": dict(self.headers),
                "payload": parsed_payload,
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        # Mock Google AI Studio (Gemini) generateContent endpoint
        if ":generateContent" in self.path or "/models/" in self.path:
            prompt = ""
            if body:
                try:
                    req_json = json.loads(body.decode("utf-8"))
                    contents = req_json.get("contents") or []
                    if contents and "parts" in contents[0]:
                        prompt = (contents[0]["parts"][0].get("text") or "").lower()
                except Exception:
                    pass

            if "category" in prompt:
                mock_response = json.dumps({"category": "Meals & Entertainment"})
            elif "vendor" in prompt:
                mock_response = json.dumps({"vendor_name": "Jollibee"})
            elif "location" in prompt:
                mock_response = json.dumps({"location": "New York, NY"})
            else:
                mock_response = json.dumps({
                    "vendor_name": "Jollibee",
                    "transaction_date": "2025-05-24",
                    "total_amount": 32.50,
                    "subtotal": 29.99,
                    "vat_amount": 2.51,
                    "tin": None,
                    "invoice_number": "340",
                    "currency": "USD",
                    "expense_category": "Meals & Entertainment",
                    "location": "New York, NY",
                    "confidence": 0.95,
                    "tax_basis": "exclusive",
                    "evidence": ["TAX: 2.51", "TOTAL: 32.50"],
                })

            resp_payload = {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": mock_response}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
            resp_bytes = json.dumps(resp_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        self.send_response(404)
        self.end_headers()


def urllib_unquote(s: str) -> str:
    import urllib.parse
    return urllib.parse.unquote(s)


class MockServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass):  # noqa: N803
        super().__init__(server_address, RequestHandlerClass)
        self.received_callbacks: list[dict[str, Any]] = []


@pytest.fixture(scope="session")
def mock_fixture_server():
    """Starts a lightweight HTTP server on an OS-assigned ephemeral port."""
    server = MockServer(("127.0.0.1", 0), FixtureRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    # Configure application Gemini base URL and key to target this mock server
    original_gemini_url = settings.gemini_base_url
    original_gemini_key = settings.gemini_api_key
    settings.gemini_base_url = base_url
    settings.gemini_api_key = "test-e2e-gemini-key"

    yield server, base_url

    settings.gemini_base_url = original_gemini_url
    settings.gemini_api_key = original_gemini_key
    server.shutdown()
    server.server_close()
