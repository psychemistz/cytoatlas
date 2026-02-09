"""Unit tests for audit logging."""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.audit import AuditEvent, AuditLogger, AuditMiddleware


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_creation(self):
        """AuditEvent is created with correct fields."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="auth_success",
            user_id=1,
            ip_address="127.0.0.1",
            method="POST",
            path="/api/v1/auth/token",
            status_code=200,
            duration_ms=45.2,
            request_id="test-request-id",
            details={"test": "data"},
        )

        assert event.event_type == "auth_success"
        assert event.user_id == 1
        assert event.status_code == 200
        assert event.duration_ms == 45.2

    def test_event_with_no_user(self):
        """AuditEvent can be created with user_id=None."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="access",
            user_id=None,
            ip_address="10.0.0.1",
            method="GET",
            path="/api/v1/health",
            status_code=200,
            duration_ms=5.0,
            request_id="req-001",
        )
        assert event.user_id is None

    def test_event_default_details(self):
        """AuditEvent details defaults to empty dict."""
        event = AuditEvent(
            timestamp="2024-01-01T00:00:00Z",
            event_type="access",
            user_id=None,
            ip_address="127.0.0.1",
            method="GET",
            path="/",
            status_code=200,
            duration_ms=1.0,
            request_id="r1",
        )
        assert event.details == {}


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_token_redaction_bearer(self, tmp_path):
        """Bearer tokens are redacted."""
        logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = logger.redact_tokens(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "Bearer ***" in redacted

    def test_token_redaction_api_key(self, tmp_path):
        """API keys are redacted."""
        logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
        text = "X-API-Key: abc123def456"
        redacted = logger.redact_tokens(text)
        assert "abc123def456" not in redacted
        assert "X-API-Key: ***" in redacted

    def test_no_token_unchanged(self, tmp_path):
        """Text without tokens is unchanged."""
        logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
        text = "Normal text without tokens"
        assert logger.redact_tokens(text) == text

    def test_log_directory_created(self, tmp_path):
        """Log directory is created if it does not exist."""
        log_path = tmp_path / "logs" / "nested" / "audit.jsonl"
        logger = AuditLogger(log_path=log_path)
        assert log_path.parent.exists()

    async def test_log_event_queues(self, tmp_path):
        """log_event adds event to the queue."""
        logger = AuditLogger(log_path=tmp_path / "audit.jsonl")

        event = AuditEvent(
            timestamp="2024-01-01T00:00:00Z",
            event_type="test",
            user_id=None,
            ip_address="127.0.0.1",
            method="GET",
            path="/test",
            status_code=200,
            duration_ms=1.0,
            request_id="test-id",
        )

        # Temporarily enable audit
        with patch("app.core.audit.settings") as mock_settings:
            mock_settings.audit_enabled = True
            await logger.log_event(event)

        assert logger.queue.qsize() == 1

    def test_write_event(self, tmp_path):
        """_write_event writes JSON line to file."""
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_path)

        event = AuditEvent(
            timestamp="2024-01-01T00:00:00Z",
            event_type="test_write",
            user_id=None,
            ip_address="127.0.0.1",
            method="GET",
            path="/test",
            status_code=200,
            duration_ms=1.0,
            request_id="test-write-id",
        )

        logger._write_event(event)
        logger.handler.stream.flush()

        # Read back
        content = log_path.read_text()
        assert "test_write" in content
        assert "test-write-id" in content

        # Should be valid JSON
        parsed = json.loads(content.strip())
        assert parsed["event_type"] == "test_write"


class TestAuditMiddleware:
    """Tests for AuditMiddleware event type detection."""

    def setup_method(self):
        self.middleware = AuditMiddleware(app=MagicMock())

    def test_rate_limit_event_type(self):
        """429 response produces rate_limit event type."""
        request = MagicMock()
        request.url.path = "/api/v1/chat/message"
        response = MagicMock()
        response.status_code = 429

        event_type = self.middleware._get_event_type(request, response)
        assert event_type == "rate_limit"

    def test_auth_failure_event_type(self):
        """4xx on auth path produces auth_failure event type."""
        request = MagicMock()
        request.url.path = "/api/v1/auth/token"
        response = MagicMock()
        response.status_code = 401

        event_type = self.middleware._get_event_type(request, response)
        assert event_type == "auth_failure"

    def test_error_event_type(self):
        """Non-auth 4xx/5xx produces error event type."""
        request = MagicMock()
        request.url.path = "/api/v1/search"
        response = MagicMock()
        response.status_code = 500

        event_type = self.middleware._get_event_type(request, response)
        assert event_type == "error"

    def test_auth_success_event_type(self):
        """Successful POST to /auth/ produces auth_success."""
        request = MagicMock()
        request.url.path = "/api/v1/auth/token"
        request.method = "POST"
        response = MagicMock()
        response.status_code = 200

        event_type = self.middleware._get_event_type(request, response)
        assert event_type == "auth_success"

    def test_dataset_submission_event_type(self):
        """POST to /submit/ produces dataset_submission."""
        request = MagicMock()
        request.url.path = "/api/v1/submit/upload"
        request.method = "POST"
        response = MagicMock()
        response.status_code = 200

        event_type = self.middleware._get_event_type(request, response)
        assert event_type == "dataset_submission"

    def test_access_event_type(self):
        """GET request with 200 produces access event type."""
        request = MagicMock()
        request.url.path = "/api/v1/health"
        request.method = "GET"
        response = MagicMock()
        response.status_code = 200

        event_type = self.middleware._get_event_type(request, response)
        assert event_type == "access"

    def test_write_methods_set(self):
        """WRITE_METHODS contains expected HTTP methods."""
        assert "POST" in AuditMiddleware.WRITE_METHODS
        assert "PUT" in AuditMiddleware.WRITE_METHODS
        assert "DELETE" in AuditMiddleware.WRITE_METHODS
        assert "PATCH" in AuditMiddleware.WRITE_METHODS
        assert "GET" not in AuditMiddleware.WRITE_METHODS
