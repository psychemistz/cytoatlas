"""Audit logging for security events and API access."""

import asyncio
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import get_settings

settings = get_settings()


@dataclass
class AuditEvent:
    """Audit event record."""

    timestamp: str
    event_type: str
    user_id: int | None
    ip_address: str
    method: str
    path: str
    status_code: int
    duration_ms: float
    request_id: str
    details: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Async audit logger with rotating file backend."""

    def __init__(self, log_path: Path | None = None):
        """Initialize audit logger."""
        self.log_path = log_path or settings.audit_log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup rotating file handler
        self.handler = RotatingFileHandler(
            filename=str(self.log_path),
            maxBytes=100 * 1024 * 1024,  # 100 MB
            backupCount=5,
            encoding="utf-8",
        )

        # Async queue for non-blocking writes
        self.queue: asyncio.Queue[AuditEvent] = asyncio.Queue()
        self.task: asyncio.Task | None = None

        # Token redaction pattern
        self.token_pattern = re.compile(
            r'(Bearer\s+|X-API-Key:\s*)([A-Za-z0-9_-]+)',
            re.IGNORECASE
        )

    async def start(self) -> None:
        """Start the background logging task."""
        if self.task is None:
            self.task = asyncio.create_task(self._process_queue())

    async def stop(self) -> None:
        """Stop the background logging task."""
        if self.task:
            # Wait for queue to drain
            await self.queue.join()
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    def redact_tokens(self, text: str) -> str:
        """Redact authentication tokens from text."""
        return self.token_pattern.sub(r'\1***', text)

    async def log_event(self, event: AuditEvent) -> None:
        """Queue an audit event for logging."""
        if not settings.audit_enabled:
            return

        # Redact sensitive data from details
        if "headers" in event.details:
            event.details["headers"] = self.redact_tokens(str(event.details["headers"]))

        await self.queue.put(event)

    async def _process_queue(self) -> None:
        """Process audit events from queue."""
        while True:
            try:
                event = await self.queue.get()
                self._write_event(event)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                # Don't let logging errors crash the app
                pass

    def _write_event(self, event: AuditEvent) -> None:
        """Write audit event to file (synchronous)."""
        try:
            record = asdict(event)
            line = json.dumps(record, default=str) + "\n"
            self.handler.stream.write(line)
            self.handler.stream.flush()
        except Exception:
            # Silent failure - don't break app for logging issues
            pass


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware to audit API requests."""

    # HTTP methods that modify data
    WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

    # Paths to always audit (auth-related)
    AUTH_PATHS = {"/api/v1/auth/", "/api/v1/users/"}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and audit if needed."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Start timing
        start_time = asyncio.get_event_loop().time()

        # Call endpoint
        response = await call_next(request)

        # Calculate duration
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        # Determine if we should audit this request
        should_audit = (
            request.method in self.WRITE_METHODS  # All write operations
            or any(request.url.path.startswith(p) for p in self.AUTH_PATHS)  # Auth endpoints
            or response.status_code == 429  # Rate limit hits
            or response.status_code >= 400  # Error responses
        )

        if should_audit:
            # Extract user ID if available
            user_id = None
            if hasattr(request.state, "user") and request.state.user:
                user_id = request.state.user.id

            # Determine event type
            event_type = self._get_event_type(request, response)

            # Create audit event
            event = AuditEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                user_id=user_id,
                ip_address=client_ip,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                request_id=request_id,
                details={
                    "query_params": dict(request.query_params),
                    "user_agent": request.headers.get("user-agent", "unknown"),
                },
            )

            # Log event
            audit_logger = get_audit_logger()
            await audit_logger.log_event(event)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response

    def _get_event_type(self, request: Request, response: Response) -> str:
        """Determine audit event type."""
        if response.status_code == 429:
            return "rate_limit"
        elif response.status_code >= 400:
            if "/auth/" in request.url.path:
                return "auth_failure"
            return "error"
        elif request.method in self.WRITE_METHODS:
            if "/auth/" in request.url.path:
                return "auth_success"
            elif "/users/" in request.url.path:
                return "user_management"
            elif "/submit/" in request.url.path:
                return "dataset_submission"
            return "data_modification"
        else:
            return "access"
