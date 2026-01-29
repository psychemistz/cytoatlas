"""Request logging middleware and utilities."""

import logging
import sys
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

settings = get_settings()


def setup_logging() -> logging.Logger:
    """Configure application logging."""
    log_level = logging.DEBUG if settings.debug else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # App logger
    logger = logging.getLogger("cytoatlas")
    logger.setLevel(log_level)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


# Global logger instance
logger = setup_logging()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests with timing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]

        # Skip logging for health checks and static files
        path = request.url.path
        if path in ["/api/v1/health", "/api/v1/health/ready"] or path.startswith("/static"):
            return await call_next(request)

        # Extract request info
        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")[:50]

        # Log request start
        logger.info(
            f"[{request_id}] --> {method} {path} | client={client_ip}"
        )

        # Process request and measure time
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
            process_time = (time.perf_counter() - start_time) * 1000  # ms

            # Log response
            status = response.status_code
            log_func = logger.info if status < 400 else logger.warning
            log_func(
                f"[{request_id}] <-- {method} {path} | "
                f"status={status} | time={process_time:.1f}ms"
            )

            # Add timing header
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.1f}ms"

            return response

        except Exception as e:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] <-- {method} {path} | "
                f"error={type(e).__name__}: {str(e)[:100]} | time={process_time:.1f}ms"
            )
            raise


class APIMetrics:
    """Simple in-memory API metrics collector."""

    def __init__(self):
        self.requests_total = 0
        self.requests_by_path: dict[str, int] = {}
        self.requests_by_status: dict[int, int] = {}
        self.response_times: list[float] = []
        self.errors_total = 0

    def record_request(self, path: str, status: int, duration_ms: float):
        """Record a request."""
        self.requests_total += 1

        # Normalize path (remove IDs)
        normalized = self._normalize_path(path)
        self.requests_by_path[normalized] = self.requests_by_path.get(normalized, 0) + 1

        self.requests_by_status[status] = self.requests_by_status.get(status, 0) + 1

        # Keep last 1000 response times
        self.response_times.append(duration_ms)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

        if status >= 500:
            self.errors_total += 1

    def _normalize_path(self, path: str) -> str:
        """Normalize path by replacing IDs with placeholders."""
        parts = path.split("/")
        normalized_parts = []
        for i, part in enumerate(parts):
            # Replace UUIDs and numeric IDs
            if len(part) == 36 and "-" in part:  # UUID
                normalized_parts.append("{id}")
            elif part.isdigit():
                normalized_parts.append("{id}")
            else:
                normalized_parts.append(part)
        return "/".join(normalized_parts)

    def get_summary(self) -> dict:
        """Get metrics summary."""
        avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        p95_time = sorted(self.response_times)[int(len(self.response_times) * 0.95)] if self.response_times else 0

        return {
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "error_rate": self.errors_total / max(self.requests_total, 1),
            "avg_response_time_ms": round(avg_time, 2),
            "p95_response_time_ms": round(p95_time, 2),
            "top_endpoints": sorted(
                self.requests_by_path.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "status_distribution": self.requests_by_status,
        }


# Global metrics instance
metrics = APIMetrics()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect API metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Skip static files
        if path.startswith("/static"):
            return await call_next(request)

        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        metrics.record_request(path, response.status_code, duration_ms)

        return response
