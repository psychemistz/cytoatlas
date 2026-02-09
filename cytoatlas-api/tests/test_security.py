"""Tests for security enhancements."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from app.config import Settings
from app.core.permissions import (
    Role,
    Permission,
    ROLE_PERMISSIONS,
    get_user_permissions,
    has_permission,
)
from app.core.security import generate_api_key, hash_api_key, verify_api_key_hash
from app.core.audit import AuditEvent, AuditLogger


class TestRBAC:
    """Test Role-Based Access Control."""

    def test_role_permissions_mapping(self):
        """Test that all roles have correct permissions."""
        # Anonymous users can only read public data
        assert ROLE_PERMISSIONS[Role.ANONYMOUS] == {Permission.READ_PUBLIC}

        # Viewers can read public data and use chat
        assert Permission.READ_PUBLIC in ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.USE_CHAT in ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.READ_PRIVATE not in ROLE_PERMISSIONS[Role.VIEWER]

        # Researchers have more permissions
        researcher_perms = ROLE_PERMISSIONS[Role.RESEARCHER]
        assert Permission.READ_PUBLIC in researcher_perms
        assert Permission.READ_PRIVATE in researcher_perms
        assert Permission.USE_CHAT in researcher_perms
        assert Permission.SUBMIT_DATASET in researcher_perms
        assert Permission.MANAGE_USERS not in researcher_perms

        # Data curators have all except user management
        curator_perms = ROLE_PERMISSIONS[Role.DATA_CURATOR]
        assert Permission.READ_PUBLIC in curator_perms
        assert Permission.READ_PRIVATE in curator_perms
        assert Permission.WRITE_DATA in curator_perms
        assert Permission.SUBMIT_DATASET in curator_perms
        assert Permission.MANAGE_USERS not in curator_perms

        # Admins have all permissions
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert len(admin_perms) == len(Permission)
        for perm in Permission:
            assert perm in admin_perms

    def test_get_user_permissions(self):
        """Test getting permissions for a role."""
        # Valid roles
        assert Permission.READ_PUBLIC in get_user_permissions("anonymous")
        assert Permission.USE_CHAT in get_user_permissions("viewer")
        assert Permission.MANAGE_USERS in get_user_permissions("admin")

        # Invalid role returns empty set
        assert get_user_permissions("invalid_role") == set()

    def test_has_permission(self):
        """Test permission checking."""
        viewer_perms = get_user_permissions("viewer")
        assert has_permission(viewer_perms, Permission.READ_PUBLIC)
        assert has_permission(viewer_perms, Permission.USE_CHAT)
        assert not has_permission(viewer_perms, Permission.WRITE_DATA)


class TestAPIKeySecurity:
    """Test API key security enhancements."""

    def test_generate_api_key(self):
        """Test API key generation returns key and prefix."""
        key, prefix = generate_api_key()

        # Key should be 43 characters (32 bytes base64url encoded)
        assert len(key) >= 32

        # Prefix should be first 8 characters
        assert len(prefix) == 8
        assert prefix == key[:8]

        # Multiple calls should generate different keys
        key2, prefix2 = generate_api_key()
        assert key != key2
        assert prefix != prefix2

    def test_api_key_hashing(self):
        """Test API key hashing and verification."""
        key, _ = generate_api_key()
        hashed = hash_api_key(key)

        # Hash should be different from original key
        assert hashed != key

        # Should verify correctly
        assert verify_api_key_hash(key, hashed)

        # Should not verify wrong key
        wrong_key, _ = generate_api_key()
        assert not verify_api_key_hash(wrong_key, hashed)


class TestProductionSecurity:
    """Test production security validation."""

    def test_production_requires_secret_key(self):
        """Test that production environment requires a valid secret key."""
        # Should raise ValueError in production with None secret_key
        with pytest.raises(ValueError, match="SECRET_KEY must be set"):
            Settings(
                environment="production",
                secret_key=None,
            )

        # Should raise ValueError in production with default secret_key
        with pytest.raises(ValueError, match="SECRET_KEY must be set"):
            Settings(
                environment="production",
                secret_key="change-me-in-production-use-openssl-rand-hex-32",
            )

        # Should work in development with None
        settings = Settings(environment="development", secret_key=None)
        assert settings.environment == "development"

    def test_cors_defaults(self):
        """Test CORS defaults to localhost only."""
        settings = Settings()
        assert settings.allowed_origins == "http://localhost:8000,http://localhost:3000"
        assert "http://localhost:8000" in settings.cors_origins
        assert "http://localhost:3000" in settings.cors_origins

    def test_audit_defaults(self):
        """Test audit logging is enabled by default."""
        settings = Settings()
        assert settings.audit_enabled is True
        assert settings.audit_log_path == Path("logs/audit.jsonl")


class TestAuditLogging:
    """Test audit logging functionality."""

    def test_audit_event_creation(self):
        """Test creating an audit event."""
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

    def test_audit_logger_token_redaction(self, tmp_path):
        """Test that tokens are redacted in audit logs."""
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_path)

        # Test Bearer token redaction
        bearer_text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = logger.redact_tokens(bearer_text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "Bearer ***" in redacted

        # Test API key redaction
        api_key_text = "X-API-Key: abc123def456"
        redacted = logger.redact_tokens(api_key_text)
        assert "abc123def456" not in redacted
        assert "X-API-Key: ***" in redacted


class TestPathTraversal:
    """Test path traversal protection."""

    async def test_path_traversal_prevention(self):
        """Test that path traversal attempts are blocked."""
        from app.services.base import BaseService

        service = BaseService()

        # Should raise ValueError for path traversal
        with pytest.raises(ValueError, match="Path traversal detected"):
            await service.load_json("../../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal detected"):
            await service.load_json("../../../secret.json", subdir="data")


class TestSecurityHeaders:
    """Test security headers middleware."""

    def test_security_headers_added(self):
        """Test that security headers are added to responses."""
        # This would require setting up a test client
        # Placeholder for actual implementation
        pass


class TestRateLimiting:
    """Test rate limiting configuration."""

    def test_rate_limit_settings(self):
        """Test rate limit settings."""
        settings = Settings()
        assert settings.rate_limit_requests == 100
        assert settings.rate_limit_window == 60


class TestMaxRequestBody:
    """Test request body size limits."""

    def test_max_request_body_setting(self):
        """Test max request body size setting."""
        settings = Settings()
        assert settings.max_request_body_mb == 100
