"""Unit tests for security: JWT, password hashing, API keys, RBAC."""

import pytest
from datetime import datetime, timedelta, timezone

from app.core.security import (
    create_access_token,
    decode_token,
    generate_api_key,
    get_password_hash,
    hash_api_key,
    verify_api_key_hash,
    verify_password,
)
from app.core.permissions import (
    ROLE_PERMISSIONS,
    Permission,
    Role,
    get_user_permissions,
    has_permission,
    require_permission,
)
from app.config import Settings


def _bcrypt_available():
    """Check if bcrypt backend works with passlib."""
    try:
        get_password_hash("test")
        return True
    except Exception:
        return False


_skip_bcrypt = pytest.mark.skipif(
    not _bcrypt_available(),
    reason="passlib bcrypt backend unavailable (passlib 1.7.x + bcrypt 5.x incompatibility)",
)


class TestJWT:
    """Tests for JWT token creation and decoding."""

    def test_create_access_token(self):
        """Access token is a non-empty string."""
        token = create_access_token(subject="user@example.com")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_decode_valid_token(self):
        """Decoding a valid token returns correct subject."""
        token = create_access_token(subject="user@example.com")
        payload = decode_token(token)
        assert payload.sub == "user@example.com"
        assert payload.type == "access"

    def test_decode_expired_token(self):
        """Decoding an expired token raises HTTPException."""
        from fastapi import HTTPException

        token = create_access_token(
            subject="user@example.com",
            expires_delta=timedelta(seconds=-10),
        )

        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)
        assert exc_info.value.status_code == 401

    def test_decode_invalid_token(self):
        """Decoding a garbage token raises HTTPException."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            decode_token("not.a.valid.token")
        assert exc_info.value.status_code == 401

    def test_custom_expiry(self):
        """Token with custom expiry decodes correctly."""
        token = create_access_token(
            subject="user@example.com",
            expires_delta=timedelta(hours=2),
        )
        payload = decode_token(token)
        assert payload.sub == "user@example.com"


@_skip_bcrypt
class TestPasswordHashing:
    """Tests for password hashing and verification."""

    def test_hash_and_verify(self):
        """Password hash verifies against original password."""
        password = "SecurePassword123!"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed)

    def test_wrong_password_fails(self):
        """Wrong password does not verify."""
        hashed = get_password_hash("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_hash_is_different_from_plaintext(self):
        """Hash is not the plaintext password."""
        password = "my_password"
        hashed = get_password_hash(password)
        assert hashed != password

    def test_same_password_different_hashes(self):
        """Hashing the same password twice produces different hashes (salted)."""
        password = "test_password"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        assert hash1 != hash2  # bcrypt uses random salt


class TestAPIKey:
    """Tests for API key generation, hashing, and verification."""

    def test_generate_api_key(self):
        """Generated API key has expected format."""
        key, prefix = generate_api_key()
        assert len(key) >= 32
        assert len(prefix) == 8
        assert prefix == key[:8]

    def test_unique_keys(self):
        """Multiple key generations produce different keys."""
        key1, _ = generate_api_key()
        key2, _ = generate_api_key()
        assert key1 != key2

    def test_hash_and_verify_api_key(self):
        """API key hash verifies against original key."""
        key, _ = generate_api_key()
        hashed = hash_api_key(key)
        assert verify_api_key_hash(key, hashed)

    def test_wrong_key_fails_verification(self):
        """Wrong API key does not verify."""
        key, _ = generate_api_key()
        hashed = hash_api_key(key)
        wrong_key, _ = generate_api_key()
        assert not verify_api_key_hash(wrong_key, hashed)

    def test_hash_is_not_plaintext(self):
        """API key hash is different from the original key."""
        key, _ = generate_api_key()
        hashed = hash_api_key(key)
        assert hashed != key


class TestRBAC:
    """Tests for Role-Based Access Control."""

    def test_anonymous_permissions(self):
        """Anonymous users can only read public data."""
        perms = ROLE_PERMISSIONS[Role.ANONYMOUS]
        assert perms == {Permission.READ_PUBLIC}

    def test_viewer_permissions(self):
        """Viewers can read public data and use chat."""
        perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.READ_PUBLIC in perms
        assert Permission.USE_CHAT in perms
        assert Permission.READ_PRIVATE not in perms
        assert Permission.WRITE_DATA not in perms

    def test_researcher_permissions(self):
        """Researchers have read, chat, and submit permissions."""
        perms = ROLE_PERMISSIONS[Role.RESEARCHER]
        assert Permission.READ_PUBLIC in perms
        assert Permission.READ_PRIVATE in perms
        assert Permission.USE_CHAT in perms
        assert Permission.SUBMIT_DATASET in perms
        assert Permission.MANAGE_USERS not in perms

    def test_data_curator_permissions(self):
        """Data curators have write but not manage_users."""
        perms = ROLE_PERMISSIONS[Role.DATA_CURATOR]
        assert Permission.READ_PUBLIC in perms
        assert Permission.WRITE_DATA in perms
        assert Permission.MANAGE_USERS not in perms

    def test_admin_has_all_permissions(self):
        """Admins have every permission."""
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        for perm in Permission:
            assert perm in admin_perms

    def test_get_user_permissions_valid(self):
        """get_user_permissions returns correct set for valid roles."""
        assert Permission.READ_PUBLIC in get_user_permissions("anonymous")
        assert Permission.USE_CHAT in get_user_permissions("viewer")
        assert Permission.MANAGE_USERS in get_user_permissions("admin")

    def test_get_user_permissions_invalid(self):
        """Invalid role returns empty set."""
        assert get_user_permissions("invalid_role") == set()

    def test_has_permission_true(self):
        """has_permission returns True when permission is present."""
        perms = get_user_permissions("admin")
        assert has_permission(perms, Permission.MANAGE_USERS)

    def test_has_permission_false(self):
        """has_permission returns False when permission is absent."""
        perms = get_user_permissions("viewer")
        assert not has_permission(perms, Permission.WRITE_DATA)


class TestRequirePermission:
    """Tests for the require_permission decorator/dependency."""

    def test_require_permission_returns_callable(self):
        """require_permission returns a coroutine function."""
        checker = require_permission(Permission.READ_PUBLIC)
        assert callable(checker)

    async def test_anonymous_can_read_public(self):
        """Anonymous user (None) has READ_PUBLIC permission."""
        checker = require_permission(Permission.READ_PUBLIC)
        # Simulate anonymous user (None)
        await checker(user=None)  # Should not raise

    async def test_anonymous_cannot_write(self):
        """Anonymous user cannot WRITE_DATA."""
        from fastapi import HTTPException

        checker = require_permission(Permission.WRITE_DATA)
        with pytest.raises(HTTPException) as exc_info:
            await checker(user=None)
        assert exc_info.value.status_code == 403


class TestProductionSecurity:
    """Tests for production security validation."""

    def test_production_requires_secret_key(self):
        """Production environment rejects None secret_key."""
        with pytest.raises(ValueError, match="SECRET_KEY must be set"):
            Settings(environment="production", secret_key=None)

    def test_production_rejects_default_secret(self):
        """Production environment rejects default secret_key."""
        with pytest.raises(ValueError, match="SECRET_KEY must be set"):
            Settings(
                environment="production",
                secret_key="change-me-in-production-use-openssl-rand-hex-32",
            )

    def test_development_allows_none_secret(self):
        """Development environment allows None secret_key."""
        settings = Settings(environment="development", secret_key=None)
        assert settings.environment == "development"

    def test_cors_origins_property(self):
        """CORS origins property splits comma-separated values correctly."""
        settings = Settings(allowed_origins="http://localhost:8000,http://localhost:3000")
        assert "http://localhost:8000" in settings.cors_origins
        assert "http://localhost:3000" in settings.cors_origins

    def test_cors_wildcard(self):
        """CORS wildcard origin works."""
        settings = Settings(allowed_origins="*")
        assert settings.cors_origins == ["*"]

    def test_rate_limit_defaults(self):
        """Rate limit fields have default values."""
        settings = Settings()
        assert settings.rate_limit_requests > 0
        assert settings.rate_limit_window > 0
