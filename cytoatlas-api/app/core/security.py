"""Authentication and security utilities."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database import get_db, get_db_optional

settings = get_settings()

# Password hashing (for user passwords)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API key hashing (PBKDF2 to avoid bcrypt's 72-byte limit)
api_key_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_v1_prefix}/auth/token")

# API Key header
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str
    exp: datetime
    type: str = "access"


class TokenResponse(BaseModel):
    """Token response schema."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def generate_api_key() -> tuple[str, str]:
    """Generate a secure API key and return both the key and its prefix.

    Returns:
        Tuple of (api_key, api_key_prefix)
    """
    api_key = secrets.token_urlsafe(32)
    api_key_prefix = api_key[:8]
    return api_key, api_key_prefix


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage.

    Uses PBKDF2-SHA256 which doesn't have length limitations like bcrypt.
    """
    return api_key_context.hash(api_key)


def verify_api_key_hash(api_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    return api_key_context.verify(api_key, hashed_key)


def create_access_token(
    subject: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode = {
        "sub": subject,
        "exp": expire,
        "type": "access",
    }
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    return encoded_jwt


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        return TokenPayload(**payload)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> "User":  # type: ignore[name-defined]
    """Get current user from JWT token."""
    from app.models.user import User
    from app.core.permissions import get_user_permissions

    payload = decode_token(token)

    result = await db.execute(select(User).where(User.email == payload.sub))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
        )

    # Attach permissions to user object (for convenience)
    user.permissions = get_user_permissions(user.role)  # type: ignore[attr-defined]

    return user


async def verify_api_key(
    api_key: Annotated[str | None, Depends(api_key_header)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> "User | None":  # type: ignore[name-defined]
    """Verify API key and return associated user (O(1) lookup via prefix index)."""
    if api_key is None:
        return None

    from app.models.user import User

    # Extract prefix for O(1) index lookup
    api_key_prefix = api_key[:8]

    # Query only users with matching prefix (indexed lookup)
    result = await db.execute(
        select(User).where(User.api_key_prefix == api_key_prefix)
    )
    users = result.scalars().all()

    # Verify hash only for matching users (typically 0-1 users)
    for user in users:
        if user.api_key_hash and verify_api_key_hash(api_key, user.api_key_hash):
            if user.is_active:
                # Update last used timestamp
                from datetime import datetime, timezone
                user.api_key_last_used = datetime.now(timezone.utc)
                await db.commit()
                return user

    return None


async def get_current_user_or_api_key(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    api_key: Annotated[str | None, Depends(api_key_header)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> "User":  # type: ignore[name-defined]
    """Get current user from either JWT token or API key."""
    # Try API key first
    if api_key:
        user = await verify_api_key(api_key, db)
        if user:
            return user

    # Try JWT token
    if token:
        return await get_current_user(token, db)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_admin(
    user: Annotated["User", Depends(get_current_user)],  # type: ignore[name-defined]
) -> "User":  # type: ignore[name-defined]
    """Require admin role."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


# Optional OAuth2 scheme that doesn't raise errors
oauth2_scheme_optional = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_v1_prefix}/auth/token",
    auto_error=False,
)


async def get_current_user_optional(
    token: Annotated[str | None, Depends(oauth2_scheme_optional)],
    api_key: Annotated[str | None, Depends(api_key_header)],
    db: Annotated[AsyncSession | None, Depends(get_db_optional)],
) -> "User | None":  # type: ignore[name-defined]
    """Get current user if authenticated, None otherwise.

    This dependency allows endpoints to work for both authenticated
    and anonymous users. Also works when database is not configured.
    """
    from app.core.permissions import get_user_permissions

    # If database is not configured, return None (anonymous user)
    if db is None:
        return None

    from app.models.user import User

    # Try API key first
    if api_key:
        user = await verify_api_key(api_key, db)
        if user:
            # Attach permissions
            user.permissions = get_user_permissions(user.role)  # type: ignore[attr-defined]
            return user

    # Try JWT token
    if token:
        try:
            payload = decode_token(token)
            result = await db.execute(select(User).where(User.email == payload.sub))
            user = result.scalar_one_or_none()
            if user and user.is_active:
                # Attach permissions
                user.permissions = get_user_permissions(user.role)  # type: ignore[attr-defined]
                return user
        except HTTPException:
            pass  # Invalid token, treat as anonymous

    return None
