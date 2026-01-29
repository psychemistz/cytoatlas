"""Authentication endpoints."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database import get_db
from app.core.security import (
    TokenResponse,
    create_access_token,
    generate_api_key,
    get_current_user,
    get_password_hash,
    hash_api_key,
    verify_password,
)
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = get_settings()


class UserCreate(BaseModel):
    """User registration schema."""

    email: EmailStr
    password: str
    full_name: str | None = None
    institution: str | None = None


class UserResponse(BaseModel):
    """User response schema."""

    id: int
    email: str
    full_name: str | None
    institution: str | None
    is_active: bool
    is_admin: bool

    class Config:
        from_attributes = True


class APIKeyResponse(BaseModel):
    """API key response schema."""

    api_key: str
    message: str = "Store this key securely. It will not be shown again."


@router.post("/token", response_model=TokenResponse)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """
    Authenticate user and return access token.

    Args:
        form_data: OAuth2 password form (username=email, password)
        db: Database session

    Returns:
        Access token response
    """
    # Get user by email
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive",
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        subject=user.email,
        expires_delta=access_token_expires,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Register a new user.

    Args:
        user_data: User registration data
        db: Database session

    Returns:
        Created user
    """
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        institution=user_data.institution,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    return user


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """
    Get current user information.

    Args:
        current_user: Authenticated user

    Returns:
        User information
    """
    return current_user


@router.post("/api-key", response_model=APIKeyResponse)
async def generate_user_api_key(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> APIKeyResponse:
    """
    Generate a new API key for the current user.

    This will replace any existing API key.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        New API key (only shown once)
    """
    # Generate new API key
    api_key = generate_api_key()

    # Hash and store
    current_user.api_key_hash = hash_api_key(api_key)
    await db.flush()

    return APIKeyResponse(api_key=api_key)


@router.delete("/api-key")
async def revoke_api_key(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Revoke current API key.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        Success message
    """
    current_user.api_key_hash = None
    await db.flush()

    return {"message": "API key revoked successfully"}
