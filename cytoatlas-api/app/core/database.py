"""Async SQLAlchemy database configuration.

Supports two backends:
- SQLite (via aiosqlite) — default for HPC, zero daemon dependencies
- PostgreSQL (via asyncpg) — optional for production deployments

When neither database_url nor sqlite_app_path is configured, the app runs
without persistence (all state is in-memory).
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
from typing import Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""

    pass


def _build_engine() -> Optional[AsyncEngine]:
    """Build async engine from configuration.

    Priority:
    1. Explicit database_url (PostgreSQL or any SQLAlchemy-supported URL)
    2. sqlite_app_path (SQLite via aiosqlite — default for HPC)
    3. None (no persistence)
    """
    if settings.use_database and settings.database_url:
        url = str(settings.database_url)
        logger.info(f"Database: Using configured URL ({url.split('@')[-1] if '@' in url else url})")
        return create_async_engine(
            url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout,
            echo=settings.debug,
        )

    # Default: SQLite via aiosqlite
    sqlite_path = settings.sqlite_app_path
    if sqlite_path:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite+aiosqlite:///{sqlite_path}"
        logger.info(f"Database: Using SQLite at {sqlite_path}")
        eng = create_async_engine(
            url,
            echo=settings.debug,
            # SQLite-specific: no pool limits needed
            pool_size=0,
            max_overflow=-1,
        )

        # Enable WAL mode and foreign keys for SQLite
        @event.listens_for(eng.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        return eng

    return None


# Build engine and session factory
engine: Optional[AsyncEngine] = _build_engine()
async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None

if engine is not None:
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


async def init_db() -> None:
    """Initialize database tables."""
    if engine is None:
        raise RuntimeError("Database not configured")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    if engine is not None:
        await engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    if async_session_factory is None:
        raise RuntimeError("Database not configured")
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions (for use outside FastAPI)."""
    if async_session_factory is None:
        raise RuntimeError("Database not configured")
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_optional() -> AsyncGenerator[AsyncSession | None, None]:
    """FastAPI dependency for optional database sessions.

    Returns None if database is not configured, allowing endpoints
    to work without a database.
    """
    if async_session_factory is None:
        yield None
        return
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def is_database_configured() -> bool:
    """Check if database is configured."""
    return async_session_factory is not None
