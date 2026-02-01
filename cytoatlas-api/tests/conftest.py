"""Pytest configuration and fixtures."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import get_settings
from app.core.database import Base, get_db
from app.main import app

settings = get_settings()

# Test database URL (use SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with database session."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_cima_data() -> dict[str, Any]:
    """Sample CIMA data for testing."""
    return {
        "correlations": {
            "age": [
                {
                    "cell_type": "CD4_T",
                    "signature": "IFNG",
                    "signature_type": "CytoSig",
                    "variable": "Age",
                    "rho": 0.25,
                    "pvalue": 0.001,
                    "qvalue": 0.01,
                    "n_samples": 100,
                }
            ],
            "bmi": [
                {
                    "cell_type": "CD4_T",
                    "signature": "IL6",
                    "signature_type": "CytoSig",
                    "variable": "BMI",
                    "rho": 0.18,
                    "pvalue": 0.05,
                    "qvalue": 0.1,
                    "n_samples": 100,
                }
            ],
        }
    }


@pytest.fixture
def sample_inflammation_data() -> dict[str, Any]:
    """Sample Inflammation data for testing."""
    return {
        "cell_type_activity": [
            {
                "cell_type": "CD8_T",
                "signature": "IFNG",
                "signature_type": "CytoSig",
                "mean_activity": 1.5,
                "n_samples": 50,
                "n_cells": 10000,
            }
        ],
        "disease_comparison": [
            {
                "cell_type": "Monocytes",
                "signature": "TNFA",
                "signature_type": "CytoSig",
                "disease": "Rheumatoid Arthritis",
                "activity_diff": 1.2,
                "mean_disease": 2.0,
                "mean_healthy": 0.8,
                "pvalue": 0.001,
                "qvalue": 0.01,
                "n_disease": 30,
                "n_healthy": 50,
            }
        ],
    }


@pytest.fixture
def sample_scatlas_data() -> dict[str, Any]:
    """Sample scAtlas data for testing."""
    return {
        "organ_signatures": [
            {
                "organ": "Lung",
                "signature": "IL17A",
                "signature_type": "CytoSig",
                "mean_activity": 0.8,
                "specificity_score": 0.6,
                "n_cells": 50000,
            }
        ],
        "cancer_comparison": {
            "data": [
                {
                    "cell_type": "T_cells",
                    "signature": "IFNG",
                    "signature_type": "CytoSig",
                    "mean_tumor": 1.5,
                    "mean_adjacent": 0.8,
                    "mean_difference": 0.7,
                    "std_difference": 0.3,
                    "n_pairs": 20,
                    "p_value": 0.01,
                }
            ],
            "cell_types": ["T_cells", "Macrophages"],
            "cytosig_signatures": ["IFNG", "IL17A"],
            "secact_signatures": [],
            "n_paired_donors": 50,
            "analysis_type": "paired_singlecell",
        },
    }
