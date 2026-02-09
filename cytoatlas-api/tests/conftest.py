"""Pytest configuration and comprehensive fixtures for CytoAtlas API tests."""

import asyncio
import json
import os
import sys
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure the app package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables before importing app modules
os.environ["ENVIRONMENT"] = "development"
os.environ["DEBUG"] = "false"
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = ""
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only-not-production"
os.environ["LLM_BASE_URL"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["RAG_ENABLED"] = "false"
os.environ["AUDIT_ENABLED"] = "false"

# Path to fixture data
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture data loaders
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def activity_summary_data() -> list[dict]:
    """Load activity summary fixture data."""
    with open(FIXTURES_DIR / "activity_summary.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def correlations_data() -> dict:
    """Load correlations fixture data."""
    with open(FIXTURES_DIR / "correlations.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def differential_data() -> list[dict]:
    """Load differential fixture data."""
    with open(FIXTURES_DIR / "differential.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def validation_data() -> dict:
    """Load validation fixture data."""
    with open(FIXTURES_DIR / "validation_results.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sample data fixtures (small, inline for unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_activity_data() -> list[dict]:
    """Small inline activity dataset for unit tests."""
    return [
        {
            "cell_type": "CD4_T",
            "signature": "IFNG",
            "signature_type": "CytoSig",
            "mean_activity": 0.85,
            "n_samples": 120,
            "n_cells": 15000,
        },
        {
            "cell_type": "CD8_T",
            "signature": "IFNG",
            "signature_type": "CytoSig",
            "mean_activity": 1.52,
            "n_samples": 115,
            "n_cells": 12000,
        },
        {
            "cell_type": "Monocytes",
            "signature": "TNF",
            "signature_type": "CytoSig",
            "mean_activity": 1.95,
            "n_samples": 130,
            "n_cells": 20000,
        },
        {
            "cell_type": "Macrophages",
            "signature": "CCL2",
            "signature_type": "SecAct",
            "mean_activity": 1.65,
            "n_samples": 95,
            "n_cells": 7000,
        },
    ]


@pytest.fixture
def sample_correlation_data() -> list[dict]:
    """Small inline correlation dataset for unit tests."""
    return [
        {
            "cell_type": "CD4_T",
            "signature": "IFNG",
            "signature_type": "CytoSig",
            "variable": "Age",
            "rho": 0.25,
            "pvalue": 0.001,
            "qvalue": 0.01,
            "n_samples": 421,
        },
        {
            "cell_type": "CD8_T",
            "signature": "IFNG",
            "signature_type": "CytoSig",
            "variable": "Age",
            "rho": 0.32,
            "pvalue": 0.0005,
            "qvalue": 0.005,
            "n_samples": 421,
        },
        {
            "cell_type": "CD4_T",
            "signature": "CCL2",
            "signature_type": "SecAct",
            "variable": "Age",
            "rho": 0.18,
            "pvalue": 0.02,
            "qvalue": 0.08,
            "n_samples": 421,
        },
    ]


@pytest.fixture
def sample_differential_data() -> list[dict]:
    """Small inline differential dataset for unit tests."""
    return [
        {
            "cell_type": "CD8_T",
            "signature": "IFNG",
            "signature_type": "CytoSig",
            "disease": "Rheumatoid Arthritis",
            "activity_diff": 1.45,
            "mean_disease": 2.10,
            "mean_healthy": 0.65,
            "pvalue": 0.00001,
            "qvalue": 0.0002,
            "n_disease": 45,
            "n_healthy": 80,
        },
        {
            "cell_type": "Monocytes",
            "signature": "TNF",
            "signature_type": "CytoSig",
            "disease": "Rheumatoid Arthritis",
            "activity_diff": 1.82,
            "mean_disease": 2.50,
            "mean_healthy": 0.68,
            "pvalue": 0.000005,
            "qvalue": 0.0001,
            "n_disease": 45,
            "n_healthy": 80,
        },
    ]


# ---------------------------------------------------------------------------
# Settings override fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_override(tmp_path):
    """Patch app settings to use test data paths."""
    test_viz_path = tmp_path / "viz_data"
    test_viz_path.mkdir(parents=True, exist_ok=True)
    test_results_path = tmp_path / "results"
    test_results_path.mkdir(parents=True, exist_ok=True)

    from app.config import get_settings

    settings = get_settings()
    original_viz = settings.viz_data_path
    original_results = settings.results_base_path

    object.__setattr__(settings, "viz_data_path", test_viz_path)
    object.__setattr__(settings, "results_base_path", test_results_path)

    yield settings

    object.__setattr__(settings, "viz_data_path", original_viz)
    object.__setattr__(settings, "results_base_path", original_results)


# ---------------------------------------------------------------------------
# HTTP test client using httpx.AsyncClient + ASGITransport
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with ASGITransport (no network I/O)."""
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Auth fixtures (JWT tokens)
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Generate JWT token headers for a regular user (viewer role)."""
    from app.core.security import create_access_token

    token = create_access_token(subject="testuser@example.com")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers() -> dict[str, str]:
    """Generate JWT token headers for an admin user."""
    from app.core.security import create_access_token

    token = create_access_token(subject="admin@example.com")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Mock LLM response fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Provide a mock LLM chat response."""
    return {
        "content": "Based on the CytoAtlas data, IFNG activity is highest in NK cells and CD8 T cells.",
        "tool_calls": None,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 30,
        },
    }


@pytest.fixture
def mock_llm_tool_response() -> dict[str, Any]:
    """Provide a mock LLM response that triggers tool use."""
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_001",
                "name": "get_activity_data",
                "arguments": {
                    "signature": "IFNG",
                    "atlas": "cima",
                    "signature_type": "CytoSig",
                },
            }
        ],
        "finish_reason": "tool_calls",
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
        },
    }


# ---------------------------------------------------------------------------
# Sample CIMA / Inflammation / scAtlas fixtures (reuse from existing)
# ---------------------------------------------------------------------------

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
    }


# ---------------------------------------------------------------------------
# Search index fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_search_index() -> dict[str, Any]:
    """Provide a minimal search index for testing."""
    return {
        "entities": {
            "cytokine:IFNG": {
                "id": "cytokine:IFNG",
                "name": "IFNG",
                "type": "cytokine",
                "description": "Interferon gamma",
                "aliases": ["IFN-gamma", "IFNG", "ifng"],
                "atlases": ["CIMA", "Inflammation", "scAtlas"],
                "atlas_count": 3,
                "hgnc_symbol": "IFNG",
            },
            "cytokine:TNF": {
                "id": "cytokine:TNF",
                "name": "TNF",
                "type": "cytokine",
                "description": "Tumor necrosis factor",
                "aliases": ["TNFA", "TNF-alpha", "tnf"],
                "atlases": ["CIMA", "Inflammation"],
                "atlas_count": 2,
                "hgnc_symbol": "TNF",
            },
            "cell_type:CD8_T": {
                "id": "cell_type:CD8_T",
                "name": "CD8_T",
                "type": "cell_type",
                "description": "CD8+ cytotoxic T cells",
                "aliases": ["CD8+ T cells", "Cytotoxic T"],
                "atlases": ["CIMA", "Inflammation", "scAtlas"],
                "atlas_count": 3,
            },
            "disease:Rheumatoid Arthritis": {
                "id": "disease:Rheumatoid Arthritis",
                "name": "Rheumatoid Arthritis",
                "type": "disease",
                "description": "Autoimmune joint disease",
                "aliases": ["RA"],
                "atlases": ["Inflammation"],
                "atlas_count": 1,
            },
            "organ:Lung": {
                "id": "organ:Lung",
                "name": "Lung",
                "type": "organ",
                "description": "Lung tissue",
                "aliases": [],
                "atlases": ["scAtlas"],
                "atlas_count": 1,
            },
            "gene:IFNG": {
                "id": "gene:IFNG",
                "name": "IFNG",
                "type": "gene",
                "description": "Gene encoding interferon gamma",
                "aliases": ["IFN-gamma", "ifng"],
                "atlases": ["CIMA"],
                "atlas_count": 1,
                "hgnc_symbol": "IFNG",
                "cytosig_name": "IFNG",
            },
        },
        "aliases": {
            "ifng": "cytokine:IFNG",
            "tnf": "cytokine:TNF",
            "cd8_t": "cell_type:CD8_T",
            "rheumatoid arthritis": "disease:Rheumatoid Arthritis",
            "lung": "organ:Lung",
        },
        "by_type": {
            "cytokine": ["cytokine:IFNG", "cytokine:TNF"],
            "cell_type": ["cell_type:CD8_T"],
            "disease": ["disease:Rheumatoid Arthritis"],
            "organ": ["organ:Lung"],
            "protein": [],
            "gene": ["gene:IFNG"],
        },
    }
