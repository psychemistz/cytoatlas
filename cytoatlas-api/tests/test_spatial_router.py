"""Tests for spatial API endpoints.

Comprehensive tests for the Spatial router (/api/v1/spatial/*) covering
summary, datasets, activity, neighborhood, technology comparison,
gene coverage, and coordinate endpoints.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import os

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("RAG_ENABLED", "false")
os.environ.setdefault("AUDIT_ENABLED", "false")

from httpx import ASGITransport, AsyncClient

from app.main import app
from app.routers.spatial import get_spatial_service

API_PREFIX = "/api/v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    """Create async test client with ASGITransport."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_spatial_service():
    """Create a mock SpatialService and register it as a dependency override.

    Yields the mock so individual tests can configure return values.
    After the test the override is removed.
    """
    mock_svc = MagicMock()
    # Pre-configure every async method as AsyncMock so they are awaitable
    mock_svc.get_summary = AsyncMock()
    mock_svc.get_technologies = AsyncMock()
    mock_svc.get_tissues = AsyncMock()
    mock_svc.get_datasets = AsyncMock()
    mock_svc.get_dataset_detail = AsyncMock()
    mock_svc.get_activity = AsyncMock()
    mock_svc.get_tissue_summary = AsyncMock()
    mock_svc.get_neighborhood = AsyncMock()
    mock_svc.get_technology_comparison = AsyncMock()
    mock_svc.get_gene_coverage = AsyncMock()
    mock_svc.get_coordinates = AsyncMock()
    mock_svc.get_coordinates_with_activity = AsyncMock()

    app.dependency_overrides[get_spatial_service] = lambda: mock_svc
    yield mock_svc
    app.dependency_overrides.pop(get_spatial_service, None)


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

SAMPLE_SUMMARY = {
    "dataset": "SpatialCorpus-110M",
    "description": "Multi-technology spatial transcriptomics atlas",
    "n_datasets": 42,
    "n_technologies": 5,
    "n_tissues": 12,
    "n_cells": 110_000_000,
    "technologies": ["MERFISH", "Visium", "CODEX", "Slide-seq", "seqFISH"],
    "tissues": ["brain", "liver", "lung", "heart", "kidney",
                "intestine", "spleen", "skin", "pancreas",
                "bone_marrow", "lymph_node", "thymus"],
    "signature_types": ["CytoSig", "SecAct"],
}

SAMPLE_TECHNOLOGIES = ["CODEX", "MERFISH", "Slide-seq", "Visium", "seqFISH"]

SAMPLE_TISSUES = ["brain", "heart", "intestine", "kidney", "liver",
                  "lung", "pancreas", "skin", "spleen"]

SAMPLE_DATASETS = [
    {
        "dataset_id": "MERFISH_brain_001",
        "technology": "MERFISH",
        "tissue": "brain",
        "n_cells": 250_000,
        "n_genes": 500,
        "source": "Allen Brain Atlas",
        "accession": "SCP001",
    },
    {
        "dataset_id": "Visium_liver_001",
        "technology": "Visium",
        "tissue": "liver",
        "n_cells": 50_000,
        "n_genes": 18_000,
        "source": "Human Cell Atlas",
        "accession": "SCP002",
    },
    {
        "dataset_id": "CODEX_spleen_001",
        "technology": "CODEX",
        "tissue": "spleen",
        "n_cells": 120_000,
        "n_genes": 40,
        "source": "HuBMAP",
        "accession": "SCP003",
    },
]

SAMPLE_DATASET_DETAIL = {
    "dataset_id": "MERFISH_brain_001",
    "technology": "MERFISH",
    "tissue": "brain",
    "n_cells": 250_000,
    "n_genes": 500,
    "source": "Allen Brain Atlas",
    "accession": "SCP001",
    "qc_metrics": {"median_genes_per_cell": 120, "median_umi_per_cell": 450},
    "gene_panel": "500-gene MERFISH panel",
    "available_analyses": ["activity", "neighborhood"],
}

SAMPLE_ACTIVITY = [
    {
        "signature": "IFNG",
        "signature_type": "CytoSig",
        "technology": "MERFISH",
        "tissue": "brain",
        "mean_activity": 0.42,
        "median_activity": 0.38,
        "n_cells": 250_000,
    },
    {
        "signature": "TNF",
        "signature_type": "CytoSig",
        "technology": "MERFISH",
        "tissue": "brain",
        "mean_activity": 1.15,
        "median_activity": 1.02,
        "n_cells": 250_000,
    },
    {
        "signature": "IL6",
        "signature_type": "CytoSig",
        "technology": "Visium",
        "tissue": "liver",
        "mean_activity": -0.31,
        "median_activity": -0.28,
        "n_cells": 50_000,
    },
]

SAMPLE_TISSUE_SUMMARY = [
    {
        "tissue": "brain",
        "signature": "IFNG",
        "signature_type": "CytoSig",
        "mean_activity": 0.42,
        "std_activity": 0.15,
        "n_technologies": 3,
    },
    {
        "tissue": "liver",
        "signature": "IL6",
        "signature_type": "CytoSig",
        "mean_activity": -0.31,
        "std_activity": 0.22,
        "n_technologies": 2,
    },
]

SAMPLE_NEIGHBORHOOD = [
    {
        "tissue": "brain",
        "signature": "IFNG",
        "signature_type": "CytoSig",
        "neighbor_signature": "TNF",
        "spatial_correlation": 0.65,
        "p_value": 0.001,
        "n_neighborhoods": 500,
    },
    {
        "tissue": "brain",
        "signature": "IL6",
        "signature_type": "CytoSig",
        "neighbor_signature": "IL1B",
        "spatial_correlation": 0.48,
        "p_value": 0.005,
        "n_neighborhoods": 500,
    },
    {
        "tissue": "liver",
        "signature": "TNF",
        "signature_type": "CytoSig",
        "neighbor_signature": "CCL2",
        "spatial_correlation": 0.55,
        "p_value": 0.002,
        "n_neighborhoods": 300,
    },
]

SAMPLE_TECHNOLOGY_COMPARISON = {
    "comparisons": [
        {
            "technology_a": "MERFISH",
            "technology_b": "Visium",
            "tissue": "brain",
            "signature_type": "CytoSig",
            "spearman_r": 0.82,
            "concordance": 0.78,
            "n_shared_signatures": 35,
        },
        {
            "technology_a": "MERFISH",
            "technology_b": "Slide-seq",
            "tissue": "brain",
            "signature_type": "CytoSig",
            "spearman_r": 0.75,
            "concordance": 0.71,
            "n_shared_signatures": 30,
        },
    ],
    "signature_type": "CytoSig",
}

SAMPLE_GENE_COVERAGE = [
    {
        "technology": "MERFISH",
        "n_genes_total": 500,
        "n_cytosig_genes": 180,
        "n_secact_genes": 320,
        "cytosig_coverage_pct": 40.9,
        "secact_coverage_pct": 25.6,
    },
    {
        "technology": "Visium",
        "n_genes_total": 18_000,
        "n_cytosig_genes": 420,
        "n_secact_genes": 1_100,
        "cytosig_coverage_pct": 95.5,
        "secact_coverage_pct": 88.1,
    },
]

SAMPLE_COORDINATES = {
    "dataset_id": "MERFISH_brain_001",
    "n_cells": 1000,
    "coordinates": [
        {"x": 100.5, "y": 200.3, "cell_type": "Neuron", "cell_id": "c001"},
        {"x": 105.2, "y": 198.7, "cell_type": "Astrocyte", "cell_id": "c002"},
        {"x": 110.0, "y": 203.1, "cell_type": "Microglia", "cell_id": "c003"},
    ],
}

SAMPLE_COORDINATES_WITH_ACTIVITY = {
    "dataset_id": "MERFISH_brain_001",
    "signature_type": "CytoSig",
    "n_cells": 1000,
    "coordinates": [
        {"x": 100.5, "y": 200.3, "cell_type": "Neuron", "cell_id": "c001",
         "IFNG": 0.85, "TNF": -0.32},
        {"x": 105.2, "y": 198.7, "cell_type": "Astrocyte", "cell_id": "c002",
         "IFNG": -0.15, "TNF": 1.42},
        {"x": 110.0, "y": 203.1, "cell_type": "Microglia", "cell_id": "c003",
         "IFNG": 1.95, "TNF": 2.10},
    ],
}


# ---------------------------------------------------------------------------
# TestSpatialSummary
# ---------------------------------------------------------------------------


class TestSpatialSummary:
    """Tests for summary and metadata endpoints."""

    async def test_get_summary_returns_200(self, client, mock_spatial_service):
        """GET /spatial/summary returns 200 with summary data."""
        mock_spatial_service.get_summary.return_value = SAMPLE_SUMMARY

        response = await client.get(f"{API_PREFIX}/spatial/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["dataset"] == "SpatialCorpus-110M"
        assert data["n_datasets"] == 42
        assert data["n_technologies"] == 5
        assert data["n_tissues"] == 12
        assert data["n_cells"] == 110_000_000
        assert "technologies" in data
        assert "tissues" in data
        assert "signature_types" in data
        mock_spatial_service.get_summary.assert_awaited_once()

    async def test_get_summary_structure(self, client, mock_spatial_service):
        """GET /spatial/summary response contains all required fields."""
        mock_spatial_service.get_summary.return_value = SAMPLE_SUMMARY

        response = await client.get(f"{API_PREFIX}/spatial/summary")

        data = response.json()
        required_keys = {
            "dataset", "description", "n_datasets", "n_technologies",
            "n_tissues", "n_cells", "technologies", "tissues",
            "signature_types",
        }
        assert required_keys.issubset(data.keys())

    async def test_get_technologies_returns_sorted_list(self, client, mock_spatial_service):
        """GET /spatial/technologies returns a sorted list of technology names."""
        mock_spatial_service.get_technologies.return_value = SAMPLE_TECHNOLOGIES

        response = await client.get(f"{API_PREFIX}/spatial/technologies")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 5
        assert "MERFISH" in data
        assert "Visium" in data
        assert data == sorted(data)
        mock_spatial_service.get_technologies.assert_awaited_once()

    async def test_get_tissues_returns_sorted_list(self, client, mock_spatial_service):
        """GET /spatial/tissues returns a sorted list of tissue names."""
        mock_spatial_service.get_tissues.return_value = SAMPLE_TISSUES

        response = await client.get(f"{API_PREFIX}/spatial/tissues")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 9
        assert "brain" in data
        assert "liver" in data
        assert data == sorted(data)
        mock_spatial_service.get_tissues.assert_awaited_once()

    async def test_get_technologies_empty(self, client, mock_spatial_service):
        """GET /spatial/technologies returns empty list when no data."""
        mock_spatial_service.get_technologies.return_value = []

        response = await client.get(f"{API_PREFIX}/spatial/technologies")

        assert response.status_code == 200
        assert response.json() == []

    async def test_get_tissues_empty(self, client, mock_spatial_service):
        """GET /spatial/tissues returns empty list when no data."""
        mock_spatial_service.get_tissues.return_value = []

        response = await client.get(f"{API_PREFIX}/spatial/tissues")

        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# TestSpatialDatasets
# ---------------------------------------------------------------------------


class TestSpatialDatasets:
    """Tests for dataset listing and detail endpoints."""

    async def test_get_datasets_no_filter(self, client, mock_spatial_service):
        """GET /spatial/datasets without filters returns all datasets."""
        mock_spatial_service.get_datasets.return_value = SAMPLE_DATASETS

        response = await client.get(f"{API_PREFIX}/spatial/datasets")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        mock_spatial_service.get_datasets.assert_awaited_once_with(
            technology=None, tissue=None,
        )

    async def test_get_datasets_filter_by_technology(self, client, mock_spatial_service):
        """GET /spatial/datasets?technology=MERFISH filters by technology."""
        merfish_only = [d for d in SAMPLE_DATASETS if d["technology"] == "MERFISH"]
        mock_spatial_service.get_datasets.return_value = merfish_only

        response = await client.get(
            f"{API_PREFIX}/spatial/datasets",
            params={"technology": "MERFISH"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["technology"] == "MERFISH"
        mock_spatial_service.get_datasets.assert_awaited_once_with(
            technology="MERFISH", tissue=None,
        )

    async def test_get_datasets_filter_by_tissue(self, client, mock_spatial_service):
        """GET /spatial/datasets?tissue=liver filters by tissue."""
        liver_only = [d for d in SAMPLE_DATASETS if d["tissue"] == "liver"]
        mock_spatial_service.get_datasets.return_value = liver_only

        response = await client.get(
            f"{API_PREFIX}/spatial/datasets",
            params={"tissue": "liver"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["tissue"] == "liver"
        mock_spatial_service.get_datasets.assert_awaited_once_with(
            technology=None, tissue="liver",
        )

    async def test_get_datasets_filter_by_both(self, client, mock_spatial_service):
        """GET /spatial/datasets?technology=MERFISH&tissue=brain filters by both."""
        mock_spatial_service.get_datasets.return_value = [SAMPLE_DATASETS[0]]

        response = await client.get(
            f"{API_PREFIX}/spatial/datasets",
            params={"technology": "MERFISH", "tissue": "brain"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        mock_spatial_service.get_datasets.assert_awaited_once_with(
            technology="MERFISH", tissue="brain",
        )

    async def test_get_datasets_empty_result(self, client, mock_spatial_service):
        """GET /spatial/datasets with no matching filter returns empty list."""
        mock_spatial_service.get_datasets.return_value = []

        response = await client.get(
            f"{API_PREFIX}/spatial/datasets",
            params={"technology": "nonexistent"},
        )

        assert response.status_code == 200
        assert response.json() == []

    async def test_get_dataset_detail_found(self, client, mock_spatial_service):
        """GET /spatial/datasets/{dataset_id} returns detail for existing dataset."""
        mock_spatial_service.get_dataset_detail.return_value = SAMPLE_DATASET_DETAIL

        response = await client.get(
            f"{API_PREFIX}/spatial/datasets/MERFISH_brain_001"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "MERFISH_brain_001"
        assert data["technology"] == "MERFISH"
        assert data["tissue"] == "brain"
        assert "qc_metrics" in data
        assert "available_analyses" in data
        mock_spatial_service.get_dataset_detail.assert_awaited_once_with(
            dataset_id="MERFISH_brain_001",
        )

    async def test_get_dataset_detail_not_found(self, client, mock_spatial_service):
        """GET /spatial/datasets/{dataset_id} returns 404 for missing dataset."""
        mock_spatial_service.get_dataset_detail.return_value = None

        response = await client.get(
            f"{API_PREFIX}/spatial/datasets/nonexistent_dataset"
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "nonexistent_dataset" in data["detail"]

    async def test_get_dataset_detail_structure(self, client, mock_spatial_service):
        """GET /spatial/datasets/{dataset_id} response has expected fields."""
        mock_spatial_service.get_dataset_detail.return_value = SAMPLE_DATASET_DETAIL

        response = await client.get(
            f"{API_PREFIX}/spatial/datasets/MERFISH_brain_001"
        )

        data = response.json()
        required_keys = {"dataset_id", "technology", "tissue", "n_cells", "n_genes"}
        assert required_keys.issubset(data.keys())


# ---------------------------------------------------------------------------
# TestSpatialActivity
# ---------------------------------------------------------------------------


class TestSpatialActivity:
    """Tests for activity analysis endpoints."""

    async def test_get_activity_default_params(self, client, mock_spatial_service):
        """GET /spatial/activity with defaults returns CytoSig activity."""
        mock_spatial_service.get_activity.return_value = SAMPLE_ACTIVITY

        response = await client.get(f"{API_PREFIX}/spatial/activity")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        mock_spatial_service.get_activity.assert_awaited_once_with(
            technology=None, tissue=None, signature_type="CytoSig",
        )

    async def test_get_activity_filter_technology(self, client, mock_spatial_service):
        """GET /spatial/activity?technology=MERFISH filters by technology."""
        merfish_activity = [a for a in SAMPLE_ACTIVITY if a["technology"] == "MERFISH"]
        mock_spatial_service.get_activity.return_value = merfish_activity

        response = await client.get(
            f"{API_PREFIX}/spatial/activity",
            params={"technology": "MERFISH"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for item in data:
            assert item["technology"] == "MERFISH"
        mock_spatial_service.get_activity.assert_awaited_once_with(
            technology="MERFISH", tissue=None, signature_type="CytoSig",
        )

    async def test_get_activity_filter_tissue(self, client, mock_spatial_service):
        """GET /spatial/activity?tissue=liver filters by tissue."""
        liver_activity = [a for a in SAMPLE_ACTIVITY if a["tissue"] == "liver"]
        mock_spatial_service.get_activity.return_value = liver_activity

        response = await client.get(
            f"{API_PREFIX}/spatial/activity",
            params={"tissue": "liver"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["tissue"] == "liver"

    async def test_get_activity_secact(self, client, mock_spatial_service):
        """GET /spatial/activity?signature_type=SecAct uses SecAct matrix."""
        mock_spatial_service.get_activity.return_value = []

        response = await client.get(
            f"{API_PREFIX}/spatial/activity",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_spatial_service.get_activity.assert_awaited_once_with(
            technology=None, tissue=None, signature_type="SecAct",
        )

    async def test_get_activity_invalid_signature_type(self, client, mock_spatial_service):
        """GET /spatial/activity?signature_type=Invalid returns 422."""
        response = await client.get(
            f"{API_PREFIX}/spatial/activity",
            params={"signature_type": "Invalid"},
        )

        assert response.status_code == 422

    async def test_get_activity_by_technology_path(self, client, mock_spatial_service):
        """GET /spatial/activity/{technology} uses technology as path param."""
        merfish_activity = [a for a in SAMPLE_ACTIVITY if a["technology"] == "MERFISH"]
        mock_spatial_service.get_activity.return_value = merfish_activity

        response = await client.get(f"{API_PREFIX}/spatial/activity/MERFISH")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        mock_spatial_service.get_activity.assert_awaited_once_with(
            technology="MERFISH", tissue=None, signature_type="CytoSig",
        )

    async def test_get_activity_by_technology_with_tissue(self, client, mock_spatial_service):
        """GET /spatial/activity/{technology}?tissue=brain filters both."""
        brain_merfish = [
            a for a in SAMPLE_ACTIVITY
            if a["technology"] == "MERFISH" and a["tissue"] == "brain"
        ]
        mock_spatial_service.get_activity.return_value = brain_merfish

        response = await client.get(
            f"{API_PREFIX}/spatial/activity/MERFISH",
            params={"tissue": "brain"},
        )

        assert response.status_code == 200
        mock_spatial_service.get_activity.assert_awaited_once_with(
            technology="MERFISH", tissue="brain", signature_type="CytoSig",
        )

    async def test_get_activity_by_technology_secact(self, client, mock_spatial_service):
        """GET /spatial/activity/{technology}?signature_type=SecAct works."""
        mock_spatial_service.get_activity.return_value = []

        response = await client.get(
            f"{API_PREFIX}/spatial/activity/Visium",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_spatial_service.get_activity.assert_awaited_once_with(
            technology="Visium", tissue=None, signature_type="SecAct",
        )

    async def test_get_activity_record_structure(self, client, mock_spatial_service):
        """Activity records contain expected fields."""
        mock_spatial_service.get_activity.return_value = [SAMPLE_ACTIVITY[0]]

        response = await client.get(f"{API_PREFIX}/spatial/activity")

        data = response.json()
        record = data[0]
        expected_keys = {
            "signature", "signature_type", "technology", "tissue",
            "mean_activity", "median_activity", "n_cells",
        }
        assert expected_keys.issubset(record.keys())

    async def test_get_tissue_summary_default(self, client, mock_spatial_service):
        """GET /spatial/tissue-summary returns tissue-level aggregation."""
        mock_spatial_service.get_tissue_summary.return_value = SAMPLE_TISSUE_SUMMARY

        response = await client.get(f"{API_PREFIX}/spatial/tissue-summary")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        mock_spatial_service.get_tissue_summary.assert_awaited_once_with(
            signature_type="CytoSig",
        )

    async def test_get_tissue_summary_secact(self, client, mock_spatial_service):
        """GET /spatial/tissue-summary?signature_type=SecAct uses SecAct."""
        mock_spatial_service.get_tissue_summary.return_value = []

        response = await client.get(
            f"{API_PREFIX}/spatial/tissue-summary",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_spatial_service.get_tissue_summary.assert_awaited_once_with(
            signature_type="SecAct",
        )

    async def test_get_tissue_summary_invalid_signature_type(self, client, mock_spatial_service):
        """GET /spatial/tissue-summary?signature_type=Bad returns 422."""
        response = await client.get(
            f"{API_PREFIX}/spatial/tissue-summary",
            params={"signature_type": "Bad"},
        )

        assert response.status_code == 422

    async def test_get_tissue_summary_structure(self, client, mock_spatial_service):
        """Tissue summary records contain expected fields."""
        mock_spatial_service.get_tissue_summary.return_value = [SAMPLE_TISSUE_SUMMARY[0]]

        response = await client.get(f"{API_PREFIX}/spatial/tissue-summary")

        data = response.json()
        record = data[0]
        assert "tissue" in record
        assert "signature" in record
        assert "signature_type" in record
        assert "mean_activity" in record

    async def test_get_technology_comparison_default(self, client, mock_spatial_service):
        """GET /spatial/technology-comparison returns comparison data."""
        mock_spatial_service.get_technology_comparison.return_value = SAMPLE_TECHNOLOGY_COMPARISON

        response = await client.get(f"{API_PREFIX}/spatial/technology-comparison")

        assert response.status_code == 200
        data = response.json()
        assert "comparisons" in data
        assert len(data["comparisons"]) == 2
        mock_spatial_service.get_technology_comparison.assert_awaited_once_with(
            signature_type="CytoSig",
        )

    async def test_get_technology_comparison_secact(self, client, mock_spatial_service):
        """GET /spatial/technology-comparison?signature_type=SecAct passes param."""
        mock_spatial_service.get_technology_comparison.return_value = {"comparisons": []}

        response = await client.get(
            f"{API_PREFIX}/spatial/technology-comparison",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_spatial_service.get_technology_comparison.assert_awaited_once_with(
            signature_type="SecAct",
        )

    async def test_get_technology_comparison_invalid_signature_type(self, client, mock_spatial_service):
        """GET /spatial/technology-comparison?signature_type=Invalid returns 422."""
        response = await client.get(
            f"{API_PREFIX}/spatial/technology-comparison",
            params={"signature_type": "Invalid"},
        )

        assert response.status_code == 422

    async def test_get_technology_comparison_structure(self, client, mock_spatial_service):
        """Technology comparison response has expected structure."""
        mock_spatial_service.get_technology_comparison.return_value = SAMPLE_TECHNOLOGY_COMPARISON

        response = await client.get(f"{API_PREFIX}/spatial/technology-comparison")

        data = response.json()
        comparison = data["comparisons"][0]
        expected_keys = {
            "technology_a", "technology_b", "tissue",
            "signature_type", "spearman_r", "concordance",
        }
        assert expected_keys.issubset(comparison.keys())


# ---------------------------------------------------------------------------
# TestSpatialNeighborhood
# ---------------------------------------------------------------------------


class TestSpatialNeighborhood:
    """Tests for neighborhood analysis endpoints."""

    async def test_get_neighborhood_default(self, client, mock_spatial_service):
        """GET /spatial/neighborhood returns neighborhood data."""
        mock_spatial_service.get_neighborhood.return_value = SAMPLE_NEIGHBORHOOD

        response = await client.get(f"{API_PREFIX}/spatial/neighborhood")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        mock_spatial_service.get_neighborhood.assert_awaited_once_with(
            tissue=None, signature_type="CytoSig",
        )

    async def test_get_neighborhood_filter_tissue(self, client, mock_spatial_service):
        """GET /spatial/neighborhood?tissue=brain filters by tissue."""
        brain_only = [n for n in SAMPLE_NEIGHBORHOOD if n["tissue"] == "brain"]
        mock_spatial_service.get_neighborhood.return_value = brain_only

        response = await client.get(
            f"{API_PREFIX}/spatial/neighborhood",
            params={"tissue": "brain"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for item in data:
            assert item["tissue"] == "brain"
        mock_spatial_service.get_neighborhood.assert_awaited_once_with(
            tissue="brain", signature_type="CytoSig",
        )

    async def test_get_neighborhood_secact(self, client, mock_spatial_service):
        """GET /spatial/neighborhood?signature_type=SecAct passes param."""
        mock_spatial_service.get_neighborhood.return_value = []

        response = await client.get(
            f"{API_PREFIX}/spatial/neighborhood",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_spatial_service.get_neighborhood.assert_awaited_once_with(
            tissue=None, signature_type="SecAct",
        )

    async def test_get_neighborhood_invalid_signature_type(self, client, mock_spatial_service):
        """GET /spatial/neighborhood?signature_type=Bad returns 422."""
        response = await client.get(
            f"{API_PREFIX}/spatial/neighborhood",
            params={"signature_type": "Bad"},
        )

        assert response.status_code == 422

    async def test_get_neighborhood_by_tissue_path(self, client, mock_spatial_service):
        """GET /spatial/neighborhood/{tissue} uses tissue as path param."""
        brain_only = [n for n in SAMPLE_NEIGHBORHOOD if n["tissue"] == "brain"]
        mock_spatial_service.get_neighborhood.return_value = brain_only

        response = await client.get(f"{API_PREFIX}/spatial/neighborhood/brain")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        mock_spatial_service.get_neighborhood.assert_awaited_once_with(
            tissue="brain", signature_type="CytoSig",
        )

    async def test_get_neighborhood_by_tissue_secact(self, client, mock_spatial_service):
        """GET /spatial/neighborhood/{tissue}?signature_type=SecAct works."""
        mock_spatial_service.get_neighborhood.return_value = []

        response = await client.get(
            f"{API_PREFIX}/spatial/neighborhood/liver",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_spatial_service.get_neighborhood.assert_awaited_once_with(
            tissue="liver", signature_type="SecAct",
        )

    async def test_get_neighborhood_by_tissue_invalid_signature_type(self, client, mock_spatial_service):
        """GET /spatial/neighborhood/{tissue}?signature_type=X returns 422."""
        response = await client.get(
            f"{API_PREFIX}/spatial/neighborhood/brain",
            params={"signature_type": "X"},
        )

        assert response.status_code == 422

    async def test_get_neighborhood_structure(self, client, mock_spatial_service):
        """Neighborhood records contain expected fields."""
        mock_spatial_service.get_neighborhood.return_value = [SAMPLE_NEIGHBORHOOD[0]]

        response = await client.get(f"{API_PREFIX}/spatial/neighborhood")

        data = response.json()
        record = data[0]
        expected_keys = {
            "tissue", "signature", "signature_type",
            "neighbor_signature", "spatial_correlation",
        }
        assert expected_keys.issubset(record.keys())

    async def test_get_neighborhood_empty(self, client, mock_spatial_service):
        """GET /spatial/neighborhood returns empty list when no data."""
        mock_spatial_service.get_neighborhood.return_value = []

        response = await client.get(f"{API_PREFIX}/spatial/neighborhood")

        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# TestSpatialGeneCoverage
# ---------------------------------------------------------------------------


class TestSpatialGeneCoverage:
    """Tests for gene coverage endpoints."""

    async def test_get_gene_coverage_no_filter(self, client, mock_spatial_service):
        """GET /spatial/gene-coverage returns all technology coverage."""
        mock_spatial_service.get_gene_coverage.return_value = SAMPLE_GENE_COVERAGE

        response = await client.get(f"{API_PREFIX}/spatial/gene-coverage")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        mock_spatial_service.get_gene_coverage.assert_awaited_once_with(
            technology=None,
        )

    async def test_get_gene_coverage_filter_technology(self, client, mock_spatial_service):
        """GET /spatial/gene-coverage?technology=MERFISH filters by tech."""
        merfish_coverage = [c for c in SAMPLE_GENE_COVERAGE if c["technology"] == "MERFISH"]
        mock_spatial_service.get_gene_coverage.return_value = merfish_coverage

        response = await client.get(
            f"{API_PREFIX}/spatial/gene-coverage",
            params={"technology": "MERFISH"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["technology"] == "MERFISH"
        mock_spatial_service.get_gene_coverage.assert_awaited_once_with(
            technology="MERFISH",
        )

    async def test_get_gene_coverage_by_technology_path(self, client, mock_spatial_service):
        """GET /spatial/gene-coverage/{technology} uses path param."""
        visium_coverage = [c for c in SAMPLE_GENE_COVERAGE if c["technology"] == "Visium"]
        mock_spatial_service.get_gene_coverage.return_value = visium_coverage

        response = await client.get(f"{API_PREFIX}/spatial/gene-coverage/Visium")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["technology"] == "Visium"
        mock_spatial_service.get_gene_coverage.assert_awaited_once_with(
            technology="Visium",
        )

    async def test_get_gene_coverage_structure(self, client, mock_spatial_service):
        """Gene coverage records contain expected fields."""
        mock_spatial_service.get_gene_coverage.return_value = [SAMPLE_GENE_COVERAGE[0]]

        response = await client.get(f"{API_PREFIX}/spatial/gene-coverage")

        data = response.json()
        record = data[0]
        expected_keys = {
            "technology", "n_genes_total",
            "n_cytosig_genes", "n_secact_genes",
            "cytosig_coverage_pct", "secact_coverage_pct",
        }
        assert expected_keys.issubset(record.keys())

    async def test_get_gene_coverage_empty(self, client, mock_spatial_service):
        """GET /spatial/gene-coverage returns empty list when no data."""
        mock_spatial_service.get_gene_coverage.return_value = []

        response = await client.get(f"{API_PREFIX}/spatial/gene-coverage")

        assert response.status_code == 200
        assert response.json() == []

    async def test_get_gene_coverage_merfish_low_coverage(self, client, mock_spatial_service):
        """Targeted gene panels (MERFISH) have lower coverage than whole-transcriptome (Visium)."""
        mock_spatial_service.get_gene_coverage.return_value = SAMPLE_GENE_COVERAGE

        response = await client.get(f"{API_PREFIX}/spatial/gene-coverage")

        data = response.json()
        merfish = next(d for d in data if d["technology"] == "MERFISH")
        visium = next(d for d in data if d["technology"] == "Visium")
        assert merfish["cytosig_coverage_pct"] < visium["cytosig_coverage_pct"]
        assert merfish["secact_coverage_pct"] < visium["secact_coverage_pct"]


# ---------------------------------------------------------------------------
# TestSpatialCoordinates
# ---------------------------------------------------------------------------


class TestSpatialCoordinates:
    """Tests for spatial coordinate endpoints."""

    async def test_get_coordinates_found(self, client, mock_spatial_service):
        """GET /spatial/coordinates/{dataset_id} returns coordinate data."""
        mock_spatial_service.get_coordinates.return_value = SAMPLE_COORDINATES

        response = await client.get(
            f"{API_PREFIX}/spatial/coordinates/MERFISH_brain_001"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "MERFISH_brain_001"
        assert "coordinates" in data
        assert len(data["coordinates"]) == 3
        mock_spatial_service.get_coordinates.assert_awaited_once_with(
            dataset_id="MERFISH_brain_001",
        )

    async def test_get_coordinates_not_found(self, client, mock_spatial_service):
        """GET /spatial/coordinates/{dataset_id} returns 404 for missing data."""
        mock_spatial_service.get_coordinates.return_value = None

        response = await client.get(
            f"{API_PREFIX}/spatial/coordinates/nonexistent_dataset"
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "nonexistent_dataset" in data["detail"]

    async def test_get_coordinates_structure(self, client, mock_spatial_service):
        """Coordinate records contain x, y, and cell_type."""
        mock_spatial_service.get_coordinates.return_value = SAMPLE_COORDINATES

        response = await client.get(
            f"{API_PREFIX}/spatial/coordinates/MERFISH_brain_001"
        )

        data = response.json()
        coord = data["coordinates"][0]
        assert "x" in coord
        assert "y" in coord
        assert "cell_type" in coord
        assert isinstance(coord["x"], (int, float))
        assert isinstance(coord["y"], (int, float))

    async def test_get_coordinates_with_activity_found(self, client, mock_spatial_service):
        """GET /spatial/coordinates/{dataset_id}/activity returns activity overlay."""
        mock_spatial_service.get_coordinates_with_activity.return_value = (
            SAMPLE_COORDINATES_WITH_ACTIVITY
        )

        response = await client.get(
            f"{API_PREFIX}/spatial/coordinates/MERFISH_brain_001/activity"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "MERFISH_brain_001"
        assert data["signature_type"] == "CytoSig"
        assert "coordinates" in data
        coord = data["coordinates"][0]
        assert "IFNG" in coord
        assert "TNF" in coord
        mock_spatial_service.get_coordinates_with_activity.assert_awaited_once_with(
            dataset_id="MERFISH_brain_001", signature_type="CytoSig",
        )

    async def test_get_coordinates_with_activity_not_found(self, client, mock_spatial_service):
        """GET /spatial/coordinates/{dataset_id}/activity returns 404 if missing."""
        mock_spatial_service.get_coordinates_with_activity.return_value = None

        response = await client.get(
            f"{API_PREFIX}/spatial/coordinates/nonexistent_dataset/activity"
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "nonexistent_dataset" in data["detail"]

    async def test_get_coordinates_with_activity_secact(self, client, mock_spatial_service):
        """GET /spatial/coordinates/{id}/activity?signature_type=SecAct passes param."""
        mock_spatial_service.get_coordinates_with_activity.return_value = {
            "dataset_id": "MERFISH_brain_001",
            "signature_type": "SecAct",
            "n_cells": 1000,
            "coordinates": [],
        }

        response = await client.get(
            f"{API_PREFIX}/spatial/coordinates/MERFISH_brain_001/activity",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["signature_type"] == "SecAct"
        mock_spatial_service.get_coordinates_with_activity.assert_awaited_once_with(
            dataset_id="MERFISH_brain_001", signature_type="SecAct",
        )

    async def test_get_coordinates_with_activity_invalid_signature_type(
        self, client, mock_spatial_service
    ):
        """GET /spatial/coordinates/{id}/activity?signature_type=Bad returns 422."""
        response = await client.get(
            f"{API_PREFIX}/spatial/coordinates/MERFISH_brain_001/activity",
            params={"signature_type": "Bad"},
        )

        assert response.status_code == 422

    async def test_get_coordinates_with_activity_structure(self, client, mock_spatial_service):
        """Activity coordinate records have x, y, cell_type, and activity values."""
        mock_spatial_service.get_coordinates_with_activity.return_value = (
            SAMPLE_COORDINATES_WITH_ACTIVITY
        )

        response = await client.get(
            f"{API_PREFIX}/spatial/coordinates/MERFISH_brain_001/activity"
        )

        data = response.json()
        coord = data["coordinates"][2]
        assert coord["cell_type"] == "Microglia"
        assert coord["IFNG"] == 1.95
        assert coord["TNF"] == 2.10
        assert isinstance(coord["x"], (int, float))
        assert isinstance(coord["y"], (int, float))

    async def test_get_coordinates_different_datasets(self, client, mock_spatial_service):
        """Different dataset IDs are passed correctly to the service."""
        mock_spatial_service.get_coordinates.return_value = {
            "dataset_id": "Visium_liver_001",
            "n_cells": 500,
            "coordinates": [{"x": 50.0, "y": 75.0, "cell_type": "Hepatocyte"}],
        }

        response = await client.get(
            f"{API_PREFIX}/spatial/coordinates/Visium_liver_001"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "Visium_liver_001"
        mock_spatial_service.get_coordinates.assert_awaited_once_with(
            dataset_id="Visium_liver_001",
        )
