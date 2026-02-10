"""Tests for perturbation API endpoints.

Covers all parse_10M and Tahoe endpoints with mocked PerturbationService.
Uses FastAPI dependency_overrides to inject mock service instances,
following the project's existing test patterns (httpx.AsyncClient + ASGITransport).
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient

# Set env before app import
import os

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("RAG_ENABLED", "false")
os.environ.setdefault("AUDIT_ENABLED", "false")

from app.main import app
from app.routers.perturbation import get_perturbation_service

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
def mock_service():
    """Create a mock PerturbationService with all methods as AsyncMocks.

    Each test overrides the return_value for the specific method it exercises.
    """
    svc = MagicMock()
    svc.get_summary = AsyncMock()
    svc.get_parse10m_summary = AsyncMock()
    svc.get_parse10m_cytokines = AsyncMock()
    svc.get_parse10m_cell_types = AsyncMock()
    svc.get_parse10m_activity = AsyncMock()
    svc.get_parse10m_treatment_effect = AsyncMock()
    svc.get_parse10m_ground_truth = AsyncMock()
    svc.get_parse10m_heatmap = AsyncMock()
    svc.get_parse10m_donor_variability = AsyncMock()
    svc.get_parse10m_cytokine_families = AsyncMock()
    svc.get_tahoe_summary = AsyncMock()
    svc.get_tahoe_drugs = AsyncMock()
    svc.get_tahoe_cell_lines = AsyncMock()
    svc.get_tahoe_activity = AsyncMock()
    svc.get_tahoe_drug_effect = AsyncMock()
    svc.get_tahoe_sensitivity_matrix = AsyncMock()
    svc.get_tahoe_dose_response = AsyncMock()
    svc.get_tahoe_pathway_activation = AsyncMock()
    return svc


@pytest.fixture(autouse=True)
def override_service(mock_service):
    """Inject mock_service via FastAPI dependency_overrides for every test."""
    app.dependency_overrides[get_perturbation_service] = lambda: mock_service
    yield
    app.dependency_overrides.pop(get_perturbation_service, None)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_combined_summary():
    return {
        "parse_10M": {
            "dataset": "parse_10M",
            "description": "Cytokine stimulation perturbation screen",
            "n_cytokines": 90,
            "n_cell_types": 18,
            "cytokines": ["IL6", "IFNG", "TNF"],
            "cell_types": ["CD4_T", "CD8_T", "Monocytes"],
            "signature_types": ["CytoSig", "SecAct"],
        },
        "tahoe": {
            "dataset": "Tahoe",
            "description": "Drug response perturbation screen",
            "n_drugs": 95,
            "n_cell_lines": 50,
            "drugs": ["Dexamethasone", "Imatinib"],
            "cell_lines": ["A549", "MCF7"],
            "signature_types": ["CytoSig", "SecAct"],
        },
        "total_datasets": 2,
        "signature_types": ["CytoSig", "SecAct"],
    }


@pytest.fixture
def sample_parse10m_summary():
    return {
        "dataset": "parse_10M",
        "description": "Cytokine stimulation perturbation screen",
        "n_cytokines": 90,
        "n_cell_types": 18,
        "cytokines": ["IL6", "IFNG", "TNF"],
        "cell_types": ["CD4_T", "CD8_T", "Monocytes"],
        "signature_types": ["CytoSig", "SecAct"],
    }


@pytest.fixture
def sample_cytokines():
    return ["CCL2", "CXCL10", "IFNG", "IL1B", "IL6", "IL10", "IL17A", "TNF"]


@pytest.fixture
def sample_cell_types():
    return ["B_cells", "CD4_T", "CD8_T", "Monocytes", "NK_cells"]


@pytest.fixture
def sample_parse10m_activity():
    return [
        {
            "cytokine": "IL6",
            "cell_type": "CD4_T",
            "signature": "IL6",
            "signature_type": "CytoSig",
            "activity": 1.85,
            "activity_ctrl": 0.12,
            "activity_diff": 1.73,
            "pvalue": 0.0001,
            "n_cells_treated": 500,
            "n_cells_ctrl": 480,
        },
        {
            "cytokine": "IL6",
            "cell_type": "CD4_T",
            "signature": "STAT3",
            "signature_type": "CytoSig",
            "activity": 1.22,
            "activity_ctrl": 0.05,
            "activity_diff": 1.17,
            "pvalue": 0.001,
            "n_cells_treated": 500,
            "n_cells_ctrl": 480,
        },
    ]


@pytest.fixture
def sample_treatment_effect():
    return [
        {
            "cytokine": "IL6",
            "cell_type": "CD4_T",
            "signature": "IL6",
            "signature_type": "CytoSig",
            "activity_diff": 1.73,
            "pvalue": 0.0001,
            "qvalue": 0.001,
        },
        {
            "cytokine": "IFNG",
            "cell_type": "CD8_T",
            "signature": "IFNG",
            "signature_type": "CytoSig",
            "activity_diff": 2.45,
            "pvalue": 0.00001,
            "qvalue": 0.0001,
        },
    ]


@pytest.fixture
def sample_ground_truth():
    return [
        {
            "cytokine": "IL6",
            "signature": "IL6",
            "signature_type": "CytoSig",
            "rank": 1,
            "activity_diff": 1.73,
            "is_top_hit": True,
            "recovery_rate": 0.95,
        },
        {
            "cytokine": "IFNG",
            "signature": "IFNG",
            "signature_type": "CytoSig",
            "rank": 1,
            "activity_diff": 2.45,
            "is_top_hit": True,
            "recovery_rate": 0.92,
        },
    ]


@pytest.fixture
def sample_heatmap_data():
    return {
        "rows": ["IL6", "IFNG", "TNF"],
        "columns": ["IL6", "IFNG", "TNF", "IL1B"],
        "values": [[1.73, 0.1, 0.05, 0.2], [0.08, 2.45, 0.03, 0.1], [0.15, 0.12, 1.95, 0.3]],
        "cell_type": "CD4_T",
        "signature_type": "CytoSig",
    }


@pytest.fixture
def sample_donor_variability():
    return [
        {
            "cytokine": "IL6",
            "cell_type": "CD4_T",
            "donor_id": "D001",
            "activity_diff": 1.85,
            "mean_activity_diff": 1.73,
            "cv": 0.15,
        },
        {
            "cytokine": "IL6",
            "cell_type": "CD4_T",
            "donor_id": "D002",
            "activity_diff": 1.60,
            "mean_activity_diff": 1.73,
            "cv": 0.15,
        },
    ]


@pytest.fixture
def sample_cytokine_families():
    return [
        {"cytokine": "IL6", "family": "Interleukins", "subfamily": "IL-6 family"},
        {"cytokine": "IFNG", "family": "Interferons", "subfamily": "Type II"},
        {"cytokine": "TNF", "family": "TNF superfamily", "subfamily": None},
        {"cytokine": "CCL2", "family": "Chemokines", "subfamily": "CC chemokines"},
    ]


@pytest.fixture
def sample_tahoe_summary():
    return {
        "dataset": "Tahoe",
        "description": "Drug response perturbation screen",
        "n_drugs": 95,
        "n_cell_lines": 50,
        "drugs": ["Dexamethasone", "Imatinib", "Paclitaxel"],
        "cell_lines": ["A549", "MCF7", "HeLa"],
        "signature_types": ["CytoSig", "SecAct"],
    }


@pytest.fixture
def sample_drugs():
    return ["Dexamethasone", "Imatinib", "Paclitaxel", "Cisplatin", "Erlotinib"]


@pytest.fixture
def sample_cell_lines():
    return ["A549", "HeLa", "MCF7", "PC3", "U2OS"]


@pytest.fixture
def sample_tahoe_activity():
    return [
        {
            "drug": "Dexamethasone",
            "cell_line": "A549",
            "signature": "IL6",
            "signature_type": "CytoSig",
            "activity": -1.2,
            "activity_ctrl": 0.3,
            "activity_diff": -1.5,
            "pvalue": 0.0005,
        },
        {
            "drug": "Dexamethasone",
            "cell_line": "A549",
            "signature": "TNF",
            "signature_type": "CytoSig",
            "activity": -0.8,
            "activity_ctrl": 0.1,
            "activity_diff": -0.9,
            "pvalue": 0.003,
        },
    ]


@pytest.fixture
def sample_drug_effect():
    return [
        {
            "drug": "Dexamethasone",
            "cell_line": "A549",
            "signature": "IL6",
            "signature_type": "CytoSig",
            "activity_diff": -1.5,
            "pvalue": 0.0005,
            "qvalue": 0.005,
        },
        {
            "drug": "Imatinib",
            "cell_line": "A549",
            "signature": "PDGF",
            "signature_type": "CytoSig",
            "activity_diff": -2.1,
            "pvalue": 0.0001,
            "qvalue": 0.001,
        },
    ]


@pytest.fixture
def sample_sensitivity_matrix():
    return {
        "rows": ["Dexamethasone", "Imatinib", "Paclitaxel"],
        "columns": ["A549", "MCF7", "HeLa"],
        "values": [[-1.5, -0.8, -1.2], [-2.1, -1.7, -0.5], [-0.3, -0.9, -1.8]],
        "signature_type": "CytoSig",
    }


@pytest.fixture
def sample_dose_response():
    return [
        {
            "drug": "Dexamethasone",
            "cell_line": "A549",
            "dose": 0.1,
            "dose_unit": "uM",
            "activity": -0.3,
            "viability": 0.95,
        },
        {
            "drug": "Dexamethasone",
            "cell_line": "A549",
            "dose": 1.0,
            "dose_unit": "uM",
            "activity": -0.9,
            "viability": 0.85,
        },
        {
            "drug": "Dexamethasone",
            "cell_line": "A549",
            "dose": 10.0,
            "dose_unit": "uM",
            "activity": -1.5,
            "viability": 0.60,
        },
    ]


@pytest.fixture
def sample_pathway_activation():
    return [
        {
            "drug": "Dexamethasone",
            "pathway": "JAK-STAT",
            "activity": -1.8,
            "pvalue": 0.0001,
            "n_signatures": 5,
        },
        {
            "drug": "Dexamethasone",
            "pathway": "NF-kB",
            "activity": -1.2,
            "pvalue": 0.001,
            "n_signatures": 8,
        },
        {
            "drug": "Imatinib",
            "pathway": "RTK-RAS",
            "activity": -2.3,
            "pvalue": 0.00005,
            "n_signatures": 6,
        },
    ]


# ===========================================================================
# TestPerturbationSummary — combined summary endpoint
# ===========================================================================


class TestPerturbationSummary:
    """Tests for GET /perturbation/summary."""

    async def test_summary_returns_200(self, client, mock_service, sample_combined_summary):
        """Combined summary returns 200 with both datasets."""
        mock_service.get_summary.return_value = sample_combined_summary

        response = await client.get(f"{API_PREFIX}/perturbation/summary")

        assert response.status_code == 200
        data = response.json()
        assert "parse_10M" in data
        assert "tahoe" in data
        assert data["total_datasets"] == 2
        assert "CytoSig" in data["signature_types"]
        assert "SecAct" in data["signature_types"]

    async def test_summary_includes_parse10m_metadata(
        self, client, mock_service, sample_combined_summary
    ):
        """Combined summary contains parse_10M dataset details."""
        mock_service.get_summary.return_value = sample_combined_summary

        response = await client.get(f"{API_PREFIX}/perturbation/summary")
        data = response.json()

        parse10m = data["parse_10M"]
        assert parse10m["dataset"] == "parse_10M"
        assert parse10m["n_cytokines"] == 90
        assert parse10m["n_cell_types"] == 18
        assert isinstance(parse10m["cytokines"], list)
        assert isinstance(parse10m["cell_types"], list)

    async def test_summary_includes_tahoe_metadata(
        self, client, mock_service, sample_combined_summary
    ):
        """Combined summary contains Tahoe dataset details."""
        mock_service.get_summary.return_value = sample_combined_summary

        response = await client.get(f"{API_PREFIX}/perturbation/summary")
        data = response.json()

        tahoe = data["tahoe"]
        assert tahoe["dataset"] == "Tahoe"
        assert tahoe["n_drugs"] == 95
        assert tahoe["n_cell_lines"] == 50
        assert isinstance(tahoe["drugs"], list)
        assert isinstance(tahoe["cell_lines"], list)

    async def test_summary_calls_service(self, client, mock_service, sample_combined_summary):
        """The endpoint delegates to service.get_summary()."""
        mock_service.get_summary.return_value = sample_combined_summary

        await client.get(f"{API_PREFIX}/perturbation/summary")

        mock_service.get_summary.assert_awaited_once()


# ===========================================================================
# TestParse10MEndpoints — all parse_10M sub-endpoints
# ===========================================================================


class TestParse10MEndpoints:
    """Tests for /perturbation/parse10m/* endpoints."""

    # -- parse10m/summary --------------------------------------------------

    async def test_parse10m_summary_returns_200(
        self, client, mock_service, sample_parse10m_summary
    ):
        """parse_10M summary returns 200 with dataset metadata."""
        mock_service.get_parse10m_summary.return_value = sample_parse10m_summary

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["dataset"] == "parse_10M"
        assert "n_cytokines" in data
        assert "n_cell_types" in data
        assert "cytokines" in data
        assert "cell_types" in data
        assert "signature_types" in data

    async def test_parse10m_summary_calls_service(self, client, mock_service, sample_parse10m_summary):
        """parse10m/summary delegates to service.get_parse10m_summary()."""
        mock_service.get_parse10m_summary.return_value = sample_parse10m_summary

        await client.get(f"{API_PREFIX}/perturbation/parse10m/summary")

        mock_service.get_parse10m_summary.assert_awaited_once()

    # -- parse10m/cytokines ------------------------------------------------

    async def test_parse10m_cytokines_returns_200(self, client, mock_service, sample_cytokines):
        """Cytokines endpoint returns 200 with sorted list."""
        mock_service.get_parse10m_cytokines.return_value = sample_cytokines

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/cytokines")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == len(sample_cytokines)
        assert "IFNG" in data
        assert "IL6" in data

    async def test_parse10m_cytokines_is_list_of_strings(self, client, mock_service, sample_cytokines):
        """Cytokines endpoint returns list[str]."""
        mock_service.get_parse10m_cytokines.return_value = sample_cytokines

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/cytokines")
        data = response.json()

        for item in data:
            assert isinstance(item, str)

    # -- parse10m/cell-types -----------------------------------------------

    async def test_parse10m_cell_types_returns_200(self, client, mock_service, sample_cell_types):
        """Cell types endpoint returns 200 with sorted list."""
        mock_service.get_parse10m_cell_types.return_value = sample_cell_types

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/cell-types")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "CD4_T" in data
        assert "CD8_T" in data

    async def test_parse10m_cell_types_is_list_of_strings(self, client, mock_service, sample_cell_types):
        """Cell types endpoint returns list[str]."""
        mock_service.get_parse10m_cell_types.return_value = sample_cell_types

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/cell-types")
        data = response.json()

        for item in data:
            assert isinstance(item, str)

    # -- parse10m/activity -------------------------------------------------

    async def test_parse10m_activity_returns_200(
        self, client, mock_service, sample_parse10m_activity
    ):
        """Activity endpoint returns 200 with activity records."""
        mock_service.get_parse10m_activity.return_value = sample_parse10m_activity

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/activity")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_parse10m_activity_with_cytokine_filter(
        self, client, mock_service, sample_parse10m_activity
    ):
        """Activity endpoint passes cytokine filter to service."""
        mock_service.get_parse10m_activity.return_value = sample_parse10m_activity

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/activity",
            params={"cytokine": "IL6"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_activity.assert_awaited_once_with(
            cytokine="IL6", cell_type=None, signature_type="CytoSig"
        )

    async def test_parse10m_activity_with_cell_type_filter(
        self, client, mock_service, sample_parse10m_activity
    ):
        """Activity endpoint passes cell_type filter to service."""
        mock_service.get_parse10m_activity.return_value = sample_parse10m_activity

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/activity",
            params={"cell_type": "CD4_T"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_activity.assert_awaited_once_with(
            cytokine=None, cell_type="CD4_T", signature_type="CytoSig"
        )

    async def test_parse10m_activity_with_signature_type_secact(
        self, client, mock_service, sample_parse10m_activity
    ):
        """Activity endpoint accepts SecAct signature type."""
        mock_service.get_parse10m_activity.return_value = sample_parse10m_activity

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/activity",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_activity.assert_awaited_once_with(
            cytokine=None, cell_type=None, signature_type="SecAct"
        )

    async def test_parse10m_activity_with_all_filters(
        self, client, mock_service, sample_parse10m_activity
    ):
        """Activity endpoint forwards all three query params."""
        mock_service.get_parse10m_activity.return_value = sample_parse10m_activity

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/activity",
            params={"cytokine": "IFNG", "cell_type": "CD8_T", "signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_activity.assert_awaited_once_with(
            cytokine="IFNG", cell_type="CD8_T", signature_type="SecAct"
        )

    async def test_parse10m_activity_invalid_signature_type(self, client, mock_service):
        """Activity endpoint rejects invalid signature_type with 422."""
        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/activity",
            params={"signature_type": "Invalid"},
        )

        assert response.status_code == 422

    async def test_parse10m_activity_default_signature_type(
        self, client, mock_service, sample_parse10m_activity
    ):
        """Activity endpoint defaults to CytoSig when signature_type is omitted."""
        mock_service.get_parse10m_activity.return_value = sample_parse10m_activity

        await client.get(f"{API_PREFIX}/perturbation/parse10m/activity")

        mock_service.get_parse10m_activity.assert_awaited_once_with(
            cytokine=None, cell_type=None, signature_type="CytoSig"
        )

    # -- parse10m/treatment-effect -----------------------------------------

    async def test_treatment_effect_returns_200(
        self, client, mock_service, sample_treatment_effect
    ):
        """Treatment effect endpoint returns 200 with effect records."""
        mock_service.get_parse10m_treatment_effect.return_value = sample_treatment_effect

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/treatment-effect")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_treatment_effect_with_cell_type_filter(
        self, client, mock_service, sample_treatment_effect
    ):
        """Treatment effect endpoint passes cell_type to service."""
        mock_service.get_parse10m_treatment_effect.return_value = sample_treatment_effect

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/treatment-effect",
            params={"cell_type": "CD4_T", "signature_type": "CytoSig"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_treatment_effect.assert_awaited_once_with(
            cell_type="CD4_T", signature_type="CytoSig"
        )

    async def test_treatment_effect_by_cell_type_path(
        self, client, mock_service, sample_treatment_effect
    ):
        """Treatment effect with path parameter forwards cell_type to service."""
        mock_service.get_parse10m_treatment_effect.return_value = sample_treatment_effect

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/treatment-effect/CD8_T",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_treatment_effect.assert_awaited_once_with(
            cell_type="CD8_T", signature_type="SecAct"
        )

    async def test_treatment_effect_invalid_signature_type(self, client, mock_service):
        """Treatment effect rejects invalid signature_type with 422."""
        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/treatment-effect",
            params={"signature_type": "BadType"},
        )

        assert response.status_code == 422

    # -- parse10m/ground-truth ---------------------------------------------

    async def test_ground_truth_returns_200(self, client, mock_service, sample_ground_truth):
        """Ground truth endpoint returns 200 with validation records."""
        mock_service.get_parse10m_ground_truth.return_value = sample_ground_truth

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/ground-truth")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_ground_truth_with_signature_type(
        self, client, mock_service, sample_ground_truth
    ):
        """Ground truth endpoint passes signature_type to service."""
        mock_service.get_parse10m_ground_truth.return_value = sample_ground_truth

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/ground-truth",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_ground_truth.assert_awaited_once_with(
            signature_type="SecAct"
        )

    async def test_ground_truth_by_type_path_cytosig(
        self, client, mock_service, sample_ground_truth
    ):
        """Ground truth path variant for CytoSig returns 200."""
        mock_service.get_parse10m_ground_truth.return_value = sample_ground_truth

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/ground-truth/CytoSig"
        )

        assert response.status_code == 200
        mock_service.get_parse10m_ground_truth.assert_awaited_once_with(
            signature_type="CytoSig"
        )

    async def test_ground_truth_by_type_path_secact(
        self, client, mock_service, sample_ground_truth
    ):
        """Ground truth path variant for SecAct returns 200."""
        mock_service.get_parse10m_ground_truth.return_value = sample_ground_truth

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/ground-truth/SecAct"
        )

        assert response.status_code == 200
        mock_service.get_parse10m_ground_truth.assert_awaited_once_with(
            signature_type="SecAct"
        )

    async def test_ground_truth_by_type_path_invalid(self, client, mock_service):
        """Ground truth path variant with invalid type returns 400."""
        mock_service.get_parse10m_ground_truth.side_effect = None

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/ground-truth/InvalidType"
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "CytoSig" in data["detail"] or "SecAct" in data["detail"]

    async def test_ground_truth_invalid_signature_type_query(self, client, mock_service):
        """Ground truth query endpoint rejects invalid signature_type with 422."""
        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/ground-truth",
            params={"signature_type": "NotValid"},
        )

        assert response.status_code == 422

    # -- parse10m/heatmap --------------------------------------------------

    async def test_heatmap_returns_200(self, client, mock_service, sample_heatmap_data):
        """Heatmap endpoint returns 200 with matrix data."""
        mock_service.get_parse10m_heatmap.return_value = sample_heatmap_data

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/heatmap")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    async def test_heatmap_with_cell_type(self, client, mock_service, sample_heatmap_data):
        """Heatmap endpoint passes cell_type filter to service."""
        mock_service.get_parse10m_heatmap.return_value = sample_heatmap_data

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/heatmap",
            params={"cell_type": "Monocytes"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_heatmap.assert_awaited_once_with(
            cell_type="Monocytes", signature_type="CytoSig"
        )

    async def test_heatmap_with_secact(self, client, mock_service, sample_heatmap_data):
        """Heatmap endpoint with SecAct signature type."""
        mock_service.get_parse10m_heatmap.return_value = sample_heatmap_data

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/heatmap",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_heatmap.assert_awaited_once_with(
            cell_type=None, signature_type="SecAct"
        )

    async def test_heatmap_invalid_signature_type(self, client, mock_service):
        """Heatmap rejects invalid signature_type with 422."""
        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/heatmap",
            params={"signature_type": "Bad"},
        )

        assert response.status_code == 422

    # -- parse10m/donor-variability ----------------------------------------

    async def test_donor_variability_returns_200(
        self, client, mock_service, sample_donor_variability
    ):
        """Donor variability endpoint returns 200 with per-donor records."""
        mock_service.get_parse10m_donor_variability.return_value = sample_donor_variability

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/donor-variability")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_donor_variability_with_cytokine(
        self, client, mock_service, sample_donor_variability
    ):
        """Donor variability passes cytokine filter to service."""
        mock_service.get_parse10m_donor_variability.return_value = sample_donor_variability

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/donor-variability",
            params={"cytokine": "IL6"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_donor_variability.assert_awaited_once_with(
            cytokine="IL6", cell_type=None
        )

    async def test_donor_variability_with_cell_type(
        self, client, mock_service, sample_donor_variability
    ):
        """Donor variability passes cell_type filter to service."""
        mock_service.get_parse10m_donor_variability.return_value = sample_donor_variability

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/donor-variability",
            params={"cell_type": "CD4_T"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_donor_variability.assert_awaited_once_with(
            cytokine=None, cell_type="CD4_T"
        )

    async def test_donor_variability_with_both_filters(
        self, client, mock_service, sample_donor_variability
    ):
        """Donor variability passes both cytokine and cell_type to service."""
        mock_service.get_parse10m_donor_variability.return_value = sample_donor_variability

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/donor-variability",
            params={"cytokine": "IFNG", "cell_type": "CD8_T"},
        )

        assert response.status_code == 200
        mock_service.get_parse10m_donor_variability.assert_awaited_once_with(
            cytokine="IFNG", cell_type="CD8_T"
        )

    # -- parse10m/cytokine-families ----------------------------------------

    async def test_cytokine_families_returns_200(
        self, client, mock_service, sample_cytokine_families
    ):
        """Cytokine families endpoint returns 200 with family groupings."""
        mock_service.get_parse10m_cytokine_families.return_value = sample_cytokine_families

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/cytokine-families")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 4

    async def test_cytokine_families_structure(
        self, client, mock_service, sample_cytokine_families
    ):
        """Cytokine families records contain cytokine and family fields."""
        mock_service.get_parse10m_cytokine_families.return_value = sample_cytokine_families

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/cytokine-families")
        data = response.json()

        for record in data:
            assert "cytokine" in record
            assert "family" in record

    async def test_cytokine_families_calls_service(
        self, client, mock_service, sample_cytokine_families
    ):
        """Cytokine families endpoint delegates to service."""
        mock_service.get_parse10m_cytokine_families.return_value = sample_cytokine_families

        await client.get(f"{API_PREFIX}/perturbation/parse10m/cytokine-families")

        mock_service.get_parse10m_cytokine_families.assert_awaited_once()


# ===========================================================================
# TestTahoeEndpoints — all Tahoe sub-endpoints
# ===========================================================================


class TestTahoeEndpoints:
    """Tests for /perturbation/tahoe/* endpoints."""

    # -- tahoe/summary -----------------------------------------------------

    async def test_tahoe_summary_returns_200(self, client, mock_service, sample_tahoe_summary):
        """Tahoe summary returns 200 with dataset metadata."""
        mock_service.get_tahoe_summary.return_value = sample_tahoe_summary

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["dataset"] == "Tahoe"
        assert "n_drugs" in data
        assert "n_cell_lines" in data
        assert "drugs" in data
        assert "cell_lines" in data

    async def test_tahoe_summary_calls_service(self, client, mock_service, sample_tahoe_summary):
        """tahoe/summary delegates to service.get_tahoe_summary()."""
        mock_service.get_tahoe_summary.return_value = sample_tahoe_summary

        await client.get(f"{API_PREFIX}/perturbation/tahoe/summary")

        mock_service.get_tahoe_summary.assert_awaited_once()

    # -- tahoe/drugs -------------------------------------------------------

    async def test_tahoe_drugs_returns_200(self, client, mock_service, sample_drugs):
        """Drugs endpoint returns 200 with sorted list."""
        mock_service.get_tahoe_drugs.return_value = sample_drugs

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/drugs")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "Dexamethasone" in data
        assert "Imatinib" in data

    async def test_tahoe_drugs_is_list_of_strings(self, client, mock_service, sample_drugs):
        """Drugs endpoint returns list[str]."""
        mock_service.get_tahoe_drugs.return_value = sample_drugs

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/drugs")
        data = response.json()

        for item in data:
            assert isinstance(item, str)

    # -- tahoe/cell-lines --------------------------------------------------

    async def test_tahoe_cell_lines_returns_200(self, client, mock_service, sample_cell_lines):
        """Cell lines endpoint returns 200 with sorted list."""
        mock_service.get_tahoe_cell_lines.return_value = sample_cell_lines

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/cell-lines")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "A549" in data
        assert "MCF7" in data

    async def test_tahoe_cell_lines_is_list_of_strings(self, client, mock_service, sample_cell_lines):
        """Cell lines endpoint returns list[str]."""
        mock_service.get_tahoe_cell_lines.return_value = sample_cell_lines

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/cell-lines")
        data = response.json()

        for item in data:
            assert isinstance(item, str)

    # -- tahoe/activity ----------------------------------------------------

    async def test_tahoe_activity_returns_200(self, client, mock_service, sample_tahoe_activity):
        """Tahoe activity endpoint returns 200 with activity records."""
        mock_service.get_tahoe_activity.return_value = sample_tahoe_activity

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/activity")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_tahoe_activity_with_drug_filter(
        self, client, mock_service, sample_tahoe_activity
    ):
        """Tahoe activity passes drug filter to service."""
        mock_service.get_tahoe_activity.return_value = sample_tahoe_activity

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/activity",
            params={"drug": "Dexamethasone"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_activity.assert_awaited_once_with(
            drug="Dexamethasone", cell_line=None, signature_type="CytoSig"
        )

    async def test_tahoe_activity_with_cell_line_filter(
        self, client, mock_service, sample_tahoe_activity
    ):
        """Tahoe activity passes cell_line filter to service."""
        mock_service.get_tahoe_activity.return_value = sample_tahoe_activity

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/activity",
            params={"cell_line": "A549"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_activity.assert_awaited_once_with(
            drug=None, cell_line="A549", signature_type="CytoSig"
        )

    async def test_tahoe_activity_with_all_filters(
        self, client, mock_service, sample_tahoe_activity
    ):
        """Tahoe activity forwards all three query params."""
        mock_service.get_tahoe_activity.return_value = sample_tahoe_activity

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/activity",
            params={"drug": "Imatinib", "cell_line": "MCF7", "signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_activity.assert_awaited_once_with(
            drug="Imatinib", cell_line="MCF7", signature_type="SecAct"
        )

    async def test_tahoe_activity_invalid_signature_type(self, client, mock_service):
        """Tahoe activity rejects invalid signature_type with 422."""
        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/activity",
            params={"signature_type": "Invalid"},
        )

        assert response.status_code == 422

    async def test_tahoe_activity_default_signature_type(
        self, client, mock_service, sample_tahoe_activity
    ):
        """Tahoe activity defaults to CytoSig when signature_type is omitted."""
        mock_service.get_tahoe_activity.return_value = sample_tahoe_activity

        await client.get(f"{API_PREFIX}/perturbation/tahoe/activity")

        mock_service.get_tahoe_activity.assert_awaited_once_with(
            drug=None, cell_line=None, signature_type="CytoSig"
        )

    # -- tahoe/drug-effect -------------------------------------------------

    async def test_drug_effect_returns_200(self, client, mock_service, sample_drug_effect):
        """Drug effect endpoint returns 200 with effect records."""
        mock_service.get_tahoe_drug_effect.return_value = sample_drug_effect

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/drug-effect")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_drug_effect_with_cell_line_filter(
        self, client, mock_service, sample_drug_effect
    ):
        """Drug effect passes cell_line to service."""
        mock_service.get_tahoe_drug_effect.return_value = sample_drug_effect

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/drug-effect",
            params={"cell_line": "A549", "signature_type": "CytoSig"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_drug_effect.assert_awaited_once_with(
            cell_line="A549", signature_type="CytoSig"
        )

    async def test_drug_effect_by_cell_line_path(
        self, client, mock_service, sample_drug_effect
    ):
        """Drug effect with path parameter forwards cell_line to service."""
        mock_service.get_tahoe_drug_effect.return_value = sample_drug_effect

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/drug-effect/MCF7",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_drug_effect.assert_awaited_once_with(
            cell_line="MCF7", signature_type="SecAct"
        )

    async def test_drug_effect_invalid_signature_type(self, client, mock_service):
        """Drug effect rejects invalid signature_type with 422."""
        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/drug-effect",
            params={"signature_type": "BadType"},
        )

        assert response.status_code == 422

    # -- tahoe/sensitivity-matrix ------------------------------------------

    async def test_sensitivity_matrix_returns_200(
        self, client, mock_service, sample_sensitivity_matrix
    ):
        """Sensitivity matrix endpoint returns 200 with matrix data."""
        mock_service.get_tahoe_sensitivity_matrix.return_value = sample_sensitivity_matrix

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/sensitivity-matrix")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    async def test_sensitivity_matrix_with_signature_type(
        self, client, mock_service, sample_sensitivity_matrix
    ):
        """Sensitivity matrix passes signature_type to service."""
        mock_service.get_tahoe_sensitivity_matrix.return_value = sample_sensitivity_matrix

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/sensitivity-matrix",
            params={"signature_type": "SecAct"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_sensitivity_matrix.assert_awaited_once_with(
            signature_type="SecAct"
        )

    async def test_sensitivity_matrix_default_cytosig(
        self, client, mock_service, sample_sensitivity_matrix
    ):
        """Sensitivity matrix defaults to CytoSig."""
        mock_service.get_tahoe_sensitivity_matrix.return_value = sample_sensitivity_matrix

        await client.get(f"{API_PREFIX}/perturbation/tahoe/sensitivity-matrix")

        mock_service.get_tahoe_sensitivity_matrix.assert_awaited_once_with(
            signature_type="CytoSig"
        )

    async def test_sensitivity_matrix_invalid_signature_type(self, client, mock_service):
        """Sensitivity matrix rejects invalid signature_type with 422."""
        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/sensitivity-matrix",
            params={"signature_type": "NotValid"},
        )

        assert response.status_code == 422

    # -- tahoe/dose-response -----------------------------------------------

    async def test_dose_response_returns_200(self, client, mock_service, sample_dose_response):
        """Dose response endpoint returns 200 with dose-response records."""
        mock_service.get_tahoe_dose_response.return_value = sample_dose_response

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/dose-response")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    async def test_dose_response_with_drug_filter(
        self, client, mock_service, sample_dose_response
    ):
        """Dose response passes drug filter to service."""
        mock_service.get_tahoe_dose_response.return_value = sample_dose_response

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/dose-response",
            params={"drug": "Dexamethasone"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_dose_response.assert_awaited_once_with(
            drug="Dexamethasone", cell_line=None
        )

    async def test_dose_response_with_cell_line_filter(
        self, client, mock_service, sample_dose_response
    ):
        """Dose response passes cell_line filter to service."""
        mock_service.get_tahoe_dose_response.return_value = sample_dose_response

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/dose-response",
            params={"cell_line": "A549"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_dose_response.assert_awaited_once_with(
            drug=None, cell_line="A549"
        )

    async def test_dose_response_with_both_filters(
        self, client, mock_service, sample_dose_response
    ):
        """Dose response passes both drug and cell_line to service."""
        mock_service.get_tahoe_dose_response.return_value = sample_dose_response

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/dose-response",
            params={"drug": "Imatinib", "cell_line": "MCF7"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_dose_response.assert_awaited_once_with(
            drug="Imatinib", cell_line="MCF7"
        )

    async def test_dose_response_by_drug_path(self, client, mock_service, sample_dose_response):
        """Dose response path variant forwards drug to service."""
        mock_service.get_tahoe_dose_response.return_value = sample_dose_response

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/dose-response/Dexamethasone"
        )

        assert response.status_code == 200
        mock_service.get_tahoe_dose_response.assert_awaited_once_with(
            drug="Dexamethasone", cell_line=None
        )

    async def test_dose_response_by_drug_path_with_cell_line(
        self, client, mock_service, sample_dose_response
    ):
        """Dose response path variant with cell_line query param."""
        mock_service.get_tahoe_dose_response.return_value = sample_dose_response

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/dose-response/Paclitaxel",
            params={"cell_line": "HeLa"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_dose_response.assert_awaited_once_with(
            drug="Paclitaxel", cell_line="HeLa"
        )

    # -- tahoe/pathway-activation ------------------------------------------

    async def test_pathway_activation_returns_200(
        self, client, mock_service, sample_pathway_activation
    ):
        """Pathway activation endpoint returns 200 with pathway records."""
        mock_service.get_tahoe_pathway_activation.return_value = sample_pathway_activation

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/pathway-activation")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    async def test_pathway_activation_with_drug_filter(
        self, client, mock_service, sample_pathway_activation
    ):
        """Pathway activation passes drug filter to service."""
        filtered = [r for r in sample_pathway_activation if r["drug"] == "Dexamethasone"]
        mock_service.get_tahoe_pathway_activation.return_value = filtered

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/pathway-activation",
            params={"drug": "Dexamethasone"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_pathway_activation.assert_awaited_once_with(
            drug="Dexamethasone"
        )
        data = response.json()
        assert len(data) == 2

    async def test_pathway_activation_by_drug_path(
        self, client, mock_service, sample_pathway_activation
    ):
        """Pathway activation path variant forwards drug to service."""
        mock_service.get_tahoe_pathway_activation.return_value = sample_pathway_activation

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/pathway-activation/Imatinib"
        )

        assert response.status_code == 200
        mock_service.get_tahoe_pathway_activation.assert_awaited_once_with(drug="Imatinib")

    async def test_pathway_activation_no_filter(
        self, client, mock_service, sample_pathway_activation
    ):
        """Pathway activation without filter returns all records."""
        mock_service.get_tahoe_pathway_activation.return_value = sample_pathway_activation

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/pathway-activation")

        assert response.status_code == 200
        mock_service.get_tahoe_pathway_activation.assert_awaited_once_with(drug=None)
        data = response.json()
        assert len(data) == 3

    async def test_pathway_activation_record_structure(
        self, client, mock_service, sample_pathway_activation
    ):
        """Pathway activation records contain expected fields."""
        mock_service.get_tahoe_pathway_activation.return_value = sample_pathway_activation

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/pathway-activation")
        data = response.json()

        for record in data:
            assert "drug" in record
            assert "pathway" in record
            assert "activity" in record


# ===========================================================================
# TestEdgeCases — empty responses, URL encoding, concurrent filters
# ===========================================================================


class TestEdgeCases:
    """Edge-case and boundary condition tests across all perturbation endpoints."""

    async def test_parse10m_activity_empty_response(self, client, mock_service):
        """Activity returns 200 with empty list when no data matches."""
        mock_service.get_parse10m_activity.return_value = []

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/activity",
            params={"cytokine": "NONEXISTENT"},
        )

        assert response.status_code == 200
        assert response.json() == []

    async def test_tahoe_activity_empty_response(self, client, mock_service):
        """Tahoe activity returns 200 with empty list when no data matches."""
        mock_service.get_tahoe_activity.return_value = []

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/activity",
            params={"drug": "NONEXISTENT"},
        )

        assert response.status_code == 200
        assert response.json() == []

    async def test_parse10m_summary_empty_dataset(self, client, mock_service):
        """parse10m/summary returns 200 even with minimal data."""
        mock_service.get_parse10m_summary.return_value = {
            "dataset": "parse_10M",
            "description": "Cytokine stimulation perturbation screen",
            "n_cytokines": 0,
            "n_cell_types": 0,
            "cytokines": [],
            "cell_types": [],
            "signature_types": ["CytoSig", "SecAct"],
        }

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["n_cytokines"] == 0
        assert data["n_cell_types"] == 0

    async def test_tahoe_dose_response_empty(self, client, mock_service):
        """Dose response returns 200 with empty list when no data."""
        mock_service.get_tahoe_dose_response.return_value = []

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/dose-response",
            params={"drug": "NONEXISTENT", "cell_line": "NONEXISTENT"},
        )

        assert response.status_code == 200
        assert response.json() == []

    async def test_url_encoded_drug_name(self, client, mock_service, sample_tahoe_activity):
        """Drug names with special characters are properly URL-decoded."""
        mock_service.get_tahoe_activity.return_value = sample_tahoe_activity

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/activity",
            params={"drug": "5-Fluorouracil"},
        )

        assert response.status_code == 200
        mock_service.get_tahoe_activity.assert_awaited_once_with(
            drug="5-Fluorouracil", cell_line=None, signature_type="CytoSig"
        )

    async def test_url_encoded_cell_type_in_path(
        self, client, mock_service, sample_treatment_effect
    ):
        """Cell type path parameter with underscores is handled correctly."""
        mock_service.get_parse10m_treatment_effect.return_value = sample_treatment_effect

        response = await client.get(
            f"{API_PREFIX}/perturbation/parse10m/treatment-effect/NK_cells"
        )

        assert response.status_code == 200
        mock_service.get_parse10m_treatment_effect.assert_awaited_once_with(
            cell_type="NK_cells", signature_type="CytoSig"
        )

    async def test_cytokines_empty_list(self, client, mock_service):
        """Cytokines returns 200 with empty list when no data available."""
        mock_service.get_parse10m_cytokines.return_value = []

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/cytokines")

        assert response.status_code == 200
        assert response.json() == []

    async def test_tahoe_drugs_empty_list(self, client, mock_service):
        """Drugs returns 200 with empty list when no data available."""
        mock_service.get_tahoe_drugs.return_value = []

        response = await client.get(f"{API_PREFIX}/perturbation/tahoe/drugs")

        assert response.status_code == 200
        assert response.json() == []

    async def test_donor_variability_no_filters(self, client, mock_service, sample_donor_variability):
        """Donor variability without filters returns all data."""
        mock_service.get_parse10m_donor_variability.return_value = sample_donor_variability

        response = await client.get(f"{API_PREFIX}/perturbation/parse10m/donor-variability")

        assert response.status_code == 200
        mock_service.get_parse10m_donor_variability.assert_awaited_once_with(
            cytokine=None, cell_type=None
        )

    async def test_dose_response_by_drug_path_url_encoding(
        self, client, mock_service, sample_dose_response
    ):
        """Dose response path variant handles URL-encoded drug names."""
        mock_service.get_tahoe_dose_response.return_value = sample_dose_response

        response = await client.get(
            f"{API_PREFIX}/perturbation/tahoe/dose-response/5-Fluorouracil"
        )

        assert response.status_code == 200
        mock_service.get_tahoe_dose_response.assert_awaited_once_with(
            drug="5-Fluorouracil", cell_line=None
        )
