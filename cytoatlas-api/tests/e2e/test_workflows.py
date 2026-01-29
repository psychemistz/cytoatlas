"""End-to-end workflow tests."""

import pytest
from httpx import AsyncClient


class TestExplorationWorkflow:
    """Test typical user exploration workflow."""

    @pytest.mark.asyncio
    async def test_cima_exploration_flow(self, client: AsyncClient):
        """Test CIMA data exploration workflow."""
        # 1. Get available cell types
        response = await client.get("/api/v1/cima/cell-types")
        # May not have data in test environment
        if response.status_code != 200:
            pytest.skip("Test data not available")

        cell_types = response.json()
        assert isinstance(cell_types, list)

        if not cell_types:
            pytest.skip("No cell types available")

        # 2. Get activity for a cell type
        cell_type = cell_types[0]
        response = await client.get(
            f"/api/v1/cima/activity/{cell_type}",
            params={"signature_type": "CytoSig"},
        )
        assert response.status_code == 200
        activity = response.json()

        # 3. Get correlations
        response = await client.get(
            "/api/v1/cima/correlations/age",
            params={"signature_type": "CytoSig", "cell_type": cell_type},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_inflammation_analysis_flow(self, client: AsyncClient):
        """Test Inflammation analysis workflow."""
        # 1. Get available diseases
        response = await client.get("/api/v1/inflammation/diseases")
        if response.status_code != 200:
            pytest.skip("Test data not available")

        diseases = response.json()
        if not diseases:
            pytest.skip("No diseases available")

        # 2. Get disease comparison
        disease = diseases[0]
        response = await client.get(
            f"/api/v1/inflammation/disease-comparison/{disease}",
            params={"signature_type": "CytoSig"},
        )
        assert response.status_code == 200

        # 3. Get treatment response
        response = await client.get(
            "/api/v1/inflammation/treatment-response",
            params={"disease": disease},
        )
        assert response.status_code == 200


class TestValidationWorkflow:
    """Test validation/credibility panel workflow."""

    @pytest.mark.asyncio
    async def test_validation_assessment_flow(self, client: AsyncClient):
        """Test validation assessment workflow."""
        # 1. Get validation summary
        response = await client.get(
            "/api/v1/validation/summary",
            params={"atlas": "cima", "signature_type": "CytoSig"},
        )
        assert response.status_code == 200
        summary = response.json()
        assert "overall_quality_score" in summary

        # 2. Get biological associations
        response = await client.get(
            "/api/v1/validation/biological-associations/cima",
            params={"signature_type": "CytoSig"},
        )
        assert response.status_code == 200
        bio = response.json()
        assert "concordance_rate" in bio

        # 3. Compare atlases
        response = await client.get(
            "/api/v1/validation/compare-atlases",
            params={"signature_type": "CytoSig"},
        )
        assert response.status_code == 200
        comparison = response.json()
        assert "comparison" in comparison


class TestExportWorkflow:
    """Test data export workflow."""

    @pytest.mark.asyncio
    async def test_export_and_verify_flow(self, client: AsyncClient):
        """Test exporting data in different formats."""
        # 1. Export as JSON
        response = await client.get(
            "/api/v1/export/cima/activity",
            params={"format": "json", "signature_type": "CytoSig"},
        )

        if response.status_code != 200:
            pytest.skip("Test data not available")

        assert "application/json" in response.headers.get("content-type", "")

        # 2. Export as CSV
        response = await client.get(
            "/api/v1/export/cima/activity",
            params={"format": "csv", "signature_type": "CytoSig"},
        )
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")
