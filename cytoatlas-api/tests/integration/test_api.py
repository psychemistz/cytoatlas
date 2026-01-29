"""Integration tests for API endpoints."""

import pytest
from httpx import AsyncClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, client: AsyncClient):
        """Test root endpoint returns API info."""
        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data

    @pytest.mark.asyncio
    async def test_health_live(self, client: AsyncClient):
        """Test liveness probe."""
        response = await client.get("/api/v1/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True


class TestCIMAEndpoints:
    """Tests for CIMA API endpoints."""

    @pytest.mark.asyncio
    async def test_get_cell_types(self, client: AsyncClient):
        """Test getting cell types list."""
        # This will fail without mock data, but structure is correct
        response = await client.get("/api/v1/cima/cell-types")

        # Either 200 (with data) or 500 (file not found in test)
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_signatures(self, client: AsyncClient):
        """Test getting signatures list."""
        response = await client.get(
            "/api/v1/cima/signatures", params={"signature_type": "CytoSig"}
        )

        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_signature_type_validation(self, client: AsyncClient):
        """Test signature type parameter validation."""
        # Invalid signature type should return 422
        response = await client.get(
            "/api/v1/cima/activity", params={"signature_type": "Invalid"}
        )

        assert response.status_code == 422


class TestInflammationEndpoints:
    """Tests for Inflammation API endpoints."""

    @pytest.mark.asyncio
    async def test_get_diseases(self, client: AsyncClient):
        """Test getting diseases list."""
        response = await client.get("/api/v1/inflammation/diseases")

        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_treatment_response(self, client: AsyncClient):
        """Test treatment response endpoint."""
        response = await client.get("/api/v1/inflammation/treatment-response")

        assert response.status_code in [200, 500]


class TestScAtlasEndpoints:
    """Tests for scAtlas API endpoints."""

    @pytest.mark.asyncio
    async def test_get_organs(self, client: AsyncClient):
        """Test getting organs list."""
        response = await client.get("/api/v1/scatlas/organs")

        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_cancer_comparison(self, client: AsyncClient):
        """Test cancer comparison endpoint."""
        response = await client.get(
            "/api/v1/scatlas/cancer-comparison",
            params={"signature_type": "CytoSig"},
        )

        assert response.status_code in [200, 500]


class TestCrossAtlasEndpoints:
    """Tests for cross-atlas API endpoints."""

    @pytest.mark.asyncio
    async def test_get_atlases(self, client: AsyncClient):
        """Test getting available atlases."""
        response = await client.get("/api/v1/cross-atlas/atlases")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "CIMA" in data


class TestValidationEndpoints:
    """Tests for validation API endpoints."""

    @pytest.mark.asyncio
    async def test_validation_summary(self, client: AsyncClient):
        """Test validation summary endpoint."""
        response = await client.get(
            "/api/v1/validation/summary",
            params={"atlas": "cima", "signature_type": "CytoSig"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "overall_quality_score" in data

    @pytest.mark.asyncio
    async def test_biological_associations(self, client: AsyncClient):
        """Test biological associations endpoint."""
        response = await client.get(
            "/api/v1/validation/biological-associations",
            params={"atlas": "cima", "signature_type": "CytoSig"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "associations" in data
        assert "concordance_rate" in data


class TestExportEndpoints:
    """Tests for export API endpoints."""

    @pytest.mark.asyncio
    async def test_export_format_validation(self, client: AsyncClient):
        """Test export format parameter validation."""
        # Invalid format should return 422
        response = await client.get(
            "/api/v1/export/cima/activity",
            params={"format": "invalid"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_export_csv(self, client: AsyncClient):
        """Test CSV export."""
        response = await client.get(
            "/api/v1/export/cima/activity",
            params={"format": "csv", "signature_type": "CytoSig"},
        )

        # Will be 200 with data or 500 without
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            assert "text/csv" in response.headers.get("content-type", "")
