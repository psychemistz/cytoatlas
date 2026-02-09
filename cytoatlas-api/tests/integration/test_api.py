"""Integration tests for API endpoints.

All tests use httpx.AsyncClient with ASGITransport.
Tests assert specific status codes -- NO assert status in [200, 500] antipattern.
"""

import pytest
from httpx import AsyncClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    async def test_liveness(self, test_client: AsyncClient):
        """Liveness probe returns 200 with alive=True."""
        response = await test_client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True

    async def test_readiness(self, test_client: AsyncClient):
        """Readiness probe returns 200."""
        response = await test_client.get("/api/v1/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data

    async def test_health_check(self, test_client: AsyncClient):
        """Health check returns 200 with proper schema."""
        response = await test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["status"] in ["healthy", "degraded"]

    async def test_metrics(self, test_client: AsyncClient):
        """Metrics endpoint returns 200."""
        response = await test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200


class TestRootEndpoint:
    """Tests for the root / endpoint."""

    async def test_root_returns_200(self, test_client: AsyncClient):
        """Root endpoint returns 200."""
        response = await test_client.get("/")
        assert response.status_code == 200


class TestCrossAtlasEndpoints:
    """Tests for cross-atlas API endpoints."""

    async def test_get_atlases(self, test_client: AsyncClient):
        """Getting available atlases returns 200."""
        response = await test_client.get("/api/v1/cross-atlas/atlases")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "CIMA" in data


class TestSearchEndpoints:
    """Tests for search API endpoints."""

    async def test_search_types(self, test_client: AsyncClient):
        """Search types endpoint returns list of types."""
        response = await test_client.get("/api/v1/search/types")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Each type should have type, name, description
        for item in data:
            assert "type" in item
            assert "name" in item

    async def test_search_requires_query(self, test_client: AsyncClient):
        """Search without query parameter returns 422."""
        response = await test_client.get("/api/v1/search")
        assert response.status_code == 422

    async def test_search_empty_query_too_short(self, test_client: AsyncClient):
        """Search with too-short query returns 422."""
        response = await test_client.get("/api/v1/search", params={"q": ""})
        assert response.status_code == 422

    async def test_autocomplete_requires_query(self, test_client: AsyncClient):
        """Autocomplete without query returns 422."""
        response = await test_client.get("/api/v1/search/autocomplete")
        assert response.status_code == 422


class TestPipelineEndpoints:
    """Tests for pipeline API endpoints."""

    async def test_pipeline_status(self, test_client: AsyncClient):
        """Pipeline status endpoint returns 200."""
        response = await test_client.get("/api/v1/pipeline/status")
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        assert "progress_percent" in data

    async def test_pipeline_stages(self, test_client: AsyncClient):
        """Pipeline stages endpoint returns 200."""
        response = await test_client.get("/api/v1/pipeline/stages")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_pipeline_stage_not_found(self, test_client: AsyncClient):
        """Non-existent pipeline stage returns 404."""
        response = await test_client.get("/api/v1/pipeline/stages/nonexistent_stage")
        assert response.status_code == 404

    async def test_pipeline_run_returns_info(self, test_client: AsyncClient):
        """Pipeline run endpoint returns info about running pipeline."""
        response = await test_client.post("/api/v1/pipeline/run")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestValidationEndpoints:
    """Tests for validation API endpoints."""

    async def test_validation_summary(self, test_client: AsyncClient):
        """Validation summary returns 200 with proper schema."""
        response = await test_client.get(
            "/api/v1/validation/summary",
            params={"atlas": "cima", "signature_type": "CytoSig"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "quality_score" in data or "overall_quality_score" in data

    async def test_validation_biological_associations(self, test_client: AsyncClient):
        """Biological associations endpoint returns 200."""
        response = await test_client.get(
            "/api/v1/validation/biological-associations/cima",
            params={"signature_type": "CytoSig"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data or "associations" in data


class TestExportEndpoints:
    """Tests for export API endpoints."""

    async def test_export_format_validation(self, test_client: AsyncClient):
        """Invalid export format returns 422."""
        response = await test_client.get(
            "/api/v1/export/cima/activity",
            params={"format": "invalid_format"},
        )
        assert response.status_code == 422

    async def test_export_signature_type_validation(self, test_client: AsyncClient):
        """Invalid signature type returns 422."""
        response = await test_client.get(
            "/api/v1/export/cima/activity",
            params={"format": "csv", "signature_type": "Invalid"},
        )
        assert response.status_code == 422


class TestChatEndpoints:
    """Tests for chat API endpoints."""

    async def test_chat_suggestions(self, test_client: AsyncClient):
        """Chat suggestions endpoint returns suggestions with mocked service."""
        from unittest.mock import MagicMock
        from app.main import app
        from app.services.chat import get_chat_service
        from app.schemas.chat import ChatSuggestionsResponse, ChatSuggestion, SuggestionCategory

        mock_service = MagicMock()
        mock_service.get_suggestions = MagicMock(return_value=ChatSuggestionsResponse(
            suggestions=[
                ChatSuggestion(
                    text="What cytokines are active in CD8 T cells?",
                    category=SuggestionCategory.EXPLORE,
                    description="Explore cytokine activity",
                ),
            ]
        ))

        app.dependency_overrides[get_chat_service] = lambda: mock_service
        try:
            response = await test_client.get("/api/v1/chat/suggestions")
            assert response.status_code == 200
            data = response.json()
            assert "suggestions" in data
            assert len(data["suggestions"]) > 0
        finally:
            app.dependency_overrides.pop(get_chat_service, None)

    async def test_chat_status(self, test_client: AsyncClient):
        """Chat status endpoint returns service info."""
        response = await test_client.get("/api/v1/chat/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "rate_limits" in data

    async def test_chat_conversations_list(self, test_client: AsyncClient):
        """Listing conversations returns 200."""
        response = await test_client.get("/api/v1/chat/conversations")
        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        assert "total" in data

    async def test_delete_conversation_not_implemented(self, test_client: AsyncClient):
        """Deleting a conversation returns 501 (not implemented)."""
        response = await test_client.delete("/api/v1/chat/conversations/1")
        assert response.status_code == 501


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    async def test_csp_header(self, test_client: AsyncClient):
        """Content-Security-Policy header is present."""
        response = await test_client.get("/api/v1/health/live")
        assert "content-security-policy" in response.headers

    async def test_x_content_type_options(self, test_client: AsyncClient):
        """X-Content-Type-Options header is nosniff."""
        response = await test_client.get("/api/v1/health/live")
        assert response.headers.get("x-content-type-options") == "nosniff"

    async def test_x_frame_options(self, test_client: AsyncClient):
        """X-Frame-Options header is DENY."""
        response = await test_client.get("/api/v1/health/live")
        assert response.headers.get("x-frame-options") == "DENY"

    async def test_referrer_policy(self, test_client: AsyncClient):
        """Referrer-Policy header is set."""
        response = await test_client.get("/api/v1/health/live")
        assert "referrer-policy" in response.headers

    async def test_permissions_policy(self, test_client: AsyncClient):
        """Permissions-Policy header is set."""
        response = await test_client.get("/api/v1/health/live")
        assert "permissions-policy" in response.headers

    async def test_api_version_header(self, test_client: AsyncClient):
        """X-API-Version header is present on API endpoints."""
        response = await test_client.get("/api/v1/health/live")
        assert response.headers.get("x-api-version") == "1.0"

    async def test_request_id_header(self, test_client: AsyncClient):
        """X-Request-ID header is present."""
        response = await test_client.get("/api/v1/health/live")
        assert "x-request-id" in response.headers
