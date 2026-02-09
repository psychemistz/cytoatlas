"""Unit tests for service layer."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.base import BaseService


class TestBaseService:
    """Tests for BaseService."""

    def test_filter_by_signature_type(self):
        """Test filtering data by signature type."""
        service = BaseService()
        data = [
            {"signature_type": "CytoSig", "value": 1},
            {"signature_type": "SecAct", "value": 2},
            {"signature_type": "CytoSig", "value": 3},
        ]

        result = service.filter_by_signature_type(data, "CytoSig")

        assert len(result) == 2
        assert all(r["signature_type"] == "CytoSig" for r in result)

    def test_filter_by_signature_type_secact(self):
        """Test filtering for SecAct signature type."""
        service = BaseService()
        data = [
            {"signature_type": "CytoSig", "value": 1},
            {"signature_type": "SecAct", "value": 2},
            {"signature_type": "CytoSig", "value": 3},
        ]

        result = service.filter_by_signature_type(data, "SecAct")

        assert len(result) == 1
        assert result[0]["value"] == 2

    def test_filter_by_signature_type_empty(self):
        """Test filtering empty data returns empty list."""
        service = BaseService()
        result = service.filter_by_signature_type([], "CytoSig")
        assert result == []

    def test_filter_by_cell_type(self):
        """Test filtering data by cell type."""
        service = BaseService()
        data = [
            {"cell_type": "CD4_T", "value": 1},
            {"cell_type": "CD8_T", "value": 2},
            {"cell_type": "CD4_T", "value": 3},
        ]

        result = service.filter_by_cell_type(data, "CD4_T")

        assert len(result) == 2
        assert all(r["cell_type"] == "CD4_T" for r in result)

    def test_filter_by_cell_type_no_match(self):
        """Test filtering returns empty when no match."""
        service = BaseService()
        data = [{"cell_type": "CD4_T", "value": 1}]
        result = service.filter_by_cell_type(data, "NonExistent")
        assert result == []

    def test_paginate(self):
        """Test pagination utility."""
        service = BaseService()
        data = list(range(100))

        # First page
        paginated, total, has_more = service.paginate(data, offset=0, limit=10)
        assert len(paginated) == 10
        assert total == 100
        assert has_more is True
        assert paginated == list(range(10))

        # Middle page
        paginated, total, has_more = service.paginate(data, offset=50, limit=10)
        assert len(paginated) == 10
        assert paginated == list(range(50, 60))
        assert has_more is True

        # Last page
        paginated, total, has_more = service.paginate(data, offset=90, limit=10)
        assert len(paginated) == 10
        assert has_more is False

        # Beyond data
        paginated, total, has_more = service.paginate(data, offset=100, limit=10)
        assert len(paginated) == 0
        assert has_more is False

    def test_paginate_small_dataset(self):
        """Test pagination with dataset smaller than limit."""
        service = BaseService()
        data = [1, 2, 3]

        paginated, total, has_more = service.paginate(data, offset=0, limit=10)
        assert len(paginated) == 3
        assert total == 3
        assert has_more is False

    def test_round_floats(self):
        """Test float rounding utility."""
        service = BaseService()
        data = [
            {"value": 1.123456789, "name": "test"},
            {"value": 2.987654321, "count": 10},
        ]

        result = service.round_floats(data, decimals=4)

        assert result[0]["value"] == 1.1235
        assert result[0]["name"] == "test"
        assert result[1]["value"] == 2.9877
        assert result[1]["count"] == 10

    def test_round_floats_preserves_integers(self):
        """Test that rounding preserves integer values."""
        service = BaseService()
        data = [{"int_val": 42, "float_val": 3.14159}]
        result = service.round_floats(data, decimals=2)
        assert result[0]["int_val"] == 42
        assert result[0]["float_val"] == 3.14

    async def test_load_json_path_traversal_prevention(self):
        """Test that path traversal attempts are blocked."""
        service = BaseService()

        with pytest.raises(ValueError, match="Path traversal detected"):
            await service.load_json("../../etc/passwd")

    async def test_load_json_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        service = BaseService()

        with pytest.raises(FileNotFoundError):
            await service.load_json("nonexistent_file_12345.json")


class TestCIMAService:
    """Tests for CIMAService."""

    async def test_get_correlations_filters_by_type(self):
        """Test that correlations are filtered by signature type."""
        from app.services.cima_service import CIMAService

        # Mock data uses "protein" for signature name and "signature" for type,
        # matching the actual cima_celltype_correlations.json format
        mock_data = {
            "age": [
                {"signature": "CytoSig", "protein": "IFNG", "rho": 0.5, "pvalue": 0.01, "cell_type": "CD4_T"},
                {"signature": "SecAct", "protein": "SIG1", "rho": 0.3, "pvalue": 0.05, "cell_type": "CD4_T"},
            ]
        }

        service = CIMAService()

        with patch.object(service, "load_json", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_data

            result = await service.get_correlations("age", "CytoSig")

            assert len(result) == 1
            assert result[0].signature_type == "CytoSig"


class TestInflammationService:
    """Tests for InflammationService."""

    async def test_get_cell_type_activity(self):
        """Test getting cell type activity."""
        from app.services.inflammation_service import InflammationService

        mock_data = [
            {"cell_type": "CD4_T", "signature": "IFNG", "signature_type": "CytoSig", "mean_activity": 1.0, "n_samples": 10, "n_cells": 1000},
            {"cell_type": "CD8_T", "signature": "IFNG", "signature_type": "CytoSig", "mean_activity": 1.5, "n_samples": 10, "n_cells": 500},
        ]

        service = InflammationService()

        with patch.object(service, "load_json", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_data

            result = await service.get_cell_type_activity("CytoSig")

            assert len(result) == 2
            assert all(r.signature_type == "CytoSig" for r in result)


class TestScAtlasService:
    """Tests for ScAtlasService."""

    async def test_get_organ_signatures(self):
        """Test getting organ signatures."""
        from app.services.scatlas_service import ScAtlasService

        mock_data = [
            {"organ": "Lung", "signature": "IFNG", "signature_type": "CytoSig", "mean_activity": 1.0, "n_cells": 1000},
            {"organ": "Liver", "signature": "IL6", "signature_type": "CytoSig", "mean_activity": 0.5, "n_cells": 2000},
        ]

        service = ScAtlasService()

        with patch.object(service, "load_json", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_data

            result = await service.get_organ_signatures("CytoSig")

            assert len(result) == 2

            # Filter by organ
            result = await service.get_organ_signatures("CytoSig", organ="Lung")
            assert len(result) == 1
            assert result[0].organ == "Lung"
