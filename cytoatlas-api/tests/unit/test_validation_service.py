"""Unit tests for validation service."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from app.services.validation_service import ValidationService


class TestValidationService:
    """Tests for ValidationService credibility assessment."""

    @pytest.fixture(autouse=True)
    def setup_service(self, tmp_path, validation_data):
        """Set up validation service with test data."""
        self.service = ValidationService()
        self.validation_dir = tmp_path / "validation"
        self.validation_dir.mkdir()

        # Write test validation data
        with open(self.validation_dir / "cima_validation.json", "w") as f:
            json.dump(validation_data, f)

        # Point service to test data
        self.service.data_dir = tmp_path
        self.service.validation_dir = self.validation_dir
        self.service._cache = {}  # Reset cache

    async def test_sample_level_validation(self):
        """Test Type 1: sample-level validation retrieval."""
        result = await self.service.get_sample_level_validation(
            "cima", "IFNG", "CytoSig"
        )
        assert result is not None
        assert result.signature == "IFNG"
        assert result.atlas == "cima"
        assert len(result.points) > 0
        assert result.stats.pearson_r > 0

    async def test_celltype_level_validation(self):
        """Test Type 2: cell type-level validation retrieval."""
        result = await self.service.get_celltype_level_validation(
            "cima", "IFNG", "CytoSig"
        )
        assert result is not None
        assert result.signature == "IFNG"
        assert len(result.points) > 0
        assert result.stats.r_squared > 0

    async def test_pseudobulk_vs_singlecell(self):
        """Test Type 3: pseudobulk vs single-cell validation."""
        result = await self.service.get_pseudobulk_vs_singlecell(
            "cima", "IFNG", "CytoSig"
        )
        assert result is not None
        assert result.stats_vs_mean.pearson_r > 0
        assert result.stats_vs_median.pearson_r > 0

    async def test_singlecell_direct_validation(self):
        """Test Type 4: single-cell direct validation."""
        result = await self.service.get_singlecell_direct_validation(
            "cima", "IFNG", "CytoSig"
        )
        assert result is not None
        assert result.n_expressing > 0
        assert result.n_non_expressing > 0
        assert result.mean_activity_expressing > result.mean_activity_non_expressing

    async def test_biological_associations(self):
        """Test Type 5: biological association validation."""
        result = await self.service.get_biological_associations(
            "cima", "CytoSig"
        )
        assert result is not None
        assert result.n_tested > 0
        assert result.n_validated > 0
        assert result.validation_rate > 0
        # Check individual results
        for r in result.results:
            assert r.signature
            assert r.expected_cell_type

    async def test_gene_coverage(self):
        """Test gene coverage retrieval."""
        result = await self.service.get_gene_coverage(
            "cima", "IFNG", "CytoSig"
        )
        assert result is not None
        assert result.coverage_pct == 90.0
        assert result.n_detected == 45
        assert result.n_missing == 5

    async def test_cv_stability(self):
        """Test cross-validation stability retrieval."""
        result = await self.service.get_cv_stability("cima", "CytoSig")
        assert len(result) > 0
        assert result[0].stability_score > 0
        assert result[0].n_folds == 5

    async def test_validation_summary(self):
        """Test overall validation summary with quality score."""
        result = await self.service.get_validation_summary("cima", "CytoSig")
        assert result is not None
        assert result.quality_score > 0
        assert result.quality_grade in ["A", "B", "C", "D", "F"]
        assert result.interpretation
        assert len(result.recommendations) > 0

    async def test_available_signatures(self):
        """Test listing available validated signatures."""
        result = await self.service.get_available_signatures("cima", "CytoSig")
        assert "IFNG" in result

    async def test_available_atlases(self):
        """Test listing atlases with validation data."""
        result = await self.service.get_available_atlases()
        assert "cima" in result

    async def test_missing_atlas_returns_none(self):
        """Missing atlas returns None for individual validations."""
        result = await self.service.get_sample_level_validation(
            "nonexistent", "IFNG", "CytoSig"
        )
        assert result is None

    async def test_missing_signature_returns_none(self):
        """Missing signature returns None."""
        result = await self.service.get_sample_level_validation(
            "cima", "NONEXISTENT_SIG", "CytoSig"
        )
        assert result is None

    async def test_validation_data_cached(self):
        """Validation data is cached after first load."""
        # First call loads from disk
        await self.service.get_biological_associations("cima", "CytoSig")
        assert "validation_cima" in self.service._cache

        # Second call uses cache
        result = await self.service.get_biological_associations("cima", "CytoSig")
        assert result is not None
