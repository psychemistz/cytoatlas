"""Data validation tests: verify fixture data matches expected schemas."""

import json
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestActivitySchema:
    """Validate activity_summary.json schema."""

    @pytest.fixture(autouse=True)
    def load_data(self, activity_summary_data):
        self.data = activity_summary_data

    def test_required_fields(self):
        """Each record has required fields."""
        required = {"cell_type", "signature", "signature_type", "mean_activity"}
        for record in self.data:
            missing = required - set(record.keys())
            assert not missing, f"Missing fields: {missing} in {record}"

    def test_signature_type_values(self):
        """signature_type is either CytoSig or SecAct."""
        for record in self.data:
            assert record["signature_type"] in ("CytoSig", "SecAct"), (
                f"Invalid signature_type: {record['signature_type']}"
            )

    def test_mean_activity_range(self):
        """Activity z-scores are in reasonable range (-5, +5)."""
        for record in self.data:
            val = record["mean_activity"]
            assert -5.0 <= val <= 5.0, (
                f"mean_activity {val} out of range for {record['signature']}"
            )

    def test_n_samples_positive(self):
        """n_samples is a positive integer."""
        for record in self.data:
            if "n_samples" in record:
                assert record["n_samples"] > 0
                assert isinstance(record["n_samples"], int)

    def test_n_cells_positive(self):
        """n_cells is a positive integer."""
        for record in self.data:
            if "n_cells" in record:
                assert record["n_cells"] > 0
                assert isinstance(record["n_cells"], int)

    def test_cell_type_non_empty(self):
        """cell_type is a non-empty string."""
        for record in self.data:
            assert record["cell_type"]
            assert isinstance(record["cell_type"], str)


class TestCorrelationsSchema:
    """Validate correlations.json schema."""

    @pytest.fixture(autouse=True)
    def load_data(self, correlations_data):
        self.data = correlations_data

    def test_has_variable_categories(self):
        """Data has expected variable categories."""
        assert "age" in self.data
        assert "bmi" in self.data

    def test_correlation_required_fields(self):
        """Each correlation record has required fields."""
        required = {"cell_type", "signature", "signature_type", "rho", "pvalue"}
        for variable, records in self.data.items():
            for record in records:
                missing = required - set(record.keys())
                assert not missing, f"Missing {missing} in {variable}: {record}"

    def test_rho_range(self):
        """Correlation rho is in [-1, 1]."""
        for variable, records in self.data.items():
            for record in records:
                rho = record["rho"]
                assert -1.0 <= rho <= 1.0, (
                    f"rho {rho} out of range in {variable}"
                )

    def test_pvalue_range(self):
        """p-values are in [0, 1]."""
        for variable, records in self.data.items():
            for record in records:
                p = record["pvalue"]
                assert 0.0 <= p <= 1.0, f"pvalue {p} out of range"

    def test_qvalue_range(self):
        """q-values (FDR) are in [0, 1]."""
        for variable, records in self.data.items():
            for record in records:
                if "qvalue" in record:
                    q = record["qvalue"]
                    assert 0.0 <= q <= 1.0, f"qvalue {q} out of range"


class TestDifferentialSchema:
    """Validate differential.json schema."""

    @pytest.fixture(autouse=True)
    def load_data(self, differential_data):
        self.data = differential_data

    def test_required_fields(self):
        """Each differential record has required fields."""
        required = {
            "cell_type", "signature", "signature_type", "disease",
            "activity_diff", "pvalue",
        }
        for record in self.data:
            missing = required - set(record.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_activity_diff_is_difference(self):
        """activity_diff equals mean_disease - mean_healthy."""
        for record in self.data:
            if "mean_disease" in record and "mean_healthy" in record:
                expected = record["mean_disease"] - record["mean_healthy"]
                assert abs(record["activity_diff"] - expected) < 0.01, (
                    f"activity_diff mismatch for {record['signature']}"
                )

    def test_uses_activity_diff_not_log2fc(self):
        """Field is 'activity_diff', not 'log2fc'."""
        for record in self.data:
            assert "activity_diff" in record
            assert "log2fc" not in record

    def test_disease_non_empty(self):
        """Disease name is a non-empty string."""
        for record in self.data:
            assert record["disease"]
            assert isinstance(record["disease"], str)


class TestValidationSchema:
    """Validate validation_results.json schema."""

    @pytest.fixture(autouse=True)
    def load_data(self, validation_data):
        self.data = validation_data

    def test_has_validation_sections(self):
        """Validation data has expected top-level sections."""
        expected_keys = {
            "sample_validations",
            "celltype_validations",
            "pseudobulk_vs_sc",
            "singlecell_validations",
            "biological_associations",
        }
        assert expected_keys.issubset(set(self.data.keys()))

    def test_sample_validation_has_stats(self):
        """Sample validations have stats with r_squared."""
        for v in self.data["sample_validations"]:
            assert "stats" in v
            assert "r_squared" in v["stats"]
            assert 0 <= v["stats"]["r_squared"] <= 1

    def test_celltype_validation_has_points(self):
        """Cell type validations have points with cell_type."""
        for v in self.data["celltype_validations"]:
            assert "points" in v
            for p in v["points"]:
                assert "cell_type" in p
                assert "expression" in p
                assert "activity" in p

    def test_biological_associations_structure(self):
        """Biological associations have proper structure."""
        bio = self.data["biological_associations"]
        assert "results" in bio
        assert "n_tested" in bio
        assert "n_validated" in bio
        assert "validation_rate" in bio

        for r in bio["results"]:
            assert "signature" in r
            assert "expected_cell_type" in r
            assert "is_validated" in r
            assert isinstance(r["is_validated"], bool)

    def test_validation_rate_valid(self):
        """Validation rate is between 0 and 1."""
        rate = self.data["biological_associations"]["validation_rate"]
        assert 0.0 <= rate <= 1.0
