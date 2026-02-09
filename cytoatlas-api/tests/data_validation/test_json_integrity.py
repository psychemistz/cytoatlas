"""Data integrity tests: No NaN, consistent naming, valid enums."""

import json
import math
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _check_no_nan(data, path="root"):
    """Recursively check that no numeric value is NaN."""
    if isinstance(data, float):
        assert not math.isnan(data), f"NaN found at {path}"
        assert not math.isinf(data), f"Inf found at {path}"
    elif isinstance(data, dict):
        for key, value in data.items():
            _check_no_nan(value, f"{path}.{key}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _check_no_nan(item, f"{path}[{i}]")


class TestNoNaN:
    """Verify no NaN values in numeric fields across all fixture files."""

    def test_activity_summary_no_nan(self, activity_summary_data):
        """activity_summary.json has no NaN values."""
        _check_no_nan(activity_summary_data, "activity_summary")

    def test_correlations_no_nan(self, correlations_data):
        """correlations.json has no NaN values."""
        _check_no_nan(correlations_data, "correlations")

    def test_differential_no_nan(self, differential_data):
        """differential.json has no NaN values."""
        _check_no_nan(differential_data, "differential")

    def test_validation_no_nan(self, validation_data):
        """validation_results.json has no NaN values."""
        _check_no_nan(validation_data, "validation")


class TestConsistentCellTypeNames:
    """Verify cell type names are consistent across datasets."""

    VALID_CELL_TYPES = {
        "CD4_T", "CD8_T", "NK_cells", "Monocytes", "Macrophages",
        "B_cells", "Dendritic_cells", "Tregs", "Th17", "T_cells",
        "Plasma_cells", "Neutrophils", "Mast_cells", "Eosinophils",
        "Basophils", "ILC", "pDC", "cDC",
    }

    def test_activity_cell_types(self, activity_summary_data):
        """Activity data uses known cell type names."""
        for record in activity_summary_data:
            ct = record["cell_type"]
            assert ct in self.VALID_CELL_TYPES, (
                f"Unknown cell type: '{ct}'. "
                f"Add to VALID_CELL_TYPES if intentional."
            )

    def test_correlations_cell_types(self, correlations_data):
        """Correlation data uses known cell type names."""
        for variable, records in correlations_data.items():
            for record in records:
                ct = record["cell_type"]
                assert ct in self.VALID_CELL_TYPES, (
                    f"Unknown cell type: '{ct}' in {variable}"
                )

    def test_differential_cell_types(self, differential_data):
        """Differential data uses known cell type names."""
        for record in differential_data:
            ct = record["cell_type"]
            assert ct in self.VALID_CELL_TYPES, (
                f"Unknown cell type: '{ct}'"
            )


class TestSignatureTypeValues:
    """Verify signature_type is always CytoSig or SecAct."""

    VALID_TYPES = {"CytoSig", "SecAct"}

    def test_activity_signature_types(self, activity_summary_data):
        """Activity data has valid signature types."""
        for record in activity_summary_data:
            assert record["signature_type"] in self.VALID_TYPES

    def test_correlations_signature_types(self, correlations_data):
        """Correlation data has valid signature types."""
        for variable, records in correlations_data.items():
            for record in records:
                assert record["signature_type"] in self.VALID_TYPES

    def test_differential_signature_types(self, differential_data):
        """Differential data has valid signature types."""
        for record in differential_data:
            assert record["signature_type"] in self.VALID_TYPES


class TestJSONFileIntegrity:
    """Verify all JSON fixture files parse correctly."""

    def test_activity_summary_parses(self):
        """activity_summary.json is valid JSON."""
        with open(FIXTURES_DIR / "activity_summary.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_correlations_parses(self):
        """correlations.json is valid JSON."""
        with open(FIXTURES_DIR / "correlations.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_differential_parses(self):
        """differential.json is valid JSON."""
        with open(FIXTURES_DIR / "differential.json") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_validation_parses(self):
        """validation_results.json is valid JSON."""
        with open(FIXTURES_DIR / "validation_results.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)


class TestNumericFieldsArePrimitive:
    """Verify numeric fields are int or float, not strings."""

    def test_activity_numerics(self, activity_summary_data):
        """Numeric fields in activity data are float or int."""
        for record in activity_summary_data:
            assert isinstance(record["mean_activity"], (int, float))
            if "n_samples" in record:
                assert isinstance(record["n_samples"], int)
            if "n_cells" in record:
                assert isinstance(record["n_cells"], int)

    def test_correlation_numerics(self, correlations_data):
        """Numeric fields in correlations are float or int."""
        for variable, records in correlations_data.items():
            for record in records:
                assert isinstance(record["rho"], (int, float))
                assert isinstance(record["pvalue"], (int, float))

    def test_differential_numerics(self, differential_data):
        """Numeric fields in differential data are float or int."""
        for record in differential_data:
            assert isinstance(record["activity_diff"], (int, float))
            assert isinstance(record["pvalue"], (int, float))
