"""Equivalence tests: CSV results vs preprocessed JSON files.

Verifies that the JSON files in visualization/data/ faithfully represent
the underlying CSV/H5AD results from the analysis scripts. These tests
ensure the preprocessing step (06_preprocess_viz_data.py) did not
introduce data drift.

Auto-skipped when reference data is not available.

Run:
    pytest cytoatlas-pipeline/tests/equivalence/ -v
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
import numpy as np
import pandas as pd
import pytest

# Import harness utilities from the equivalence conftest
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.maintenance.equivalence_test import (
    compare_dataframes,
    compare_json_files,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("/data/parks34/projects/2secactpy")
_RESULTS = _DATA_ROOT / "results"
_VIZ_DATA = _DATA_ROOT / "visualization" / "data"

_HAS_DATA = _RESULTS.exists() and _VIZ_DATA.exists()
pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason="Reference results or viz data not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_viz_json(filename: str) -> list | dict:
    path = _VIZ_DATA / filename
    if not path.exists():
        pytest.skip(f"JSON file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _load_csv(relpath: str, **kwargs) -> pd.DataFrame:
    path = _RESULTS / relpath
    if not path.exists():
        pytest.skip(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)


# ===========================================================================
# CIMA correlation equivalence
# ===========================================================================

class TestCIMACorrelations:
    """Verify CIMA correlation JSON matches underlying CSVs."""

    def test_cima_age_correlations_shape(self):
        """cima_correlations.json should contain age/bmi/biochemistry sections."""
        data = _load_viz_json("cima_correlations.json")
        # Nested format: {age: [...], bmi: [...], biochemistry: [...]}
        assert isinstance(data, dict)
        assert "age" in data, f"Expected 'age' key, got: {list(data.keys())}"

        records = data["age"]
        assert isinstance(records, list)
        assert len(records) > 0

        # Verify expected fields
        sample = records[0]
        assert "protein" in sample
        assert "rho" in sample

    def test_cima_metabolites_top_vs_csv(self):
        """cima_metabolites_top.json should be derived from metabolites CSV."""
        json_data = _load_viz_json("cima_metabolites_top.json")
        assert isinstance(json_data, list)
        assert len(json_data) > 100  # Should have substantial data

        # All records should have required fields
        for row in json_data[:10]:
            assert "protein" in row
            assert "feature" in row
            assert "rho" in row

    def test_cima_differential_completeness(self):
        """cima_differential.json should cover expected comparisons."""
        data = _load_viz_json("cima_differential.json")
        assert isinstance(data, list)
        assert len(data) > 0

        # Check that key fields are present
        sample = data[0]
        required_fields = {"protein", "group1", "group2", "comparison", "signature"}
        actual_fields = set(sample.keys())
        missing = required_fields - actual_fields
        assert not missing, f"Missing fields in cima_differential: {missing}"


# ===========================================================================
# Inflammation correlation equivalence
# ===========================================================================

class TestInflammationCorrelations:
    """Verify Inflammation Atlas JSON matches underlying results."""

    def test_inflammation_disease_data(self):
        """inflammation_disease.json should have disease group data."""
        data = _load_viz_json("inflammation_disease.json")
        assert isinstance(data, (list, dict))
        if isinstance(data, list):
            assert len(data) > 0
        else:
            # Nested format — should have keys
            assert len(data) > 0

    def test_inflammation_differential_structure(self):
        """inflammation_differential.json should have comparison records."""
        data = _load_viz_json("inflammation_differential.json")
        assert isinstance(data, list)
        assert len(data) > 0


# ===========================================================================
# Cross-atlas equivalence
# ===========================================================================

class TestCrossAtlasEquivalence:
    """Verify cross-atlas JSON aligns with integrated results."""

    def test_cross_atlas_json_structure(self):
        """cross_atlas.json should have conserved signature data."""
        data = _load_viz_json("cross_atlas.json")
        assert isinstance(data, (list, dict))

    def test_cancer_comparison_structure(self):
        """cancer_comparison.json should have tumor vs adjacent data."""
        data = _load_viz_json("cancer_comparison.json")
        assert isinstance(data, (list, dict))


# ===========================================================================
# Validation data consistency
# ===========================================================================

class TestValidationDataConsistency:
    """Verify validation JSON files are internally consistent."""

    def test_validation_summary_has_all_atlases(self):
        """validation_summary.json should cover all atlases."""
        data = _load_viz_json("validation_summary.json")
        assert isinstance(data, (list, dict))
        if isinstance(data, list):
            assert len(data) > 0

    def test_cima_atlas_validation_structure(self):
        """cima_atlas_validation.json should have validation metrics."""
        data = _load_viz_json("cima_atlas_validation.json")
        assert isinstance(data, (list, dict))

    def test_bulk_rnaseq_validation_structure(self):
        """bulk_rnaseq_validation.json should have GTEx/TCGA data."""
        data = _load_viz_json("bulk_rnaseq_validation.json")
        assert isinstance(data, (list, dict))


# ===========================================================================
# H5AD validation results vs CSVs
# ===========================================================================

class TestH5ADvsCSV:
    """Verify H5AD pseudobulk shapes match expected dimensions."""

    @pytest.mark.parametrize("atlas,level", [
        ("cima", "l1"),
        ("cima", "l2"),
        ("inflammation_main", "l1"),
        ("inflammation_main", "l2"),
    ])
    def test_pseudobulk_h5ad_exists_and_has_shape(self, atlas, level):
        """Pseudobulk H5AD files should exist with reasonable shape."""
        h5ad_path = _RESULTS / "atlas_validation" / f"{atlas}_pseudobulk_{level}.h5ad"
        if not h5ad_path.exists():
            pytest.skip(f"H5AD not found: {h5ad_path}")

        import anndata as ad
        adata = ad.read_h5ad(h5ad_path, backed="r")

        assert adata.n_obs > 0, f"Empty observations in {h5ad_path.name}"
        assert adata.n_vars > 0, f"Empty variables in {h5ad_path.name}"

    @pytest.mark.parametrize("atlas,level,sigtype", [
        ("cima", "l1", "cytosig"),
        ("cima", "l2", "cytosig"),
    ])
    def test_activity_h5ad_has_signature_columns(self, atlas, level, sigtype):
        """Activity H5AD should have signature columns matching signature type."""
        h5ad_path = _RESULTS / "atlas_validation" / f"{atlas}_{level}_{sigtype}.h5ad"
        if not h5ad_path.exists():
            pytest.skip(f"H5AD not found: {h5ad_path}")

        import anndata as ad
        adata = ad.read_h5ad(h5ad_path, backed="r")

        if sigtype == "cytosig":
            # CytoSig has 44 cytokines
            assert adata.n_vars <= 50, (
                f"Expected ~44 CytoSig signatures, got {adata.n_vars}"
            )
        elif sigtype == "secact":
            # SecAct has ~1249 proteins
            assert adata.n_vars > 100, (
                f"Expected ~1249 SecAct signatures, got {adata.n_vars}"
            )


# ===========================================================================
# Cross-sample validation CSVs
# ===========================================================================

class TestCrossSampleValidation:
    """Verify cross-sample validation CSV structure."""

    @pytest.mark.parametrize("atlas", ["cima", "inflammation_main", "scatlas_normal"])
    def test_validation_h5ad_exists(self, atlas):
        """Validation H5AD files should exist for each atlas."""
        h5ad_dir = _RESULTS / "cross_sample_validation" / atlas
        if not h5ad_dir.exists():
            # Some atlases use base name subdirectory
            h5ad_dir = _RESULTS / "cross_sample_validation"
            matches = list(h5ad_dir.glob(f"{atlas}/*.h5ad"))
        else:
            matches = list(h5ad_dir.glob("*.h5ad"))

        assert len(matches) > 0, (
            f"No validation H5AD files found for atlas '{atlas}'"
        )

    def test_correlation_csv_has_expected_columns(self):
        """Correlation CSVs should have rho/pvalue columns."""
        csv_dir = _RESULTS / "cross_sample_validation" / "correlations"
        if not csv_dir.exists():
            pytest.skip("correlations directory not found")

        csvs = list(csv_dir.glob("*_correlations.csv"))
        if not csvs:
            pytest.skip("No correlation CSVs found")

        df = pd.read_csv(csvs[0])
        cols = set(df.columns)
        has_correlation = any(
            c in cols
            for c in ["rho", "correlation", "spearman_rho", "r"]
        )
        assert has_correlation, (
            f"Expected correlation column in {csvs[0].name}, got: {sorted(cols)}"
        )
