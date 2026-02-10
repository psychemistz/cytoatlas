"""Shared fixtures for equivalence tests."""

import sys
from pathlib import Path

import pytest

# Add project root to path so we can import the maintenance harness
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.maintenance.equivalence_test import (
    ComparisonResult,
    compare_dataframes,
    compare_h5ad_files,
    compare_json_files,
)


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def results_dir(project_root):
    """Return the results/ directory."""
    return project_root / "results"


@pytest.fixture
def scripts_dir(project_root):
    """Return the scripts/ directory."""
    return project_root / "scripts"


@pytest.fixture
def viz_data_dir(project_root):
    """Return the visualization/data/ directory."""
    return project_root / "visualization" / "data"


def assert_equivalent(result: ComparisonResult):
    """Assert that a comparison result shows equivalence."""
    assert result.equal, f"Not equivalent:\n{result}"
