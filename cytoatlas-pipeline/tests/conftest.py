"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


@pytest.fixture
def sample_activity():
    """Create sample activity matrix."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(44, 100),  # 44 signatures, 100 samples
        index=[f"sig_{i}" for i in range(44)],
        columns=[f"sample_{i}" for i in range(100)],
    )


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(20, 80, 100),
        "sex": np.random.choice(["M", "F"], 100),
        "condition": np.random.choice(["healthy", "disease"], 100),
        "cell_type": np.random.choice(["CD4+ T cell", "CD8+ T cell", "B cell"], 100),
    }, index=[f"sample_{i}" for i in range(100)])


@pytest.fixture
def sample_expression():
    """Create sample expression matrix."""
    np.random.seed(42)
    return pd.DataFrame(
        np.abs(np.random.randn(1000, 100)),  # 1000 genes, 100 samples
        index=[f"gene_{i}" for i in range(1000)],
        columns=[f"sample_{i}" for i in range(100)],
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_h5ad(temp_dir):
    """Create mock H5AD file."""
    try:
        import anndata as ad

        np.random.seed(42)
        n_cells = 1000
        n_genes = 500

        X = np.random.randn(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "cell_type": np.random.choice(["T cell", "B cell", "NK cell"], n_cells),
            "sample": np.random.choice([f"sample_{i}" for i in range(10)], n_cells),
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        adata = ad.AnnData(X=X, obs=obs, var=var)

        path = temp_dir / "test.h5ad"
        adata.write_h5ad(path)

        return path
    except ImportError:
        pytest.skip("anndata not installed")
