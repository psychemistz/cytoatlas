"""Tests for correlation analysis modules."""

import pytest
import numpy as np
import pandas as pd


class TestPearsonCorrelation:
    """Test Pearson correlation."""

    def test_basic_correlation(self):
        from cytoatlas_pipeline.correlation.pearson import pearson_correlation

        # Perfect positive correlation - need 2D arrays (features x samples)
        x = np.array([[1, 2, 3, 4, 5]])  # 1 feature x 5 samples
        y = np.array([[2, 4, 6, 8, 10]])

        rho, pval = pearson_correlation(x, y)
        assert abs(rho[0, 0] - 1.0) < 1e-6
        assert pval[0, 0] < 0.01

    def test_negative_correlation(self):
        from cytoatlas_pipeline.correlation.pearson import pearson_correlation

        x = np.array([[1, 2, 3, 4, 5]])
        y = np.array([[5, 4, 3, 2, 1]])

        rho, pval = pearson_correlation(x, y)
        assert abs(rho[0, 0] - (-1.0)) < 1e-6

    def test_no_correlation(self):
        from cytoatlas_pipeline.correlation.pearson import pearson_correlation

        np.random.seed(42)
        x = np.random.randn(1, 1000)  # 1 feature x 1000 samples
        y = np.random.randn(1, 1000)

        rho, pval = pearson_correlation(x, y)
        assert abs(rho[0, 0]) < 0.1  # Should be close to 0


class TestSpearmanCorrelation:
    """Test Spearman correlation."""

    def test_monotonic_relationship(self):
        from cytoatlas_pipeline.correlation.spearman import spearman_correlation

        # Perfect monotonic (but not linear) - 2D arrays
        x = np.array([[1, 2, 3, 4, 5]])
        y = np.array([[1, 4, 9, 16, 25]])  # Quadratic

        rho, pval = spearman_correlation(x, y)
        assert abs(rho[0, 0] - 1.0) < 1e-6  # Spearman should be 1

    def test_with_ties(self):
        from cytoatlas_pipeline.correlation.spearman import spearman_correlation

        x = np.array([[1, 2, 2, 3, 4]])  # Has ties
        y = np.array([[1, 2, 3, 4, 5]])

        rho, pval = spearman_correlation(x, y)
        assert rho[0, 0] > 0.8  # Should still be strongly positive


class TestContinuousCorrelator:
    """Test continuous variable correlator."""

    def test_correlate_with_metadata(self):
        from cytoatlas_pipeline.correlation.continuous import ContinuousCorrelator

        np.random.seed(42)

        # Create test data
        activity = pd.DataFrame(
            np.random.randn(5, 100),
            index=[f"sig_{i}" for i in range(5)],
            columns=[f"sample_{i}" for i in range(100)],
        )

        # Features DataFrame indexed by sample
        features = pd.DataFrame({
            "age": np.random.randint(20, 80, 100),
            "bmi": np.random.uniform(18, 35, 100),
        }, index=activity.columns)

        correlator = ContinuousCorrelator()
        result = correlator.correlate(activity, features)

        # Result should be a CorrelationResult object
        assert hasattr(result, 'rho')
        assert "age" in result.rho.columns
        assert "bmi" in result.rho.columns
        assert len(result.rho.index) == 5  # 5 signatures


class TestFuzzyMatcher:
    """Test fuzzy string matching."""

    def test_levenshtein_distance(self):
        from cytoatlas_pipeline.search.fuzzy import FuzzyMatcher

        matcher = FuzzyMatcher()

        # Same string
        assert matcher.levenshtein_distance("test", "test") == 0

        # One character insert
        assert matcher.levenshtein_distance("test", "tests") == 1

        # One character delete
        assert matcher.levenshtein_distance("test", "est") == 1

    def test_fuzzy_match(self):
        from cytoatlas_pipeline.search.fuzzy import FuzzyMatcher

        matcher = FuzzyMatcher(max_distance=2)
        candidates = ["IFNG", "IFNA", "TNF", "IL2", "IL6"]

        matches = matcher.match("IFN", candidates)
        # Should match IFNG and IFNA (distance 1 each)
        match_names = [m.match for m in matches]
        assert "IFNG" in match_names or "IFNA" in match_names

    def test_best_match(self):
        from cytoatlas_pipeline.search.fuzzy import FuzzyMatcher

        matcher = FuzzyMatcher(max_distance=3)
        candidates = ["CD4+ T cell", "CD8+ T cell", "B cell", "NK cell"]

        best = matcher.best_match("CD4 T cell", candidates)
        assert best is not None
        assert "CD4" in best.match
