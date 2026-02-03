"""Tests for differential analysis modules."""

import pytest
import numpy as np
import pandas as pd


class TestWilcoxonTest:
    """Test Wilcoxon rank-sum test."""

    def test_significant_difference(self):
        from cytoatlas_pipeline.differential.wilcoxon import WilcoxonTest

        # Create clearly different distributions - 2D (features x samples)
        np.random.seed(42)
        group1 = np.random.randn(1, 50) + 3  # 1 feature, 50 samples, mean ~3
        group2 = np.random.randn(1, 50)  # 1 feature, 50 samples, mean ~0

        test = WilcoxonTest()
        stat, pval = test.test(group1, group2)

        assert pval[0] < 0.001  # Should be highly significant

    def test_no_difference(self):
        from cytoatlas_pipeline.differential.wilcoxon import WilcoxonTest

        np.random.seed(42)
        group1 = np.random.randn(1, 50)  # 1 feature, 50 samples
        group2 = np.random.randn(1, 50)

        test = WilcoxonTest()
        stat, pval = test.test(group1, group2)

        assert pval[0] > 0.05  # Should not be significant


class TestEffectSize:
    """Test effect size calculations."""

    def test_activity_difference(self):
        from cytoatlas_pipeline.differential.effect_size import EffectSizeCalculator

        calc = EffectSizeCalculator()

        # 2D arrays (features x samples)
        group1 = np.array([[3, 4, 5]])  # 1 feature, 3 samples, mean = 4
        group2 = np.array([[1, 2, 3]])  # 1 feature, 3 samples, mean = 2

        diff = calc.activity_difference(group1, group2)
        assert diff[0] == pytest.approx(2.0)

        # Negative difference
        diff_neg = calc.activity_difference(group2, group1)
        assert diff_neg[0] == pytest.approx(-2.0)

    def test_cohens_d(self):
        from cytoatlas_pipeline.differential.effect_size import EffectSizeCalculator

        calc = EffectSizeCalculator()

        # Large effect size - 2D arrays
        group1 = np.array([[10, 11, 12, 13, 14]])
        group2 = np.array([[0, 1, 2, 3, 4]])

        d = calc.cohens_d(group1, group2)
        assert d[0] > 2.0  # Large effect

    def test_negative_values_handled(self):
        """Activity values can be negative (z-scores)."""
        from cytoatlas_pipeline.differential.effect_size import EffectSizeCalculator

        calc = EffectSizeCalculator()

        # Both groups negative - 2D arrays
        group1 = np.array([[-2, -1.5, -1]])  # Mean = -1.5
        group2 = np.array([[-4, -3.5, -3]])  # Mean = -3.5

        diff = calc.activity_difference(group1, group2)
        assert diff[0] == pytest.approx(2.0)  # -1.5 - (-3.5) = 2


class TestFDRCorrection:
    """Test FDR correction."""

    def test_bh_correction(self):
        from cytoatlas_pipeline.differential.fdr import FDRCorrector

        # Mix of significant and non-significant p-values
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])

        corrector = FDRCorrector(method="fdr_bh")
        qvals = corrector.correct(pvals)

        # Q-values should be >= p-values
        assert np.all(qvals >= pvals)

        # Should maintain order
        assert np.all(np.diff(qvals) >= -1e-10)  # Allow small numerical error

    def test_bonferroni(self):
        from cytoatlas_pipeline.differential.fdr import FDRCorrector

        pvals = np.array([0.01, 0.02, 0.03])

        corrector = FDRCorrector(method="bonferroni")
        qvals = corrector.correct(pvals)

        # Bonferroni multiplies by n
        assert qvals[0] == pytest.approx(0.03)  # 0.01 * 3


class TestStratifiedDifferential:
    """Test stratified differential analysis."""

    def test_simple_comparison(self):
        from cytoatlas_pipeline.differential.stratified import StratifiedDifferential

        np.random.seed(42)

        # Create test data
        activity = pd.DataFrame(
            np.random.randn(10, 20),
            index=[f"sig_{i}" for i in range(10)],
            columns=[f"sample_{i}" for i in range(20)],
        )

        # Make first 10 samples "disease", rest "healthy"
        activity.iloc[:, :10] += 2  # Shift disease samples up

        metadata = pd.DataFrame({
            "condition": ["disease"] * 10 + ["healthy"] * 10,
        }, index=activity.columns)

        diff = StratifiedDifferential()
        result = diff.compare(
            activity=activity,
            metadata=metadata,
            group_col="condition",
            group1_value="disease",
            group2_value="healthy",
        )

        # All signatures should show positive activity_diff
        assert all(result.activity_diff > 0)
