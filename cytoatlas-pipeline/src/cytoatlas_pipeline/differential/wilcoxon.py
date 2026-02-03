"""
Wilcoxon rank-sum test with GPU acceleration.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from cytoatlas_pipeline.core.gpu_manager import GPUManager, get_gpu_manager


class WilcoxonTest:
    """
    GPU-accelerated Wilcoxon rank-sum test.

    Non-parametric test for comparing two groups.

    Example:
        >>> test = WilcoxonTest()
        >>> statistic, pvalue = test.test(group1_activity, group2_activity)
    """

    def __init__(
        self,
        gpu_manager: Optional[GPUManager] = None,
        backend: Literal["auto", "numpy", "cupy"] = "auto",
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    ):
        """
        Initialize Wilcoxon test.

        Args:
            gpu_manager: GPU manager instance.
            backend: Computation backend.
            alternative: Alternative hypothesis.
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.alternative = alternative

        if backend == "auto":
            self.use_gpu = self.gpu_manager.is_gpu_available
        elif backend == "cupy":
            self.use_gpu = True
        else:
            self.use_gpu = False

    def test(
        self,
        group1: Union[np.ndarray, pd.DataFrame],
        group2: Union[np.ndarray, pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform Wilcoxon rank-sum test.

        Args:
            group1: First group (features x samples1).
            group2: Second group (features x samples2).

        Returns:
            Tuple of (statistic, pvalue) arrays.
        """
        # Convert to numpy
        if isinstance(group1, pd.DataFrame):
            group1 = group1.values
        if isinstance(group2, pd.DataFrame):
            group2 = group2.values

        # Ensure 2D
        if group1.ndim == 1:
            group1 = group1.reshape(-1, 1)
        if group2.ndim == 1:
            group2 = group2.reshape(-1, 1)

        n_features = group1.shape[0]
        n1 = group1.shape[1]
        n2 = group2.shape[1]

        if self.use_gpu:
            return self._test_gpu(group1, group2)
        else:
            return self._test_numpy(group1, group2)

    def _test_numpy(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy Wilcoxon rank-sum test."""
        n_features = group1.shape[0]

        statistics = np.zeros(n_features)
        pvalues = np.zeros(n_features)

        for i in range(n_features):
            # Remove NaN
            x = group1[i, ~np.isnan(group1[i])]
            y = group2[i, ~np.isnan(group2[i])]

            if len(x) < 2 or len(y) < 2:
                statistics[i] = np.nan
                pvalues[i] = np.nan
                continue

            stat, pval = stats.mannwhitneyu(
                x, y, alternative=self.alternative
            )
            statistics[i] = stat
            pvalues[i] = pval

        return statistics, pvalues

    def _test_gpu(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated Wilcoxon test.

        Uses GPU for rank computation, falls back to CPU for p-values.
        """
        xp = self.gpu_manager.xp

        n_features = group1.shape[0]
        n1 = group1.shape[1]
        n2 = group2.shape[1]

        # Combine groups
        combined = np.hstack([group1, group2])

        # Transfer to GPU
        combined_gpu = self.gpu_manager.to_gpu(combined)

        # Compute ranks using argsort
        # For each feature, rank the combined samples
        statistics = np.zeros(n_features)
        pvalues = np.zeros(n_features)

        for i in range(n_features):
            row = combined_gpu[i]

            # Argsort-based ranking (not handling ties perfectly)
            order = xp.argsort(row)
            ranks = xp.empty_like(order, dtype=xp.float64)
            ranks[order] = xp.arange(1, len(row) + 1)

            # Sum of ranks for group 1
            R1 = float(ranks[:n1].sum())

            # Mann-Whitney U statistic
            U1 = R1 - n1 * (n1 + 1) / 2
            U2 = n1 * n2 - U1

            statistics[i] = min(U1, U2)

            # Compute p-value on CPU using normal approximation
            mean_U = n1 * n2 / 2
            std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

            if std_U > 0:
                z = (U1 - mean_U) / std_U
                if self.alternative == "two-sided":
                    pvalues[i] = 2 * stats.norm.sf(abs(z))
                elif self.alternative == "greater":
                    pvalues[i] = stats.norm.sf(z)
                else:
                    pvalues[i] = stats.norm.cdf(z)
            else:
                pvalues[i] = 1.0

        self.gpu_manager.free_memory()

        return statistics, pvalues


def wilcoxon_test(
    group1: Union[np.ndarray, pd.DataFrame],
    group2: Union[np.ndarray, pd.DataFrame],
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Wilcoxon rank-sum test.

    Args:
        group1: First group (features x samples1).
        group2: Second group (features x samples2).
        alternative: Alternative hypothesis.
        backend: Computation backend.

    Returns:
        Tuple of (statistic, pvalue) arrays.
    """
    test = WilcoxonTest(backend=backend, alternative=alternative)
    return test.test(group1, group2)
