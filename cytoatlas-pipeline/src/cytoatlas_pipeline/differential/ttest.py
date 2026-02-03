"""
T-test variants for differential analysis.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from cytoatlas_pipeline.core.gpu_manager import GPUManager, get_gpu_manager


class TTest:
    """
    T-test for comparing two groups.

    Supports independent samples t-test and Welch's t-test.

    Example:
        >>> test = TTest(equal_var=False)  # Welch's t-test
        >>> statistic, pvalue = test.test(group1, group2)
    """

    def __init__(
        self,
        gpu_manager: Optional[GPUManager] = None,
        backend: Literal["auto", "numpy", "cupy"] = "auto",
        equal_var: bool = False,
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    ):
        """
        Initialize t-test.

        Args:
            gpu_manager: GPU manager instance.
            backend: Computation backend.
            equal_var: Assume equal variance (False = Welch's).
            alternative: Alternative hypothesis.
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.equal_var = equal_var
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
        Perform t-test.

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

        if self.use_gpu:
            return self._test_gpu(group1, group2)
        else:
            return self._test_numpy(group1, group2)

    def _test_numpy(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy t-test."""
        n_features = group1.shape[0]
        n1 = group1.shape[1]
        n2 = group2.shape[1]

        # Compute means and variances
        mean1 = np.nanmean(group1, axis=1)
        mean2 = np.nanmean(group2, axis=1)
        var1 = np.nanvar(group1, axis=1, ddof=1)
        var2 = np.nanvar(group2, axis=1, ddof=1)

        if self.equal_var:
            # Pooled variance
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = np.sqrt(var1/n1 + var2/n2)
            # Welch-Satterthwaite df
            num = (var1/n1 + var2/n2)**2
            denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
            df = num / np.where(denom > 0, denom, 1)

        # T-statistic
        t_stat = (mean1 - mean2) / np.where(se > 0, se, 1)

        # P-values
        if self.alternative == "two-sided":
            pvalues = 2 * stats.t.sf(np.abs(t_stat), df)
        elif self.alternative == "greater":
            pvalues = stats.t.sf(t_stat, df)
        else:
            pvalues = stats.t.cdf(t_stat, df)

        return t_stat, pvalues

    def _test_gpu(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated t-test."""
        xp = self.gpu_manager.xp

        n1 = group1.shape[1]
        n2 = group2.shape[1]

        # Transfer to GPU
        g1_gpu = self.gpu_manager.to_gpu(group1)
        g2_gpu = self.gpu_manager.to_gpu(group2)

        # Compute means and variances
        mean1 = xp.nanmean(g1_gpu, axis=1)
        mean2 = xp.nanmean(g2_gpu, axis=1)
        var1 = xp.nanvar(g1_gpu, axis=1, ddof=1)
        var2 = xp.nanvar(g2_gpu, axis=1, ddof=1)

        if self.equal_var:
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = xp.sqrt(pooled_var * (1/n1 + 1/n2))
            df_val = n1 + n2 - 2
        else:
            se = xp.sqrt(var1/n1 + var2/n2)
            num = (var1/n1 + var2/n2)**2
            denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
            df_gpu = num / xp.where(denom > 0, denom, 1)
            df_val = self.gpu_manager.to_cpu(df_gpu)

        t_stat_gpu = (mean1 - mean2) / xp.where(se > 0, se, 1)
        t_stat = self.gpu_manager.to_cpu(t_stat_gpu)

        self.gpu_manager.free_memory()

        # P-values on CPU
        if self.alternative == "two-sided":
            pvalues = 2 * stats.t.sf(np.abs(t_stat), df_val)
        elif self.alternative == "greater":
            pvalues = stats.t.sf(t_stat, df_val)
        else:
            pvalues = stats.t.cdf(t_stat, df_val)

        return t_stat, pvalues


def ttest(
    group1: Union[np.ndarray, pd.DataFrame],
    group2: Union[np.ndarray, pd.DataFrame],
    equal_var: bool = False,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform t-test.

    Args:
        group1: First group (features x samples1).
        group2: Second group (features x samples2).
        equal_var: Assume equal variance.
        alternative: Alternative hypothesis.
        backend: Computation backend.

    Returns:
        Tuple of (statistic, pvalue) arrays.
    """
    test = TTest(backend=backend, equal_var=equal_var, alternative=alternative)
    return test.test(group1, group2)
