"""
Partial correlation with confound adjustment.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from cytoatlas_pipeline.core.gpu_manager import GPUManager, get_gpu_manager


class PartialCorrelator:
    """
    Partial correlation controlling for confounding variables.

    Computes correlation between X and Y after removing the effect
    of confounding variables Z.

    Example:
        >>> correlator = PartialCorrelator()
        >>> # Correlate activity with outcome, controlling for age and sex
        >>> rho, pval = correlator.correlate(activity, outcome, confounds)
    """

    def __init__(
        self,
        gpu_manager: Optional[GPUManager] = None,
        backend: Literal["auto", "numpy", "cupy"] = "auto",
    ):
        """
        Initialize partial correlator.

        Args:
            gpu_manager: GPU manager instance.
            backend: Computation backend.
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()

        if backend == "auto":
            self.use_gpu = self.gpu_manager.is_gpu_available
        elif backend == "cupy":
            self.use_gpu = True
        else:
            self.use_gpu = False

    def correlate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        Y: Union[np.ndarray, pd.DataFrame],
        Z: Union[np.ndarray, pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute partial correlation between X and Y controlling for Z.

        Args:
            X: First matrix (features x samples).
            Y: Second matrix (features x samples).
            Z: Confounding variables (confounds x samples).

        Returns:
            Tuple of (correlation, pvalue) matrices.
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values
        if isinstance(Z, pd.DataFrame):
            Z = Z.values

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        # Transpose to samples x features
        X = X.T
        Y = Y.T
        Z = Z.T

        return self._correlate_numpy(X, Y, Z)

    def _correlate_numpy(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy partial correlation."""
        n_samples = X.shape[0]
        n_confounds = Z.shape[1]

        # Residualize X and Y with respect to Z
        # X_resid = X - Z @ (Z'Z)^{-1} Z' X
        Z_centered = Z - Z.mean(axis=0, keepdims=True)

        # Add intercept
        Z_design = np.column_stack([np.ones(n_samples), Z_centered])

        # Compute projection matrix
        ZtZ_inv = np.linalg.pinv(Z_design.T @ Z_design)
        proj = Z_design @ ZtZ_inv @ Z_design.T

        # Residualize
        X_resid = X - proj @ X
        Y_resid = Y - proj @ Y

        # Compute Pearson correlation on residuals
        X_centered = X_resid - X_resid.mean(axis=0, keepdims=True)
        Y_centered = Y_resid - Y_resid.mean(axis=0, keepdims=True)

        X_std = X_resid.std(axis=0, ddof=1, keepdims=True)
        Y_std = Y_resid.std(axis=0, ddof=1, keepdims=True)

        X_std = np.where(X_std == 0, 1, X_std)
        Y_std = np.where(Y_std == 0, 1, Y_std)

        rho = (X_centered.T @ Y_centered) / (n_samples - 1) / (X_std.T @ Y_std)

        # P-values with adjusted df
        df = n_samples - n_confounds - 2
        t_stat = rho * np.sqrt(df / (1 - rho**2 + 1e-10))
        pval = 2 * stats.t.sf(np.abs(t_stat), df=max(1, df))

        return rho, pval


def partial_correlation(
    X: Union[np.ndarray, pd.DataFrame],
    Y: Union[np.ndarray, pd.DataFrame],
    Z: Union[np.ndarray, pd.DataFrame],
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute partial correlation controlling for Z.

    Args:
        X: First matrix (features x samples).
        Y: Second matrix (features x samples).
        Z: Confounding variables (confounds x samples).
        backend: Computation backend.

    Returns:
        Tuple of (correlation, pvalue) matrices.
    """
    correlator = PartialCorrelator(backend=backend)
    return correlator.correlate(X, Y, Z)
