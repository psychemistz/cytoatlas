"""
Pearson correlation with GPU acceleration.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from cytoatlas_pipeline.core.gpu_manager import GPUManager, get_gpu_manager


class PearsonCorrelator:
    """
    GPU-accelerated Pearson correlation.

    Example:
        >>> correlator = PearsonCorrelator()
        >>> rho, pval = correlator.correlate(activity, features)
    """

    def __init__(
        self,
        gpu_manager: Optional[GPUManager] = None,
        backend: Literal["auto", "numpy", "cupy"] = "auto",
    ):
        """
        Initialize Pearson correlator.

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Pearson correlation between X and Y.

        Args:
            X: First matrix (features x samples).
            Y: Second matrix (features x samples).

        Returns:
            Tuple of (correlation, pvalue) matrices.
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Transpose to samples x features
        X = X.T
        Y = Y.T

        if self.use_gpu:
            return self._correlate_gpu(X, Y)
        else:
            return self._correlate_numpy(X, Y)

    def _correlate_numpy(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy Pearson correlation."""
        n_samples = X.shape[0]
        n_x = X.shape[1]
        n_y = Y.shape[1]

        # Center data
        X_centered = X - X.mean(axis=0, keepdims=True)
        Y_centered = Y - Y.mean(axis=0, keepdims=True)

        # Standard deviations
        X_std = X.std(axis=0, ddof=1, keepdims=True)
        Y_std = Y.std(axis=0, ddof=1, keepdims=True)

        # Prevent division by zero
        X_std = np.where(X_std == 0, 1, X_std)
        Y_std = np.where(Y_std == 0, 1, Y_std)

        # Correlation
        rho = (X_centered.T @ Y_centered) / (n_samples - 1) / (X_std.T @ Y_std)

        # P-values using t-distribution
        t_stat = rho * np.sqrt((n_samples - 2) / (1 - rho**2 + 1e-10))
        pval = 2 * stats.t.sf(np.abs(t_stat), df=n_samples - 2)

        return rho, pval

    def _correlate_gpu(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """CuPy GPU Pearson correlation."""
        xp = self.gpu_manager.xp

        n_samples = X.shape[0]

        # Transfer to GPU
        X_gpu = self.gpu_manager.to_gpu(X)
        Y_gpu = self.gpu_manager.to_gpu(Y)

        # Center data
        X_centered = X_gpu - X_gpu.mean(axis=0, keepdims=True)
        Y_centered = Y_gpu - Y_gpu.mean(axis=0, keepdims=True)

        # Standard deviations
        X_std = X_gpu.std(axis=0, ddof=1, keepdims=True)
        Y_std = Y_gpu.std(axis=0, ddof=1, keepdims=True)

        # Prevent division by zero
        X_std = xp.where(X_std == 0, 1, X_std)
        Y_std = xp.where(Y_std == 0, 1, Y_std)

        # Correlation
        rho_gpu = (X_centered.T @ Y_centered) / (n_samples - 1) / (X_std.T @ Y_std)

        # Transfer back
        rho = self.gpu_manager.to_cpu(rho_gpu)

        # P-values on CPU (scipy doesn't have GPU version)
        t_stat = rho * np.sqrt((n_samples - 2) / (1 - rho**2 + 1e-10))
        pval = 2 * stats.t.sf(np.abs(t_stat), df=n_samples - 2)

        # Cleanup
        self.gpu_manager.free_memory()

        return rho, pval


def pearson_correlation(
    X: Union[np.ndarray, pd.DataFrame],
    Y: Union[np.ndarray, pd.DataFrame],
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Pearson correlation between X and Y.

    Convenience function for PearsonCorrelator.

    Args:
        X: First matrix (features x samples).
        Y: Second matrix (features x samples).
        backend: Computation backend.

    Returns:
        Tuple of (correlation, pvalue) matrices.
    """
    correlator = PearsonCorrelator(backend=backend)
    return correlator.correlate(X, Y)
