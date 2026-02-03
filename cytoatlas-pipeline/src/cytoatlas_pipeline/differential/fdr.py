"""
FDR correction for multiple testing.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


class FDRCorrector:
    """
    FDR correction for differential analysis.

    Supports multiple correction methods from statsmodels.

    Example:
        >>> corrector = FDRCorrector(method="fdr_bh")
        >>> qvalues = corrector.correct(pvalues)
    """

    METHODS = [
        "bonferroni",
        "sidak",
        "holm-sidak",
        "holm",
        "simes-hochberg",
        "hommel",
        "fdr_bh",  # Benjamini-Hochberg
        "fdr_by",  # Benjamini-Yekutieli
        "fdr_tsbh",  # Two-stage BH
        "fdr_tsbky",  # Two-stage BY
    ]

    def __init__(
        self,
        method: str = "fdr_bh",
        alpha: float = 0.05,
    ):
        """
        Initialize FDR corrector.

        Args:
            method: Correction method.
            alpha: Significance threshold.
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown method: {method}. Available: {self.METHODS}"
            )
        self.method = method
        self.alpha = alpha

    def correct(
        self,
        pvalues: Union[np.ndarray, pd.DataFrame, pd.Series],
    ) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Apply FDR correction.

        Args:
            pvalues: P-values (any shape).

        Returns:
            Corrected q-values (same shape as input).
        """
        if isinstance(pvalues, pd.DataFrame):
            # Apply to flattened, reshape back
            flat_pvals = pvalues.values.ravel()
            _, flat_qvals, _, _ = multipletests(
                flat_pvals, alpha=self.alpha, method=self.method
            )
            qvalues = pd.DataFrame(
                flat_qvals.reshape(pvalues.shape),
                index=pvalues.index,
                columns=pvalues.columns,
            )
            return qvalues

        elif isinstance(pvalues, pd.Series):
            _, qvals, _, _ = multipletests(
                pvalues.values, alpha=self.alpha, method=self.method
            )
            return pd.Series(qvals, index=pvalues.index)

        else:
            original_shape = pvalues.shape
            flat_pvals = pvalues.ravel()
            _, flat_qvals, _, _ = multipletests(
                flat_pvals, alpha=self.alpha, method=self.method
            )
            return flat_qvals.reshape(original_shape)

    def get_significant_mask(
        self,
        pvalues: Union[np.ndarray, pd.DataFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get mask of significant results after correction.

        Args:
            pvalues: P-values.

        Returns:
            Boolean mask of significant results.
        """
        qvalues = self.correct(pvalues)
        return qvalues < self.alpha


def apply_fdr(
    pvalues: Union[np.ndarray, pd.DataFrame, pd.Series],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Apply FDR correction to p-values.

    Convenience function for FDRCorrector.

    Args:
        pvalues: P-values.
        method: Correction method.
        alpha: Significance threshold.

    Returns:
        Corrected q-values.
    """
    corrector = FDRCorrector(method=method, alpha=alpha)
    return corrector.correct(pvalues)
