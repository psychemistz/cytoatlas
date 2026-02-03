"""
Ridge regression wrapper for activity inference.

Provides a high-level interface to SecActpy's ridge regression
with automatic data preprocessing and result formatting.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sp

# Add SecActpy to path
SECACTPY_PATH = Path("/vf/users/parks34/projects/1ridgesig/SecActpy")
if str(SECACTPY_PATH) not in sys.path:
    sys.path.insert(0, str(SECACTPY_PATH))


@dataclass
class ActivityResult:
    """Result of activity inference."""

    beta: pd.DataFrame
    """Activity coefficients (signatures x samples)."""

    se: pd.DataFrame
    """Standard errors (signatures x samples)."""

    zscore: pd.DataFrame
    """Z-scores (signatures x samples)."""

    pvalue: pd.DataFrame
    """P-values (signatures x samples)."""

    signature_names: list[str]
    """Names of signatures."""

    sample_names: list[str]
    """Names of samples."""

    n_signatures: int
    """Number of signatures."""

    n_samples: int
    """Number of samples."""

    n_genes_used: int
    """Number of genes used in inference."""

    gene_overlap: float
    """Fraction of signature genes found in expression."""

    method: str
    """Backend used (numpy/cupy)."""

    time_seconds: float
    """Execution time."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def get_significant(
        self,
        pvalue_threshold: float = 0.05,
        zscore_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Get significant activity scores.

        Args:
            pvalue_threshold: P-value cutoff.
            zscore_threshold: Optional z-score cutoff.

        Returns:
            DataFrame with significant scores only.
        """
        mask = self.pvalue < pvalue_threshold
        if zscore_threshold is not None:
            mask = mask & (np.abs(self.zscore) > zscore_threshold)

        result = self.zscore.copy()
        result[~mask] = np.nan
        return result

    def to_anndata_uns(self) -> dict[str, Any]:
        """Convert to dict for storing in AnnData.uns."""
        return {
            "activity_beta": self.beta.to_dict(),
            "activity_zscore": self.zscore.to_dict(),
            "activity_pvalue": self.pvalue.to_dict(),
            "activity_method": self.method,
            "activity_n_genes": self.n_genes_used,
            "activity_gene_overlap": self.gene_overlap,
        }


class RidgeInference:
    """
    Ridge regression-based activity inference.

    Wraps SecActpy's ridge regression with preprocessing and result formatting.

    Example:
        >>> inference = RidgeInference(lambda_=5e5, n_rand=1000)
        >>> result = inference.run(expression_df, signature_df)
        >>> print(f"Inferred {result.n_signatures} signatures")
    """

    def __init__(
        self,
        lambda_: float = 5e5,
        n_rand: int = 1000,
        seed: int = 0,
        backend: Literal["auto", "numpy", "cupy"] = "auto",
        batch_size: int = 5000,
        use_cache: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize ridge inference.

        Args:
            lambda_: Ridge regularization parameter.
            n_rand: Number of permutations.
            seed: Random seed.
            backend: Computation backend.
            batch_size: Batch size for large datasets.
            use_cache: Cache permutation tables.
            verbose: Print progress.
        """
        self.lambda_ = lambda_
        self.n_rand = n_rand
        self.seed = seed
        self.backend = backend
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose

    def run(
        self,
        expression: Union[pd.DataFrame, np.ndarray],
        signature: pd.DataFrame,
        expression_gene_names: Optional[list[str]] = None,
        sample_names: Optional[list[str]] = None,
    ) -> ActivityResult:
        """
        Run activity inference.

        Args:
            expression: Expression matrix (genes x samples).
            signature: Signature matrix (genes x signatures).
            expression_gene_names: Gene names for expression (if ndarray).
            sample_names: Sample names (if ndarray).

        Returns:
            ActivityResult with inferred activities.
        """
        # Import SecActpy
        try:
            from secactpy import ridge, ridge_batch
        except ImportError:
            raise ImportError(
                "SecActpy not found. Ensure it's installed or path is correct."
            )

        # Convert to DataFrames if needed
        if isinstance(expression, np.ndarray):
            if expression_gene_names is None:
                raise ValueError("expression_gene_names required for ndarray input")
            if sample_names is None:
                sample_names = [f"sample_{i}" for i in range(expression.shape[1])]
            expression = pd.DataFrame(
                expression, index=expression_gene_names, columns=sample_names
            )

        # Align genes
        expr_aligned, sig_aligned, gene_overlap = self._align_genes(
            expression, signature
        )

        n_genes = len(expr_aligned)
        n_samples = expr_aligned.shape[1]
        n_signatures = sig_aligned.shape[1]

        if self.verbose:
            print(f"Activity inference: {n_genes} genes, {n_samples} samples, {n_signatures} signatures")
            print(f"Gene overlap: {gene_overlap:.1%}")

        # Scale data
        expr_scaled = self._scale_data(expr_aligned)
        sig_scaled = self._scale_data(sig_aligned)

        # Run ridge regression
        if n_samples > self.batch_size:
            result = ridge_batch(
                X=sig_scaled.values,
                Y=expr_scaled.values,
                lambda_=self.lambda_,
                n_rand=self.n_rand,
                seed=self.seed,
                batch_size=self.batch_size,
                backend=self.backend,
                use_cache=self.use_cache,
                verbose=self.verbose,
            )
        else:
            result = ridge(
                X=sig_scaled.values,
                Y=expr_scaled.values,
                lambda_=self.lambda_,
                n_rand=self.n_rand,
                seed=self.seed,
                backend=self.backend,
                verbose=self.verbose,
            )

        # Format results
        signature_names = list(sig_aligned.columns)
        sample_names_out = list(expr_aligned.columns)

        return ActivityResult(
            beta=pd.DataFrame(result["beta"], index=signature_names, columns=sample_names_out),
            se=pd.DataFrame(result["se"], index=signature_names, columns=sample_names_out),
            zscore=pd.DataFrame(result["zscore"], index=signature_names, columns=sample_names_out),
            pvalue=pd.DataFrame(result["pvalue"], index=signature_names, columns=sample_names_out),
            signature_names=signature_names,
            sample_names=sample_names_out,
            n_signatures=n_signatures,
            n_samples=n_samples,
            n_genes_used=n_genes,
            gene_overlap=gene_overlap,
            method=result.get("method", self.backend),
            time_seconds=result.get("time", 0.0),
        )

    def _align_genes(
        self,
        expression: pd.DataFrame,
        signature: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        """Align genes between expression and signature."""
        # Normalize gene names to uppercase
        expr_copy = expression.copy()
        expr_copy.index = expr_copy.index.str.upper()
        expr_copy = expr_copy[~expr_copy.index.duplicated(keep="first")]

        sig_copy = signature.copy()
        sig_copy.index = sig_copy.index.str.upper()
        sig_copy = sig_copy[~sig_copy.index.duplicated(keep="first")]

        # Find common genes
        common_genes = list(set(expr_copy.index) & set(sig_copy.index))
        common_genes = sorted(common_genes)

        if len(common_genes) < 10:
            raise ValueError(
                f"Too few common genes: {len(common_genes)}. "
                f"Expression has {len(expr_copy)} genes, "
                f"signature has {len(sig_copy)} genes."
            )

        # Compute overlap
        gene_overlap = len(common_genes) / len(sig_copy)

        # Align
        expr_aligned = expr_copy.loc[common_genes]
        sig_aligned = sig_copy.loc[common_genes]

        return expr_aligned, sig_aligned, gene_overlap

    def _scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalize each column."""
        scaled = (df - df.mean()) / df.std(ddof=1)
        return scaled.fillna(0)


def run_ridge_inference(
    expression: Union[pd.DataFrame, np.ndarray],
    signature: pd.DataFrame,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 0,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    batch_size: int = 5000,
    verbose: bool = False,
    expression_gene_names: Optional[list[str]] = None,
    sample_names: Optional[list[str]] = None,
) -> ActivityResult:
    """
    Convenience function for activity inference.

    Args:
        expression: Expression matrix (genes x samples).
        signature: Signature matrix (genes x signatures).
        lambda_: Ridge regularization parameter.
        n_rand: Number of permutations.
        seed: Random seed.
        backend: Computation backend.
        batch_size: Batch size for large datasets.
        verbose: Print progress.
        expression_gene_names: Gene names for expression (if ndarray).
        sample_names: Sample names (if ndarray).

    Returns:
        ActivityResult with inferred activities.
    """
    inference = RidgeInference(
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        verbose=verbose,
    )
    return inference.run(
        expression,
        signature,
        expression_gene_names=expression_gene_names,
        sample_names=sample_names,
    )
