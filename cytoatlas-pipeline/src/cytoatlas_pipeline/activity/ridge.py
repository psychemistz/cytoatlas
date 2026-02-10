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
SECACTPY_PATH = Path("/data/parks34/projects/1ridgesig/SecActpy")
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


@dataclass
class MultiSignatureResult:
    """Result of multi-signature activity inference."""

    results: dict[str, ActivityResult]
    """Results per signature type."""

    signature_names: list[str]
    """Names of signature types processed."""

    total_time_seconds: float
    """Total execution time."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def __getitem__(self, key: str) -> ActivityResult:
        """Get result by signature name."""
        return self.results[key]

    def keys(self) -> list[str]:
        """Get available signature names."""
        return list(self.results.keys())


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


def run_multi_signature_inference(
    expression: Union[pd.DataFrame, np.ndarray],
    signatures: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 42,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    batch_size: int = 5000,
    verbose: bool = True,
    expression_gene_names: Optional[list[str]] = None,
    sample_names: Optional[list[str]] = None,
    save_h5ad: bool = True,
) -> MultiSignatureResult:
    """
    Run activity inference with multiple signature types.

    Processes CytoSig, LinCytoSig, and SecAct signatures in sequence,
    optionally saving results as H5AD files.

    Args:
        expression: Expression matrix (genes x samples).
        signatures: List of signatures to run. Default: ["cytosig", "lincytosig", "secact"]
        output_dir: Directory to save H5AD outputs.
        lambda_: Ridge regularization parameter.
        n_rand: Number of permutations.
        seed: Random seed.
        backend: Computation backend.
        batch_size: Batch size for large datasets.
        verbose: Print progress.
        expression_gene_names: Gene names if expression is ndarray.
        sample_names: Sample names if expression is ndarray.
        save_h5ad: Whether to save results as H5AD files.

    Returns:
        MultiSignatureResult with results per signature type.
    """
    import time
    import anndata as ad

    # Import signature loaders
    try:
        from secactpy import load_cytosig, load_secact, load_lincytosig
    except ImportError:
        raise ImportError("SecActpy not found. Ensure it's installed or path is correct.")

    # Default signatures
    if signatures is None:
        signatures = ["cytosig", "lincytosig", "secact"]

    # Signature loaders
    SIGNATURE_LOADERS = {
        "cytosig": load_cytosig,
        "lincytosig": load_lincytosig,
        "secact": load_secact,
    }

    # Validate signatures
    for sig in signatures:
        if sig not in SIGNATURE_LOADERS:
            raise ValueError(f"Unknown signature: {sig}. Available: {list(SIGNATURE_LOADERS.keys())}")

    # Convert expression if needed
    if isinstance(expression, np.ndarray):
        if expression_gene_names is None:
            raise ValueError("expression_gene_names required for ndarray input")
        if sample_names is None:
            sample_names = [f"sample_{i}" for i in range(expression.shape[1])]
        expression_df = pd.DataFrame(
            expression, index=expression_gene_names, columns=sample_names
        )
    else:
        expression_df = expression
        sample_names = list(expression_df.columns)

    # Create output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference for each signature type
    results = {}
    total_start = time.time()

    for sig_name in signatures:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {sig_name.upper()} inference...")
            print('='*60)

        sig_start = time.time()

        # Load signature matrix
        sig_matrix = SIGNATURE_LOADERS[sig_name]()
        if verbose:
            print(f"  Signature shape: {sig_matrix.shape}")

        # Run inference
        inference = RidgeInference(
            lambda_=lambda_,
            n_rand=n_rand,
            seed=seed,
            backend=backend,
            batch_size=batch_size,
            verbose=verbose,
        )
        result = inference.run(expression_df, sig_matrix)
        results[sig_name] = result

        if verbose:
            print(f"  Gene overlap: {result.gene_overlap:.1%}")
            print(f"  Time: {time.time() - sig_start:.1f}s")

        # Save as H5AD
        if save_h5ad and output_dir is not None:
            h5ad_path = output_dir / f"{sig_name}_activity.h5ad"

            # Create AnnData (samples × signatures)
            activity_adata = ad.AnnData(
                X=result.zscore.T.values.astype(np.float32),
                obs=pd.DataFrame(index=result.sample_names),
                var=pd.DataFrame(index=result.signature_names),
            )
            activity_adata.layers['beta'] = result.beta.T.values.astype(np.float32)
            activity_adata.layers['se'] = result.se.T.values.astype(np.float32)
            activity_adata.layers['pvalue'] = result.pvalue.T.values.astype(np.float32)
            activity_adata.uns['signature'] = sig_name
            activity_adata.uns['gene_overlap'] = result.gene_overlap
            activity_adata.uns['n_genes_used'] = result.n_genes_used
            activity_adata.uns['method'] = result.method

            activity_adata.write_h5ad(h5ad_path, compression='gzip')
            if verbose:
                print(f"  Saved: {h5ad_path}")

    total_time = time.time() - total_start

    return MultiSignatureResult(
        results=results,
        signature_names=signatures,
        total_time_seconds=total_time,
        metadata={
            'n_samples': len(sample_names),
            'n_genes_input': len(expression_df),
            'lambda': lambda_,
            'n_rand': n_rand,
            'seed': seed,
        },
    )
