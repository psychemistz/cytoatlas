"""
Correlation with continuous variables (age, BMI, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from cytoatlas_pipeline.correlation.spearman import SpearmanCorrelator


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""

    rho: pd.DataFrame
    """Correlation coefficients (signatures x features)."""

    pvalue: pd.DataFrame
    """P-values (signatures x features)."""

    qvalue: pd.DataFrame
    """FDR-corrected q-values (signatures x features)."""

    n_samples: int
    """Number of samples used."""

    method: str
    """Correlation method used."""

    feature_names: list[str]
    """Names of correlated features."""

    signature_names: list[str]
    """Names of signatures."""

    stats: dict[str, Any] = field(default_factory=dict)
    """Additional statistics."""

    def get_significant(
        self,
        qvalue_threshold: float = 0.05,
        rho_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Get significant correlations.

        Args:
            qvalue_threshold: FDR threshold.
            rho_threshold: Minimum absolute correlation.

        Returns:
            DataFrame with significant results.
        """
        results = []

        for sig in self.signature_names:
            for feat in self.feature_names:
                q = self.qvalue.loc[sig, feat]
                r = self.rho.loc[sig, feat]

                if q < qvalue_threshold:
                    if rho_threshold is None or abs(r) >= rho_threshold:
                        results.append({
                            "signature": sig,
                            "feature": feat,
                            "rho": r,
                            "pvalue": self.pvalue.loc[sig, feat],
                            "qvalue": q,
                        })

        return pd.DataFrame(results)

    def to_long_format(self) -> pd.DataFrame:
        """Convert to long format DataFrame."""
        records = []
        for sig in self.signature_names:
            for feat in self.feature_names:
                records.append({
                    "signature": sig,
                    "feature": feat,
                    "rho": self.rho.loc[sig, feat],
                    "pvalue": self.pvalue.loc[sig, feat],
                    "qvalue": self.qvalue.loc[sig, feat],
                })
        return pd.DataFrame(records)


class ContinuousCorrelator:
    """
    Correlates activity with continuous phenotype variables.

    Handles binning, stratification, and FDR correction.

    Example:
        >>> correlator = ContinuousCorrelator()
        >>> result = correlator.correlate(
        ...     activity=activity_df,
        ...     features=metadata[["age", "bmi"]],
        ...     method="spearman"
        ... )
    """

    def __init__(
        self,
        method: Literal["spearman", "pearson"] = "spearman",
        fdr_method: str = "fdr_bh",
        min_samples: int = 10,
        backend: Literal["auto", "numpy", "cupy"] = "auto",
    ):
        """
        Initialize continuous correlator.

        Args:
            method: Correlation method.
            fdr_method: FDR correction method.
            min_samples: Minimum samples required.
            backend: Computation backend.
        """
        self.method = method
        self.fdr_method = fdr_method
        self.min_samples = min_samples
        self.backend = backend

    def correlate(
        self,
        activity: pd.DataFrame,
        features: pd.DataFrame,
        sample_mapping: Optional[dict[str, str]] = None,
    ) -> CorrelationResult:
        """
        Correlate activity with continuous features.

        Args:
            activity: Activity matrix (signatures x samples).
            features: Feature matrix (samples x features).
            sample_mapping: Map activity columns to feature index.

        Returns:
            CorrelationResult.
        """
        # Align samples
        if sample_mapping is not None:
            # Map activity columns to feature rows
            common = [c for c in activity.columns if sample_mapping.get(c) in features.index]
            activity_aligned = activity[common]
            feature_idx = [sample_mapping[c] for c in common]
            features_aligned = features.loc[feature_idx]
        else:
            # Direct matching
            common = list(set(activity.columns) & set(features.index))
            if len(common) < self.min_samples:
                raise ValueError(
                    f"Too few common samples: {len(common)} < {self.min_samples}"
                )
            activity_aligned = activity[common]
            features_aligned = features.loc[common]

        n_samples = len(common)

        # Compute correlations
        if self.method == "spearman":
            correlator = SpearmanCorrelator(backend=self.backend)
        else:
            from cytoatlas_pipeline.correlation.pearson import PearsonCorrelator
            correlator = PearsonCorrelator(backend=self.backend)

        # activity: signatures x samples, features.T: features x samples
        rho, pval = correlator.correlate(
            activity_aligned.values,
            features_aligned.T.values,
        )

        # Create DataFrames
        signature_names = list(activity_aligned.index)
        feature_names = list(features_aligned.columns)

        rho_df = pd.DataFrame(rho, index=signature_names, columns=feature_names)
        pval_df = pd.DataFrame(pval, index=signature_names, columns=feature_names)

        # FDR correction
        qval_flat = multipletests(pval.ravel(), method=self.fdr_method)[1]
        qval_df = pd.DataFrame(
            qval_flat.reshape(pval.shape),
            index=signature_names,
            columns=feature_names,
        )

        return CorrelationResult(
            rho=rho_df,
            pvalue=pval_df,
            qvalue=qval_df,
            n_samples=n_samples,
            method=self.method,
            feature_names=feature_names,
            signature_names=signature_names,
        )

    def correlate_binned(
        self,
        activity: pd.DataFrame,
        feature: pd.Series,
        n_bins: int = 5,
    ) -> pd.DataFrame:
        """
        Correlate activity with binned continuous feature.

        Useful for age/BMI binned analyses.

        Args:
            activity: Activity matrix (signatures x samples).
            feature: Continuous feature series.
            n_bins: Number of bins.

        Returns:
            DataFrame with bin-wise statistics.
        """
        # Create bins
        bins = pd.qcut(feature, n_bins, labels=False, duplicates="drop")

        results = []
        for bin_idx in sorted(bins.unique()):
            bin_mask = bins == bin_idx
            bin_samples = feature.index[bin_mask]

            # Get activity for bin
            common = list(set(activity.columns) & set(bin_samples))
            if len(common) < 3:
                continue

            activity_bin = activity[common]
            feature_bin = feature.loc[common]

            # Mean and range for this bin
            bin_info = {
                "bin": int(bin_idx),
                "n_samples": len(common),
                "feature_min": feature_bin.min(),
                "feature_max": feature_bin.max(),
                "feature_mean": feature_bin.mean(),
            }

            # Activity statistics per signature
            for sig in activity_bin.index:
                bin_info[f"{sig}_mean"] = activity_bin.loc[sig].mean()
                bin_info[f"{sig}_std"] = activity_bin.loc[sig].std()

            results.append(bin_info)

        return pd.DataFrame(results)


def correlate_with_continuous(
    activity: pd.DataFrame,
    features: pd.DataFrame,
    method: Literal["spearman", "pearson"] = "spearman",
    fdr_method: str = "fdr_bh",
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> CorrelationResult:
    """
    Convenience function for continuous correlation.

    Args:
        activity: Activity matrix (signatures x samples).
        features: Feature matrix (samples x features).
        method: Correlation method.
        fdr_method: FDR correction method.
        backend: Computation backend.

    Returns:
        CorrelationResult.
    """
    correlator = ContinuousCorrelator(
        method=method,
        fdr_method=fdr_method,
        backend=backend,
    )
    return correlator.correlate(activity, features)
