"""
Blood biochemistry and metabolite correlation analysis.

Specialized for CIMA-style biochemistry marker correlations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cytoatlas_pipeline.correlation.continuous import (
    ContinuousCorrelator,
    CorrelationResult,
)


# Common biochemistry markers
BIOCHEMISTRY_MARKERS = [
    "ALT",
    "AST",
    "ALP",
    "GGT",
    "Albumin",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "BUN",
    "Creatinine",
    "eGFR",
    "Uric_Acid",
    "Glucose",
    "HbA1c",
    "Total_Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "CRP",
    "ESR",
]


@dataclass
class BiochemistryResult(CorrelationResult):
    """Result with biochemistry-specific metadata."""

    marker_categories: dict[str, str] = None
    """Category for each marker (liver, kidney, etc.)."""

    def __post_init__(self):
        if self.marker_categories is None:
            self.marker_categories = {}

    def get_category_summary(self) -> pd.DataFrame:
        """Summarize correlations by marker category."""
        records = []
        for cat in set(self.marker_categories.values()):
            markers = [m for m, c in self.marker_categories.items() if c == cat]
            markers_in_result = [m for m in markers if m in self.feature_names]

            if not markers_in_result:
                continue

            # Count significant correlations
            for sig in self.signature_names:
                n_sig = sum(
                    self.qvalue.loc[sig, m] < 0.05 for m in markers_in_result
                )
                mean_rho = self.rho.loc[sig, markers_in_result].mean()

                records.append({
                    "signature": sig,
                    "category": cat,
                    "n_markers": len(markers_in_result),
                    "n_significant": n_sig,
                    "mean_rho": mean_rho,
                })

        return pd.DataFrame(records)


class BiochemistryCorrelator:
    """
    Specialized correlator for blood biochemistry markers.

    Handles marker categories and provides domain-specific analysis.

    Example:
        >>> correlator = BiochemistryCorrelator()
        >>> result = correlator.correlate(
        ...     activity=activity_df,
        ...     biochemistry=biochemistry_df,
        ...     agg_metadata=pseudobulk_meta
        ... )
    """

    # Marker categories
    MARKER_CATEGORIES = {
        "ALT": "liver",
        "AST": "liver",
        "ALP": "liver",
        "GGT": "liver",
        "Albumin": "liver",
        "Total_Bilirubin": "liver",
        "Direct_Bilirubin": "liver",
        "BUN": "kidney",
        "Creatinine": "kidney",
        "eGFR": "kidney",
        "Uric_Acid": "kidney",
        "Glucose": "metabolic",
        "HbA1c": "metabolic",
        "Total_Cholesterol": "lipid",
        "HDL": "lipid",
        "LDL": "lipid",
        "Triglycerides": "lipid",
        "CRP": "inflammatory",
        "ESR": "inflammatory",
    }

    def __init__(
        self,
        method: Literal["spearman", "pearson"] = "spearman",
        fdr_method: str = "fdr_bh",
        min_samples: int = 10,
        backend: Literal["auto", "numpy", "cupy"] = "auto",
    ):
        """
        Initialize biochemistry correlator.

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
        biochemistry: pd.DataFrame,
        agg_metadata: Optional[pd.DataFrame] = None,
        sample_col: str = "sample",
        cell_type: Optional[str] = None,
    ) -> BiochemistryResult:
        """
        Correlate activity with biochemistry markers.

        Args:
            activity: Activity matrix (signatures x pseudobulk samples).
            biochemistry: Biochemistry data (samples x markers).
            agg_metadata: Aggregation metadata with sample mapping.
            sample_col: Column in metadata with sample IDs.
            cell_type: Filter to specific cell type.

        Returns:
            BiochemistryResult with correlations.
        """
        # Build sample mapping from pseudobulk to biochemistry
        if agg_metadata is not None:
            # Map activity columns to original sample IDs
            if cell_type is not None:
                mask = agg_metadata["cell_type"] == cell_type
                valid_cols = agg_metadata.index[mask]
                activity = activity[[c for c in activity.columns if c in valid_cols]]
                agg_metadata = agg_metadata.loc[valid_cols]

            sample_mapping = dict(zip(agg_metadata.index, agg_metadata[sample_col]))
        else:
            sample_mapping = None

        # Use base correlator
        correlator = ContinuousCorrelator(
            method=self.method,
            fdr_method=self.fdr_method,
            min_samples=self.min_samples,
            backend=self.backend,
        )

        result = correlator.correlate(activity, biochemistry, sample_mapping)

        # Get marker categories for markers in result
        marker_categories = {
            m: self.MARKER_CATEGORIES.get(m, "other")
            for m in result.feature_names
        }

        return BiochemistryResult(
            rho=result.rho,
            pvalue=result.pvalue,
            qvalue=result.qvalue,
            n_samples=result.n_samples,
            method=result.method,
            feature_names=result.feature_names,
            signature_names=result.signature_names,
            stats=result.stats,
            marker_categories=marker_categories,
        )

    def correlate_by_celltype(
        self,
        activity: pd.DataFrame,
        biochemistry: pd.DataFrame,
        agg_metadata: pd.DataFrame,
        sample_col: str = "sample",
        cell_type_col: str = "cell_type",
    ) -> dict[str, BiochemistryResult]:
        """
        Correlate for each cell type separately.

        Args:
            activity: Activity matrix.
            biochemistry: Biochemistry data.
            agg_metadata: Aggregation metadata.
            sample_col: Sample column.
            cell_type_col: Cell type column.

        Returns:
            Dict mapping cell type to BiochemistryResult.
        """
        results = {}
        cell_types = agg_metadata[cell_type_col].unique()

        for ct in cell_types:
            try:
                result = self.correlate(
                    activity=activity,
                    biochemistry=biochemistry,
                    agg_metadata=agg_metadata,
                    sample_col=sample_col,
                    cell_type=ct,
                )
                results[ct] = result
            except ValueError:
                # Skip if too few samples
                continue

        return results


def correlate_with_biochemistry(
    activity: pd.DataFrame,
    biochemistry: pd.DataFrame,
    agg_metadata: Optional[pd.DataFrame] = None,
    method: Literal["spearman", "pearson"] = "spearman",
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> BiochemistryResult:
    """
    Convenience function for biochemistry correlation.

    Args:
        activity: Activity matrix.
        biochemistry: Biochemistry data.
        agg_metadata: Aggregation metadata.
        method: Correlation method.
        backend: Computation backend.

    Returns:
        BiochemistryResult.
    """
    correlator = BiochemistryCorrelator(method=method, backend=backend)
    return correlator.correlate(activity, biochemistry, agg_metadata)
