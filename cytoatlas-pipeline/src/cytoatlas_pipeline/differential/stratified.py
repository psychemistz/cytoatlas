"""
Stratified differential analysis.

Multi-level stratification for complex comparisons
(e.g., disease vs healthy, stratified by cell type).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd

from cytoatlas_pipeline.differential.wilcoxon import WilcoxonTest
from cytoatlas_pipeline.differential.ttest import TTest
from cytoatlas_pipeline.differential.effect_size import EffectSizeCalculator
from cytoatlas_pipeline.differential.fdr import FDRCorrector


@dataclass
class DifferentialResult:
    """Result of differential analysis."""

    activity_diff: pd.DataFrame
    """Activity difference (group1 - group2)."""

    pvalue: pd.DataFrame
    """P-values."""

    qvalue: pd.DataFrame
    """FDR-corrected q-values."""

    statistic: pd.DataFrame
    """Test statistics."""

    cohens_d: Optional[pd.DataFrame] = None
    """Cohen's d effect size."""

    n_group1: int = 0
    """Samples in group 1."""

    n_group2: int = 0
    """Samples in group 2."""

    method: str = "wilcoxon"
    """Test method used."""

    comparison: str = ""
    """Description of comparison."""

    stratification: Optional[str] = None
    """Stratification level (e.g., cell type)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def get_significant(
        self,
        qvalue_threshold: float = 0.05,
        activity_diff_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Get significant differential signatures.

        Args:
            qvalue_threshold: FDR threshold.
            activity_diff_threshold: Minimum absolute activity difference.

        Returns:
            DataFrame with significant results.
        """
        results = []

        for sig in self.activity_diff.index:
            q = self.qvalue.loc[sig].values[0] if isinstance(self.qvalue.loc[sig], pd.Series) else self.qvalue.loc[sig]
            diff = self.activity_diff.loc[sig].values[0] if isinstance(self.activity_diff.loc[sig], pd.Series) else self.activity_diff.loc[sig]

            if q < qvalue_threshold:
                if activity_diff_threshold is None or abs(diff) >= activity_diff_threshold:
                    results.append({
                        "signature": sig,
                        "activity_diff": diff,
                        "pvalue": self.pvalue.loc[sig].values[0] if isinstance(self.pvalue.loc[sig], pd.Series) else self.pvalue.loc[sig],
                        "qvalue": q,
                        "cohens_d": self.cohens_d.loc[sig].values[0] if self.cohens_d is not None else None,
                    })

        return pd.DataFrame(results)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single DataFrame."""
        df = pd.DataFrame({
            "activity_diff": self.activity_diff.iloc[:, 0] if self.activity_diff.shape[1] == 1 else self.activity_diff.mean(axis=1),
            "pvalue": self.pvalue.iloc[:, 0] if self.pvalue.shape[1] == 1 else self.pvalue.mean(axis=1),
            "qvalue": self.qvalue.iloc[:, 0] if self.qvalue.shape[1] == 1 else self.qvalue.mean(axis=1),
            "statistic": self.statistic.iloc[:, 0] if self.statistic.shape[1] == 1 else self.statistic.mean(axis=1),
        })

        if self.cohens_d is not None:
            df["cohens_d"] = self.cohens_d.iloc[:, 0] if self.cohens_d.shape[1] == 1 else self.cohens_d.mean(axis=1)

        df["neg_log10_pval"] = -np.log10(df["pvalue"].clip(lower=1e-300))

        return df


class StratifiedDifferential:
    """
    Performs stratified differential analysis.

    Supports multi-level comparisons with consistent result formatting.

    Example:
        >>> diff = StratifiedDifferential()
        >>> result = diff.compare(
        ...     activity=activity_df,
        ...     metadata=metadata_df,
        ...     group_col="disease",
        ...     group1_value="disease",
        ...     group2_value="healthy",
        ...     stratify_by="cell_type"
        ... )
    """

    def __init__(
        self,
        method: Literal["wilcoxon", "ttest"] = "wilcoxon",
        fdr_method: str = "fdr_bh",
        min_samples: int = 3,
        backend: Literal["auto", "numpy", "cupy"] = "auto",
    ):
        """
        Initialize stratified differential.

        Args:
            method: Statistical test method.
            fdr_method: FDR correction method.
            min_samples: Minimum samples per group.
            backend: Computation backend.
        """
        self.method = method
        self.fdr_method = fdr_method
        self.min_samples = min_samples
        self.backend = backend

        self.effect_calc = EffectSizeCalculator()
        self.fdr_corrector = FDRCorrector(method=fdr_method)

    def compare(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        group_col: str,
        group1_value: Any,
        group2_value: Any,
        stratify_by: Optional[str] = None,
    ) -> Union[DifferentialResult, dict[str, DifferentialResult]]:
        """
        Compare groups, optionally stratified.

        Args:
            activity: Activity matrix (signatures x samples).
            metadata: Sample metadata.
            group_col: Column defining groups.
            group1_value: Value for group 1.
            group2_value: Value for group 2.
            stratify_by: Column to stratify by (e.g., cell type).

        Returns:
            DifferentialResult or dict of results per stratum.
        """
        if stratify_by is not None:
            return self._compare_stratified(
                activity, metadata, group_col, group1_value, group2_value, stratify_by
            )
        else:
            return self._compare_simple(
                activity, metadata, group_col, group1_value, group2_value
            )

    def _compare_simple(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        group_col: str,
        group1_value: Any,
        group2_value: Any,
    ) -> DifferentialResult:
        """Simple two-group comparison."""
        # Align activity and metadata
        common = list(set(activity.columns) & set(metadata.index))
        activity = activity[common]
        metadata = metadata.loc[common]

        # Split by group
        group1_mask = metadata[group_col] == group1_value
        group2_mask = metadata[group_col] == group2_value

        group1_samples = metadata.index[group1_mask].tolist()
        group2_samples = metadata.index[group2_mask].tolist()

        if len(group1_samples) < self.min_samples or len(group2_samples) < self.min_samples:
            raise ValueError(
                f"Insufficient samples: group1={len(group1_samples)}, "
                f"group2={len(group2_samples)}, min={self.min_samples}"
            )

        group1_data = activity[group1_samples]
        group2_data = activity[group2_samples]

        # Statistical test
        if self.method == "wilcoxon":
            test = WilcoxonTest(backend=self.backend)
        else:
            test = TTest(backend=self.backend)

        stat, pval = test.test(group1_data, group2_data)

        # Effect sizes
        activity_diff = self.effect_calc.activity_difference(group1_data, group2_data)
        cohens_d = self.effect_calc.cohens_d(group1_data, group2_data)

        # FDR correction
        qval = self.fdr_corrector.correct(pval)

        # Format results
        signature_names = list(activity.index)

        return DifferentialResult(
            activity_diff=pd.DataFrame(
                activity_diff, index=signature_names, columns=["activity_diff"]
            ),
            pvalue=pd.DataFrame(pval, index=signature_names, columns=["pvalue"]),
            qvalue=pd.DataFrame(qval, index=signature_names, columns=["qvalue"]),
            statistic=pd.DataFrame(stat, index=signature_names, columns=["statistic"]),
            cohens_d=pd.DataFrame(cohens_d, index=signature_names, columns=["cohens_d"]),
            n_group1=len(group1_samples),
            n_group2=len(group2_samples),
            method=self.method,
            comparison=f"{group1_value}_vs_{group2_value}",
        )

    def _compare_stratified(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        group_col: str,
        group1_value: Any,
        group2_value: Any,
        stratify_by: str,
    ) -> dict[str, DifferentialResult]:
        """Stratified comparison (e.g., by cell type)."""
        results = {}
        strata = metadata[stratify_by].unique()

        for stratum in strata:
            stratum_mask = metadata[stratify_by] == stratum
            stratum_meta = metadata.loc[stratum_mask]
            stratum_samples = stratum_meta.index.tolist()
            stratum_activity = activity[[s for s in stratum_samples if s in activity.columns]]

            if len(stratum_activity.columns) < self.min_samples * 2:
                continue

            try:
                result = self._compare_simple(
                    stratum_activity, stratum_meta, group_col, group1_value, group2_value
                )
                result.stratification = stratum
                results[stratum] = result
            except ValueError:
                # Skip strata with insufficient samples
                continue

        return results


def run_differential(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
    group1_value: Any,
    group2_value: Any,
    stratify_by: Optional[str] = None,
    method: Literal["wilcoxon", "ttest"] = "wilcoxon",
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> Union[DifferentialResult, dict[str, DifferentialResult]]:
    """
    Convenience function for differential analysis.

    Args:
        activity: Activity matrix.
        metadata: Sample metadata.
        group_col: Group column.
        group1_value: Group 1 value.
        group2_value: Group 2 value.
        stratify_by: Stratification column.
        method: Test method.
        backend: Computation backend.

    Returns:
        DifferentialResult or dict per stratum.
    """
    diff = StratifiedDifferential(method=method, backend=backend)
    return diff.compare(
        activity, metadata, group_col, group1_value, group2_value, stratify_by
    )
