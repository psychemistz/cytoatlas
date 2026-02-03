"""
Tumor vs adjacent tissue comparison.
"""

from __future__ import annotations

import pandas as pd

from cytoatlas_pipeline.differential.stratified import (
    StratifiedDifferential,
    DifferentialResult,
)


class TumorAdjacentAnalyzer:
    """Analyzes tumor vs adjacent tissue activity differences."""

    def __init__(self, method: str = "wilcoxon"):
        self.method = method

    def compare(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        tissue_col: str = "tissue_type",
        tumor_label: str = "tumor",
        adjacent_label: str = "adjacent",
        stratify_by: str = None,
    ) -> DifferentialResult | dict[str, DifferentialResult]:
        """Compare tumor vs adjacent tissue."""
        diff = StratifiedDifferential(method=self.method)
        return diff.compare(
            activity=activity,
            metadata=metadata,
            group_col=tissue_col,
            group1_value=tumor_label,
            group2_value=adjacent_label,
            stratify_by=stratify_by,
        )


def tumor_vs_adjacent(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    tissue_col: str = "tissue_type",
) -> DifferentialResult:
    """Convenience function for tumor vs adjacent comparison."""
    analyzer = TumorAdjacentAnalyzer()
    return analyzer.compare(activity, metadata, tissue_col)
