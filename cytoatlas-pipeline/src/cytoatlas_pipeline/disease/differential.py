"""
Disease vs healthy differential analysis.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from cytoatlas_pipeline.differential.stratified import (
    StratifiedDifferential,
    DifferentialResult,
)


class DiseaseDifferential:
    """Disease vs healthy differential analysis."""

    def __init__(
        self,
        healthy_label: str = "healthy",
        method: str = "wilcoxon",
    ):
        self.healthy_label = healthy_label
        self.method = method

    def compare(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        disease: str,
        disease_col: str = "disease",
        cell_type_col: Optional[str] = None,
    ) -> DifferentialResult | dict[str, DifferentialResult]:
        """Compare disease vs healthy."""
        diff = StratifiedDifferential(method=self.method)
        return diff.compare(
            activity=activity,
            metadata=metadata,
            group_col=disease_col,
            group1_value=disease,
            group2_value=self.healthy_label,
            stratify_by=cell_type_col,
        )


def disease_vs_healthy(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    disease: str,
    disease_col: str = "disease",
    healthy_label: str = "healthy",
) -> DifferentialResult:
    """Convenience function for disease vs healthy comparison."""
    diff = DiseaseDifferential(healthy_label=healthy_label)
    return diff.compare(activity, metadata, disease, disease_col)
