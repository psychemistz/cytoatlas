"""
Immune infiltration analysis in tumor microenvironment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class InfiltrationResult:
    """Result of immune infiltration analysis."""

    sample: str
    immune_fraction: float
    cell_type_proportions: dict[str, float]
    dominant_immune_type: str
    activity_score: float


class ImmuneInfiltrationAnalyzer:
    """Analyzes immune cell infiltration in tumors."""

    IMMUNE_TYPES = [
        "T cell", "B cell", "NK cell", "Macrophage", "Monocyte",
        "Dendritic cell", "Neutrophil", "Mast cell"
    ]

    def __init__(
        self,
        immune_types: list[str] = None,
    ):
        self.immune_types = immune_types or self.IMMUNE_TYPES

    def analyze(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        cell_type_col: str = "cell_type",
        sample_col: str = "sample",
    ) -> list[InfiltrationResult]:
        """Analyze immune infiltration per sample."""
        common = list(set(activity.columns) & set(metadata.index))
        activity = activity[common]
        metadata = metadata.loc[common]

        results = []
        for sample in metadata[sample_col].unique():
            sample_mask = metadata[sample_col] == sample
            sample_meta = metadata.loc[sample_mask]

            # Count cell types
            ct_counts = sample_meta[cell_type_col].value_counts()
            total_cells = ct_counts.sum()

            # Immune fraction
            immune_counts = sum(
                ct_counts.get(it, 0)
                for it in self.immune_types
                if any(it.lower() in ct.lower() for ct in ct_counts.index)
            )
            immune_fraction = immune_counts / total_cells if total_cells > 0 else 0

            # Proportions
            proportions = (ct_counts / total_cells).to_dict()

            # Dominant immune type
            immune_props = {
                ct: prop for ct, prop in proportions.items()
                if any(it.lower() in ct.lower() for it in self.immune_types)
            }
            dominant = max(immune_props, key=immune_props.get) if immune_props else "None"

            # Activity score (mean of immune-related signatures)
            sample_cols = sample_meta.index.tolist()
            sample_activity = activity[[c for c in sample_cols if c in activity.columns]]
            activity_score = sample_activity.mean().mean()

            results.append(InfiltrationResult(
                sample=sample,
                immune_fraction=immune_fraction,
                cell_type_proportions=proportions,
                dominant_immune_type=dominant,
                activity_score=activity_score,
            ))

        return results


def compute_infiltration(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    cell_type_col: str = "cell_type",
    sample_col: str = "sample",
) -> list[InfiltrationResult]:
    """Convenience function for infiltration analysis."""
    analyzer = ImmuneInfiltrationAnalyzer()
    return analyzer.analyze(activity, metadata, cell_type_col, sample_col)
