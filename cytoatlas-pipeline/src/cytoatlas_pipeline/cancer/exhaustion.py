"""
T cell exhaustion analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from cytoatlas_pipeline.differential.stratified import (
    StratifiedDifferential,
    DifferentialResult,
)


# Exhaustion markers
EXHAUSTION_SIGNATURES = ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX"]


@dataclass
class ExhaustionResult:
    """Result of exhaustion analysis."""

    cell_type: str
    exhaustion_score: float
    n_exhausted: int
    n_non_exhausted: int
    differential: Optional[DifferentialResult]


class ExhaustionAnalyzer:
    """Analyzes T cell exhaustion states."""

    def __init__(
        self,
        exhaustion_signatures: list[str] = None,
        score_threshold: float = 0.5,
    ):
        self.exhaustion_signatures = exhaustion_signatures or EXHAUSTION_SIGNATURES
        self.score_threshold = score_threshold

    def analyze(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        cell_type_col: str = "cell_type",
        t_cell_types: list[str] = None,
    ) -> list[ExhaustionResult]:
        """Analyze exhaustion in T cell populations."""
        if t_cell_types is None:
            t_cell_types = ["CD8+ T cell", "CD4+ T cell", "T cell"]

        common = list(set(activity.columns) & set(metadata.index))
        activity = activity[common]
        metadata = metadata.loc[common]

        results = []

        for ct in t_cell_types:
            ct_mask = metadata[cell_type_col].str.contains(ct, case=False, na=False)
            if ct_mask.sum() < 10:
                continue

            ct_samples = metadata.index[ct_mask].tolist()
            ct_activity = activity[[s for s in ct_samples if s in activity.columns]]

            # Compute exhaustion score
            exhaustion_sigs = [
                s for s in self.exhaustion_signatures
                if s in ct_activity.index
            ]
            if not exhaustion_sigs:
                continue

            exhaustion_scores = ct_activity.loc[exhaustion_sigs].mean(axis=0)
            mean_score = exhaustion_scores.mean()

            # Classify cells
            exhausted = exhaustion_scores > self.score_threshold
            n_exhausted = exhausted.sum()
            n_non_exhausted = len(exhausted) - n_exhausted

            # Differential if enough cells in both groups
            diff_result = None
            if n_exhausted >= 5 and n_non_exhausted >= 5:
                # Add exhaustion label to metadata
                meta_copy = metadata.loc[ct_samples].copy()
                meta_copy["exhausted"] = exhausted.values

                try:
                    diff = StratifiedDifferential()
                    diff_result = diff.compare(
                        activity=ct_activity,
                        metadata=meta_copy,
                        group_col="exhausted",
                        group1_value=True,
                        group2_value=False,
                    )
                except ValueError:
                    pass

            results.append(ExhaustionResult(
                cell_type=ct,
                exhaustion_score=mean_score,
                n_exhausted=n_exhausted,
                n_non_exhausted=n_non_exhausted,
                differential=diff_result,
            ))

        return results


def analyze_exhaustion(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    cell_type_col: str = "cell_type",
) -> list[ExhaustionResult]:
    """Convenience function for exhaustion analysis."""
    analyzer = ExhaustionAnalyzer()
    return analyzer.analyze(activity, metadata, cell_type_col)
