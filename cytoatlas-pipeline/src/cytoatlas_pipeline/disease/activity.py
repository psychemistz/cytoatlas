"""
Disease-specific activity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DiseaseActivityResult:
    """Result of disease activity analysis."""

    disease: str
    mean_activity: pd.Series
    std_activity: pd.Series
    n_samples: int
    top_signatures: list[str]


class DiseaseActivityAnalyzer:
    """Analyzes disease-specific activity patterns."""

    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples

    def analyze(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        disease_col: str = "disease",
    ) -> dict[str, DiseaseActivityResult]:
        """Analyze activity patterns per disease."""
        common = list(set(activity.columns) & set(metadata.index))
        activity = activity[common]
        metadata = metadata.loc[common]

        results = {}
        for disease in metadata[disease_col].unique():
            mask = metadata[disease_col] == disease
            samples = metadata.index[mask].tolist()

            if len(samples) < self.min_samples:
                continue

            disease_activity = activity[samples]
            mean_act = disease_activity.mean(axis=1)
            std_act = disease_activity.std(axis=1)

            top_sigs = mean_act.nlargest(10).index.tolist()

            results[disease] = DiseaseActivityResult(
                disease=disease,
                mean_activity=mean_act,
                std_activity=std_act,
                n_samples=len(samples),
                top_signatures=top_sigs,
            )

        return results


def compute_disease_activity(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    disease_col: str = "disease",
) -> dict[str, DiseaseActivityResult]:
    """Convenience function for disease activity analysis."""
    analyzer = DiseaseActivityAnalyzer()
    return analyzer.analyze(activity, metadata, disease_col)
