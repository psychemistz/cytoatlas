"""
Tissue specificity scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SpecificityResult:
    """Tissue specificity scores for a signature."""

    signature: str
    tau_score: float  # Tissue specificity index (0-1)
    max_organ: str
    max_activity: float
    organ_scores: dict[str, float]


class TissueSpecificityScorer:
    """Computes tissue specificity scores using tau index."""

    def __init__(self, min_expression: float = 0.0):
        self.min_expression = min_expression

    def compute_tau(self, values: np.ndarray) -> float:
        """Compute tissue specificity index (tau).

        Tau ranges from 0 (ubiquitous) to 1 (tissue-specific).
        Formula: tau = sum(1 - x_i/x_max) / (n - 1)

        Parameters
        ----------
        values : np.ndarray
            Activity values across tissues

        Returns
        -------
        float
            Tau specificity index
        """
        values = np.asarray(values)
        n = len(values)

        if n <= 1:
            return 0.0

        # Shift to positive if needed (activity can be negative)
        values_shifted = values - values.min() + 1e-10
        x_max = values_shifted.max()

        if x_max <= 0:
            return 0.0

        tau = np.sum(1 - values_shifted / x_max) / (n - 1)
        return float(tau)

    def score(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        organ_col: str = "organ",
    ) -> list[SpecificityResult]:
        """Compute specificity scores for all signatures.

        Parameters
        ----------
        activity : pd.DataFrame
            Activity matrix (signatures × samples)
        metadata : pd.DataFrame
            Sample metadata with organ annotations
        organ_col : str
            Column containing organ labels

        Returns
        -------
        list[SpecificityResult]
            Specificity scores per signature
        """
        # Align samples
        common = list(set(activity.columns) & set(metadata.index))
        activity = activity[common]
        metadata = metadata.loc[common]

        # Compute mean activity per organ
        organ_means = {}
        for organ in metadata[organ_col].unique():
            organ_samples = metadata.index[metadata[organ_col] == organ].tolist()
            organ_activity = activity[[s for s in organ_samples if s in activity.columns]]
            if organ_activity.shape[1] > 0:
                organ_means[organ] = organ_activity.mean(axis=1)

        if not organ_means:
            return []

        organ_df = pd.DataFrame(organ_means)

        results = []
        for sig in activity.index:
            values = organ_df.loc[sig].values
            tau = self.compute_tau(values)

            organ_scores = organ_df.loc[sig].to_dict()
            max_organ = max(organ_scores, key=organ_scores.get)
            max_activity = organ_scores[max_organ]

            results.append(SpecificityResult(
                signature=sig,
                tau_score=tau,
                max_organ=max_organ,
                max_activity=max_activity,
                organ_scores=organ_scores,
            ))

        # Sort by specificity
        results.sort(key=lambda x: x.tau_score, reverse=True)

        return results

    def get_organ_specific(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        organ_col: str = "organ",
        tau_threshold: float = 0.6,
    ) -> dict[str, list[str]]:
        """Get organ-specific signatures above tau threshold.

        Returns
        -------
        dict[str, list[str]]
            Signatures grouped by their most specific organ
        """
        results = self.score(activity, metadata, organ_col)

        organ_specific = {}
        for r in results:
            if r.tau_score >= tau_threshold:
                organ = r.max_organ
                if organ not in organ_specific:
                    organ_specific[organ] = []
                organ_specific[organ].append(r.signature)

        return organ_specific


def compute_specificity(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    organ_col: str = "organ",
) -> list[SpecificityResult]:
    """Convenience function for specificity scoring."""
    scorer = TissueSpecificityScorer()
    return scorer.score(activity, metadata, organ_col)
