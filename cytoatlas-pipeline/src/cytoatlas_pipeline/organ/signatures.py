"""
Organ-specific signature analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OrganSignature:
    """Signature activity profile for an organ."""

    organ: str
    mean_activity: pd.Series
    std_activity: pd.Series
    n_samples: int
    top_signatures: list[str]
    suppressed_signatures: list[str]


class OrganSignatureAnalyzer:
    """Analyzes organ-specific cytokine/protein activity patterns."""

    def __init__(
        self,
        top_n: int = 10,
        min_samples: int = 5,
    ):
        self.top_n = top_n
        self.min_samples = min_samples

    def analyze(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        organ_col: str = "organ",
    ) -> dict[str, OrganSignature]:
        """Compute organ-specific signatures.

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
        dict[str, OrganSignature]
            Organ signatures keyed by organ name
        """
        # Align samples
        common = list(set(activity.columns) & set(metadata.index))
        activity = activity[common]
        metadata = metadata.loc[common]

        results = {}
        organs = metadata[organ_col].unique()

        for organ in organs:
            organ_mask = metadata[organ_col] == organ
            organ_samples = metadata.index[organ_mask].tolist()

            if len(organ_samples) < self.min_samples:
                continue

            organ_activity = activity[[s for s in organ_samples if s in activity.columns]]

            if organ_activity.shape[1] < self.min_samples:
                continue

            # Compute statistics
            mean_act = organ_activity.mean(axis=1)
            std_act = organ_activity.std(axis=1)

            # Top activated and suppressed
            top_sigs = mean_act.nlargest(self.top_n).index.tolist()
            suppressed_sigs = mean_act.nsmallest(self.top_n).index.tolist()

            results[organ] = OrganSignature(
                organ=organ,
                mean_activity=mean_act,
                std_activity=std_act,
                n_samples=organ_activity.shape[1],
                top_signatures=top_sigs,
                suppressed_signatures=suppressed_sigs,
            )

        return results

    def compare_organs(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        organ_col: str = "organ",
    ) -> pd.DataFrame:
        """Compare activity between all organ pairs.

        Returns
        -------
        pd.DataFrame
            Pairwise organ similarity matrix (correlation)
        """
        signatures = self.analyze(activity, metadata, organ_col)

        if len(signatures) < 2:
            return pd.DataFrame()

        organs = list(signatures.keys())
        n_organs = len(organs)

        # Build mean activity matrix
        mean_matrix = pd.DataFrame(
            {org: signatures[org].mean_activity for org in organs}
        )

        # Compute pairwise correlations
        similarity = mean_matrix.corr(method="spearman")

        return similarity


def compute_organ_signatures(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    organ_col: str = "organ",
) -> dict[str, OrganSignature]:
    """Convenience function for organ signature analysis."""
    analyzer = OrganSignatureAnalyzer()
    return analyzer.analyze(activity, metadata, organ_col)
