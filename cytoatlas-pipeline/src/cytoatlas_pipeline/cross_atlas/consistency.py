"""
Cross-atlas consistency scoring.

Measures agreement between atlases for activity patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ConsistencyResult:
    """Result of consistency analysis."""

    signature: str
    """Signature name."""

    pairwise_correlations: dict[str, float]
    """Pairwise correlations between atlases."""

    mean_correlation: float
    """Mean pairwise correlation."""

    cv: float
    """Coefficient of variation across atlases."""

    rank_concordance: float
    """Kendall's W for rank concordance."""

    consistency_score: float
    """Overall consistency score (0-1)."""


class ConsistencyScorer:
    """
    Computes consistency scores for cross-atlas comparisons.

    Example:
        >>> scorer = ConsistencyScorer()
        >>> results = scorer.compute(activities)
    """

    def __init__(
        self,
        cell_type_col: Optional[str] = "cell_type",
    ):
        """
        Initialize consistency scorer.

        Args:
            cell_type_col: Cell type column for grouping.
        """
        self.cell_type_col = cell_type_col

    def compute(
        self,
        activities: dict[str, pd.DataFrame],
        metadata: Optional[dict[str, pd.DataFrame]] = None,
    ) -> list[ConsistencyResult]:
        """
        Compute consistency for each signature.

        Args:
            activities: Dict of atlas name to activity matrix.
            metadata: Optional metadata per atlas.

        Returns:
            List of consistency results.
        """
        # Get common signatures
        all_signatures = None
        for df in activities.values():
            if all_signatures is None:
                all_signatures = set(df.index)
            else:
                all_signatures &= set(df.index)

        if not all_signatures:
            return []

        atlas_names = list(activities.keys())
        results = []

        for signature in all_signatures:
            # Get cell-type means if metadata available
            if metadata and self.cell_type_col:
                atlas_ct_means = self._get_celltype_means(
                    signature, activities, metadata
                )
            else:
                # Use overall means
                atlas_ct_means = {
                    atlas: {"overall": df.loc[signature].mean()}
                    for atlas, df in activities.items()
                    if signature in df.index
                }

            if len(atlas_ct_means) < 2:
                continue

            # Compute pairwise correlations
            pairwise_corrs = {}
            corr_values = []

            for i, a1 in enumerate(atlas_names):
                for a2 in atlas_names[i+1:]:
                    if a1 not in atlas_ct_means or a2 not in atlas_ct_means:
                        continue

                    # Find common cell types
                    common_cts = set(atlas_ct_means[a1].keys()) & set(atlas_ct_means[a2].keys())
                    if len(common_cts) < 3:
                        continue

                    v1 = [atlas_ct_means[a1][ct] for ct in common_cts]
                    v2 = [atlas_ct_means[a2][ct] for ct in common_cts]

                    r, _ = stats.pearsonr(v1, v2)
                    pair_name = f"{a1}_vs_{a2}"
                    pairwise_corrs[pair_name] = r
                    corr_values.append(r)

            if not corr_values:
                continue

            # Mean correlation
            mean_corr = np.mean(corr_values)

            # CV of activity across atlases
            atlas_means = [
                np.mean(list(atlas_ct_means[a].values()))
                for a in atlas_ct_means
            ]
            cv = np.std(atlas_means) / (np.abs(np.mean(atlas_means)) + 1e-10)

            # Rank concordance (simplified Kendall's W)
            rank_concordance = self._compute_rank_concordance(
                signature, atlas_ct_means
            )

            # Overall consistency score
            consistency = (
                0.4 * max(0, mean_corr) +
                0.3 * (1 - min(cv, 1)) +
                0.3 * rank_concordance
            )

            results.append(ConsistencyResult(
                signature=signature,
                pairwise_correlations=pairwise_corrs,
                mean_correlation=mean_corr,
                cv=cv,
                rank_concordance=rank_concordance,
                consistency_score=consistency,
            ))

        # Sort by consistency
        results.sort(key=lambda x: x.consistency_score, reverse=True)

        return results

    def _get_celltype_means(
        self,
        signature: str,
        activities: dict[str, pd.DataFrame],
        metadata: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, float]]:
        """Get cell-type means for each atlas."""
        atlas_ct_means = {}

        for atlas_name, activity_df in activities.items():
            if atlas_name not in metadata or signature not in activity_df.index:
                continue

            meta = metadata[atlas_name]
            if self.cell_type_col not in meta.columns:
                continue

            ct_means = {}
            for ct in meta[self.cell_type_col].unique():
                ct_mask = meta[self.cell_type_col] == ct
                ct_samples = meta.index[ct_mask].tolist()
                ct_samples = [s for s in ct_samples if s in activity_df.columns]
                if ct_samples:
                    ct_means[ct.lower()] = activity_df.loc[signature, ct_samples].mean()

            if ct_means:
                atlas_ct_means[atlas_name] = ct_means

        return atlas_ct_means

    def _compute_rank_concordance(
        self,
        signature: str,
        atlas_ct_means: dict[str, dict[str, float]],
    ) -> float:
        """Compute rank concordance (simplified Kendall's W)."""
        if len(atlas_ct_means) < 2:
            return 0.0

        # Find common cell types
        common_cts = None
        for ct_means in atlas_ct_means.values():
            if common_cts is None:
                common_cts = set(ct_means.keys())
            else:
                common_cts &= set(ct_means.keys())

        if not common_cts or len(common_cts) < 3:
            return 0.0

        common_cts = sorted(common_cts)
        n_cts = len(common_cts)
        n_atlases = len(atlas_ct_means)

        # Get ranks for each atlas
        ranks_matrix = np.zeros((n_cts, n_atlases))
        for j, atlas in enumerate(atlas_ct_means.keys()):
            values = [atlas_ct_means[atlas].get(ct, 0) for ct in common_cts]
            ranks_matrix[:, j] = stats.rankdata(values)

        # Compute Kendall's W
        R = ranks_matrix.sum(axis=1)
        R_mean = R.mean()
        S = ((R - R_mean) ** 2).sum()
        W = 12 * S / (n_atlases ** 2 * (n_cts ** 3 - n_cts))

        return min(W, 1.0)


def compute_consistency(
    activities: dict[str, pd.DataFrame],
    metadata: Optional[dict[str, pd.DataFrame]] = None,
) -> list[ConsistencyResult]:
    """
    Convenience function for consistency scoring.

    Args:
        activities: Dict of atlas name to activity matrix.
        metadata: Optional metadata per atlas.

    Returns:
        List of consistency results.
    """
    scorer = ConsistencyScorer()
    return scorer.compute(activities, metadata)
