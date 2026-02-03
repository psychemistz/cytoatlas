"""
Activity harmonization across atlases.

Batch correction and normalization for cross-atlas comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class HarmonizationResult:
    """Result of activity harmonization."""

    harmonized: pd.DataFrame
    """Harmonized activity matrix."""

    batch_effects: dict[str, float]
    """Estimated batch effects per atlas."""

    method: str
    """Harmonization method used."""

    n_atlases: int
    """Number of atlases harmonized."""


class ActivityHarmonizer:
    """
    Harmonizes activities across multiple atlases.

    Removes batch effects while preserving biological variation.

    Example:
        >>> harmonizer = ActivityHarmonizer(method="combat")
        >>> result = harmonizer.harmonize({
        ...     "CIMA": cima_activity,
        ...     "Inflammation": inflam_activity,
        ...     "scAtlas": scatlas_activity
        ... })
    """

    def __init__(
        self,
        method: Literal["zscore", "quantile", "combat"] = "zscore",
        reference_atlas: Optional[str] = None,
    ):
        """
        Initialize harmonizer.

        Args:
            method: Harmonization method.
            reference_atlas: Reference atlas for batch correction.
        """
        self.method = method
        self.reference_atlas = reference_atlas

    def harmonize(
        self,
        activities: dict[str, pd.DataFrame],
    ) -> HarmonizationResult:
        """
        Harmonize activities from multiple atlases.

        Args:
            activities: Dict of atlas name to activity matrix.

        Returns:
            HarmonizationResult with harmonized activities.
        """
        if self.method == "zscore":
            return self._harmonize_zscore(activities)
        elif self.method == "quantile":
            return self._harmonize_quantile(activities)
        elif self.method == "combat":
            return self._harmonize_combat(activities)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _harmonize_zscore(
        self,
        activities: dict[str, pd.DataFrame],
    ) -> HarmonizationResult:
        """Z-score normalization per atlas."""
        harmonized_dfs = []
        batch_effects = {}

        for atlas_name, df in activities.items():
            # Z-score per signature (row)
            mean = df.mean(axis=1)
            std = df.std(axis=1, ddof=1)
            std = std.replace(0, 1)

            df_zscore = df.subtract(mean, axis=0).divide(std, axis=0)
            df_zscore.columns = [f"{atlas_name}_{c}" for c in df_zscore.columns]

            harmonized_dfs.append(df_zscore)
            batch_effects[atlas_name] = float(mean.mean())

        # Combine
        harmonized = pd.concat(harmonized_dfs, axis=1)

        return HarmonizationResult(
            harmonized=harmonized,
            batch_effects=batch_effects,
            method="zscore",
            n_atlases=len(activities),
        )

    def _harmonize_quantile(
        self,
        activities: dict[str, pd.DataFrame],
    ) -> HarmonizationResult:
        """Quantile normalization."""
        # Combine all atlases
        all_dfs = []
        atlas_mapping = {}

        for atlas_name, df in activities.items():
            df_copy = df.copy()
            df_copy.columns = [f"{atlas_name}_{c}" for c in df_copy.columns]
            all_dfs.append(df_copy)
            for col in df_copy.columns:
                atlas_mapping[col] = atlas_name

        combined = pd.concat(all_dfs, axis=1)

        # Quantile normalize
        ranks = combined.rank(axis=1, method="average")
        means = combined.mean(axis=1)

        harmonized = ranks.apply(
            lambda row: np.interp(
                row.values,
                np.arange(1, len(row) + 1),
                np.sort(combined.loc[row.name].values)
            ),
            axis=1,
            result_type="broadcast"
        )

        # Estimate batch effects
        batch_effects = {}
        for atlas_name in activities.keys():
            atlas_cols = [c for c in harmonized.columns if c.startswith(atlas_name)]
            batch_effects[atlas_name] = float(combined[atlas_cols].mean().mean())

        return HarmonizationResult(
            harmonized=harmonized,
            batch_effects=batch_effects,
            method="quantile",
            n_atlases=len(activities),
        )

    def _harmonize_combat(
        self,
        activities: dict[str, pd.DataFrame],
    ) -> HarmonizationResult:
        """Combat batch correction (simplified)."""
        # Combine all atlases
        all_dfs = []
        batch_labels = []

        for atlas_name, df in activities.items():
            df_copy = df.copy()
            df_copy.columns = [f"{atlas_name}_{c}" for c in df_copy.columns]
            all_dfs.append(df_copy)
            batch_labels.extend([atlas_name] * df_copy.shape[1])

        combined = pd.concat(all_dfs, axis=1)

        # Simplified ComBat: per-batch mean/variance adjustment
        unique_batches = list(activities.keys())
        ref_batch = self.reference_atlas or unique_batches[0]

        batch_effects = {}
        harmonized = combined.copy()

        for batch in unique_batches:
            batch_cols = [c for c in combined.columns if c.startswith(batch)]

            if batch == ref_batch:
                batch_effects[batch] = 0.0
                continue

            ref_cols = [c for c in combined.columns if c.startswith(ref_batch)]

            # Compute shift to match reference
            batch_mean = combined[batch_cols].mean(axis=1)
            ref_mean = combined[ref_cols].mean(axis=1)

            shift = ref_mean - batch_mean
            batch_effects[batch] = float(shift.mean())

            # Apply correction
            harmonized[batch_cols] = combined[batch_cols].add(shift, axis=0)

        return HarmonizationResult(
            harmonized=harmonized,
            batch_effects=batch_effects,
            method="combat",
            n_atlases=len(activities),
        )


def harmonize_activities(
    activities: dict[str, pd.DataFrame],
    method: Literal["zscore", "quantile", "combat"] = "zscore",
) -> HarmonizationResult:
    """
    Convenience function for activity harmonization.

    Args:
        activities: Dict of atlas name to activity matrix.
        method: Harmonization method.

    Returns:
        HarmonizationResult.
    """
    harmonizer = ActivityHarmonizer(method=method)
    return harmonizer.harmonize(activities)
