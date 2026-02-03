"""
Pseudobulk vs single-cell validation.

Compares pseudobulk aggregated activities with single-cell results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PseudobulkSCResult:
    """Result of pseudobulk vs single-cell comparison."""

    signature: str
    """Signature name."""

    correlation: float
    """Correlation between pseudobulk and SC mean."""

    pvalue: float
    """P-value for correlation."""

    mean_concordance: float
    """Concordance of means."""

    median_concordance: float
    """Concordance of medians."""

    n_groups: int
    """Number of groups compared."""


class PseudobulkSCValidator:
    """
    Validates consistency between pseudobulk and single-cell activity.

    For each cell type x sample group:
    - Computes pseudobulk activity (from aggregated expression)
    - Computes mean/median of single-cell activities
    - Assesses concordance

    Example:
        >>> validator = PseudobulkSCValidator()
        >>> result = validator.validate(
        ...     pseudobulk_activity=pb_activity,
        ...     singlecell_activity=sc_activity,
        ...     sc_metadata=sc_metadata,
        ...     group_cols=["sample", "cell_type"]
        ... )
    """

    def __init__(self, min_cells_per_group: int = 10, min_groups: int = 10):
        """
        Initialize validator.

        Args:
            min_cells_per_group: Minimum cells per group.
            min_groups: Minimum groups for comparison.
        """
        self.min_cells_per_group = min_cells_per_group
        self.min_groups = min_groups

    def validate(
        self,
        pseudobulk_activity: pd.DataFrame,
        singlecell_activity: pd.DataFrame,
        sc_metadata: pd.DataFrame,
        group_cols: list[str] = ["sample", "cell_type"],
    ) -> list[PseudobulkSCResult]:
        """
        Compare pseudobulk vs single-cell activities.

        Args:
            pseudobulk_activity: Pseudobulk activity (signatures x pb_samples).
            singlecell_activity: Single-cell activity (signatures x cells).
            sc_metadata: Single-cell metadata with group columns.
            group_cols: Columns defining groups (must match pb column names).

        Returns:
            List of comparison results.
        """
        # Group single cells
        sc_groups = sc_metadata.groupby(group_cols, observed=True).groups

        results = []

        for signature in pseudobulk_activity.index:
            if signature not in singlecell_activity.index:
                continue

            pb_vals = []
            sc_means = []
            sc_medians = []

            for group_key, cell_indices in sc_groups.items():
                if len(cell_indices) < self.min_cells_per_group:
                    continue

                # Build pseudobulk column name
                if isinstance(group_key, tuple):
                    pb_col = "_".join(str(g) for g in group_key)
                else:
                    pb_col = str(group_key)

                if pb_col not in pseudobulk_activity.columns:
                    continue

                # Get pseudobulk value
                pb_val = pseudobulk_activity.loc[signature, pb_col]

                # Get single-cell values
                cell_ids = [str(idx) for idx in cell_indices]
                sc_cells = [c for c in cell_ids if c in singlecell_activity.columns]

                if len(sc_cells) < self.min_cells_per_group:
                    continue

                sc_vals = singlecell_activity.loc[signature, sc_cells].values
                sc_mean = np.nanmean(sc_vals)
                sc_median = np.nanmedian(sc_vals)

                pb_vals.append(pb_val)
                sc_means.append(sc_mean)
                sc_medians.append(sc_median)

            if len(pb_vals) < self.min_groups:
                continue

            # Compute concordance
            pb_arr = np.array(pb_vals)
            sc_mean_arr = np.array(sc_means)
            sc_median_arr = np.array(sc_medians)

            r_mean, p_mean = stats.pearsonr(pb_arr, sc_mean_arr)
            r_median, _ = stats.pearsonr(pb_arr, sc_median_arr)

            results.append(PseudobulkSCResult(
                signature=signature,
                correlation=r_mean,
                pvalue=p_mean,
                mean_concordance=r_mean,
                median_concordance=r_median,
                n_groups=len(pb_vals),
            ))

        return results


def validate_pseudobulk_vs_sc(
    pseudobulk_activity: pd.DataFrame,
    singlecell_activity: pd.DataFrame,
    sc_metadata: pd.DataFrame,
    group_cols: list[str] = ["sample", "cell_type"],
) -> list[PseudobulkSCResult]:
    """
    Convenience function for pseudobulk vs SC validation.

    Args:
        pseudobulk_activity: Pseudobulk activity matrix.
        singlecell_activity: Single-cell activity matrix.
        sc_metadata: Single-cell metadata.
        group_cols: Group definition columns.

    Returns:
        List of comparison results.
    """
    validator = PseudobulkSCValidator()
    return validator.validate(
        pseudobulk_activity, singlecell_activity, sc_metadata, group_cols
    )
