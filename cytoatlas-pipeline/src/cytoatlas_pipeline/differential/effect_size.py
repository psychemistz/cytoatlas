"""
Effect size computation for activity differences.

IMPORTANT: Activity values are z-scores and can be negative.
We use activity_diff (simple difference), NOT log2 fold change.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


class EffectSizeCalculator:
    """
    Computes effect sizes for differential activity.

    CRITICAL: Activity values are z-scores (can be negative).
    We compute activity_diff = mean_group1 - mean_group2, NOT log2FC.

    Example:
        >>> calc = EffectSizeCalculator()
        >>> diff = calc.activity_difference(group1, group2)
        >>> cohens_d = calc.cohens_d(group1, group2)
    """

    def activity_difference(
        self,
        group1: Union[np.ndarray, pd.DataFrame],
        group2: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Compute activity difference between groups.

        activity_diff = mean(group1) - mean(group2)

        Args:
            group1: First group (features x samples).
            group2: Second group (features x samples).

        Returns:
            Array of activity differences.
        """
        if isinstance(group1, pd.DataFrame):
            group1 = group1.values
        if isinstance(group2, pd.DataFrame):
            group2 = group2.values

        mean1 = np.nanmean(group1, axis=1)
        mean2 = np.nanmean(group2, axis=1)

        return mean1 - mean2

    def cohens_d(
        self,
        group1: Union[np.ndarray, pd.DataFrame],
        group2: Union[np.ndarray, pd.DataFrame],
        pooled: bool = True,
    ) -> np.ndarray:
        """
        Compute Cohen's d effect size.

        d = (mean1 - mean2) / pooled_std

        Args:
            group1: First group.
            group2: Second group.
            pooled: Use pooled standard deviation.

        Returns:
            Array of Cohen's d values.
        """
        if isinstance(group1, pd.DataFrame):
            group1 = group1.values
        if isinstance(group2, pd.DataFrame):
            group2 = group2.values

        n1 = group1.shape[1]
        n2 = group2.shape[1]

        mean1 = np.nanmean(group1, axis=1)
        mean2 = np.nanmean(group2, axis=1)
        var1 = np.nanvar(group1, axis=1, ddof=1)
        var2 = np.nanvar(group2, axis=1, ddof=1)

        if pooled:
            # Pooled standard deviation
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            pooled_std = np.sqrt(pooled_var)
        else:
            # Simple average of stds
            pooled_std = np.sqrt((var1 + var2) / 2)

        # Avoid division by zero
        pooled_std = np.where(pooled_std > 0, pooled_std, 1)

        return (mean1 - mean2) / pooled_std

    def hedges_g(
        self,
        group1: Union[np.ndarray, pd.DataFrame],
        group2: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Compute Hedges' g (bias-corrected Cohen's d).

        Args:
            group1: First group.
            group2: Second group.

        Returns:
            Array of Hedges' g values.
        """
        d = self.cohens_d(group1, group2)

        n1 = group1.shape[1] if group1.ndim > 1 else len(group1)
        n2 = group2.shape[1] if group2.ndim > 1 else len(group2)

        # Correction factor
        df = n1 + n2 - 2
        correction = 1 - (3 / (4 * df - 1))

        return d * correction

    def glass_delta(
        self,
        group1: Union[np.ndarray, pd.DataFrame],
        group2: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Compute Glass's delta (using control group std).

        delta = (mean1 - mean2) / std(group2)

        Args:
            group1: Treatment group.
            group2: Control group (used for std).

        Returns:
            Array of Glass's delta values.
        """
        if isinstance(group1, pd.DataFrame):
            group1 = group1.values
        if isinstance(group2, pd.DataFrame):
            group2 = group2.values

        mean1 = np.nanmean(group1, axis=1)
        mean2 = np.nanmean(group2, axis=1)
        std2 = np.nanstd(group2, axis=1, ddof=1)

        std2 = np.where(std2 > 0, std2, 1)

        return (mean1 - mean2) / std2


def compute_activity_diff(
    group1: Union[np.ndarray, pd.DataFrame],
    group2: Union[np.ndarray, pd.DataFrame],
) -> np.ndarray:
    """
    Compute activity difference (NOT log2FC).

    Activity values are z-scores that can be negative.

    Example:
        >>> # exhausted T cells vs non-exhausted
        >>> # exhausted mean = -2, non-exhausted mean = -4
        >>> # activity_diff = -2 - (-4) = +2
        >>> # Correctly shows higher activity in exhausted

    Args:
        group1: First group (features x samples).
        group2: Second group (features x samples).

    Returns:
        Array of activity differences.
    """
    calc = EffectSizeCalculator()
    return calc.activity_difference(group1, group2)


def compute_cohens_d(
    group1: Union[np.ndarray, pd.DataFrame],
    group2: Union[np.ndarray, pd.DataFrame],
    pooled: bool = True,
) -> np.ndarray:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group.
        group2: Second group.
        pooled: Use pooled std.

    Returns:
        Array of Cohen's d values.
    """
    calc = EffectSizeCalculator()
    return calc.cohens_d(group1, group2, pooled=pooled)
