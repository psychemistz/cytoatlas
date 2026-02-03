"""
Meta-analysis across multiple atlases.

Combines results from multiple atlases using weighted averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MetaAnalysisResult:
    """Result of meta-analysis."""

    combined_effect: pd.Series
    """Combined effect sizes."""

    combined_se: pd.Series
    """Combined standard errors."""

    combined_pvalue: pd.Series
    """Combined p-values."""

    heterogeneity_q: pd.Series
    """Cochran's Q for heterogeneity."""

    heterogeneity_i2: pd.Series
    """I² heterogeneity index."""

    n_studies: int
    """Number of studies combined."""

    weights: dict[str, pd.Series]
    """Weights per study."""


class MetaAnalyzer:
    """
    Meta-analysis combining results across atlases.

    Supports fixed-effect and random-effects models.

    Example:
        >>> analyzer = MetaAnalyzer(method="random")
        >>> result = analyzer.analyze(effects, standard_errors)
    """

    def __init__(
        self,
        method: Literal["fixed", "random"] = "random",
        weight_by: Literal["inverse_variance", "sample_size"] = "inverse_variance",
    ):
        """
        Initialize meta-analyzer.

        Args:
            method: Meta-analysis method.
            weight_by: Weighting scheme.
        """
        self.method = method
        self.weight_by = weight_by

    def analyze(
        self,
        effects: dict[str, pd.DataFrame],
        standard_errors: Optional[dict[str, pd.DataFrame]] = None,
        sample_sizes: Optional[dict[str, int]] = None,
    ) -> MetaAnalysisResult:
        """
        Perform meta-analysis.

        Args:
            effects: Effect sizes per atlas (signatures x samples).
            standard_errors: Standard errors per atlas.
            sample_sizes: Sample sizes per atlas.

        Returns:
            MetaAnalysisResult.
        """
        # Get common signatures
        all_signatures = None
        for df in effects.values():
            if all_signatures is None:
                all_signatures = set(df.index)
            else:
                all_signatures &= set(df.index)

        signatures = sorted(all_signatures)
        atlas_names = list(effects.keys())
        n_studies = len(atlas_names)

        # Compute mean effect per atlas per signature
        atlas_means = {}
        atlas_ses = {}

        for atlas_name, effect_df in effects.items():
            # Mean across samples
            atlas_means[atlas_name] = effect_df.loc[signatures].mean(axis=1)

            # SE: either provided or computed from effect variance
            if standard_errors and atlas_name in standard_errors:
                se_df = standard_errors[atlas_name]
                atlas_ses[atlas_name] = se_df.loc[signatures].mean(axis=1)
            else:
                # Approximate SE from std
                atlas_ses[atlas_name] = effect_df.loc[signatures].std(axis=1) / np.sqrt(
                    effect_df.shape[1]
                )

        # Stack into arrays
        means_matrix = pd.DataFrame(atlas_means).loc[signatures]
        ses_matrix = pd.DataFrame(atlas_ses).loc[signatures]

        # Compute weights
        if self.weight_by == "inverse_variance":
            # w_i = 1 / SE_i^2
            weights_matrix = 1 / (ses_matrix ** 2 + 1e-10)
        else:
            # Weight by sample size
            if sample_sizes:
                weights_matrix = pd.DataFrame({
                    atlas: pd.Series(
                        [sample_sizes.get(atlas, 1)] * len(signatures),
                        index=signatures
                    )
                    for atlas in atlas_names
                })
            else:
                weights_matrix = pd.DataFrame(
                    np.ones((len(signatures), n_studies)),
                    index=signatures,
                    columns=atlas_names,
                )

        # Normalize weights
        weight_sums = weights_matrix.sum(axis=1)
        weights_norm = weights_matrix.div(weight_sums, axis=0)

        # Combined effect: weighted average
        combined_effect = (means_matrix * weights_norm).sum(axis=1)

        # Combined SE
        if self.method == "fixed":
            combined_se = np.sqrt(1 / weights_matrix.sum(axis=1))
        else:
            # Random effects: add between-study variance
            tau2 = self._estimate_tau2(means_matrix, ses_matrix)
            combined_se = np.sqrt(
                1 / (1 / (ses_matrix ** 2 + tau2 + 1e-10)).sum(axis=1)
            )

        # Z-test p-values
        z = combined_effect / (combined_se + 1e-10)
        combined_pvalue = 2 * stats.norm.sf(np.abs(z))

        # Heterogeneity tests
        Q, I2 = self._compute_heterogeneity(means_matrix, ses_matrix, combined_effect)

        return MetaAnalysisResult(
            combined_effect=combined_effect,
            combined_se=combined_se,
            combined_pvalue=pd.Series(combined_pvalue, index=signatures),
            heterogeneity_q=Q,
            heterogeneity_i2=I2,
            n_studies=n_studies,
            weights={k: weights_norm[k] for k in atlas_names},
        )

    def _estimate_tau2(
        self,
        means: pd.DataFrame,
        ses: pd.DataFrame,
    ) -> pd.Series:
        """Estimate between-study variance (DerSimonian-Laird)."""
        k = means.shape[1]
        weights = 1 / (ses ** 2 + 1e-10)

        # Weighted mean
        theta = (means * weights).sum(axis=1) / weights.sum(axis=1)

        # Q statistic
        Q = (weights * (means.subtract(theta, axis=0) ** 2)).sum(axis=1)

        # Expected Q under homogeneity
        C = weights.sum(axis=1) - (weights ** 2).sum(axis=1) / weights.sum(axis=1)

        # Tau^2 estimate
        tau2 = (Q - (k - 1)) / C
        tau2 = tau2.clip(lower=0)

        return tau2

    def _compute_heterogeneity(
        self,
        means: pd.DataFrame,
        ses: pd.DataFrame,
        combined: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Compute heterogeneity statistics."""
        k = means.shape[1]
        weights = 1 / (ses ** 2 + 1e-10)

        # Cochran's Q
        Q = (weights * (means.subtract(combined, axis=0) ** 2)).sum(axis=1)

        # I² = (Q - df) / Q * 100%
        I2 = ((Q - (k - 1)) / Q).clip(lower=0, upper=1)

        return Q, I2


def run_meta_analysis(
    effects: dict[str, pd.DataFrame],
    standard_errors: Optional[dict[str, pd.DataFrame]] = None,
    method: Literal["fixed", "random"] = "random",
) -> MetaAnalysisResult:
    """
    Convenience function for meta-analysis.

    Args:
        effects: Effect sizes per atlas.
        standard_errors: Standard errors per atlas.
        method: Meta-analysis method.

    Returns:
        MetaAnalysisResult.
    """
    analyzer = MetaAnalyzer(method=method)
    return analyzer.analyze(effects, standard_errors)
