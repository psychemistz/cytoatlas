"""
Bootstrap resampling aggregation.

Provides uncertainty estimates through bootstrap resampling of cells
within each aggregation group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sp

from cytoatlas_pipeline.aggregation.base import (
    AggregatedData,
    AggregationConfig,
    AggregationStrategy,
)


@dataclass
class ResamplingConfig(AggregationConfig):
    """Configuration for bootstrap resampling."""

    n_bootstrap: int = 100
    """Number of bootstrap samples."""

    sample_fraction: float = 0.8
    """Fraction of cells to sample in each bootstrap."""

    seed: int = 42
    """Random seed for reproducibility."""

    return_all_samples: bool = False
    """Return all bootstrap samples (vs just mean/std)."""


@dataclass
class BootstrapResult:
    """Result of bootstrap aggregation."""

    mean: pd.DataFrame
    """Mean across bootstrap samples."""

    std: pd.DataFrame
    """Standard deviation across bootstrap samples."""

    ci_lower: pd.DataFrame
    """Lower confidence interval (2.5%)."""

    ci_upper: pd.DataFrame
    """Upper confidence interval (97.5%)."""

    samples: Optional[list[pd.DataFrame]] = None
    """Individual bootstrap samples (if return_all_samples=True)."""


class BootstrapAggregator(AggregationStrategy):
    """
    Bootstrap resampling aggregation with uncertainty estimates.

    Creates multiple aggregated profiles by resampling cells,
    providing confidence intervals for expression estimates.

    Example:
        >>> config = ResamplingConfig(n_bootstrap=100)
        >>> aggregator = BootstrapAggregator(config)
        >>> result = aggregator.aggregate_with_ci(X, obs, var)
        >>> print(f"CI width: {(result.ci_upper - result.ci_lower).mean().mean():.3f}")
    """

    def __init__(self, config: Optional[ResamplingConfig] = None):
        """
        Initialize bootstrap aggregator.

        Args:
            config: Resampling configuration.
        """
        if config is None:
            config = ResamplingConfig()
        super().__init__(config)
        self.config: ResamplingConfig = config

    def aggregate(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: pd.DataFrame,
        var: pd.DataFrame,
    ) -> AggregatedData:
        """
        Aggregate with bootstrap (returns mean).

        Args:
            X: Expression matrix (cells x genes).
            obs: Cell metadata.
            var: Gene metadata.

        Returns:
            Mean aggregated data across bootstrap samples.
        """
        result = self.aggregate_with_ci(X, obs, var)

        # Return mean as the primary result
        metadata_df = self._create_metadata(obs)

        return AggregatedData(
            expression=result.mean,
            metadata=metadata_df,
            gene_names=list(var.index),
            n_units=result.mean.shape[1],
            n_genes=len(var),
            aggregation_type="bootstrap_mean",
            config=self.config,
            stats={
                "n_bootstrap": self.config.n_bootstrap,
                "mean_ci_width": float((result.ci_upper - result.ci_lower).mean().mean()),
            },
        )

    def aggregate_with_ci(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: pd.DataFrame,
        var: pd.DataFrame,
    ) -> BootstrapResult:
        """
        Aggregate with full confidence intervals.

        Args:
            X: Expression matrix (cells x genes).
            obs: Cell metadata.
            var: Gene metadata.

        Returns:
            BootstrapResult with mean, std, and CIs.
        """
        rng = np.random.default_rng(self.config.seed)

        cell_type_col = self.config.cell_type_col
        sample_col = self.config.sample_col

        # Get groups
        obs_reset = obs.reset_index(drop=True)
        groups = obs_reset.groupby([sample_col, cell_type_col], observed=True).groups

        gene_names = list(var.index)
        n_genes = len(gene_names)

        # Identify valid groups
        valid_groups = {
            k: v for k, v in groups.items() if len(v) >= self.config.min_cells
        }
        col_names = [f"{s}_{ct}" for (s, ct) in valid_groups.keys()]
        n_groups = len(col_names)

        # Bootstrap samples storage
        bootstrap_samples = np.zeros((self.config.n_bootstrap, n_genes, n_groups))

        for b in range(self.config.n_bootstrap):
            for i, ((sample, cell_type), indices) in enumerate(valid_groups.items()):
                idx_array = np.array(list(indices), dtype=np.int64)

                # Resample
                n_sample = max(1, int(len(idx_array) * self.config.sample_fraction))
                resampled_idx = rng.choice(idx_array, size=n_sample, replace=True)

                # Sum expression
                if sp.issparse(X):
                    group_sum = np.asarray(X[resampled_idx].sum(axis=0)).ravel()
                else:
                    group_sum = X[resampled_idx].sum(axis=0)

                bootstrap_samples[b, :, i] = group_sum

        # Process each bootstrap sample
        processed_samples = []
        for b in range(self.config.n_bootstrap):
            df = pd.DataFrame(
                bootstrap_samples[b], index=gene_names, columns=col_names
            )
            df = self.process_expression(df)
            processed_samples.append(df.values)

        processed_array = np.stack(processed_samples, axis=0)

        # Compute statistics
        mean_vals = np.mean(processed_array, axis=0)
        std_vals = np.std(processed_array, axis=0, ddof=1)
        ci_lower_vals = np.percentile(processed_array, 2.5, axis=0)
        ci_upper_vals = np.percentile(processed_array, 97.5, axis=0)

        # Create DataFrames
        mean_df = pd.DataFrame(mean_vals, index=gene_names, columns=col_names)
        std_df = pd.DataFrame(std_vals, index=gene_names, columns=col_names)
        ci_lower_df = pd.DataFrame(ci_lower_vals, index=gene_names, columns=col_names)
        ci_upper_df = pd.DataFrame(ci_upper_vals, index=gene_names, columns=col_names)

        # Optionally return all samples
        all_samples = None
        if self.config.return_all_samples:
            all_samples = [
                pd.DataFrame(processed_array[b], index=gene_names, columns=col_names)
                for b in range(self.config.n_bootstrap)
            ]

        return BootstrapResult(
            mean=mean_df,
            std=std_df,
            ci_lower=ci_lower_df,
            ci_upper=ci_upper_df,
            samples=all_samples,
        )

    def _create_metadata(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Create metadata DataFrame from obs."""
        cell_type_col = self.config.cell_type_col
        sample_col = self.config.sample_col

        obs_reset = obs.reset_index(drop=True)
        groups = obs_reset.groupby([sample_col, cell_type_col], observed=True).groups

        metadata_rows = []
        for (sample, cell_type), indices in groups.items():
            if len(indices) >= self.config.min_cells:
                metadata_rows.append({
                    "sample": sample,
                    "cell_type": cell_type,
                    "n_cells": len(indices),
                })

        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.index = [
            f"{r['sample']}_{r['cell_type']}" for _, r in metadata_df.iterrows()
        ]

        return metadata_df
