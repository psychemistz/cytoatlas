"""
Pseudobulk aggregation.

Aggregates single-cell expression by cell type and sample combinations,
creating a "pseudo-bulk" expression profile for each combination.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sp

from cytoatlas_pipeline.aggregation.base import (
    AggregatedData,
    AggregationConfig,
    AggregationStrategy,
)


class PseudobulkAggregator(AggregationStrategy):
    """
    Aggregates expression by cell type × sample combinations.

    This is the standard approach for differential analysis, where
    each biological sample contributes one data point per cell type.

    Example:
        >>> aggregator = PseudobulkAggregator()
        >>> result = aggregator.aggregate(X, obs, var)
        >>> print(f"Created {result.n_units} pseudobulk samples")
    """

    def __init__(
        self,
        config: Optional[AggregationConfig] = None,
        method: str = "sum",
    ):
        """
        Initialize pseudobulk aggregator.

        Args:
            config: Aggregation configuration.
            method: Aggregation method ("sum", "mean", "median").
        """
        super().__init__(config)
        self.method = method

    def aggregate(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: pd.DataFrame,
        var: pd.DataFrame,
    ) -> AggregatedData:
        """
        Aggregate by cell type × sample.

        Args:
            X: Expression matrix (cells x genes).
            obs: Cell metadata with cell_type and sample columns.
            var: Gene metadata.

        Returns:
            Pseudobulk aggregated data.
        """
        cell_type_col = self.config.cell_type_col
        sample_col = self.config.sample_col

        if cell_type_col not in obs.columns:
            raise ValueError(f"Cell type column '{cell_type_col}' not in obs")
        if sample_col not in obs.columns:
            raise ValueError(f"Sample column '{sample_col}' not in obs")

        # Get groups
        obs_reset = obs.reset_index(drop=True)
        groups = obs_reset.groupby([sample_col, cell_type_col], observed=True).groups

        gene_names = list(var.index)
        aggregated = {}
        metadata_rows = []
        stats = {"cells_per_group": {}}

        for (sample, cell_type), indices in groups.items():
            n_cells = len(indices)

            if n_cells < self.config.min_cells:
                continue

            col_name = f"{sample}_{cell_type}"
            idx_array = np.array(list(indices), dtype=np.int64)

            # Aggregate
            if sp.issparse(X):
                if self.method == "sum":
                    group_expr = np.asarray(X[idx_array].sum(axis=0)).ravel()
                elif self.method == "mean":
                    group_expr = np.asarray(X[idx_array].mean(axis=0)).ravel()
                else:
                    group_expr = np.median(X[idx_array].toarray(), axis=0)
            else:
                if self.method == "sum":
                    group_expr = X[idx_array].sum(axis=0)
                elif self.method == "mean":
                    group_expr = X[idx_array].mean(axis=0)
                else:
                    group_expr = np.median(X[idx_array], axis=0)

            # Ensure 1D
            group_expr = np.asarray(group_expr).ravel()

            aggregated[col_name] = group_expr
            metadata_rows.append({
                "sample": sample,
                "cell_type": cell_type,
                "n_cells": n_cells,
            })
            stats["cells_per_group"][col_name] = n_cells

        # Create expression DataFrame
        expr_df = pd.DataFrame(aggregated, index=gene_names)

        # Process expression
        expr_df = self.process_expression(expr_df)

        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.index = [f"{r['sample']}_{r['cell_type']}" for _, r in metadata_df.iterrows()]

        return AggregatedData(
            expression=expr_df,
            metadata=metadata_df,
            gene_names=gene_names,
            n_units=len(aggregated),
            n_genes=len(gene_names),
            aggregation_type="pseudobulk",
            config=self.config,
            stats=stats,
        )


def aggregate_pseudobulk(
    X: Union[np.ndarray, sp.spmatrix],
    obs: pd.DataFrame,
    var: pd.DataFrame,
    cell_type_col: str = "cell_type",
    sample_col: str = "sample",
    min_cells: int = 10,
    normalize: bool = True,
    log_transform: bool = True,
    differential: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for pseudobulk aggregation.

    Args:
        X: Expression matrix (cells x genes).
        obs: Cell metadata.
        var: Gene metadata.
        cell_type_col: Column for cell type.
        sample_col: Column for sample.
        min_cells: Minimum cells per group.
        normalize: Apply TPM normalization.
        log_transform: Apply log2 transform.
        differential: Subtract row mean.

    Returns:
        Tuple of (expression_df, metadata_df).
    """
    config = AggregationConfig(
        cell_type_col=cell_type_col,
        sample_col=sample_col,
        min_cells=min_cells,
        normalize=normalize,
        log_transform=log_transform,
        differential=differential,
    )

    aggregator = PseudobulkAggregator(config)
    result = aggregator.aggregate(X, obs, var)

    return result.expression, result.metadata
