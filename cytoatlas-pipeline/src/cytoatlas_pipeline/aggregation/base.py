"""
Base interfaces for aggregation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sp


@dataclass
class AggregationConfig:
    """Configuration for aggregation strategies."""

    cell_type_col: str = "cell_type"
    """Column for cell type annotations."""

    sample_col: str = "sample"
    """Column for sample identifiers."""

    min_cells: int = 10
    """Minimum cells required for aggregation."""

    normalize: bool = True
    """Normalize aggregated expression (TPM)."""

    log_transform: bool = True
    """Apply log2(x + 1) transformation."""

    differential: bool = True
    """Subtract mean across samples (for ridge regression)."""


@dataclass
class AggregatedData:
    """
    Result of aggregation operation.

    Contains expression matrix and associated metadata.
    """

    expression: pd.DataFrame
    """Expression matrix (genes x aggregation units)."""

    metadata: pd.DataFrame
    """Metadata for each aggregation unit (column of expression)."""

    gene_names: list[str]
    """Gene names (rows)."""

    n_units: int
    """Number of aggregation units (columns)."""

    n_genes: int
    """Number of genes (rows)."""

    aggregation_type: str
    """Type of aggregation performed."""

    config: AggregationConfig
    """Configuration used."""

    stats: dict[str, Any] = field(default_factory=dict)
    """Additional statistics (e.g., cells per unit)."""

    def filter_genes(self, gene_names: list[str]) -> "AggregatedData":
        """Filter to specific genes."""
        gene_mask = self.expression.index.str.upper().isin(
            [g.upper() for g in gene_names]
        )
        expr_filtered = self.expression.loc[gene_mask].copy()

        return AggregatedData(
            expression=expr_filtered,
            metadata=self.metadata.copy(),
            gene_names=list(expr_filtered.index),
            n_units=self.n_units,
            n_genes=len(expr_filtered),
            aggregation_type=self.aggregation_type,
            config=self.config,
            stats=self.stats,
        )

    def to_numpy(self) -> tuple[np.ndarray, list[str], list[str]]:
        """Convert to numpy array with row/column names."""
        return (
            self.expression.values,
            list(self.expression.index),
            list(self.expression.columns),
        )

    def get_groups(self, column: str) -> dict[str, list[str]]:
        """Get column names grouped by metadata column."""
        groups = {}
        for col_name, row in self.metadata.iterrows():
            group_val = str(row[column])
            if group_val not in groups:
                groups[group_val] = []
            groups[group_val].append(col_name)
        return groups


class AggregationStrategy(ABC):
    """
    Abstract base class for aggregation strategies.

    Aggregation transforms single-cell expression data into
    summarized expression matrices suitable for downstream analysis.
    """

    def __init__(self, config: Optional[AggregationConfig] = None):
        """
        Initialize aggregation strategy.

        Args:
            config: Aggregation configuration.
        """
        self.config = config or AggregationConfig()

    @abstractmethod
    def aggregate(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: pd.DataFrame,
        var: pd.DataFrame,
    ) -> AggregatedData:
        """
        Aggregate expression data.

        Args:
            X: Expression matrix (cells x genes).
            obs: Cell metadata.
            var: Gene metadata.

        Returns:
            Aggregated data.
        """
        ...

    def normalize_expression(
        self,
        expr: Union[np.ndarray, pd.DataFrame],
        target_sum: float = 1e6,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        TPM normalize expression.

        Args:
            expr: Expression matrix (genes x samples).
            target_sum: Target sum per sample.

        Returns:
            Normalized expression.
        """
        if isinstance(expr, pd.DataFrame):
            col_sums = expr.sum(axis=0)
            return expr.div(col_sums, axis=1) * target_sum
        else:
            col_sums = expr.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            return expr / col_sums * target_sum

    def log_transform(
        self,
        expr: Union[np.ndarray, pd.DataFrame],
        pseudocount: float = 1.0,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Log2 transform expression.

        Args:
            expr: Expression matrix.
            pseudocount: Pseudocount to add before log.

        Returns:
            Log-transformed expression.
        """
        return np.log2(expr + pseudocount)

    def compute_differential(
        self,
        expr: Union[np.ndarray, pd.DataFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Compute differential expression (subtract row mean).

        Args:
            expr: Expression matrix (genes x samples).

        Returns:
            Differential expression.
        """
        if isinstance(expr, pd.DataFrame):
            row_means = expr.mean(axis=1)
            return expr.subtract(row_means, axis=0)
        else:
            row_means = expr.mean(axis=1, keepdims=True)
            return expr - row_means

    def process_expression(
        self,
        expr: Union[np.ndarray, pd.DataFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply full expression processing pipeline.

        Args:
            expr: Raw expression (genes x samples).

        Returns:
            Processed expression.
        """
        if self.config.normalize:
            expr = self.normalize_expression(expr)

        if self.config.log_transform:
            expr = self.log_transform(expr)

        if self.config.differential:
            expr = self.compute_differential(expr)

        return expr
