"""
Cell-type level validation.

Validates activity patterns across cell types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CellTypeLevelResult:
    """Result of cell-type level validation."""

    signature: str
    """Signature name."""

    correlation: float
    """Correlation between expression and activity across cell types."""

    pvalue: float
    """P-value for correlation."""

    n_celltypes: int
    """Number of cell types."""

    top_celltype: str
    """Cell type with highest activity."""

    top_activity: float
    """Activity in top cell type."""


class CellTypeLevelValidator:
    """
    Cell-type level validation.

    Validates that activity patterns are consistent across cell types,
    i.e., cell types with high target gene expression also have high activity.

    Example:
        >>> validator = CellTypeLevelValidator()
        >>> results = validator.validate(
        ...     activity=activity_df,
        ...     expression=expression_df,
        ...     metadata=metadata_df,
        ...     cell_type_col="cell_type"
        ... )
    """

    def __init__(self, min_celltypes: int = 5):
        """
        Initialize validator.

        Args:
            min_celltypes: Minimum cell types required.
        """
        self.min_celltypes = min_celltypes

    def validate(
        self,
        activity: pd.DataFrame,
        expression: pd.DataFrame,
        metadata: pd.DataFrame,
        cell_type_col: str = "cell_type",
    ) -> list[CellTypeLevelResult]:
        """
        Validate activity vs expression across cell types.

        Args:
            activity: Activity matrix (signatures x samples).
            expression: Expression matrix (genes x samples).
            metadata: Sample metadata with cell type.
            cell_type_col: Cell type column name.

        Returns:
            List of validation results.
        """
        # Align samples
        common = list(
            set(activity.columns) & set(expression.columns) & set(metadata.index)
        )
        activity = activity[common]
        expression = expression[common]
        metadata = metadata.loc[common]

        # Compute mean per cell type
        cell_types = metadata[cell_type_col].unique()
        if len(cell_types) < self.min_celltypes:
            return []

        results = []

        # Get expression genes that match signatures
        expr_genes_upper = {g.upper(): g for g in expression.index}

        for signature in activity.index:
            sig_upper = signature.upper()
            if sig_upper not in expr_genes_upper:
                continue

            target_gene = expr_genes_upper[sig_upper]

            # Compute cell type means
            ct_activity = {}
            ct_expression = {}

            for ct in cell_types:
                ct_mask = metadata[cell_type_col] == ct
                ct_samples = metadata.index[ct_mask].tolist()

                if len(ct_samples) < 3:
                    continue

                ct_activity[ct] = activity.loc[signature, ct_samples].mean()
                ct_expression[ct] = expression.loc[target_gene, ct_samples].mean()

            if len(ct_activity) < self.min_celltypes:
                continue

            # Correlate
            act_vals = np.array(list(ct_activity.values()))
            expr_vals = np.array(list(ct_expression.values()))
            cts = list(ct_activity.keys())

            r, p = stats.pearsonr(act_vals, expr_vals)

            # Find top cell type
            top_idx = np.argmax(act_vals)

            results.append(CellTypeLevelResult(
                signature=signature,
                correlation=r,
                pvalue=p,
                n_celltypes=len(ct_activity),
                top_celltype=cts[top_idx],
                top_activity=act_vals[top_idx],
            ))

        return results


def validate_celltype_level(
    activity: pd.DataFrame,
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    cell_type_col: str = "cell_type",
) -> list[CellTypeLevelResult]:
    """
    Convenience function for cell-type level validation.

    Args:
        activity: Activity matrix.
        expression: Expression matrix.
        metadata: Sample metadata.
        cell_type_col: Cell type column.

    Returns:
        List of validation results.
    """
    validator = CellTypeLevelValidator()
    return validator.validate(activity, expression, metadata, cell_type_col)
