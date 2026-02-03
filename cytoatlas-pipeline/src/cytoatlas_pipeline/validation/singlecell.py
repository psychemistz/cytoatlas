"""
Single-cell validation: Expressing vs non-expressing cells.

Validates that cells expressing the target gene have higher activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SingleCellResult:
    """Result of single-cell validation."""

    signature: str
    """Signature name."""

    auc: float
    """Area under ROC curve."""

    activity_diff: float
    """Activity difference (expressing - non-expressing)."""

    pvalue: float
    """P-value from Wilcoxon test."""

    n_expressing: int
    """Number of expressing cells."""

    n_non_expressing: int
    """Number of non-expressing cells."""

    expression_threshold: float
    """Threshold used for expressing/non-expressing."""


class SingleCellValidator:
    """
    Single-cell validation via expressing vs non-expressing comparison.

    For each signature, compares activity between cells that express
    the target gene and cells that don't.

    Example:
        >>> validator = SingleCellValidator()
        >>> results = validator.validate(
        ...     activity=sc_activity,
        ...     expression=sc_expression,
        ... )
    """

    def __init__(
        self,
        expression_threshold: float = 0.0,
        min_cells_per_group: int = 50,
    ):
        """
        Initialize validator.

        Args:
            expression_threshold: Threshold for "expressing" (usually 0).
            min_cells_per_group: Minimum cells per group.
        """
        self.expression_threshold = expression_threshold
        self.min_cells_per_group = min_cells_per_group

    def validate(
        self,
        activity: pd.DataFrame,
        expression: pd.DataFrame,
        signature_genes: Optional[dict[str, str]] = None,
    ) -> list[SingleCellResult]:
        """
        Validate single-cell activities.

        Args:
            activity: Activity matrix (signatures x cells).
            expression: Expression matrix (genes x cells).
            signature_genes: Signature to gene mapping (auto if None).

        Returns:
            List of validation results.
        """
        # Align cells
        common_cells = list(set(activity.columns) & set(expression.columns))
        activity = activity[common_cells]
        expression = expression[common_cells]

        # Build gene mapping
        if signature_genes is None:
            expr_genes_upper = {g.upper(): g for g in expression.index}
            signature_genes = {}
            for sig in activity.index:
                if sig.upper() in expr_genes_upper:
                    signature_genes[sig] = expr_genes_upper[sig.upper()]

        results = []

        for signature, target_gene in signature_genes.items():
            if signature not in activity.index or target_gene not in expression.index:
                continue

            # Get expression and activity
            expr = expression.loc[target_gene].values
            act = activity.loc[signature].values

            # Split by expression
            expressing = expr > self.expression_threshold
            non_expressing = ~expressing

            n_expr = expressing.sum()
            n_non_expr = non_expressing.sum()

            if n_expr < self.min_cells_per_group or n_non_expr < self.min_cells_per_group:
                continue

            act_expr = act[expressing]
            act_non_expr = act[non_expressing]

            # Activity difference
            activity_diff = np.mean(act_expr) - np.mean(act_non_expr)

            # Wilcoxon test
            stat, pval = stats.mannwhitneyu(
                act_expr, act_non_expr, alternative="greater"
            )

            # AUC (from U statistic)
            auc = stat / (n_expr * n_non_expr)

            results.append(SingleCellResult(
                signature=signature,
                auc=auc,
                activity_diff=activity_diff,
                pvalue=pval,
                n_expressing=n_expr,
                n_non_expressing=n_non_expr,
                expression_threshold=self.expression_threshold,
            ))

        return results


def validate_singlecell(
    activity: pd.DataFrame,
    expression: pd.DataFrame,
    signature_genes: Optional[dict[str, str]] = None,
    expression_threshold: float = 0.0,
) -> list[SingleCellResult]:
    """
    Convenience function for single-cell validation.

    Args:
        activity: Activity matrix (signatures x cells).
        expression: Expression matrix (genes x cells).
        signature_genes: Signature to gene mapping.
        expression_threshold: Expression threshold.

    Returns:
        List of validation results.
    """
    validator = SingleCellValidator(expression_threshold=expression_threshold)
    return validator.validate(activity, expression, signature_genes)
