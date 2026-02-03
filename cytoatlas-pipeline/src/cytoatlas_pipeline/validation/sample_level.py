"""
Sample-level validation: Expression vs activity regression.

Validates that target gene expression correlates with inferred activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SampleLevelResult:
    """Result of sample-level validation."""

    signature: str
    """Signature name."""

    r2: float
    """R-squared from regression."""

    correlation: float
    """Pearson correlation."""

    pvalue: float
    """P-value for correlation."""

    slope: float
    """Regression slope."""

    intercept: float
    """Regression intercept."""

    n_samples: int
    """Number of samples."""

    target_gene: Optional[str] = None
    """Target gene used for validation."""


class SampleLevelValidator:
    """
    Sample-level validation via expression-activity regression.

    For each signature, validates that the target gene expression
    correlates with the inferred activity.

    Example:
        >>> validator = SampleLevelValidator()
        >>> results = validator.validate(
        ...     activity=activity_df,
        ...     expression=expression_df,
        ...     signature_genes={"IL6": "IL6"}
        ... )
    """

    def __init__(self, min_samples: int = 10):
        """
        Initialize validator.

        Args:
            min_samples: Minimum samples required.
        """
        self.min_samples = min_samples

    def validate(
        self,
        activity: pd.DataFrame,
        expression: pd.DataFrame,
        signature_genes: dict[str, str],
    ) -> list[SampleLevelResult]:
        """
        Validate activities against gene expression.

        Args:
            activity: Activity matrix (signatures x samples).
            expression: Expression matrix (genes x samples).
            signature_genes: Map of signature name to target gene.

        Returns:
            List of validation results.
        """
        # Align samples
        common_samples = list(set(activity.columns) & set(expression.columns))
        if len(common_samples) < self.min_samples:
            raise ValueError(f"Insufficient common samples: {len(common_samples)}")

        activity = activity[common_samples]
        expression = expression[common_samples]

        results = []

        for signature, target_gene in signature_genes.items():
            if signature not in activity.index:
                continue

            # Handle case-insensitive gene matching
            expr_genes_upper = {g.upper(): g for g in expression.index}
            if target_gene.upper() not in expr_genes_upper:
                continue

            actual_gene = expr_genes_upper[target_gene.upper()]

            # Get vectors
            act = activity.loc[signature].values
            expr = expression.loc[actual_gene].values

            # Remove NaN
            mask = ~(np.isnan(act) | np.isnan(expr))
            if mask.sum() < self.min_samples:
                continue

            act = act[mask]
            expr = expr[mask]

            # Regression
            slope, intercept, r, p, se = stats.linregress(expr, act)

            results.append(SampleLevelResult(
                signature=signature,
                r2=r**2,
                correlation=r,
                pvalue=p,
                slope=slope,
                intercept=intercept,
                n_samples=len(act),
                target_gene=actual_gene,
            ))

        return results

    def validate_all(
        self,
        activity: pd.DataFrame,
        expression: pd.DataFrame,
    ) -> list[SampleLevelResult]:
        """
        Validate all signatures where signature name matches a gene.

        Args:
            activity: Activity matrix.
            expression: Expression matrix.

        Returns:
            List of validation results.
        """
        # Auto-detect signature-gene mappings
        expr_genes_upper = {g.upper(): g for g in expression.index}
        signature_genes = {}

        for sig in activity.index:
            sig_upper = sig.upper()
            if sig_upper in expr_genes_upper:
                signature_genes[sig] = expr_genes_upper[sig_upper]

        return self.validate(activity, expression, signature_genes)


def validate_sample_level(
    activity: pd.DataFrame,
    expression: pd.DataFrame,
    signature_genes: Optional[dict[str, str]] = None,
) -> list[SampleLevelResult]:
    """
    Convenience function for sample-level validation.

    Args:
        activity: Activity matrix.
        expression: Expression matrix.
        signature_genes: Signature to gene mapping (auto if None).

    Returns:
        List of validation results.
    """
    validator = SampleLevelValidator()
    if signature_genes is not None:
        return validator.validate(activity, expression, signature_genes)
    else:
        return validator.validate_all(activity, expression)
