"""
Gene coverage validation.

Validates signature gene detection in expression data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GeneCoverageResult:
    """Result of gene coverage validation."""

    signature_name: str
    """Signature name (e.g., 'CytoSig')."""

    n_signature_genes: int
    """Total genes in signature."""

    n_detected: int
    """Genes detected in expression data."""

    coverage_fraction: float
    """Fraction of signature genes detected."""

    missing_genes: list[str]
    """Genes in signature but not detected."""

    low_expression_genes: list[str]
    """Genes detected but with low expression."""


class GeneCoverageValidator:
    """
    Validates signature gene coverage in expression data.

    Checks what fraction of signature genes are present and
    expressed in the dataset.

    Example:
        >>> validator = GeneCoverageValidator()
        >>> result = validator.validate(
        ...     expression=expression_df,
        ...     signature=signature_df
        ... )
    """

    def __init__(
        self,
        min_mean_expression: float = 0.1,
        min_cells_expressing: float = 0.01,
    ):
        """
        Initialize gene coverage validator.

        Args:
            min_mean_expression: Minimum mean expression to count as "expressed".
            min_cells_expressing: Minimum fraction of cells expressing gene.
        """
        self.min_mean_expression = min_mean_expression
        self.min_cells_expressing = min_cells_expressing

    def validate(
        self,
        expression: pd.DataFrame,
        signature: pd.DataFrame,
        signature_name: str = "signature",
    ) -> GeneCoverageResult:
        """
        Validate gene coverage.

        Args:
            expression: Expression matrix (genes x samples).
            signature: Signature matrix (genes x signatures).
            signature_name: Name for this signature.

        Returns:
            GeneCoverageResult.
        """
        # Normalize gene names
        expr_genes = set(g.upper() for g in expression.index)
        sig_genes = set(g.upper() for g in signature.index)

        # Coverage
        common = expr_genes & sig_genes
        missing = sig_genes - expr_genes

        # Check expression levels
        low_expression = []
        expr_upper_map = {g.upper(): g for g in expression.index}

        for gene in common:
            actual_gene = expr_upper_map.get(gene)
            if actual_gene is None:
                continue

            gene_expr = expression.loc[actual_gene].values
            mean_expr = np.mean(gene_expr)
            frac_expressing = np.mean(gene_expr > 0)

            if mean_expr < self.min_mean_expression:
                low_expression.append(actual_gene)
            elif frac_expressing < self.min_cells_expressing:
                low_expression.append(actual_gene)

        return GeneCoverageResult(
            signature_name=signature_name,
            n_signature_genes=len(sig_genes),
            n_detected=len(common),
            coverage_fraction=len(common) / len(sig_genes) if sig_genes else 0,
            missing_genes=sorted(missing),
            low_expression_genes=low_expression,
        )

    def validate_multiple(
        self,
        expression: pd.DataFrame,
        signatures: dict[str, pd.DataFrame],
    ) -> dict[str, GeneCoverageResult]:
        """
        Validate coverage for multiple signatures.

        Args:
            expression: Expression matrix.
            signatures: Dict of signature name to signature matrix.

        Returns:
            Dict of signature name to GeneCoverageResult.
        """
        results = {}
        for name, sig in signatures.items():
            results[name] = self.validate(expression, sig, name)
        return results


def validate_gene_coverage(
    expression: pd.DataFrame,
    signature: pd.DataFrame,
    signature_name: str = "signature",
) -> GeneCoverageResult:
    """
    Convenience function for gene coverage validation.

    Args:
        expression: Expression matrix.
        signature: Signature matrix.
        signature_name: Name for signature.

    Returns:
        GeneCoverageResult.
    """
    validator = GeneCoverageValidator()
    return validator.validate(expression, signature, signature_name)
