"""
Composite quality scoring.

Combines multiple validation metrics into a single quality score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationSummary:
    """Summary of all validation results."""

    quality_score: float
    """Composite quality score (0-1)."""

    sample_level_r2: float
    """Mean R² from sample-level validation."""

    celltype_correlation: float
    """Mean correlation from cell-type validation."""

    gene_coverage: float
    """Gene coverage fraction."""

    cv_stability: float
    """Mean CV stability correlation."""

    biological_concordance: float
    """Fraction of biologically concordant signatures."""

    n_signatures_validated: int
    """Number of signatures with validation data."""

    component_scores: dict[str, float] = field(default_factory=dict)
    """Individual component scores."""

    confidence: str = "low"
    """Confidence level (low/medium/high)."""

    warnings: list[str] = field(default_factory=list)
    """Validation warnings."""


class QualityScorer:
    """
    Computes composite quality score from validation results.

    Weights:
    - Sample-level R²: 20%
    - Cell-type correlation: 20%
    - Gene coverage: 20%
    - CV stability: 20%
    - Biological concordance: 20%

    Example:
        >>> scorer = QualityScorer()
        >>> summary = scorer.compute(
        ...     sample_results=sample_results,
        ...     celltype_results=celltype_results,
        ...     gene_coverage_result=gene_result,
        ...     cv_results=cv_results,
        ...     biological_results=bio_results
        ... )
    """

    DEFAULT_WEIGHTS = {
        "sample_level": 0.20,
        "celltype_level": 0.20,
        "gene_coverage": 0.20,
        "cv_stability": 0.20,
        "biological": 0.20,
    }

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize quality scorer.

        Args:
            weights: Custom weights for components.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

    def compute(
        self,
        sample_results: Optional[list] = None,
        celltype_results: Optional[list] = None,
        gene_coverage_result: Optional[Any] = None,
        cv_results: Optional[list] = None,
        biological_results: Optional[list] = None,
    ) -> ValidationSummary:
        """
        Compute quality score from validation results.

        Args:
            sample_results: Sample-level validation results.
            celltype_results: Cell-type level results.
            gene_coverage_result: Gene coverage result.
            cv_results: CV stability results.
            biological_results: Biological validation results.

        Returns:
            ValidationSummary with composite score.
        """
        warnings = []
        component_scores = {}

        # Sample-level
        if sample_results:
            r2_values = [r.r2 for r in sample_results if hasattr(r, 'r2')]
            sample_level_r2 = np.mean(r2_values) if r2_values else 0.0
            component_scores["sample_level"] = min(sample_level_r2, 1.0)
        else:
            sample_level_r2 = 0.0
            component_scores["sample_level"] = 0.0
            warnings.append("No sample-level validation data")

        # Cell-type level
        if celltype_results:
            corrs = [r.correlation for r in celltype_results if hasattr(r, 'correlation')]
            celltype_corr = np.mean(corrs) if corrs else 0.0
            component_scores["celltype_level"] = max(0, min(celltype_corr, 1.0))
        else:
            celltype_corr = 0.0
            component_scores["celltype_level"] = 0.0
            warnings.append("No cell-type level validation data")

        # Gene coverage
        if gene_coverage_result:
            gene_cov = gene_coverage_result.coverage_fraction
            component_scores["gene_coverage"] = gene_cov
        else:
            gene_cov = 0.0
            component_scores["gene_coverage"] = 0.0
            warnings.append("No gene coverage data")

        # CV stability
        if cv_results:
            stabilities = [r.mean_correlation for r in cv_results if hasattr(r, 'mean_correlation')]
            cv_stability = np.mean(stabilities) if stabilities else 0.0
            component_scores["cv_stability"] = max(0, min(cv_stability, 1.0))
        else:
            cv_stability = 0.0
            component_scores["cv_stability"] = 0.0
            warnings.append("No CV stability data")

        # Biological concordance
        if biological_results:
            concordant = sum(1 for r in biological_results if r.concordant)
            bio_concordance = concordant / len(biological_results)
            component_scores["biological"] = bio_concordance
        else:
            bio_concordance = 0.0
            component_scores["biological"] = 0.0
            warnings.append("No biological validation data")

        # Compute weighted score
        quality_score = sum(
            component_scores.get(comp, 0.0) * weight
            for comp, weight in self.weights.items()
        )

        # Count validated signatures
        n_validated = 0
        if sample_results:
            n_validated = max(n_validated, len(sample_results))
        if celltype_results:
            n_validated = max(n_validated, len(celltype_results))

        # Determine confidence
        if quality_score >= 0.7 and len(warnings) <= 1:
            confidence = "high"
        elif quality_score >= 0.5 and len(warnings) <= 2:
            confidence = "medium"
        else:
            confidence = "low"

        return ValidationSummary(
            quality_score=quality_score,
            sample_level_r2=sample_level_r2,
            celltype_correlation=celltype_corr,
            gene_coverage=gene_cov,
            cv_stability=cv_stability,
            biological_concordance=bio_concordance,
            n_signatures_validated=n_validated,
            component_scores=component_scores,
            confidence=confidence,
            warnings=warnings,
        )


def compute_quality_score(
    sample_results: Optional[list] = None,
    celltype_results: Optional[list] = None,
    gene_coverage_result: Optional[Any] = None,
    cv_results: Optional[list] = None,
    biological_results: Optional[list] = None,
) -> ValidationSummary:
    """
    Convenience function for quality scoring.

    Args:
        sample_results: Sample-level validation results.
        celltype_results: Cell-type level results.
        gene_coverage_result: Gene coverage result.
        cv_results: CV stability results.
        biological_results: Biological validation results.

    Returns:
        ValidationSummary with composite score.
    """
    scorer = QualityScorer()
    return scorer.compute(
        sample_results=sample_results,
        celltype_results=celltype_results,
        gene_coverage_result=gene_coverage_result,
        cv_results=cv_results,
        biological_results=biological_results,
    )
