"""
Cross-validation stability validation.

Assesses robustness of activity inference through CV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CVStabilityResult:
    """Result of CV stability validation."""

    signature: str
    """Signature name."""

    mean_correlation: float
    """Mean correlation across CV folds."""

    std_correlation: float
    """Std of correlations."""

    min_correlation: float
    """Minimum correlation."""

    max_correlation: float
    """Maximum correlation."""

    n_folds: int
    """Number of CV folds."""


class CVStabilityValidator:
    """
    Cross-validation stability assessment.

    Runs activity inference on CV folds and measures consistency.

    Example:
        >>> validator = CVStabilityValidator(n_folds=5)
        >>> results = validator.validate(
        ...     expression=expression_df,
        ...     signature=signature_df,
        ...     inference_fn=run_inference
        ... )
    """

    def __init__(
        self,
        n_folds: int = 5,
        seed: int = 42,
    ):
        """
        Initialize CV stability validator.

        Args:
            n_folds: Number of CV folds.
            seed: Random seed.
        """
        self.n_folds = n_folds
        self.seed = seed

    def validate(
        self,
        expression: pd.DataFrame,
        signature: pd.DataFrame,
        full_activity: Optional[pd.DataFrame] = None,
    ) -> list[CVStabilityResult]:
        """
        Validate stability through cross-validation.

        If full_activity is provided, compares CV fold activities
        to the full-data activities. Otherwise, compares between folds.

        Args:
            expression: Expression matrix (genes x samples).
            signature: Signature matrix (genes x signatures).
            full_activity: Full-data activity (optional).

        Returns:
            List of CV stability results.
        """
        rng = np.random.default_rng(self.seed)

        n_samples = expression.shape[1]
        sample_order = rng.permutation(n_samples)
        fold_size = n_samples // self.n_folds

        # Store fold activities
        fold_activities = []

        for fold in range(self.n_folds):
            # Train indices (all except this fold)
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_folds - 1 else n_samples

            test_idx = sample_order[test_start:test_end]
            train_idx = np.concatenate([
                sample_order[:test_start],
                sample_order[test_end:],
            ])

            # Get train expression
            train_samples = [expression.columns[i] for i in train_idx]
            train_expr = expression[train_samples]

            # Run inference on train set
            fold_activity = self._run_inference(train_expr, signature)
            fold_activities.append(fold_activity)

        # Compute stability for each signature
        results = []
        signatures = signature.columns.tolist()

        for sig in signatures:
            correlations = []

            if full_activity is not None and sig in full_activity.index:
                # Compare each fold to full activity
                for fold_act in fold_activities:
                    if sig not in fold_act.index:
                        continue
                    common = list(
                        set(fold_act.columns) & set(full_activity.columns)
                    )
                    if len(common) < 10:
                        continue
                    r, _ = stats.pearsonr(
                        fold_act.loc[sig, common].values,
                        full_activity.loc[sig, common].values,
                    )
                    correlations.append(r)
            else:
                # Compare between folds
                for i in range(len(fold_activities)):
                    for j in range(i + 1, len(fold_activities)):
                        if sig not in fold_activities[i].index:
                            continue
                        if sig not in fold_activities[j].index:
                            continue
                        common = list(
                            set(fold_activities[i].columns) &
                            set(fold_activities[j].columns)
                        )
                        if len(common) < 10:
                            continue
                        r, _ = stats.pearsonr(
                            fold_activities[i].loc[sig, common].values,
                            fold_activities[j].loc[sig, common].values,
                        )
                        correlations.append(r)

            if not correlations:
                continue

            results.append(CVStabilityResult(
                signature=sig,
                mean_correlation=np.mean(correlations),
                std_correlation=np.std(correlations),
                min_correlation=np.min(correlations),
                max_correlation=np.max(correlations),
                n_folds=self.n_folds,
            ))

        return results

    def _run_inference(
        self,
        expression: pd.DataFrame,
        signature: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run activity inference on a subset."""
        # Import here to avoid circular imports
        from cytoatlas_pipeline.activity.ridge import run_ridge_inference

        try:
            result = run_ridge_inference(
                expression,
                signature,
                n_rand=100,  # Faster for CV
                verbose=False,
            )
            return result.zscore
        except Exception:
            # Return empty if inference fails
            return pd.DataFrame(index=signature.columns)


def validate_cv_stability(
    expression: pd.DataFrame,
    signature: pd.DataFrame,
    full_activity: Optional[pd.DataFrame] = None,
    n_folds: int = 5,
) -> list[CVStabilityResult]:
    """
    Convenience function for CV stability validation.

    Args:
        expression: Expression matrix.
        signature: Signature matrix.
        full_activity: Full-data activity (optional).
        n_folds: Number of CV folds.

    Returns:
        List of CV stability results.
    """
    validator = CVStabilityValidator(n_folds=n_folds)
    return validator.validate(expression, signature, full_activity)
