"""
Conserved signature detection across atlases.

Identifies signatures with consistent patterns across multiple atlases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ConservedSignature:
    """A conserved signature across atlases."""

    signature: str
    """Signature name."""

    n_atlases: int
    """Number of atlases where detected."""

    mean_activity: float
    """Mean activity across atlases."""

    std_activity: float
    """Std of activity across atlases."""

    conservation_score: float
    """Conservation score (0-1)."""

    atlas_activities: dict[str, float]
    """Activity per atlas."""

    top_celltype_concordance: float
    """Concordance of top cell type across atlases."""


class ConservedSignatureDetector:
    """
    Detects signatures conserved across multiple atlases.

    Example:
        >>> detector = ConservedSignatureDetector(min_atlases=2)
        >>> conserved = detector.detect({
        ...     "CIMA": cima_activity,
        ...     "Inflammation": inflam_activity,
        ... })
    """

    def __init__(
        self,
        min_atlases: int = 2,
        min_correlation: float = 0.5,
        cv_threshold: float = 0.3,
    ):
        """
        Initialize detector.

        Args:
            min_atlases: Minimum atlases for conservation.
            min_correlation: Minimum cross-atlas correlation.
            cv_threshold: Maximum CV for conservation.
        """
        self.min_atlases = min_atlases
        self.min_correlation = min_correlation
        self.cv_threshold = cv_threshold

    def detect(
        self,
        activities: dict[str, pd.DataFrame],
        metadata: Optional[dict[str, pd.DataFrame]] = None,
    ) -> list[ConservedSignature]:
        """
        Detect conserved signatures.

        Args:
            activities: Dict of atlas name to activity matrix.
            metadata: Optional metadata per atlas (for cell type info).

        Returns:
            List of conserved signatures.
        """
        # Get common signatures
        all_signatures = None
        for df in activities.values():
            if all_signatures is None:
                all_signatures = set(df.index)
            else:
                all_signatures &= set(df.index)

        if not all_signatures:
            return []

        results = []

        for signature in all_signatures:
            # Collect activity across atlases
            atlas_means = {}
            for atlas_name, df in activities.items():
                if signature in df.index:
                    atlas_means[atlas_name] = df.loc[signature].mean()

            if len(atlas_means) < self.min_atlases:
                continue

            # Compute statistics
            values = np.array(list(atlas_means.values()))
            mean_act = np.mean(values)
            std_act = np.std(values)
            cv = std_act / abs(mean_act) if mean_act != 0 else float('inf')

            # Conservation score based on CV
            if cv <= self.cv_threshold:
                conservation = 1.0 - (cv / self.cv_threshold)
            else:
                conservation = 0.0

            # Cell type concordance (if metadata available)
            top_ct_concordance = self._compute_celltype_concordance(
                signature, activities, metadata
            ) if metadata else 0.0

            if conservation > 0 or top_ct_concordance > 0.5:
                results.append(ConservedSignature(
                    signature=signature,
                    n_atlases=len(atlas_means),
                    mean_activity=mean_act,
                    std_activity=std_act,
                    conservation_score=conservation,
                    atlas_activities=atlas_means,
                    top_celltype_concordance=top_ct_concordance,
                ))

        # Sort by conservation score
        results.sort(key=lambda x: x.conservation_score, reverse=True)

        return results

    def _compute_celltype_concordance(
        self,
        signature: str,
        activities: dict[str, pd.DataFrame],
        metadata: dict[str, pd.DataFrame],
    ) -> float:
        """Compute cell type concordance across atlases."""
        top_celltypes = []

        for atlas_name, activity_df in activities.items():
            if atlas_name not in metadata or signature not in activity_df.index:
                continue

            meta = metadata[atlas_name]
            if "cell_type" not in meta.columns:
                continue

            # Get mean activity per cell type
            ct_means = {}
            for ct in meta["cell_type"].unique():
                ct_mask = meta["cell_type"] == ct
                ct_samples = meta.index[ct_mask].tolist()
                ct_samples = [s for s in ct_samples if s in activity_df.columns]
                if ct_samples:
                    ct_means[ct] = activity_df.loc[signature, ct_samples].mean()

            if ct_means:
                top_ct = max(ct_means, key=ct_means.get)
                top_celltypes.append(top_ct.lower())

        if len(top_celltypes) < 2:
            return 0.0

        # Compute pairwise similarity
        from collections import Counter
        counts = Counter(top_celltypes)
        most_common_count = counts.most_common(1)[0][1]

        return most_common_count / len(top_celltypes)


def detect_conserved_signatures(
    activities: dict[str, pd.DataFrame],
    min_atlases: int = 2,
) -> list[ConservedSignature]:
    """
    Convenience function for conserved signature detection.

    Args:
        activities: Dict of atlas name to activity matrix.
        min_atlases: Minimum atlases for conservation.

    Returns:
        List of conserved signatures.
    """
    detector = ConservedSignatureDetector(min_atlases=min_atlases)
    return detector.detect(activities)
