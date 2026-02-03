"""
Biological validation: Known marker validation.

Validates that expected cell type-cytokine relationships are detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# Known biological markers (cytokine -> expected cell types)
KNOWN_MARKERS = {
    # T cell signatures
    "IFNG": ["CD8+ T cell", "NK cell", "Th1"],
    "IL17A": ["Th17", "gamma-delta T cell", "ILC3"],
    "IL17F": ["Th17"],
    "IL2": ["T cell", "CD4+ T cell", "Treg"],
    "IL4": ["Th2", "basophil", "mast cell"],
    "IL5": ["Th2", "ILC2", "eosinophil"],
    "IL13": ["Th2", "ILC2"],
    "IL10": ["Treg", "macrophage", "B cell"],
    "IL21": ["Tfh", "Th17"],
    "IL22": ["Th17", "Th22", "ILC3", "NK cell"],

    # Inflammatory
    "TNF": ["monocyte", "macrophage", "dendritic cell"],
    "TNFA": ["monocyte", "macrophage", "dendritic cell"],
    "IL1B": ["monocyte", "macrophage", "dendritic cell"],
    "IL6": ["monocyte", "macrophage", "fibroblast"],
    "IL8": ["neutrophil", "monocyte", "epithelial"],
    "CXCL8": ["neutrophil", "monocyte", "epithelial"],

    # Type I IFN
    "IFNA": ["pDC", "plasmacytoid dendritic"],
    "IFNB": ["fibroblast", "macrophage"],

    # Growth factors
    "TGFB1": ["Treg", "macrophage", "fibroblast"],
    "VEGFA": ["macrophage", "fibroblast", "endothelial"],

    # Chemokines
    "CCL2": ["monocyte", "macrophage", "fibroblast"],
    "CCL5": ["T cell", "NK cell"],
    "CXCL10": ["monocyte", "macrophage", "endothelial"],
}


@dataclass
class BiologicalResult:
    """Result of biological validation."""

    signature: str
    """Signature/cytokine name."""

    expected_celltypes: list[str]
    """Expected cell types for this signature."""

    top_celltype: str
    """Cell type with highest mean activity."""

    top_activity: float
    """Activity in top cell type."""

    expected_in_top_n: int
    """Number of expected cell types in top N."""

    rank_of_expected: list[int]
    """Ranks of expected cell types."""

    concordant: bool
    """Whether top cell type is in expected list."""


class BiologicalValidator:
    """
    Validates activities against known biology.

    Checks whether signatures show expected activity patterns
    in known producer cell types.

    Example:
        >>> validator = BiologicalValidator()
        >>> results = validator.validate(
        ...     activity=activity_df,
        ...     metadata=metadata_df,
        ...     cell_type_col="cell_type"
        ... )
    """

    def __init__(
        self,
        known_markers: Optional[dict[str, list[str]]] = None,
        top_n: int = 3,
    ):
        """
        Initialize biological validator.

        Args:
            known_markers: Known marker map (uses default if None).
            top_n: Consider top N cell types for validation.
        """
        self.known_markers = known_markers or KNOWN_MARKERS
        self.top_n = top_n

    def validate(
        self,
        activity: pd.DataFrame,
        metadata: pd.DataFrame,
        cell_type_col: str = "cell_type",
    ) -> list[BiologicalResult]:
        """
        Validate activities against known biology.

        Args:
            activity: Activity matrix (signatures x samples).
            metadata: Sample metadata with cell type.
            cell_type_col: Cell type column name.

        Returns:
            List of biological validation results.
        """
        # Align
        common = list(set(activity.columns) & set(metadata.index))
        activity = activity[common]
        metadata = metadata.loc[common]

        # Compute mean activity per cell type
        cell_types = metadata[cell_type_col].unique()
        ct_activity = {}

        for ct in cell_types:
            ct_mask = metadata[cell_type_col] == ct
            ct_samples = metadata.index[ct_mask].tolist()
            if len(ct_samples) >= 3:
                ct_activity[ct] = activity[ct_samples].mean(axis=1)

        if not ct_activity:
            return []

        # Create cell type activity matrix
        ct_df = pd.DataFrame(ct_activity)

        results = []

        for signature in activity.index:
            sig_upper = signature.upper()

            # Check if we have known markers
            expected = None
            for marker, expected_cts in self.known_markers.items():
                if marker.upper() == sig_upper:
                    expected = expected_cts
                    break

            if expected is None:
                continue

            # Get activity ranking
            sig_activity = ct_df.loc[signature].sort_values(ascending=False)
            ranked_cts = list(sig_activity.index)

            # Find ranks of expected cell types (case-insensitive partial match)
            ranks = []
            for exp_ct in expected:
                exp_lower = exp_ct.lower()
                for i, ct in enumerate(ranked_cts):
                    if exp_lower in ct.lower():
                        ranks.append(i + 1)
                        break

            # Count expected in top N
            top_cts = set(ct.lower() for ct in ranked_cts[:self.top_n])
            expected_in_top = sum(
                1 for exp_ct in expected
                if any(exp_ct.lower() in tc for tc in top_cts)
            )

            # Check concordance
            top_ct = ranked_cts[0]
            concordant = any(
                exp.lower() in top_ct.lower() for exp in expected
            )

            results.append(BiologicalResult(
                signature=signature,
                expected_celltypes=expected,
                top_celltype=top_ct,
                top_activity=float(sig_activity.iloc[0]),
                expected_in_top_n=expected_in_top,
                rank_of_expected=ranks,
                concordant=concordant,
            ))

        return results


def validate_biological(
    activity: pd.DataFrame,
    metadata: pd.DataFrame,
    cell_type_col: str = "cell_type",
    known_markers: Optional[dict[str, list[str]]] = None,
) -> list[BiologicalResult]:
    """
    Convenience function for biological validation.

    Args:
        activity: Activity matrix.
        metadata: Sample metadata.
        cell_type_col: Cell type column.
        known_markers: Known marker map.

    Returns:
        List of biological validation results.
    """
    validator = BiologicalValidator(known_markers=known_markers)
    return validator.validate(activity, metadata, cell_type_col)
