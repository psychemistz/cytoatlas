"""
Cell type mapping across atlases.

Aligns cell type annotations between different atlases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# Default cross-atlas cell type mapping
DEFAULT_MAPPING = {
    # T cells
    "T cell": ["T cell", "T cells", "T_cell", "Tcell"],
    "CD4+ T cell": ["CD4+ T", "CD4 T", "CD4+T", "CD4_T", "CD4+ T cell"],
    "CD8+ T cell": ["CD8+ T", "CD8 T", "CD8+T", "CD8_T", "CD8+ T cell"],
    "Treg": ["Treg", "T regulatory", "Regulatory T", "CD4+CD25+ T"],
    "NK cell": ["NK", "NK cell", "Natural killer", "NKcell"],

    # B cells
    "B cell": ["B cell", "B cells", "B_cell", "Bcell"],
    "Plasma cell": ["Plasma", "Plasma cell", "Plasmacyte"],

    # Myeloid
    "Monocyte": ["Mono", "Monocyte", "Monocytes", "CD14+"],
    "Macrophage": ["Mac", "Macrophage", "MΦ", "M1", "M2"],
    "Dendritic cell": ["DC", "Dendritic", "cDC", "pDC", "mDC"],
    "Neutrophil": ["Neutro", "Neutrophil", "PMN"],

    # Other
    "Fibroblast": ["Fib", "Fibroblast", "CAF", "myCAF", "iCAF"],
    "Endothelial": ["Endo", "Endothelial", "EC", "VEC"],
    "Epithelial": ["Epi", "Epithelial", "Epithelium"],
}


@dataclass
class MappingResult:
    """Result of cell type mapping."""

    mapped_types: dict[str, str]
    """Mapping from original to canonical type."""

    unmapped: list[str]
    """Cell types that couldn't be mapped."""

    similarity_scores: dict[str, float]
    """Similarity scores for fuzzy matches."""


class CellTypeMapper:
    """
    Maps cell type annotations across atlases.

    Uses exact matching, fuzzy matching, and manual overrides.

    Example:
        >>> mapper = CellTypeMapper()
        >>> mapped = mapper.map(cell_types, target_atlas="CIMA")
    """

    def __init__(
        self,
        mapping: Optional[dict[str, list[str]]] = None,
        case_sensitive: bool = False,
    ):
        """
        Initialize cell type mapper.

        Args:
            mapping: Custom mapping dictionary.
            case_sensitive: Whether matching is case-sensitive.
        """
        self.mapping = mapping or DEFAULT_MAPPING
        self.case_sensitive = case_sensitive

        # Build reverse lookup
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Build reverse lookup for fast matching."""
        self._lookup = {}
        for canonical, variants in self.mapping.items():
            for variant in variants:
                key = variant if self.case_sensitive else variant.lower()
                self._lookup[key] = canonical

    def map_single(self, cell_type: str) -> tuple[Optional[str], float]:
        """
        Map a single cell type to canonical form.

        Args:
            cell_type: Cell type to map.

        Returns:
            Tuple of (canonical_type, confidence).
        """
        key = cell_type if self.case_sensitive else cell_type.lower()

        # Exact match
        if key in self._lookup:
            return self._lookup[key], 1.0

        # Partial match
        for variant, canonical in self._lookup.items():
            if variant in key or key in variant:
                return canonical, 0.8

        # Fuzzy match (simple Levenshtein-like)
        best_match = None
        best_score = 0.0

        for variant, canonical in self._lookup.items():
            score = self._similarity(key, variant)
            if score > best_score and score > 0.6:
                best_score = score
                best_match = canonical

        if best_match:
            return best_match, best_score

        return None, 0.0

    def _similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity (Jaccard on character bigrams)."""
        if not s1 or not s2:
            return 0.0

        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s) - 1))

        b1, b2 = bigrams(s1), bigrams(s2)
        if not b1 or not b2:
            return 0.0

        return len(b1 & b2) / len(b1 | b2)

    def map(
        self,
        cell_types: list[str],
        min_confidence: float = 0.5,
    ) -> MappingResult:
        """
        Map multiple cell types.

        Args:
            cell_types: Cell types to map.
            min_confidence: Minimum confidence for mapping.

        Returns:
            MappingResult.
        """
        mapped = {}
        unmapped = []
        scores = {}

        for ct in cell_types:
            canonical, confidence = self.map_single(ct)
            if canonical and confidence >= min_confidence:
                mapped[ct] = canonical
                scores[ct] = confidence
            else:
                unmapped.append(ct)

        return MappingResult(
            mapped_types=mapped,
            unmapped=unmapped,
            similarity_scores=scores,
        )

    def add_mapping(
        self,
        canonical: str,
        variants: list[str],
    ) -> None:
        """
        Add custom mapping.

        Args:
            canonical: Canonical cell type name.
            variants: Alternative names.
        """
        if canonical not in self.mapping:
            self.mapping[canonical] = []
        self.mapping[canonical].extend(variants)
        self._build_lookup()


def map_cell_types(
    cell_types: list[str],
    mapping: Optional[dict[str, list[str]]] = None,
) -> MappingResult:
    """
    Convenience function for cell type mapping.

    Args:
        cell_types: Cell types to map.
        mapping: Custom mapping dictionary.

    Returns:
        MappingResult.
    """
    mapper = CellTypeMapper(mapping=mapping)
    return mapper.map(cell_types)
