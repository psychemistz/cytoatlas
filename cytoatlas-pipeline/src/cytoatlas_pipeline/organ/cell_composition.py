"""
Cell type composition analysis per organ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CompositionResult:
    """Cell type composition for an organ."""

    organ: str
    cell_type_proportions: dict[str, float]
    dominant_type: str
    n_cells: int
    diversity_index: float  # Shannon diversity


class CellCompositionAnalyzer:
    """Analyzes cell type composition across organs."""

    def __init__(self, min_cells: int = 100):
        self.min_cells = min_cells

    def compute_shannon_diversity(self, proportions: np.ndarray) -> float:
        """Compute Shannon diversity index.

        H = -sum(p * log(p))

        Higher values indicate more diverse cell populations.
        """
        proportions = np.asarray(proportions)
        proportions = proportions[proportions > 0]

        if len(proportions) == 0:
            return 0.0

        return float(-np.sum(proportions * np.log(proportions)))

    def analyze(
        self,
        metadata: pd.DataFrame,
        organ_col: str = "organ",
        cell_type_col: str = "cell_type",
    ) -> list[CompositionResult]:
        """Compute cell type composition per organ.

        Parameters
        ----------
        metadata : pd.DataFrame
            Cell metadata with organ and cell type annotations
        organ_col : str
            Column containing organ labels
        cell_type_col : str
            Column containing cell type labels

        Returns
        -------
        list[CompositionResult]
            Composition analysis per organ
        """
        results = []

        for organ in metadata[organ_col].unique():
            organ_cells = metadata[metadata[organ_col] == organ]

            if len(organ_cells) < self.min_cells:
                continue

            # Count cell types
            ct_counts = organ_cells[cell_type_col].value_counts()
            total = ct_counts.sum()

            proportions = (ct_counts / total).to_dict()
            dominant = ct_counts.idxmax()

            # Shannon diversity
            prop_values = np.array(list(proportions.values()))
            diversity = self.compute_shannon_diversity(prop_values)

            results.append(CompositionResult(
                organ=organ,
                cell_type_proportions=proportions,
                dominant_type=dominant,
                n_cells=total,
                diversity_index=diversity,
            ))

        # Sort by diversity
        results.sort(key=lambda x: x.diversity_index, reverse=True)

        return results

    def compare_compositions(
        self,
        metadata: pd.DataFrame,
        organ_col: str = "organ",
        cell_type_col: str = "cell_type",
    ) -> pd.DataFrame:
        """Create composition matrix (organs × cell types).

        Returns
        -------
        pd.DataFrame
            Proportion matrix (organs × cell types)
        """
        results = self.analyze(metadata, organ_col, cell_type_col)

        if not results:
            return pd.DataFrame()

        # Collect all cell types
        all_types = set()
        for r in results:
            all_types.update(r.cell_type_proportions.keys())

        # Build matrix
        matrix = {}
        for r in results:
            row = {ct: r.cell_type_proportions.get(ct, 0.0) for ct in all_types}
            matrix[r.organ] = row

        return pd.DataFrame(matrix).T

    def get_enriched_types(
        self,
        metadata: pd.DataFrame,
        organ_col: str = "organ",
        cell_type_col: str = "cell_type",
        enrichment_threshold: float = 1.5,
    ) -> dict[str, list[str]]:
        """Find cell types enriched in each organ vs overall.

        Parameters
        ----------
        enrichment_threshold : float
            Minimum fold enrichment to report

        Returns
        -------
        dict[str, list[str]]
            Enriched cell types per organ
        """
        composition_matrix = self.compare_compositions(
            metadata, organ_col, cell_type_col
        )

        if composition_matrix.empty:
            return {}

        # Overall proportions
        overall = metadata[cell_type_col].value_counts(normalize=True)

        enriched = {}
        for organ in composition_matrix.index:
            organ_props = composition_matrix.loc[organ]
            enriched_types = []

            for ct in organ_props.index:
                if ct in overall.index and overall[ct] > 0:
                    fold = organ_props[ct] / overall[ct]
                    if fold >= enrichment_threshold:
                        enriched_types.append(ct)

            if enriched_types:
                enriched[organ] = enriched_types

        return enriched


def analyze_composition(
    metadata: pd.DataFrame,
    organ_col: str = "organ",
    cell_type_col: str = "cell_type",
) -> list[CompositionResult]:
    """Convenience function for composition analysis."""
    analyzer = CellCompositionAnalyzer()
    return analyzer.analyze(metadata, organ_col, cell_type_col)
