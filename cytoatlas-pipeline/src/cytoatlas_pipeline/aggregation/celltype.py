"""
Cell type hierarchy and aggregation.

Provides cell type mapping between annotation levels (L1/L2/L3)
and aggregation at each level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sp

from cytoatlas_pipeline.aggregation.base import (
    AggregatedData,
    AggregationConfig,
    AggregationStrategy,
)


@dataclass
class CellTypeLevel:
    """Definition of a cell type annotation level."""

    name: str
    """Level name (e.g., 'L1', 'L2', 'L3')."""

    column: str
    """Column name in obs."""

    n_types: int = 0
    """Number of unique types at this level."""


class CellTypeHierarchy:
    """
    Manages cell type hierarchy across annotation levels.

    Provides mapping between coarse (L1) and fine (L3) cell types.

    Example:
        >>> hierarchy = CellTypeHierarchy()
        >>> hierarchy.add_mapping("T cell", "CD4+ T cell", "Naive CD4+ T cell")
        >>> hierarchy.add_mapping("T cell", "CD4+ T cell", "Memory CD4+ T cell")
        >>>
        >>> parent = hierarchy.get_parent("Naive CD4+ T cell", to_level="L1")
        >>> # Returns "T cell"
    """

    # Default CytoAtlas cell type hierarchy
    DEFAULT_HIERARCHY = {
        # Lymphoid
        "T cell": {
            "CD4+ T cell": ["Naive CD4+ T", "Memory CD4+ T", "Treg", "Th1", "Th2", "Th17", "Tfh"],
            "CD8+ T cell": ["Naive CD8+ T", "Memory CD8+ T", "Effector CD8+ T", "Exhausted CD8+ T"],
            "γδ T cell": ["Vδ1 T cell", "Vδ2 T cell"],
            "NKT cell": ["Type I NKT", "Type II NKT"],
        },
        "NK cell": {
            "CD56bright NK": ["CD56bright NK"],
            "CD56dim NK": ["CD56dim NK"],
        },
        "B cell": {
            "Naive B cell": ["Naive B cell"],
            "Memory B cell": ["Memory B cell"],
            "Plasma cell": ["Plasma cell", "Plasmablast"],
        },
        # Myeloid
        "Monocyte": {
            "Classical monocyte": ["CD14+ monocyte"],
            "Non-classical monocyte": ["CD16+ monocyte"],
            "Intermediate monocyte": ["Intermediate monocyte"],
        },
        "Dendritic cell": {
            "cDC1": ["cDC1"],
            "cDC2": ["cDC2"],
            "pDC": ["pDC"],
        },
        "Macrophage": {
            "M1 macrophage": ["M1 macrophage"],
            "M2 macrophage": ["M2 macrophage"],
            "Tissue macrophage": ["Alveolar macrophage", "Kupffer cell", "Microglia"],
        },
        "Granulocyte": {
            "Neutrophil": ["Neutrophil"],
            "Eosinophil": ["Eosinophil"],
            "Basophil": ["Basophil"],
            "Mast cell": ["Mast cell"],
        },
    }

    def __init__(self, hierarchy: Optional[Dict[str, Dict[str, list]]] = None):
        """
        Initialize cell type hierarchy.

        Args:
            hierarchy: Custom hierarchy dict. None uses default.
        """
        self.hierarchy = hierarchy or self.DEFAULT_HIERARCHY

        # Build reverse mappings
        self._l3_to_l2: Dict[str, str] = {}
        self._l3_to_l1: Dict[str, str] = {}
        self._l2_to_l1: Dict[str, str] = {}

        for l1, l2_dict in self.hierarchy.items():
            for l2, l3_list in l2_dict.items():
                self._l2_to_l1[l2] = l1
                for l3 in l3_list:
                    self._l3_to_l2[l3] = l2
                    self._l3_to_l1[l3] = l1

    @property
    def l1_types(self) -> list[str]:
        """Get L1 (coarse) cell types."""
        return list(self.hierarchy.keys())

    @property
    def l2_types(self) -> list[str]:
        """Get L2 (intermediate) cell types."""
        return list(self._l2_to_l1.keys())

    @property
    def l3_types(self) -> list[str]:
        """Get L3 (fine) cell types."""
        return list(self._l3_to_l2.keys())

    def get_parent(self, cell_type: str, to_level: str = "L1") -> Optional[str]:
        """
        Get parent cell type at specified level.

        Args:
            cell_type: Cell type name.
            to_level: Target level ("L1" or "L2").

        Returns:
            Parent cell type, or None if not found.
        """
        if to_level == "L1":
            # Check if it's L3
            if cell_type in self._l3_to_l1:
                return self._l3_to_l1[cell_type]
            # Check if it's L2
            if cell_type in self._l2_to_l1:
                return self._l2_to_l1[cell_type]
            # Might already be L1
            if cell_type in self.hierarchy:
                return cell_type
        elif to_level == "L2":
            if cell_type in self._l3_to_l2:
                return self._l3_to_l2[cell_type]
            # Might already be L2
            if cell_type in self._l2_to_l1:
                return cell_type

        return None

    def get_children(self, cell_type: str) -> list[str]:
        """
        Get child cell types.

        Args:
            cell_type: Cell type name.

        Returns:
            List of child cell types.
        """
        # L1 -> L2
        if cell_type in self.hierarchy:
            return list(self.hierarchy[cell_type].keys())

        # L2 -> L3
        for l1, l2_dict in self.hierarchy.items():
            if cell_type in l2_dict:
                return l2_dict[cell_type]

        return []

    def map_to_level(
        self,
        cell_types: pd.Series,
        target_level: str = "L1",
    ) -> pd.Series:
        """
        Map cell type series to target level.

        Args:
            cell_types: Series of cell type annotations.
            target_level: Target level ("L1", "L2", or "L3").

        Returns:
            Mapped cell types.
        """
        if target_level == "L1":
            return cell_types.map(
                lambda x: self.get_parent(x, "L1") or x
            )
        elif target_level == "L2":
            return cell_types.map(
                lambda x: self.get_parent(x, "L2") or x
            )
        else:
            return cell_types

    def add_mapping(self, l1: str, l2: str, l3: str) -> None:
        """
        Add a cell type mapping.

        Args:
            l1: L1 (coarse) cell type.
            l2: L2 (intermediate) cell type.
            l3: L3 (fine) cell type.
        """
        if l1 not in self.hierarchy:
            self.hierarchy[l1] = {}
        if l2 not in self.hierarchy[l1]:
            self.hierarchy[l1][l2] = []
        if l3 not in self.hierarchy[l1][l2]:
            self.hierarchy[l1][l2].append(l3)

        # Update reverse mappings
        self._l2_to_l1[l2] = l1
        self._l3_to_l2[l3] = l2
        self._l3_to_l1[l3] = l1


class CellTypeAggregator(AggregationStrategy):
    """
    Aggregates expression by cell type.

    Supports aggregation at multiple hierarchy levels.

    Example:
        >>> aggregator = CellTypeAggregator(level="L2")
        >>> result = aggregator.aggregate(X, obs, var)
    """

    def __init__(
        self,
        config: Optional[AggregationConfig] = None,
        level: str = "L2",
        hierarchy: Optional[CellTypeHierarchy] = None,
    ):
        """
        Initialize cell type aggregator.

        Args:
            config: Aggregation configuration.
            level: Cell type level to aggregate at ("L1", "L2", "L3").
            hierarchy: Cell type hierarchy (None uses default).
        """
        super().__init__(config)
        self.level = level
        self.hierarchy = hierarchy or CellTypeHierarchy()

    def aggregate(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: pd.DataFrame,
        var: pd.DataFrame,
    ) -> AggregatedData:
        """
        Aggregate by cell type.

        Args:
            X: Expression matrix (cells x genes).
            obs: Cell metadata.
            var: Gene metadata.

        Returns:
            Aggregated data at specified level.
        """
        # Get cell types at target level
        if self.level in ("L1", "L2") and self.config.cell_type_col in obs.columns:
            cell_types = self.hierarchy.map_to_level(
                obs[self.config.cell_type_col],
                self.level,
            )
        else:
            cell_types = obs[self.config.cell_type_col]

        # Group cells by type
        unique_types = cell_types.unique()
        gene_names = list(var.index)

        aggregated = {}
        metadata_rows = []
        stats = {"cells_per_type": {}}

        for ct in unique_types:
            mask = (cell_types == ct).values
            n_cells = mask.sum()

            if n_cells < self.config.min_cells:
                continue

            # Sum expression
            if sp.issparse(X):
                ct_sum = np.asarray(X[mask].sum(axis=0)).ravel()
            else:
                ct_sum = X[mask].sum(axis=0)

            aggregated[ct] = ct_sum
            metadata_rows.append({
                "cell_type": ct,
                "level": self.level,
                "n_cells": n_cells,
            })
            stats["cells_per_type"][ct] = n_cells

        # Create expression DataFrame
        expr_df = pd.DataFrame(aggregated, index=gene_names)

        # Process expression
        expr_df = self.process_expression(expr_df)

        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.index = metadata_df["cell_type"]

        return AggregatedData(
            expression=expr_df,
            metadata=metadata_df,
            gene_names=gene_names,
            n_units=len(aggregated),
            n_genes=len(gene_names),
            aggregation_type=f"celltype_{self.level}",
            config=self.config,
            stats=stats,
        )
