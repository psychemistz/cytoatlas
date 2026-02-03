"""
Organ and tissue analysis pipeline.

Specialized analyses for organ-specific signatures and tissue composition.
"""

from cytoatlas_pipeline.organ.signatures import (
    OrganSignatureAnalyzer,
    compute_organ_signatures,
)
from cytoatlas_pipeline.organ.specificity import (
    TissueSpecificityScorer,
    compute_specificity,
)
from cytoatlas_pipeline.organ.cell_composition import (
    CellCompositionAnalyzer,
    analyze_composition,
)

__all__ = [
    "OrganSignatureAnalyzer",
    "compute_organ_signatures",
    "TissueSpecificityScorer",
    "compute_specificity",
    "CellCompositionAnalyzer",
    "analyze_composition",
]
