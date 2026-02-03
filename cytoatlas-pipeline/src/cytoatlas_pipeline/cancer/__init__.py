"""
Cancer analysis pipeline.

Specialized analyses for tumor microenvironment and cancer patterns.
"""

from cytoatlas_pipeline.cancer.tumor_adjacent import (
    TumorAdjacentAnalyzer,
    tumor_vs_adjacent,
)
from cytoatlas_pipeline.cancer.infiltration import (
    ImmuneInfiltrationAnalyzer,
    compute_infiltration,
)
from cytoatlas_pipeline.cancer.exhaustion import (
    ExhaustionAnalyzer,
    analyze_exhaustion,
)

__all__ = [
    "TumorAdjacentAnalyzer",
    "tumor_vs_adjacent",
    "ImmuneInfiltrationAnalyzer",
    "compute_infiltration",
    "ExhaustionAnalyzer",
    "analyze_exhaustion",
]
