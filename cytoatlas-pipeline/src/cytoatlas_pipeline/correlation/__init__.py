"""
Correlation analysis pipeline.

GPU-accelerated correlation computation for activity vs phenotype associations.
"""

from cytoatlas_pipeline.correlation.pearson import (
    pearson_correlation,
    PearsonCorrelator,
)
from cytoatlas_pipeline.correlation.spearman import (
    spearman_correlation,
    SpearmanCorrelator,
)
from cytoatlas_pipeline.correlation.partial import (
    partial_correlation,
    PartialCorrelator,
)
from cytoatlas_pipeline.correlation.continuous import (
    ContinuousCorrelator,
    correlate_with_continuous,
    CorrelationResult,
)
from cytoatlas_pipeline.correlation.biochemistry import (
    BiochemistryCorrelator,
    correlate_with_biochemistry,
)

__all__ = [
    # Pearson
    "pearson_correlation",
    "PearsonCorrelator",
    # Spearman
    "spearman_correlation",
    "SpearmanCorrelator",
    # Partial
    "partial_correlation",
    "PartialCorrelator",
    # Continuous
    "ContinuousCorrelator",
    "correlate_with_continuous",
    "CorrelationResult",
    # Biochemistry
    "BiochemistryCorrelator",
    "correlate_with_biochemistry",
]
