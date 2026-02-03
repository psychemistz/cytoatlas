"""
Validation pipeline for activity inference credibility assessment.

Implements 5-type validation:
1. Sample-level: Expression vs activity regression
2. Cell-type level: Pseudobulk correlation
3. Pseudobulk vs single-cell: Aggregation comparison
4. Single-cell: Expressing vs non-expressing cells
5. Biological: Known marker validation
"""

from cytoatlas_pipeline.validation.sample_level import (
    SampleLevelValidator,
    validate_sample_level,
)
from cytoatlas_pipeline.validation.celltype_level import (
    CellTypeLevelValidator,
    validate_celltype_level,
)
from cytoatlas_pipeline.validation.pseudobulk_vs_sc import (
    PseudobulkSCValidator,
    validate_pseudobulk_vs_sc,
)
from cytoatlas_pipeline.validation.singlecell import (
    SingleCellValidator,
    validate_singlecell,
)
from cytoatlas_pipeline.validation.biological import (
    BiologicalValidator,
    validate_biological,
    KNOWN_MARKERS,
)
from cytoatlas_pipeline.validation.gene_coverage import (
    GeneCoverageValidator,
    validate_gene_coverage,
)
from cytoatlas_pipeline.validation.cv_stability import (
    CVStabilityValidator,
    validate_cv_stability,
)
from cytoatlas_pipeline.validation.quality_score import (
    QualityScorer,
    compute_quality_score,
    ValidationSummary,
)

__all__ = [
    # Sample level
    "SampleLevelValidator",
    "validate_sample_level",
    # Cell type level
    "CellTypeLevelValidator",
    "validate_celltype_level",
    # Pseudobulk vs SC
    "PseudobulkSCValidator",
    "validate_pseudobulk_vs_sc",
    # Single cell
    "SingleCellValidator",
    "validate_singlecell",
    # Biological
    "BiologicalValidator",
    "validate_biological",
    "KNOWN_MARKERS",
    # Gene coverage
    "GeneCoverageValidator",
    "validate_gene_coverage",
    # CV stability
    "CVStabilityValidator",
    "validate_cv_stability",
    # Quality score
    "QualityScorer",
    "compute_quality_score",
    "ValidationSummary",
]
