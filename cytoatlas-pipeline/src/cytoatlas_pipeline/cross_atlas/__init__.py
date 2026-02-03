"""
Cross-atlas integration pipeline.

Harmonizes and compares activities across multiple atlases.
"""

from cytoatlas_pipeline.cross_atlas.harmonization import (
    ActivityHarmonizer,
    harmonize_activities,
)
from cytoatlas_pipeline.cross_atlas.celltype_mapping import (
    CellTypeMapper,
    map_cell_types,
)
from cytoatlas_pipeline.cross_atlas.conserved import (
    ConservedSignatureDetector,
    detect_conserved_signatures,
)
from cytoatlas_pipeline.cross_atlas.meta_analysis import (
    MetaAnalyzer,
    run_meta_analysis,
)
from cytoatlas_pipeline.cross_atlas.consistency import (
    ConsistencyScorer,
    compute_consistency,
)

__all__ = [
    # Harmonization
    "ActivityHarmonizer",
    "harmonize_activities",
    # Cell type mapping
    "CellTypeMapper",
    "map_cell_types",
    # Conserved signatures
    "ConservedSignatureDetector",
    "detect_conserved_signatures",
    # Meta-analysis
    "MetaAnalyzer",
    "run_meta_analysis",
    # Consistency
    "ConsistencyScorer",
    "compute_consistency",
]
