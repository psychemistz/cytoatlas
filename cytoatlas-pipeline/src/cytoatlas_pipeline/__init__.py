"""
CytoAtlas Pipeline - GPU-Accelerated Data Processing for Single-Cell Cytokine Activity Analysis.

This package provides comprehensive pipelines for:
- Activity inference (ridge regression with permutation testing)
- Correlation analysis (Pearson, Spearman, partial)
- Differential analysis (Wilcoxon, t-test, effect size)
- Validation (5-type credibility assessment)
- Cross-atlas integration
- Disease and cancer analysis
- Organ/tissue analysis
- Search indexing
- Multi-format export (JSON, CSV, H5AD, Parquet)

Example:
    >>> from cytoatlas_pipeline import Pipeline, Config
    >>> from cytoatlas_pipeline.ingest import LocalH5ADSource
    >>>
    >>> config = Config(gpu_devices=[0], batch_size=10000)
    >>> pipeline = Pipeline(config)
    >>>
    >>> source = LocalH5ADSource("/path/to/data.h5ad")
    >>> results = pipeline.process(
    ...     source=source,
    ...     signatures=["CytoSig"],
    ...     analyses=["activity", "correlation"]
    ... )
"""

__version__ = "0.1.0"

# Core infrastructure
from cytoatlas_pipeline.core.config import Config, PipelineConfig
from cytoatlas_pipeline.core.gpu_manager import GPUManager, get_gpu_manager
from cytoatlas_pipeline.core.checkpoint import CheckpointManager
from cytoatlas_pipeline.core.memory import estimate_memory, MemoryEstimator
from cytoatlas_pipeline.core.cache import ResultCache

# Subpackages are imported as needed to avoid heavy startup cost
# Use explicit imports for specific functionality:
#   from cytoatlas_pipeline.ingest import LocalH5ADSource
#   from cytoatlas_pipeline.activity import RidgeInference
#   from cytoatlas_pipeline.correlation import spearman_correlation
#   from cytoatlas_pipeline.differential import wilcoxon_test
#   from cytoatlas_pipeline.validation import QualityScorer
#   from cytoatlas_pipeline.cross_atlas import ActivityHarmonizer
#   from cytoatlas_pipeline.disease import DiseaseActivityAnalyzer
#   from cytoatlas_pipeline.cancer import ExhaustionAnalyzer
#   from cytoatlas_pipeline.organ import OrganSignatureAnalyzer
#   from cytoatlas_pipeline.search import SearchIndexer
#   from cytoatlas_pipeline.export import JSONWriter

# Main Pipeline class
from cytoatlas_pipeline.pipeline import Pipeline, PipelineResult, create_pipeline

__all__ = [
    # Version
    "__version__",
    # Pipeline
    "Pipeline",
    "PipelineResult",
    "create_pipeline",
    # Core
    "Config",
    "PipelineConfig",
    "GPUManager",
    "get_gpu_manager",
    "CheckpointManager",
    "MemoryEstimator",
    "estimate_memory",
    "ResultCache",
]
