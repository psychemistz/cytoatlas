"""
Batch processing module for atlas-level pseudobulk generation.

This module provides GPU-accelerated streaming aggregation for large
single-cell atlases (millions of cells) with memory-efficient batch processing.
"""

from cytoatlas_pipeline.batch.atlas_config import (
    AtlasConfig,
    PseudobulkConfig,
    ActivityConfig,
    ATLAS_REGISTRY,
    get_atlas_config,
)
from cytoatlas_pipeline.batch.streaming_aggregator import (
    StreamingPseudobulkAggregator,
    aggregate_atlas_pseudobulk,
)

__all__ = [
    # Config classes
    "AtlasConfig",
    "PseudobulkConfig",
    "ActivityConfig",
    "ATLAS_REGISTRY",
    "get_atlas_config",
    # Aggregation
    "StreamingPseudobulkAggregator",
    "aggregate_atlas_pseudobulk",
]
