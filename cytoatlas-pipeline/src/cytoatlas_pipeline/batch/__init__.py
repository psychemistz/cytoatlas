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
from cytoatlas_pipeline.batch.multi_level_aggregator import (
    MultiLevelAggregator,
    aggregate_all_levels,
)

__all__ = [
    # Config classes
    "AtlasConfig",
    "PseudobulkConfig",
    "ActivityConfig",
    "ATLAS_REGISTRY",
    "get_atlas_config",
    # Single-level aggregation
    "StreamingPseudobulkAggregator",
    "aggregate_atlas_pseudobulk",
    # Multi-level aggregation (single pass)
    "MultiLevelAggregator",
    "aggregate_all_levels",
]
