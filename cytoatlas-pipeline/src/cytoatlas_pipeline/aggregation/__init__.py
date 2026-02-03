"""
Aggregation strategies for single-cell data.

Provides various methods for aggregating single-cell expression:
- Cell type hierarchy (L1/L2/L3 levels)
- Pseudobulk (cell type x sample)
- Bootstrap resampling
- Single-cell streaming
"""

from cytoatlas_pipeline.aggregation.base import (
    AggregationStrategy,
    AggregatedData,
    AggregationConfig,
)
from cytoatlas_pipeline.aggregation.celltype import (
    CellTypeHierarchy,
    CellTypeAggregator,
)
from cytoatlas_pipeline.aggregation.pseudobulk import (
    PseudobulkAggregator,
    aggregate_pseudobulk,
)
from cytoatlas_pipeline.aggregation.resampling import (
    BootstrapAggregator,
    ResamplingConfig,
)
from cytoatlas_pipeline.aggregation.singlecell import (
    SingleCellStreamer,
    StreamingConfig,
)

__all__ = [
    # Base
    "AggregationStrategy",
    "AggregatedData",
    "AggregationConfig",
    # Cell type
    "CellTypeHierarchy",
    "CellTypeAggregator",
    # Pseudobulk
    "PseudobulkAggregator",
    "aggregate_pseudobulk",
    # Resampling
    "BootstrapAggregator",
    "ResamplingConfig",
    # Single cell
    "SingleCellStreamer",
    "StreamingConfig",
]
