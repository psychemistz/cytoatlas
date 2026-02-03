"""
Data ingestion layer.

Provides unified interfaces for loading data from various sources:
- Local H5AD files (backed mode for large files)
- cellxgene Census (cloud-based single-cell data)
- Remote H5AD files (streaming)
- Various formats (Loom, 10X, etc.)
"""

from cytoatlas_pipeline.ingest.base import DataSource, DataChunk, DataSourceConfig
from cytoatlas_pipeline.ingest.local_h5ad import LocalH5ADSource
from cytoatlas_pipeline.ingest.cellxgene import CellxgeneCensusSource
from cytoatlas_pipeline.ingest.remote_h5ad import RemoteH5ADSource
from cytoatlas_pipeline.ingest.formats import (
    LoomSource,
    TenXSource,
    convert_to_anndata,
)

__all__ = [
    # Base
    "DataSource",
    "DataChunk",
    "DataSourceConfig",
    # Sources
    "LocalH5ADSource",
    "CellxgeneCensusSource",
    "RemoteH5ADSource",
    "LoomSource",
    "TenXSource",
    # Utilities
    "convert_to_anndata",
]
