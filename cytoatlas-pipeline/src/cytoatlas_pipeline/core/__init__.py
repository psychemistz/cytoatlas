"""
Core infrastructure for cytoatlas-pipeline.

Provides:
- Configuration management
- GPU resource management
- Checkpointing and recovery
- Memory estimation
- Result caching
"""

from cytoatlas_pipeline.core.config import Config, PipelineConfig
from cytoatlas_pipeline.core.gpu_manager import GPUManager, get_gpu_manager
from cytoatlas_pipeline.core.checkpoint import CheckpointManager, Checkpoint
from cytoatlas_pipeline.core.memory import estimate_memory, MemoryEstimator
from cytoatlas_pipeline.core.cache import ResultCache

__all__ = [
    "Config",
    "PipelineConfig",
    "GPUManager",
    "get_gpu_manager",
    "CheckpointManager",
    "Checkpoint",
    "MemoryEstimator",
    "estimate_memory",
    "ResultCache",
]
