"""
Activity inference pipeline.

GPU-accelerated signature activity scoring using ridge regression.
"""

from cytoatlas_pipeline.activity.ridge import (
    RidgeInference,
    ActivityResult,
    run_ridge_inference,
)
from cytoatlas_pipeline.activity.parallel import (
    ParallelRidgeInference,
    MultiGPUConfig,
)
from cytoatlas_pipeline.activity.streaming import (
    StreamingResultWriter,
    ActivityStreamWriter,
)
from cytoatlas_pipeline.activity.signatures import (
    SignatureLoader,
    load_cytosig,
    load_secact,
    load_custom_signature,
)

__all__ = [
    # Ridge inference
    "RidgeInference",
    "ActivityResult",
    "run_ridge_inference",
    # Parallel processing
    "ParallelRidgeInference",
    "MultiGPUConfig",
    # Streaming
    "StreamingResultWriter",
    "ActivityStreamWriter",
    # Signatures
    "SignatureLoader",
    "load_cytosig",
    "load_secact",
    "load_custom_signature",
]
