"""
Pipeline configuration management.

Provides dataclass-based configuration with validation and serialization.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal, Any
import json
import os


@dataclass
class GPUConfig:
    """GPU-specific configuration."""

    devices: list[int] = field(default_factory=lambda: [0])
    """GPU device IDs to use."""

    memory_fraction: float = 0.9
    """Fraction of GPU memory to use (0-1)."""

    allow_growth: bool = True
    """Allow memory to grow dynamically."""

    fallback_to_cpu: bool = True
    """Fall back to CPU if GPU unavailable."""


@dataclass
class BatchConfig:
    """Batch processing configuration."""

    batch_size: int = 10000
    """Number of samples per batch."""

    min_batch_size: int = 100
    """Minimum batch size."""

    max_batch_size: int = 50000
    """Maximum batch size."""

    auto_batch_size: bool = True
    """Automatically determine batch size based on available memory."""


@dataclass
class RidgeConfig:
    """Ridge regression configuration."""

    lambda_: float = 5e5
    """Ridge regularization parameter."""

    n_rand: int = 1000
    """Number of permutations for significance testing."""

    seed: int = 0
    """Random seed for reproducibility."""

    use_gsl_rng: bool = False
    """Use GSL-compatible RNG for exact R/RidgeR reproducibility."""

    use_cache: bool = True
    """Cache permutation tables for reuse."""


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    enabled: bool = True
    """Enable checkpointing."""

    interval_seconds: int = 300
    """Checkpoint interval in seconds."""

    checkpoint_dir: Optional[Path] = None
    """Directory for checkpoint files."""

    max_checkpoints: int = 3
    """Maximum number of checkpoints to keep."""


@dataclass
class CacheConfig:
    """Result caching configuration."""

    enabled: bool = True
    """Enable result caching."""

    cache_dir: Optional[Path] = None
    """Directory for cached results."""

    max_size_gb: float = 10.0
    """Maximum cache size in GB."""

    ttl_hours: int = 24
    """Time-to-live for cached results in hours."""


@dataclass
class Config:
    """
    Main pipeline configuration.

    Example:
        >>> config = Config(
        ...     gpu_devices=[0, 1],
        ...     batch_size=10000,
        ...     n_rand=1000,
        ... )
        >>> pipeline = Pipeline(config)
    """

    # Convenience shortcuts (these override sub-config values)
    gpu_devices: list[int] = field(default_factory=lambda: [0])
    """GPU device IDs to use."""

    batch_size: int = 10000
    """Number of samples per batch."""

    n_rand: int = 1000
    """Number of permutations for significance testing."""

    seed: int = 0
    """Random seed for reproducibility."""

    # Sub-configurations
    gpu: GPUConfig = field(default_factory=GPUConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Output settings
    output_dir: Optional[Path] = None
    """Base output directory."""

    output_format: Literal["json", "csv", "h5ad", "parquet"] = "json"
    """Default output format."""

    compression: Optional[str] = "gzip"
    """Compression for output files."""

    # Logging
    verbose: bool = False
    """Enable verbose logging."""

    log_file: Optional[Path] = None
    """Log file path."""

    def __post_init__(self):
        """Synchronize shortcut values with sub-configs."""
        # Sync GPU devices
        self.gpu.devices = self.gpu_devices

        # Sync batch size
        self.batch.batch_size = self.batch_size

        # Sync ridge parameters
        self.ridge.n_rand = self.n_rand
        self.ridge.seed = self.seed

        # Convert paths
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if self.checkpoint.checkpoint_dir is not None:
            self.checkpoint.checkpoint_dir = Path(self.checkpoint.checkpoint_dir)
        if self.cache.cache_dir is not None:
            self.cache.cache_dir = Path(self.cache.cache_dir)
        if self.log_file is not None:
            self.log_file = Path(self.log_file)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        # Convert Path objects to strings
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, Path):
                        d[key][k] = str(v)
        return d

    def to_json(self, path: Path | str) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Create from dictionary."""
        # Handle nested configs
        if "gpu" in d and isinstance(d["gpu"], dict):
            d["gpu"] = GPUConfig(**d["gpu"])
        if "batch" in d and isinstance(d["batch"], dict):
            d["batch"] = BatchConfig(**d["batch"])
        if "ridge" in d and isinstance(d["ridge"], dict):
            d["ridge"] = RidgeConfig(**d["ridge"])
        if "checkpoint" in d and isinstance(d["checkpoint"], dict):
            d["checkpoint"] = CheckpointConfig(**d["checkpoint"])
        if "cache" in d and isinstance(d["cache"], dict):
            d["cache"] = CacheConfig(**d["cache"])
        return cls(**d)

    @classmethod
    def from_json(cls, path: Path | str) -> "Config":
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            gpu_devices=[int(x) for x in os.getenv("CYTOATLAS_GPU_DEVICES", "0").split(",")],
            batch_size=int(os.getenv("CYTOATLAS_BATCH_SIZE", "10000")),
            n_rand=int(os.getenv("CYTOATLAS_N_RAND", "1000")),
            seed=int(os.getenv("CYTOATLAS_SEED", "0")),
            verbose=os.getenv("CYTOATLAS_VERBOSE", "").lower() in ("1", "true", "yes"),
        )


# Alias for backwards compatibility
PipelineConfig = Config
