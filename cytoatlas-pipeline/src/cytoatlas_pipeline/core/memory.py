"""
Memory estimation utilities.

Helps determine optimal batch sizes and predict memory requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np


@dataclass
class MemoryEstimate:
    """Memory estimation results."""

    total_gb: float
    """Total estimated memory in GB."""

    peak_gb: float
    """Peak memory usage in GB."""

    per_batch_gb: float
    """Memory per batch in GB."""

    components: dict[str, float]
    """Breakdown by component (in GB)."""

    recommended_batch_size: int
    """Recommended batch size for available memory."""


class MemoryEstimator:
    """
    Estimates memory requirements for pipeline operations.

    Based on SecActpy's batch.py memory estimation patterns.

    Example:
        >>> estimator = MemoryEstimator(available_gb=16.0, safety_factor=0.8)
        >>> estimate = estimator.estimate_activity(
        ...     n_genes=20000,
        ...     n_features=50,
        ...     n_samples=100000,
        ...     n_rand=1000
        ... )
        >>> print(f"Recommended batch size: {estimate.recommended_batch_size}")
    """

    BYTES_PER_FLOAT64 = 8
    BYTES_PER_FLOAT32 = 4
    BYTES_PER_INT64 = 8
    BYTES_PER_INT32 = 4

    def __init__(
        self,
        available_gb: float = 8.0,
        safety_factor: float = 0.7,
        use_float32: bool = False,
    ):
        """
        Initialize memory estimator.

        Args:
            available_gb: Available memory in GB.
            safety_factor: Fraction of available memory to use (0-1).
            use_float32: Use float32 instead of float64.
        """
        self.available_gb = available_gb
        self.safety_factor = safety_factor
        self.bytes_per_float = self.BYTES_PER_FLOAT32 if use_float32 else self.BYTES_PER_FLOAT64

    def _to_gb(self, bytes_: int) -> float:
        """Convert bytes to gigabytes."""
        return bytes_ / (1024**3)

    def estimate_activity(
        self,
        n_genes: int,
        n_features: int,
        n_samples: int,
        n_rand: int = 1000,
        batch_size: Optional[int] = None,
        include_gpu: bool = False,
    ) -> MemoryEstimate:
        """
        Estimate memory for activity inference.

        Args:
            n_genes: Number of genes.
            n_features: Number of features/proteins.
            n_samples: Number of samples.
            n_rand: Number of permutations.
            batch_size: Batch size (None for auto).
            include_gpu: Include GPU memory in estimate.

        Returns:
            MemoryEstimate with breakdown and recommendations.
        """
        bpf = self.bytes_per_float

        # Fixed memory components
        T_bytes = n_features * n_genes * bpf  # Projection matrix
        perm_bytes = n_rand * n_genes * self.BYTES_PER_INT32  # Permutation table

        # Per-sample memory
        Y_per_sample = n_genes * bpf  # Expression data
        result_per_sample = 4 * n_features * bpf  # beta, se, zscore, pvalue

        # Working memory per batch
        def working_bytes(bs: int) -> int:
            return (
                3 * n_features * bs * bpf  # aver, aver_sq, pvalue_counts
                + n_genes * bs * bpf  # Y batch
                + n_features * bs * bpf  # beta batch
            )

        # Determine batch size if not provided
        if batch_size is None:
            batch_size = self.estimate_batch_size(
                n_genes=n_genes,
                n_features=n_features,
                n_rand=n_rand,
            )

        # Calculate totals
        fixed_bytes = T_bytes + perm_bytes
        batch_bytes = working_bytes(batch_size)
        total_result_bytes = n_samples * result_per_sample

        components = {
            "T_matrix": self._to_gb(T_bytes),
            "permutation_table": self._to_gb(perm_bytes),
            "results": self._to_gb(total_result_bytes),
            "working_per_batch": self._to_gb(batch_bytes),
        }

        total_gb = self._to_gb(fixed_bytes + batch_bytes + total_result_bytes)
        peak_gb = self._to_gb(fixed_bytes + batch_bytes)
        per_batch_gb = self._to_gb(batch_bytes)

        return MemoryEstimate(
            total_gb=total_gb,
            peak_gb=peak_gb,
            per_batch_gb=per_batch_gb,
            components=components,
            recommended_batch_size=batch_size,
        )

    def estimate_batch_size(
        self,
        n_genes: int,
        n_features: int,
        n_rand: int = 1000,
        min_batch: int = 100,
        max_batch: int = 50000,
    ) -> int:
        """
        Estimate optimal batch size given available memory.

        Args:
            n_genes: Number of genes.
            n_features: Number of features.
            n_rand: Number of permutations.
            min_batch: Minimum batch size.
            max_batch: Maximum batch size.

        Returns:
            Recommended batch size.
        """
        bpf = self.bytes_per_float
        available_bytes = self.available_gb * (1024**3) * self.safety_factor

        # Fixed memory
        T_bytes = n_features * n_genes * bpf
        perm_bytes = n_rand * n_genes * self.BYTES_PER_INT32
        fixed_bytes = T_bytes + perm_bytes

        # Available for batches
        batch_budget = available_bytes - fixed_bytes

        if batch_budget <= 0:
            warnings.warn(
                f"Available memory ({self.available_gb}GB) may be insufficient. "
                f"T matrix alone requires {self._to_gb(T_bytes):.2f}GB."
            )
            return min_batch

        # Memory per sample in batch
        per_sample_bytes = (
            n_genes * bpf  # Y
            + 4 * n_features * bpf  # accumulators
        )

        batch_size = int(batch_budget / per_sample_bytes)
        batch_size = max(min_batch, min(max_batch, batch_size))

        return batch_size

    def estimate_correlation(
        self,
        n_signatures: int,
        n_samples: int,
        n_features: int,
        batch_size: Optional[int] = None,
    ) -> MemoryEstimate:
        """
        Estimate memory for correlation analysis.

        Args:
            n_signatures: Number of activity signatures.
            n_samples: Number of samples.
            n_features: Number of features to correlate with.
            batch_size: Batch size for computation.

        Returns:
            MemoryEstimate.
        """
        bpf = self.bytes_per_float

        if batch_size is None:
            batch_size = min(n_samples, 10000)

        # Activity matrix
        activity_bytes = n_signatures * n_samples * bpf

        # Feature matrix
        feature_bytes = n_features * n_samples * bpf

        # Output correlation matrix
        output_bytes = n_signatures * n_features * bpf

        # Working memory (ranks, etc.)
        working_bytes = 2 * n_signatures * batch_size * bpf

        components = {
            "activity_matrix": self._to_gb(activity_bytes),
            "feature_matrix": self._to_gb(feature_bytes),
            "output": self._to_gb(output_bytes),
            "working": self._to_gb(working_bytes),
        }

        total_gb = sum(components.values())
        peak_gb = total_gb  # All needed simultaneously for correlation

        return MemoryEstimate(
            total_gb=total_gb,
            peak_gb=peak_gb,
            per_batch_gb=self._to_gb(working_bytes),
            components=components,
            recommended_batch_size=batch_size,
        )

    def estimate_differential(
        self,
        n_signatures: int,
        n_samples_group1: int,
        n_samples_group2: int,
        n_permutations: int = 1000,
    ) -> MemoryEstimate:
        """
        Estimate memory for differential analysis.

        Args:
            n_signatures: Number of signatures.
            n_samples_group1: Samples in group 1.
            n_samples_group2: Samples in group 2.
            n_permutations: Permutations for statistical test.

        Returns:
            MemoryEstimate.
        """
        bpf = self.bytes_per_float
        n_total = n_samples_group1 + n_samples_group2

        # Activity data
        activity_bytes = n_signatures * n_total * bpf

        # Ranks for Wilcoxon
        rank_bytes = n_signatures * n_total * bpf

        # Permutation counts
        perm_bytes = n_signatures * bpf

        # Output
        output_bytes = n_signatures * 4 * bpf  # statistic, pvalue, effect_size, qvalue

        components = {
            "activity_data": self._to_gb(activity_bytes),
            "ranks": self._to_gb(rank_bytes),
            "permutation_stats": self._to_gb(perm_bytes),
            "output": self._to_gb(output_bytes),
        }

        total_gb = sum(components.values())

        return MemoryEstimate(
            total_gb=total_gb,
            peak_gb=total_gb,
            per_batch_gb=self._to_gb(rank_bytes),
            components=components,
            recommended_batch_size=n_total,  # Usually done in one pass
        )


def estimate_memory(
    n_genes: int,
    n_features: int,
    n_samples: int,
    n_rand: int = 1000,
    batch_size: Optional[int] = None,
    include_gpu: bool = False,
) -> dict[str, float]:
    """
    Quick memory estimation for activity inference.

    Convenience function matching SecActpy's estimate_memory interface.

    Args:
        n_genes: Number of genes.
        n_features: Number of features.
        n_samples: Number of samples.
        n_rand: Number of permutations.
        batch_size: Batch size (None for full dataset).
        include_gpu: Include GPU memory estimates.

    Returns:
        Dict with memory estimates in GB.
    """
    estimator = MemoryEstimator()
    estimate = estimator.estimate_activity(
        n_genes=n_genes,
        n_features=n_features,
        n_samples=n_samples,
        n_rand=n_rand,
        batch_size=batch_size,
        include_gpu=include_gpu,
    )

    result = {
        "T_matrix": estimate.components.get("T_matrix", 0),
        "results": estimate.components.get("results", 0),
        "working": estimate.components.get("working_per_batch", 0),
        "per_batch": estimate.per_batch_gb,
        "total": estimate.total_gb,
    }

    if include_gpu:
        result["gpu_per_batch"] = estimate.per_batch_gb

    return result
