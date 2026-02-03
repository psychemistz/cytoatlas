"""
Multi-GPU parallel activity inference.

Distributes computation across multiple GPUs for large datasets.
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from cytoatlas_pipeline.activity.ridge import ActivityResult, RidgeInference


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU processing."""

    devices: list[int]
    """GPU device IDs to use."""

    samples_per_device: Optional[int] = None
    """Samples to process per device (auto if None)."""

    overlap_samples: int = 0
    """Overlapping samples between devices (for validation)."""


class ParallelRidgeInference:
    """
    Multi-GPU parallel activity inference.

    Splits samples across multiple GPUs for parallel processing.

    Example:
        >>> config = MultiGPUConfig(devices=[0, 1, 2, 3])
        >>> parallel = ParallelRidgeInference(config)
        >>> result = parallel.run(expression, signature)
    """

    def __init__(
        self,
        gpu_config: MultiGPUConfig,
        lambda_: float = 5e5,
        n_rand: int = 1000,
        seed: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize parallel inference.

        Args:
            gpu_config: Multi-GPU configuration.
            lambda_: Ridge regularization parameter.
            n_rand: Number of permutations.
            seed: Random seed.
            verbose: Print progress.
        """
        self.gpu_config = gpu_config
        self.lambda_ = lambda_
        self.n_rand = n_rand
        self.seed = seed
        self.verbose = verbose

    def run(
        self,
        expression: pd.DataFrame,
        signature: pd.DataFrame,
    ) -> ActivityResult:
        """
        Run parallel activity inference.

        Args:
            expression: Expression matrix (genes x samples).
            signature: Signature matrix (genes x signatures).

        Returns:
            Combined ActivityResult.
        """
        n_samples = expression.shape[1]
        n_devices = len(self.gpu_config.devices)

        # Calculate samples per device
        if self.gpu_config.samples_per_device is not None:
            samples_per_device = self.gpu_config.samples_per_device
        else:
            samples_per_device = math.ceil(n_samples / n_devices)

        # Create sample splits
        splits = []
        for i, device_id in enumerate(self.gpu_config.devices):
            start_idx = i * samples_per_device
            end_idx = min(start_idx + samples_per_device, n_samples)

            if start_idx >= n_samples:
                break

            splits.append({
                "device_id": device_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "sample_cols": list(expression.columns[start_idx:end_idx]),
            })

        if self.verbose:
            print(f"Parallel inference: {n_samples} samples across {len(splits)} devices")

        # Run in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(splits)) as executor:
            futures = {
                executor.submit(
                    self._run_on_device,
                    expression[split["sample_cols"]],
                    signature,
                    split["device_id"],
                ): split
                for split in splits
            }

            for future in concurrent.futures.as_completed(futures):
                split = futures[future]
                try:
                    result = future.result()
                    results.append((split["start_idx"], result))
                except Exception as e:
                    raise RuntimeError(
                        f"Failed on device {split['device_id']}: {e}"
                    ) from e

        # Sort by start index and combine
        results.sort(key=lambda x: x[0])

        return self._combine_results([r[1] for r in results])

    def _run_on_device(
        self,
        expression: pd.DataFrame,
        signature: pd.DataFrame,
        device_id: int,
    ) -> ActivityResult:
        """Run inference on a specific GPU."""
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        inference = RidgeInference(
            lambda_=self.lambda_,
            n_rand=self.n_rand,
            seed=self.seed,
            backend="cupy",
            verbose=self.verbose,
        )

        return inference.run(expression, signature)

    def _combine_results(self, results: list[ActivityResult]) -> ActivityResult:
        """Combine results from multiple devices."""
        # Concatenate DataFrames horizontally
        beta = pd.concat([r.beta for r in results], axis=1)
        se = pd.concat([r.se for r in results], axis=1)
        zscore = pd.concat([r.zscore for r in results], axis=1)
        pvalue = pd.concat([r.pvalue for r in results], axis=1)

        return ActivityResult(
            beta=beta,
            se=se,
            zscore=zscore,
            pvalue=pvalue,
            signature_names=results[0].signature_names,
            sample_names=list(beta.columns),
            n_signatures=results[0].n_signatures,
            n_samples=beta.shape[1],
            n_genes_used=results[0].n_genes_used,
            gene_overlap=results[0].gene_overlap,
            method="multi_gpu",
            time_seconds=max(r.time_seconds for r in results),
            metadata={"n_devices": len(results)},
        )
