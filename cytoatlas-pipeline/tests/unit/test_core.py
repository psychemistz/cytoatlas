"""Tests for core infrastructure modules."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


class TestConfig:
    """Test configuration classes."""

    def test_default_config(self):
        from cytoatlas_pipeline.core.config import Config
        config = Config()
        assert config.gpu is not None
        assert config.batch is not None

    def test_gpu_config(self):
        from cytoatlas_pipeline.core.config import GPUConfig
        config = GPUConfig(devices=[0, 1], memory_fraction=0.8)
        assert config.devices == [0, 1]
        assert config.memory_fraction == 0.8

    def test_batch_config(self):
        from cytoatlas_pipeline.core.config import BatchConfig
        config = BatchConfig(batch_size=5000)
        assert config.batch_size == 5000


class TestGPUManager:
    """Test GPU manager."""

    def test_fallback_to_numpy(self):
        from cytoatlas_pipeline.core.gpu_manager import GPUManager
        from cytoatlas_pipeline.core.config import GPUConfig

        # Force CPU mode with empty devices
        config = GPUConfig(devices=[], fallback_to_cpu=True)
        manager = GPUManager(config)

        xp = manager.xp  # Uses xp property
        assert xp is np

    def test_to_numpy(self):
        from cytoatlas_pipeline.core.gpu_manager import GPUManager
        from cytoatlas_pipeline.core.config import GPUConfig

        config = GPUConfig(fallback_to_cpu=True)
        manager = GPUManager(config)

        arr = np.array([1, 2, 3])
        result = manager.to_cpu(arr)
        np.testing.assert_array_equal(result, arr)


class TestCheckpointManager:
    """Test checkpoint functionality."""

    def test_save_load_checkpoint(self):
        from cytoatlas_pipeline.core.checkpoint import CheckpointManager, Checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir), job_id="test_job")

            checkpoint = Checkpoint(
                job_id="test_job",
                pipeline_name="test",
                step_name="loading",
                step_index=1,
                total_steps=5,
                progress=0.5,
                state={"value": 42},
            )
            manager.save(checkpoint, force=True)

            loaded = manager.load_latest()
            assert loaded is not None
            assert loaded.step_index == 1
            assert loaded.step_name == "loading"

    def test_checkpoint_cleanup(self):
        from cytoatlas_pipeline.core.checkpoint import CheckpointManager, Checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir), job_id="cleanup_test", max_checkpoints=2)

            # Create multiple checkpoints
            for i in range(5):
                checkpoint = Checkpoint(
                    job_id="cleanup_test",
                    pipeline_name="test",
                    step_name=f"step_{i}",
                    step_index=i,
                    total_steps=5,
                    progress=i / 5,
                )
                manager.save(checkpoint, force=True)

            # Only max_checkpoints should remain
            checkpoints = list(manager.job_dir.glob("checkpoint_*.json"))
            assert len(checkpoints) <= 2


class TestMemoryEstimator:
    """Test memory estimation."""

    def test_estimate_activity(self):
        from cytoatlas_pipeline.core.memory import MemoryEstimator

        estimator = MemoryEstimator(available_gb=8.0)

        estimate = estimator.estimate_activity(
            n_genes=20000,
            n_features=50,
            n_samples=10000,
            n_rand=1000,
        )

        assert estimate.total_gb > 0
        assert estimate.peak_gb > 0
        assert estimate.recommended_batch_size > 0

    def test_estimate_batch_size(self):
        from cytoatlas_pipeline.core.memory import MemoryEstimator

        estimator = MemoryEstimator(available_gb=8.0)

        batch_size = estimator.estimate_batch_size(
            n_genes=20000,
            n_features=50,
            n_rand=1000,
        )

        assert batch_size > 0
        assert batch_size <= 50000  # max_batch default


class TestResultCache:
    """Test result caching."""

    def test_cache_operations(self):
        from cytoatlas_pipeline.core.cache import ResultCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResultCache(cache_dir=tmpdir)

            # Store
            cache.set("test_key", {"data": [1, 2, 3]})

            # Retrieve
            result = cache.get("test_key")
            assert result is not None
            assert result["data"] == [1, 2, 3]

            # Miss
            assert cache.get("nonexistent") is None

    def test_cache_expiration(self):
        from cytoatlas_pipeline.core.cache import ResultCache
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            # Very short TTL (0.0001 hours = 0.36 seconds)
            cache = ResultCache(cache_dir=tmpdir, ttl_hours=0.0001)

            cache.set("expiring_key", "value")
            time.sleep(0.5)

            assert cache.get("expiring_key") is None
