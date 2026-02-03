"""
Checkpoint and recovery system.

Provides atomic checkpointing for crash recovery in long-running pipelines.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import hashlib


@dataclass
class Checkpoint:
    """
    Represents a pipeline checkpoint.

    Contains all state needed to resume a pipeline from this point.
    """

    job_id: str
    """Unique job identifier."""

    pipeline_name: str
    """Name of the pipeline being run."""

    step_name: str
    """Current step name."""

    step_index: int
    """Current step index (0-based)."""

    total_steps: int
    """Total number of steps."""

    progress: float
    """Progress within current step (0-1)."""

    batch_index: int = 0
    """Current batch index within step."""

    total_batches: int = 1
    """Total batches in current step."""

    state: dict[str, Any] = field(default_factory=dict)
    """Arbitrary state data."""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    """Checkpoint creation timestamp."""

    elapsed_seconds: float = 0.0
    """Total elapsed time at checkpoint."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (config hash, versions, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(**d)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Checkpoint":
        """Create from JSON string."""
        return cls.from_dict(json.loads(s))


class CheckpointManager:
    """
    Manages checkpoint creation, storage, and recovery.

    Provides atomic checkpoint saves to prevent corruption from crashes.

    Example:
        >>> manager = CheckpointManager("/path/to/checkpoints", job_id="job123")
        >>>
        >>> # Save checkpoint
        >>> checkpoint = Checkpoint(
        ...     job_id="job123",
        ...     pipeline_name="activity",
        ...     step_name="batch_processing",
        ...     step_index=2,
        ...     total_steps=5,
        ...     progress=0.5,
        ...     state={"processed_batches": 50}
        ... )
        >>> manager.save(checkpoint)
        >>>
        >>> # Later, recover
        >>> latest = manager.load_latest()
        >>> if latest:
        ...     resume_from = latest.state["processed_batches"]
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        job_id: str,
        max_checkpoints: int = 3,
        interval_seconds: int = 300,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files.
            job_id: Unique job identifier.
            max_checkpoints: Maximum checkpoints to retain.
            interval_seconds: Minimum interval between checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.job_id = job_id
        self.max_checkpoints = max_checkpoints
        self.interval_seconds = interval_seconds

        # Create checkpoint directory
        self.job_dir = self.checkpoint_dir / job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)

        # Track last checkpoint time
        self._last_checkpoint_time: float = 0.0
        self._start_time: float = time.time()

    def _get_checkpoint_path(self, index: int) -> Path:
        """Get path for checkpoint with given index."""
        return self.job_dir / f"checkpoint_{index:05d}.json"

    def _get_next_index(self) -> int:
        """Get next checkpoint index."""
        existing = list(self.job_dir.glob("checkpoint_*.json"))
        if not existing:
            return 0
        indices = [int(p.stem.split("_")[1]) for p in existing]
        return max(indices) + 1

    def save(self, checkpoint: Checkpoint, force: bool = False) -> bool:
        """
        Save a checkpoint atomically.

        Args:
            checkpoint: Checkpoint to save.
            force: Force save even if interval hasn't passed.

        Returns:
            True if saved, False if skipped (interval not passed).
        """
        current_time = time.time()

        # Check interval (unless forced)
        if not force and (current_time - self._last_checkpoint_time) < self.interval_seconds:
            return False

        # Update elapsed time
        checkpoint.elapsed_seconds = current_time - self._start_time

        # Get next index
        index = self._get_next_index()
        checkpoint_path = self._get_checkpoint_path(index)

        # Write atomically using temp file
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.job_dir, prefix=".checkpoint_", suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, "w") as f:
                f.write(checkpoint.to_json())

            # Atomic rename
            shutil.move(temp_path, checkpoint_path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

        self._last_checkpoint_time = current_time

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return True

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoints = sorted(self.job_dir.glob("checkpoint_*.json"))
        if len(checkpoints) > self.max_checkpoints:
            for cp in checkpoints[: -self.max_checkpoints]:
                cp.unlink()

    def load_latest(self) -> Optional[Checkpoint]:
        """
        Load the most recent checkpoint.

        Returns:
            Latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = sorted(self.job_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None

        latest_path = checkpoints[-1]
        with open(latest_path) as f:
            return Checkpoint.from_json(f.read())

    def load(self, index: int) -> Optional[Checkpoint]:
        """
        Load checkpoint by index.

        Args:
            index: Checkpoint index.

        Returns:
            Checkpoint, or None if not found.
        """
        checkpoint_path = self._get_checkpoint_path(index)
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path) as f:
            return Checkpoint.from_json(f.read())

    def list_checkpoints(self) -> list[tuple[int, Checkpoint]]:
        """
        List all checkpoints.

        Returns:
            List of (index, checkpoint) tuples.
        """
        checkpoints = []
        for path in sorted(self.job_dir.glob("checkpoint_*.json")):
            index = int(path.stem.split("_")[1])
            with open(path) as f:
                checkpoint = Checkpoint.from_json(f.read())
            checkpoints.append((index, checkpoint))
        return checkpoints

    def clear(self) -> None:
        """Remove all checkpoints for this job."""
        for path in self.job_dir.glob("checkpoint_*.json"):
            path.unlink()

    def exists(self) -> bool:
        """Check if any checkpoints exist."""
        return any(self.job_dir.glob("checkpoint_*.json"))

    def should_checkpoint(self) -> bool:
        """Check if enough time has passed for a new checkpoint."""
        return (time.time() - self._last_checkpoint_time) >= self.interval_seconds

    @staticmethod
    def compute_config_hash(config: dict[str, Any]) -> str:
        """
        Compute hash of configuration for change detection.

        Args:
            config: Configuration dictionary.

        Returns:
            MD5 hash of configuration.
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


class CheckpointContext:
    """
    Context manager for checkpointing within a step.

    Example:
        >>> manager = CheckpointManager("/checkpoints", "job123")
        >>> with CheckpointContext(manager, "processing", 0, 5) as ctx:
        ...     for i, batch in enumerate(batches):
        ...         process(batch)
        ...         ctx.update(progress=i/len(batches), state={"batch": i})
    """

    def __init__(
        self,
        manager: CheckpointManager,
        step_name: str,
        step_index: int,
        total_steps: int,
        pipeline_name: str = "pipeline",
    ):
        """
        Initialize checkpoint context.

        Args:
            manager: Checkpoint manager.
            step_name: Name of current step.
            step_index: Index of current step.
            total_steps: Total number of steps.
            pipeline_name: Name of pipeline.
        """
        self.manager = manager
        self.step_name = step_name
        self.step_index = step_index
        self.total_steps = total_steps
        self.pipeline_name = pipeline_name

        self._progress: float = 0.0
        self._batch_index: int = 0
        self._total_batches: int = 1
        self._state: dict[str, Any] = {}

    def __enter__(self) -> "CheckpointContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Save final checkpoint on successful completion
        if exc_type is None:
            self.update(progress=1.0, force=True)

    def update(
        self,
        progress: Optional[float] = None,
        batch_index: Optional[int] = None,
        total_batches: Optional[int] = None,
        state: Optional[dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """
        Update checkpoint state and potentially save.

        Args:
            progress: Current progress (0-1).
            batch_index: Current batch index.
            total_batches: Total number of batches.
            state: State data to merge.
            force: Force save even if interval hasn't passed.

        Returns:
            True if checkpoint was saved.
        """
        if progress is not None:
            self._progress = progress
        if batch_index is not None:
            self._batch_index = batch_index
        if total_batches is not None:
            self._total_batches = total_batches
        if state is not None:
            self._state.update(state)

        checkpoint = Checkpoint(
            job_id=self.manager.job_id,
            pipeline_name=self.pipeline_name,
            step_name=self.step_name,
            step_index=self.step_index,
            total_steps=self.total_steps,
            progress=self._progress,
            batch_index=self._batch_index,
            total_batches=self._total_batches,
            state=self._state.copy(),
        )

        return self.manager.save(checkpoint, force=force)
