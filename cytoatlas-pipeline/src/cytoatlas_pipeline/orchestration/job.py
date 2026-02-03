"""
Job definition and state management.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import json


class JobState(Enum):
    """Job execution states."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress tracking for a job."""

    current_step: int = 0
    total_steps: int = 0
    current_phase: str = ""
    message: str = ""
    percentage: float = 0.0

    def update(
        self,
        step: Optional[int] = None,
        phase: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update progress."""
        if step is not None:
            self.current_step = step
        if phase is not None:
            self.current_phase = phase
        if message is not None:
            self.message = message

        if self.total_steps > 0:
            self.percentage = (self.current_step / self.total_steps) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_phase": self.current_phase,
            "message": self.message,
            "percentage": self.percentage,
        }


@dataclass
class JobResult:
    """Result of a completed job."""

    success: bool
    output_paths: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    traceback: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output_paths": {k: str(v) for k, v in self.output_paths.items()},
            "metrics": self.metrics,
            "error": self.error,
            "traceback": self.traceback,
        }


@dataclass
class Job:
    """Pipeline job definition and state."""

    id: str
    name: str
    pipeline_type: str  # activity, correlation, differential, validation, full
    config: dict[str, Any]

    state: JobState = JobState.PENDING
    progress: JobProgress = field(default_factory=JobProgress)
    result: Optional[JobResult] = None

    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    checkpoint_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Celery task info
    celery_task_id: Optional[str] = None
    worker_id: Optional[str] = None

    def start(self) -> None:
        """Mark job as started."""
        self.state = JobState.RUNNING
        self.started_at = datetime.now()

    def complete(self, result: JobResult) -> None:
        """Mark job as completed."""
        self.state = JobState.COMPLETED if result.success else JobState.FAILED
        self.result = result
        self.completed_at = datetime.now()

    def fail(self, error: str, traceback: Optional[str] = None) -> None:
        """Mark job as failed."""
        self.state = JobState.FAILED
        self.result = JobResult(
            success=False,
            error=error,
            traceback=traceback,
        )
        self.completed_at = datetime.now()

    def pause(self) -> None:
        """Pause job execution."""
        self.state = JobState.PAUSED

    def resume(self) -> None:
        """Resume paused job."""
        self.state = JobState.RUNNING

    def cancel(self) -> None:
        """Cancel job."""
        self.state = JobState.CANCELLED
        self.completed_at = datetime.now()

    @property
    def is_running(self) -> bool:
        """Check if job is actively running."""
        return self.state == JobState.RUNNING

    @property
    def is_finished(self) -> bool:
        """Check if job has finished (success or failure)."""
        return self.state in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        )

    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at is None:
            return None

        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    def to_dict(self) -> dict:
        """Serialize job to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "pipeline_type": self.pipeline_type,
            "config": self.config,
            "state": self.state.value,
            "progress": self.progress.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "celery_task_id": self.celery_task_id,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        """Deserialize job from dictionary."""
        job = cls(
            id=data["id"],
            name=data["name"],
            pipeline_type=data["pipeline_type"],
            config=data["config"],
        )

        job.state = JobState(data["state"])

        if data.get("progress"):
            job.progress = JobProgress(**data["progress"])

        if data.get("result"):
            result_data = data["result"]
            job.result = JobResult(
                success=result_data["success"],
                output_paths={k: Path(v) for k, v in result_data.get("output_paths", {}).items()},
                metrics=result_data.get("metrics", {}),
                error=result_data.get("error"),
                traceback=result_data.get("traceback"),
            )

        job.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.fromisoformat(data["completed_at"])

        if data.get("checkpoint_path"):
            job.checkpoint_path = Path(data["checkpoint_path"])
        if data.get("output_dir"):
            job.output_dir = Path(data["output_dir"])

        job.celery_task_id = data.get("celery_task_id")
        job.worker_id = data.get("worker_id")

        return job

    def save(self, path: Path) -> None:
        """Save job state to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Job:
        """Load job state from file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_job(
    name: str,
    pipeline_type: str,
    config: dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Job:
    """Create a new job.

    Parameters
    ----------
    name : str
        Human-readable job name
    pipeline_type : str
        Type of pipeline (activity, correlation, etc.)
    config : dict
        Pipeline configuration
    output_dir : Path, optional
        Output directory

    Returns
    -------
    Job
        New job instance
    """
    job_id = str(uuid.uuid4())

    job = Job(
        id=job_id,
        name=name,
        pipeline_type=pipeline_type,
        config=config,
    )

    if output_dir:
        job.output_dir = output_dir
        job.checkpoint_path = output_dir / f"checkpoint_{job_id}.json"

    return job
