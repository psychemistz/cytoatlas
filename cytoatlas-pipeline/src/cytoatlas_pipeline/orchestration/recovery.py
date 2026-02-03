"""
Crash recovery and job resumption.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from cytoatlas_pipeline.orchestration.job import Job, JobState, JobResult
from cytoatlas_pipeline.core.checkpoint import CheckpointManager


@dataclass
class RecoveryInfo:
    """Information about a recoverable job."""

    job_id: str
    job_path: Path
    checkpoint_path: Optional[Path]
    last_phase: str
    last_step: int
    failed_at: datetime
    error: Optional[str]
    can_resume: bool


class RecoveryManager:
    """Manages job recovery after crashes or failures.

    Features:
    - Detects incomplete jobs
    - Validates checkpoints
    - Resumes from last good state
    - Handles partial output cleanup
    """

    def __init__(
        self,
        jobs_dir: Path,
        max_retries: int = 3,
    ):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

    def scan_for_incomplete(self) -> list[RecoveryInfo]:
        """Scan for incomplete or failed jobs.

        Returns
        -------
        list[RecoveryInfo]
            List of recoverable jobs
        """
        incomplete = []

        for job_file in self.jobs_dir.glob("job_*.json"):
            try:
                job = Job.load(job_file)

                # Check if job is incomplete
                if job.state in (JobState.RUNNING, JobState.PAUSED):
                    # Job was interrupted
                    checkpoint_exists = (
                        job.checkpoint_path and
                        job.checkpoint_path.exists()
                    )

                    recovery_info = RecoveryInfo(
                        job_id=job.id,
                        job_path=job_file,
                        checkpoint_path=job.checkpoint_path if checkpoint_exists else None,
                        last_phase=job.progress.current_phase,
                        last_step=job.progress.current_step,
                        failed_at=job.started_at or job.created_at,
                        error=None,
                        can_resume=checkpoint_exists,
                    )
                    incomplete.append(recovery_info)

                elif job.state == JobState.FAILED:
                    # Job failed, might be retryable
                    checkpoint_exists = (
                        job.checkpoint_path and
                        job.checkpoint_path.exists()
                    )

                    recovery_info = RecoveryInfo(
                        job_id=job.id,
                        job_path=job_file,
                        checkpoint_path=job.checkpoint_path if checkpoint_exists else None,
                        last_phase=job.progress.current_phase,
                        last_step=job.progress.current_step,
                        failed_at=job.completed_at or datetime.now(),
                        error=job.result.error if job.result else None,
                        can_resume=checkpoint_exists,
                    )
                    incomplete.append(recovery_info)

            except (json.JSONDecodeError, KeyError) as e:
                # Corrupted job file
                incomplete.append(RecoveryInfo(
                    job_id=job_file.stem.replace("job_", ""),
                    job_path=job_file,
                    checkpoint_path=None,
                    last_phase="unknown",
                    last_step=0,
                    failed_at=datetime.now(),
                    error=f"Corrupted job file: {e}",
                    can_resume=False,
                ))

        return incomplete

    def validate_checkpoint(self, checkpoint_path: Path) -> tuple[bool, str]:
        """Validate checkpoint integrity.

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        if not checkpoint_path.exists():
            return False, "Checkpoint file not found"

        try:
            manager = CheckpointManager(checkpoint_path.parent)
            checkpoint = manager.load(checkpoint_path.name)

            if checkpoint is None:
                return False, "Failed to load checkpoint"

            # Verify required fields
            required = ["step", "phase", "timestamp"]
            for field in required:
                if field not in checkpoint:
                    return False, f"Missing required field: {field}"

            return True, "Checkpoint valid"

        except Exception as e:
            return False, f"Checkpoint validation error: {e}"

    def recover_job(
        self,
        recovery_info: RecoveryInfo,
        force: bool = False,
    ) -> tuple[Job, dict[str, Any]]:
        """Recover a job from its last checkpoint.

        Parameters
        ----------
        recovery_info : RecoveryInfo
            Recovery information
        force : bool
            Force recovery even without valid checkpoint

        Returns
        -------
        tuple[Job, dict]
            (recovered_job, checkpoint_data)

        Raises
        ------
        ValueError
            If job cannot be recovered
        """
        if not recovery_info.can_resume and not force:
            raise ValueError(
                f"Job {recovery_info.job_id} cannot be resumed: no valid checkpoint"
            )

        # Load job
        job = Job.load(recovery_info.job_path)

        # Load checkpoint data
        checkpoint_data = {}
        if recovery_info.checkpoint_path:
            is_valid, msg = self.validate_checkpoint(recovery_info.checkpoint_path)
            if is_valid:
                manager = CheckpointManager(recovery_info.checkpoint_path.parent)
                checkpoint_data = manager.load(recovery_info.checkpoint_path.name) or {}

        # Reset job state
        job.state = JobState.RUNNING
        job.started_at = datetime.now()

        # Update progress from checkpoint
        if checkpoint_data:
            job.progress.current_step = checkpoint_data.get("step", 0)
            job.progress.current_phase = checkpoint_data.get("phase", "")

        return job, checkpoint_data

    def cleanup_failed_job(
        self,
        job_id: str,
        remove_outputs: bool = False,
    ) -> None:
        """Clean up a failed job.

        Parameters
        ----------
        job_id : str
            Job ID to clean up
        remove_outputs : bool
            Also remove output files
        """
        job_path = self.jobs_dir / f"job_{job_id}.json"

        if job_path.exists():
            job = Job.load(job_path)

            # Remove checkpoint
            if job.checkpoint_path and job.checkpoint_path.exists():
                job.checkpoint_path.unlink()

            # Remove outputs if requested
            if remove_outputs and job.output_dir and job.output_dir.exists():
                import shutil
                shutil.rmtree(job.output_dir)

            # Remove job file
            job_path.unlink()

    def retry_job(
        self,
        recovery_info: RecoveryInfo,
        new_config: Optional[dict[str, Any]] = None,
    ) -> Job:
        """Create a new job as retry of a failed one.

        Parameters
        ----------
        recovery_info : RecoveryInfo
            Failed job info
        new_config : dict, optional
            Updated configuration

        Returns
        -------
        Job
            New job for retry
        """
        from cytoatlas_pipeline.orchestration.job import create_job

        # Load original job
        original = Job.load(recovery_info.job_path)

        # Create new job with updated config
        config = original.config.copy()
        if new_config:
            config.update(new_config)

        # Track retry count
        config["_retry_count"] = config.get("_retry_count", 0) + 1
        config["_original_job_id"] = original.id

        new_job = create_job(
            name=f"{original.name} (retry {config['_retry_count']})",
            pipeline_type=original.pipeline_type,
            config=config,
            output_dir=original.output_dir,
        )

        # Copy checkpoint reference if valid
        if recovery_info.can_resume:
            new_job.checkpoint_path = recovery_info.checkpoint_path

        return new_job

    def save_job(self, job: Job) -> Path:
        """Save job state to jobs directory."""
        path = self.jobs_dir / f"job_{job.id}.json"
        job.save(path)
        return path


def recover_job(jobs_dir: Path, job_id: str) -> tuple[Job, dict[str, Any]]:
    """Convenience function to recover a specific job.

    Parameters
    ----------
    jobs_dir : Path
        Directory containing job files
    job_id : str
        ID of job to recover

    Returns
    -------
    tuple[Job, dict]
        (recovered_job, checkpoint_data)
    """
    manager = RecoveryManager(jobs_dir)

    incomplete = manager.scan_for_incomplete()
    for info in incomplete:
        if info.job_id == job_id:
            return manager.recover_job(info)

    raise ValueError(f"Job {job_id} not found or not recoverable")
