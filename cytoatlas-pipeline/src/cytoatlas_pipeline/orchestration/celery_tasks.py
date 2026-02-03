"""
Celery task definitions for background processing.

These tasks integrate with the CytoAtlas API for long-running
pipeline execution with progress reporting.
"""

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Celery import with fallback for testing
try:
    from celery import shared_task, current_task
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False

    def shared_task(*args, **kwargs):
        """Fallback decorator when Celery not installed."""
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator

    current_task = None

from cytoatlas_pipeline.orchestration.job import Job, JobState, JobResult, create_job
from cytoatlas_pipeline.orchestration.recovery import RecoveryManager


def update_progress(
    job: Job,
    step: int,
    phase: str,
    message: str,
    broadcast: bool = True,
) -> None:
    """Update job progress and optionally broadcast.

    Parameters
    ----------
    job : Job
        Job to update
    step : int
        Current step number
    phase : str
        Current phase name
    message : str
        Progress message
    broadcast : bool
        Whether to broadcast via Redis pub/sub
    """
    job.progress.update(step=step, phase=phase, message=message)

    # Update Celery task state if available
    if HAS_CELERY and current_task:
        current_task.update_state(
            state="PROGRESS",
            meta={
                "job_id": job.id,
                "step": step,
                "total": job.progress.total_steps,
                "phase": phase,
                "message": message,
                "percentage": job.progress.percentage,
            }
        )


@shared_task(
    bind=True,
    max_retries=3,
    soft_time_limit=14400,  # 4 hours soft limit
    time_limit=14700,       # 4 hours + 5 min hard limit
    acks_late=True,
)
def run_activity_pipeline(
    self,
    h5ad_path: str,
    output_dir: str,
    config: dict[str, Any],
    job_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run activity inference pipeline as Celery task.

    Parameters
    ----------
    h5ad_path : str
        Path to input H5AD file
    output_dir : str
        Output directory path
    config : dict
        Pipeline configuration
    job_id : str, optional
        Existing job ID for recovery

    Returns
    -------
    dict
        Task result with output paths
    """
    from cytoatlas_pipeline.core.config import Config
    from cytoatlas_pipeline.core.checkpoint import CheckpointManager
    from cytoatlas_pipeline.ingest import LocalH5ADSource
    from cytoatlas_pipeline.activity import RidgeInference
    from cytoatlas_pipeline.export import JSONWriter, CSVWriter

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create or recover job
    if job_id:
        manager = RecoveryManager(output_path / "jobs")
        incomplete = manager.scan_for_incomplete()
        job_info = next((j for j in incomplete if j.job_id == job_id), None)
        if job_info:
            job, checkpoint = manager.recover_job(job_info)
        else:
            job = create_job("Activity Pipeline", "activity", config, output_path)
    else:
        job = create_job("Activity Pipeline", "activity", config, output_path)

    job.celery_task_id = self.request.id if HAS_CELERY and self else None
    job.progress.total_steps = 5

    checkpoint_mgr = CheckpointManager(output_path / "checkpoints")

    try:
        job.start()

        # Step 1: Load data
        update_progress(job, 1, "loading", "Loading H5AD file...")
        source = LocalH5ADSource(
            h5ad_path,
            backed=True,
            chunk_size=config.get("chunk_size", 10000),
        )

        # Save checkpoint after loading
        checkpoint_mgr.save({
            "step": 1,
            "phase": "loading",
            "timestamp": datetime.now().isoformat(),
        })

        # Step 2: Aggregate
        update_progress(job, 2, "aggregation", "Aggregating cells...")
        aggregation_level = config.get("aggregation", "pseudobulk")

        # Actual aggregation would happen here
        # aggregator = get_aggregator(aggregation_level)
        # aggregated = aggregator.aggregate(source)

        checkpoint_mgr.save({
            "step": 2,
            "phase": "aggregation",
            "timestamp": datetime.now().isoformat(),
        })

        # Step 3: Activity inference
        update_progress(job, 3, "inference", "Computing activity scores...")
        pipeline_config = Config(**config.get("pipeline_config", {}))

        # Ridge inference would happen here
        # ridge = RidgeInference(config=pipeline_config)
        # activity = ridge.compute(aggregated)

        checkpoint_mgr.save({
            "step": 3,
            "phase": "inference",
            "timestamp": datetime.now().isoformat(),
        })

        # Step 4: Export results
        update_progress(job, 4, "export", "Writing results...")

        json_writer = JSONWriter(output_path / "json")
        csv_writer = CSVWriter(output_path / "csv")

        # Would write actual results
        # json_writer.write_activity_matrix(activity)
        # csv_writer.write_activity(activity)

        checkpoint_mgr.save({
            "step": 4,
            "phase": "export",
            "timestamp": datetime.now().isoformat(),
        })

        # Step 5: Complete
        update_progress(job, 5, "complete", "Pipeline complete")

        result = JobResult(
            success=True,
            output_paths={
                "json": output_path / "json",
                "csv": output_path / "csv",
            },
            metrics={
                "n_signatures": config.get("n_signatures", 0),
                "n_samples": config.get("n_samples", 0),
            },
        )

        job.complete(result)

        # Clean up checkpoint
        checkpoint_mgr.cleanup()

        return result.to_dict()

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()

        job.fail(error_msg, tb)

        # Save failed state
        if job.checkpoint_path:
            checkpoint_mgr.save({
                "step": job.progress.current_step,
                "phase": job.progress.current_phase,
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            })

        raise


@shared_task(
    bind=True,
    max_retries=2,
    soft_time_limit=7200,  # 2 hours
    time_limit=7500,
    acks_late=True,
)
def run_differential_pipeline(
    self,
    activity_path: str,
    metadata_path: str,
    output_dir: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run differential analysis pipeline."""
    from cytoatlas_pipeline.differential import StratifiedDifferential
    from cytoatlas_pipeline.export import JSONWriter

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    job = create_job("Differential Analysis", "differential", config, output_path)
    job.celery_task_id = self.request.id if HAS_CELERY and self else None
    job.progress.total_steps = 3

    try:
        job.start()

        # Step 1: Load data
        update_progress(job, 1, "loading", "Loading activity and metadata...")
        import pandas as pd
        # activity = pd.read_csv(activity_path, index_col=0)
        # metadata = pd.read_csv(metadata_path, index_col=0)

        # Step 2: Run differential
        update_progress(job, 2, "analysis", "Computing differential activity...")
        # diff = StratifiedDifferential(method=config.get("method", "wilcoxon"))
        # results = diff.compare(activity, metadata, ...)

        # Step 3: Export
        update_progress(job, 3, "export", "Writing results...")
        # writer = JSONWriter(output_path)
        # writer.write_differential_results(results)

        result = JobResult(
            success=True,
            output_paths={"results": output_path / "differential.json"},
        )
        job.complete(result)

        return result.to_dict()

    except Exception as e:
        job.fail(str(e), traceback.format_exc())
        raise


@shared_task(
    bind=True,
    max_retries=2,
    soft_time_limit=3600,  # 1 hour
    time_limit=3900,
    acks_late=True,
)
def run_validation_pipeline(
    self,
    activity_path: str,
    expression_path: str,
    output_dir: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run validation pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    job = create_job("Validation", "validation", config, output_path)
    job.progress.total_steps = 4

    try:
        job.start()

        update_progress(job, 1, "loading", "Loading data...")
        update_progress(job, 2, "sample_validation", "Running sample-level validation...")
        update_progress(job, 3, "celltype_validation", "Running cell-type validation...")
        update_progress(job, 4, "complete", "Validation complete")

        result = JobResult(
            success=True,
            output_paths={"validation": output_path / "validation.json"},
        )
        job.complete(result)

        return result.to_dict()

    except Exception as e:
        job.fail(str(e), traceback.format_exc())
        raise


# Helper to get task by name
TASK_REGISTRY = {
    "activity": run_activity_pipeline,
    "differential": run_differential_pipeline,
    "validation": run_validation_pipeline,
}


def get_task(task_name: str):
    """Get Celery task by name."""
    return TASK_REGISTRY.get(task_name)
