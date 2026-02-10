"""Pipeline status and metadata service."""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.services.base import BaseService

logger = logging.getLogger(__name__)
settings = get_settings()


class PipelineService(BaseService):
    """
    Service for pipeline status and metadata.

    Reads pipeline definition and status files to provide
    real-time pipeline execution information.
    """

    def __init__(self):
        """Initialize pipeline service."""
        super().__init__(db=None)

        # Paths - use results_base_path as the anchor
        self.pipeline_yaml = settings.results_base_path.parent / "cytoatlas-pipeline" / "pipeline.yaml"
        self.status_file = settings.results_base_path / ".pipeline_status.json"

    async def get_status(self) -> dict[str, Any]:
        """
        Get overall pipeline status.

        Returns:
            Status dictionary with stage statuses and overall progress
        """
        # Load status file if exists
        if self.status_file.exists():
            try:
                with open(self.status_file) as f:
                    status = json.load(f)
            except Exception:
                status = self._get_empty_status()
        else:
            status = self._get_empty_status()

        # Calculate summary
        stage_statuses = status.get("stages", {})
        total_stages = len(stage_statuses)

        completed = sum(1 for s in stage_statuses.values() if s == "completed")
        running = sum(1 for s in stage_statuses.values() if s == "running")
        failed = sum(1 for s in stage_statuses.values() if s == "failed")
        pending = sum(1 for s in stage_statuses.values() if s == "pending")

        overall_status = "idle"
        if running > 0:
            overall_status = "running"
        elif failed > 0:
            overall_status = "failed"
        elif completed == total_stages and total_stages > 0:
            overall_status = "completed"

        progress = (completed / total_stages * 100) if total_stages > 0 else 0

        return {
            "overall_status": overall_status,
            "progress_percent": progress,
            "total_stages": total_stages,
            "completed": completed,
            "running": running,
            "failed": failed,
            "pending": pending,
            "stages": stage_statuses,
            "start_time": status.get("start_time"),
            "end_time": status.get("end_time"),
            "last_updated": status.get("last_updated"),
            "timings": status.get("timings", {}),
            "errors": status.get("errors", {}),
        }

    async def get_all_stages(self) -> list[dict]:
        """
        Get list of all pipeline stages with metadata.

        Returns:
            List of stage dictionaries
        """
        stages = await self._load_pipeline_definition()
        status = await self.get_status()
        stage_statuses = status["stages"]
        timings = status.get("timings", {})

        result = []
        for name, stage_def in stages.items():
            result.append({
                "name": name,
                "script": stage_def.get("script"),
                "inputs": stage_def.get("inputs", []),
                "outputs": stage_def.get("outputs", []),
                "depends_on": stage_def.get("depends_on", []),
                "gpu": stage_def.get("gpu", False),
                "time_estimate": stage_def.get("time_estimate", "unknown"),
                "status": stage_statuses.get(name, "pending"),
                "elapsed_seconds": timings.get(name),
            })

        return result

    async def get_stage(self, name: str) -> dict | None:
        """
        Get detailed information about a specific stage.

        Args:
            name: Stage name

        Returns:
            Stage dictionary or None if not found
        """
        stages = await self.get_all_stages()
        for stage in stages:
            if stage["name"] == name:
                return stage
        return None

    async def get_stage_names(self) -> list[str]:
        """
        Get list of stage names.

        Returns:
            List of stage names
        """
        stages = await self._load_pipeline_definition()
        return list(stages.keys())

    async def _load_pipeline_definition(self) -> dict[str, Any]:
        """
        Load pipeline definition from YAML.

        Returns:
            Dictionary mapping stage name to stage definition
        """
        if not self.pipeline_yaml.exists():
            return {}

        try:
            with open(self.pipeline_yaml) as f:
                config = yaml.safe_load(f)
            return config.get("stages", {})
        except Exception:
            return {}

    async def run_stage(self, stage_name: str) -> dict[str, Any]:
        """Run a pipeline stage via SLURM sbatch.

        Args:
            stage_name: Name of the stage to run.

        Returns:
            Dictionary with job submission info.

        Raises:
            ValueError: If stage not found or no script configured.
        """
        stages = await self._load_pipeline_definition()
        if stage_name not in stages:
            available = list(stages.keys())
            raise ValueError(f"Stage '{stage_name}' not found. Available: {available}")

        stage_def = stages[stage_name]
        script = stage_def.get("script")
        if not script:
            raise ValueError(f"Stage '{stage_name}' has no script configured")

        # Resolve script path
        script_path = settings.results_base_path.parent / script
        if not script_path.exists():
            raise ValueError(f"Script not found: {script_path}")

        # Submit via sbatch
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr}")

            # Parse job ID from sbatch output (e.g., "Submitted batch job 12345")
            job_id = None
            if "Submitted batch job" in result.stdout:
                job_id = result.stdout.strip().split()[-1]

            # Update status file
            self._update_stage_status(stage_name, "submitted")

            logger.info("Submitted SLURM job for stage %s: job_id=%s", stage_name, job_id)

            return {
                "stage": stage_name,
                "status": "submitted",
                "job_id": job_id,
                "script": str(script_path),
                "submitted_at": datetime.utcnow().isoformat(),
            }

        except FileNotFoundError:
            # sbatch not available — try Celery as fallback
            return await self._run_via_celery(stage_name, stage_def)

    async def _run_via_celery(self, stage_name: str, stage_def: dict) -> dict[str, Any]:
        """Fall back to Celery task submission when SLURM is unavailable."""
        try:
            from app.tasks.celery_app import celery_app

            task = celery_app.send_task(
                "app.tasks.process_atlas.process_h5ad",
                kwargs={"stage_name": stage_name},
            )

            return {
                "stage": stage_name,
                "status": "queued",
                "task_id": task.id,
                "backend": "celery",
                "submitted_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            raise RuntimeError(
                f"Neither SLURM nor Celery available for pipeline execution: {e}"
            )

    def _update_stage_status(self, stage_name: str, status: str) -> None:
        """Update the stage status in the status file."""
        if self.status_file.exists():
            try:
                with open(self.status_file) as f:
                    data = json.load(f)
            except Exception:
                data = self._get_empty_status()
        else:
            data = self._get_empty_status()

        data["stages"][stage_name] = status
        data["last_updated"] = datetime.utcnow().isoformat()

        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w") as f:
            json.dump(data, f, indent=2)

    def _get_empty_status(self) -> dict[str, Any]:
        """
        Get empty status structure.

        Returns:
            Empty status dictionary
        """
        return {
            "stages": {},
            "start_time": None,
            "end_time": None,
            "last_updated": None,
            "timings": {},
            "errors": {},
        }
