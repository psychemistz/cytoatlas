"""Pipeline status and metadata service."""

import json
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.services.base import BaseService

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

        # Paths
        self.pipeline_yaml = Path(settings.root_path).parent / "cytoatlas-pipeline" / "pipeline.yaml"
        self.status_file = Path(settings.root_path).parent / "results" / ".pipeline_status.json"

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
