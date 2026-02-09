"""
Pipeline runner with checkpointing and status tracking.

Orchestrates pipeline execution with crash recovery.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from cytoatlas_pipeline.core.checkpoint import CheckpointManager
from cytoatlas_pipeline.core.paths import PathResolver
from cytoatlas_pipeline.orchestration.dependency_graph import DependencyGraph, Stage

logger = logging.getLogger(__name__)


class StageStatus:
    """Stage execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineRunner:
    """
    Pipeline runner with checkpointing and status tracking.

    Example:
        >>> graph = DependencyGraph.from_yaml("pipeline.yaml")
        >>> config = PathResolver()
        >>> runner = PipelineRunner(graph, config)
        >>> runner.run(stages=["pilot", "cima"], resume=True)
    """

    def __init__(
        self,
        graph: DependencyGraph,
        config: PathResolver,
        checkpoint_manager: Optional[CheckpointManager] = None,
        status_file: Optional[Path | str] = None,
    ):
        """
        Initialize pipeline runner.

        Args:
            graph: Dependency graph
            config: Path resolver for configuration
            checkpoint_manager: Optional checkpoint manager
            status_file: Path to status file (default: results/.pipeline_status.json)
        """
        self.graph = graph
        self.config = config
        self.checkpoint_manager = checkpoint_manager

        # Status tracking
        if status_file is None:
            status_file = config.results / ".pipeline_status.json"
        self.status_file = Path(status_file)

        # Load existing status or initialize
        self.status = self._load_status()

    def _load_status(self) -> dict[str, Any]:
        """Load pipeline status from file."""
        if self.status_file.exists():
            try:
                with open(self.status_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load status file: {e}")

        # Initialize new status
        return {
            "stages": {name: StageStatus.PENDING for name in self.graph.stages},
            "start_time": None,
            "end_time": None,
            "last_updated": None,
            "timings": {},
            "errors": {},
        }

    def _save_status(self) -> None:
        """Save pipeline status to file."""
        self.status["last_updated"] = datetime.now().isoformat()

        # Ensure parent directory exists
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically
        temp_path = self.status_file.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self.status, f, indent=2)

        temp_path.replace(self.status_file)

    def get_status(self) -> dict[str, Any]:
        """
        Get current pipeline status.

        Returns:
            Status dictionary
        """
        return self.status.copy()

    def get_stage_status(self, stage_name: str) -> str:
        """
        Get status of a specific stage.

        Args:
            stage_name: Stage name

        Returns:
            Status string (pending, running, completed, failed)
        """
        return self.status["stages"].get(stage_name, StageStatus.PENDING)

    def get_completed_stages(self) -> set[str]:
        """
        Get set of completed stage names.

        Returns:
            Set of completed stage names
        """
        return {
            name
            for name, status in self.status["stages"].items()
            if status == StageStatus.COMPLETED
        }

    def run(
        self,
        stages: Optional[list[str]] = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """
        Run pipeline.

        Args:
            stages: Specific stages to run (None = all stages)
            resume: Whether to skip completed stages

        Returns:
            Final status dictionary
        """
        # Validate graph
        errors = self.graph.validate()
        if errors:
            logger.error("Pipeline validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid pipeline configuration")

        # Determine which stages to run
        if stages is None:
            # Run all stages in topological order
            stages_to_run = self.graph.topological_sort()
        else:
            # Validate requested stages
            for stage in stages:
                if stage not in self.graph.stages:
                    raise ValueError(f"Unknown stage: {stage}")

            # Add dependencies
            all_deps = set()
            for stage in stages:
                all_deps.update(self.graph.get_all_dependencies(stage))
                all_deps.add(stage)

            # Sort by execution order
            full_order = self.graph.topological_sort()
            stages_to_run = [s for s in full_order if s in all_deps]

        # Resume: skip completed stages
        if resume:
            completed = self.get_completed_stages()
            stages_to_run = [s for s in stages_to_run if s not in completed]
            if completed:
                logger.info(f"Resuming pipeline, skipping completed: {completed}")

        if not stages_to_run:
            logger.info("No stages to run (all completed)")
            return self.get_status()

        # Initialize status
        if self.status["start_time"] is None:
            self.status["start_time"] = datetime.now().isoformat()

        logger.info(f"Running pipeline stages: {stages_to_run}")

        # Run stages
        for stage_name in stages_to_run:
            success = self._run_stage(stage_name)

            if not success:
                logger.error(f"Pipeline failed at stage: {stage_name}")
                self.status["end_time"] = datetime.now().isoformat()
                self._save_status()
                return self.get_status()

        # Pipeline completed
        self.status["end_time"] = datetime.now().isoformat()
        self._save_status()

        logger.info("Pipeline completed successfully")
        return self.get_status()

    def _run_stage(self, stage_name: str) -> bool:
        """
        Run a single stage.

        Args:
            stage_name: Stage name

        Returns:
            True if successful, False otherwise
        """
        stage = self.graph.get_stage(stage_name)
        if stage is None:
            logger.error(f"Stage not found: {stage_name}")
            return False

        # Check dependencies
        completed = self.get_completed_stages()
        missing_deps = [d for d in stage.depends_on if d not in completed]
        if missing_deps:
            logger.error(
                f"Cannot run '{stage_name}': missing dependencies {missing_deps}"
            )
            self.status["stages"][stage_name] = StageStatus.FAILED
            self.status["errors"][stage_name] = f"Missing dependencies: {missing_deps}"
            self._save_status()
            return False

        logger.info(f"Running stage: {stage_name} ({stage.time_estimate})")
        self.status["stages"][stage_name] = StageStatus.RUNNING
        self._save_status()

        # Record start time
        start_time = time.time()

        # Build command
        script_path = Path(stage.script)
        if not script_path.is_absolute():
            # Relative to project root
            project_root = self.config.results.parent
            script_path = project_root / script_path

        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            self.status["stages"][stage_name] = StageStatus.FAILED
            self.status["errors"][stage_name] = f"Script not found: {script_path}"
            self._save_status()
            return False

        # Run script
        try:
            cmd = ["python", str(script_path)]
            logger.info(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Log output
            if result.stdout:
                logger.info(f"[{stage_name}] {result.stdout}")
            if result.stderr:
                logger.warning(f"[{stage_name}] {result.stderr}")

            # Success
            elapsed = time.time() - start_time
            self.status["stages"][stage_name] = StageStatus.COMPLETED
            self.status["timings"][stage_name] = elapsed
            self._save_status()

            logger.info(f"Stage '{stage_name}' completed in {elapsed:.1f}s")
            return True

        except subprocess.CalledProcessError as e:
            # Failed
            elapsed = time.time() - start_time
            self.status["stages"][stage_name] = StageStatus.FAILED
            self.status["timings"][stage_name] = elapsed
            self.status["errors"][stage_name] = str(e)
            self._save_status()

            logger.error(f"Stage '{stage_name}' failed after {elapsed:.1f}s")
            logger.error(f"Error: {e}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")

            return False

        except Exception as e:
            # Unexpected error
            elapsed = time.time() - start_time
            self.status["stages"][stage_name] = StageStatus.FAILED
            self.status["timings"][stage_name] = elapsed
            self.status["errors"][stage_name] = str(e)
            self._save_status()

            logger.error(f"Unexpected error in stage '{stage_name}': {e}")
            return False
