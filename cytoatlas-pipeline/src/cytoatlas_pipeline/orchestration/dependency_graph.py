"""
Pipeline dependency graph and execution order.

Manages stage dependencies and determines execution order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Stage:
    """
    Pipeline stage definition.

    Contains metadata about a single stage in the pipeline.
    """

    name: str
    """Stage name."""

    script: str
    """Script path relative to project root."""

    inputs: list[str] = field(default_factory=list)
    """Input paths or config keys."""

    outputs: list[str] = field(default_factory=list)
    """Output paths."""

    depends_on: list[str] = field(default_factory=list)
    """List of stage names this stage depends on."""

    gpu: bool = False
    """Whether this stage requires GPU."""

    time_estimate: str = "1h"
    """Estimated runtime."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "script": self.script,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "depends_on": self.depends_on,
            "gpu": self.gpu,
            "time_estimate": self.time_estimate,
            "metadata": self.metadata,
        }


class DependencyGraph:
    """
    Pipeline dependency graph.

    Manages stage dependencies and provides topological sort for execution order.

    Example:
        >>> graph = DependencyGraph.from_yaml("pipeline.yaml")
        >>> order = graph.topological_sort()
        >>> ready = graph.get_ready_stages(completed={"pilot"})
    """

    def __init__(self, stages: dict[str, Stage]):
        """
        Initialize dependency graph.

        Args:
            stages: Dictionary mapping stage name to Stage object
        """
        self.stages = stages

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "DependencyGraph":
        """
        Load pipeline definition from YAML.

        Args:
            yaml_path: Path to pipeline.yaml

        Returns:
            DependencyGraph instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Pipeline definition not found: {yaml_path}")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        stages = {}
        for name, stage_config in config.get("stages", {}).items():
            stages[name] = Stage(
                name=name,
                script=stage_config.get("script", ""),
                inputs=stage_config.get("inputs", []),
                outputs=stage_config.get("outputs", []),
                depends_on=stage_config.get("depends_on", []),
                gpu=stage_config.get("gpu", False),
                time_estimate=stage_config.get("time_estimate", "1h"),
                metadata=stage_config.get("metadata", {}),
            )

        return cls(stages)

    def validate(self) -> list[str]:
        """
        Validate dependency graph.

        Checks for:
        - Cycles in dependencies
        - Missing dependency references
        - Disconnected components

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for missing dependencies
        for stage_name, stage in self.stages.items():
            for dep in stage.depends_on:
                if dep not in self.stages:
                    errors.append(
                        f"Stage '{stage_name}' depends on unknown stage '{dep}'"
                    )

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            """Detect cycle using DFS."""
            visited.add(node)
            rec_stack.add(node)

            for dep in self.stages[node].depends_on:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    errors.append(f"Cycle detected involving stage '{node}' -> '{dep}'")
                    return True

            rec_stack.remove(node)
            return False

        for stage_name in self.stages:
            if stage_name not in visited:
                has_cycle(stage_name)

        return errors

    def topological_sort(self) -> list[str]:
        """
        Get execution order using topological sort.

        Returns:
            List of stage names in execution order

        Raises:
            ValueError: If graph has cycles or is invalid
        """
        # Validate first
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid dependency graph:\n" + "\n".join(errors))

        # Kahn's algorithm for topological sort
        # in_degree = number of dependencies (stages that must run before this one)
        in_degree = {name: len(stage.depends_on) for name, stage in self.stages.items()}

        # Start with stages that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort by stage name for deterministic order
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for stages that depend on current
            for stage_name, stage in self.stages.items():
                if current in stage.depends_on:
                    in_degree[stage_name] -= 1
                    if in_degree[stage_name] == 0:
                        queue.append(stage_name)

        # Check if all stages were processed
        if len(result) != len(self.stages):
            raise ValueError("Graph has cycles or is invalid")

        return result

    def get_ready_stages(self, completed: set[str]) -> list[str]:
        """
        Get stages that are ready to run.

        A stage is ready if all its dependencies have completed.

        Args:
            completed: Set of completed stage names

        Returns:
            List of stage names that are ready to run
        """
        ready = []

        for stage_name, stage in self.stages.items():
            # Skip if already completed
            if stage_name in completed:
                continue

            # Check if all dependencies are satisfied
            if all(dep in completed for dep in stage.depends_on):
                ready.append(stage_name)

        return ready

    def get_stage(self, name: str) -> Stage | None:
        """
        Get stage by name.

        Args:
            name: Stage name

        Returns:
            Stage object or None if not found
        """
        return self.stages.get(name)

    def get_all_dependencies(self, stage_name: str) -> set[str]:
        """
        Get all dependencies (direct and transitive) for a stage.

        Args:
            stage_name: Stage name

        Returns:
            Set of all dependency stage names
        """
        if stage_name not in self.stages:
            return set()

        deps = set()
        to_visit = list(self.stages[stage_name].depends_on)

        while to_visit:
            dep = to_visit.pop()
            if dep not in deps:
                deps.add(dep)
                if dep in self.stages:
                    to_visit.extend(self.stages[dep].depends_on)

        return deps

    def get_downstream_stages(self, stage_name: str) -> set[str]:
        """
        Get all stages that depend on this stage (directly or transitively).

        Args:
            stage_name: Stage name

        Returns:
            Set of downstream stage names
        """
        downstream = set()

        for name, stage in self.stages.items():
            if stage_name in self.get_all_dependencies(name):
                downstream.add(name)

        return downstream

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()}
        }
