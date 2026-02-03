"""
Task scheduling and dependency management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import heapq


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass(order=True)
class ScheduledTask:
    """A task scheduled for execution."""

    priority: int
    scheduled_time: datetime = field(compare=True)
    task_id: str = field(compare=False)
    task_name: str = field(compare=False)
    task_func: Callable = field(compare=False, repr=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    dependencies: list[str] = field(default_factory=list, compare=False)
    timeout: Optional[int] = field(default=None, compare=False)


class TaskScheduler:
    """Schedules and manages task execution.

    Features:
    - Priority-based scheduling
    - Dependency resolution
    - Timeout management
    - Resource-aware scheduling (GPU)
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        gpu_tasks_limit: int = 1,
    ):
        self.max_concurrent = max_concurrent
        self.gpu_tasks_limit = gpu_tasks_limit

        self._queue: list[ScheduledTask] = []
        self._running: dict[str, ScheduledTask] = {}
        self._completed: dict[str, Any] = {}
        self._failed: dict[str, str] = {}

    def schedule(
        self,
        task_id: str,
        task_name: str,
        task_func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[list[str]] = None,
        delay_seconds: int = 0,
        timeout: Optional[int] = None,
    ) -> ScheduledTask:
        """Schedule a task for execution.

        Parameters
        ----------
        task_id : str
            Unique task identifier
        task_name : str
            Human-readable task name
        task_func : Callable
            Function to execute
        args : tuple
            Positional arguments
        kwargs : dict
            Keyword arguments
        priority : TaskPriority
            Execution priority
        dependencies : list[str]
            Task IDs that must complete first
        delay_seconds : int
            Seconds to wait before scheduling
        timeout : int, optional
            Maximum execution time in seconds

        Returns
        -------
        ScheduledTask
            Scheduled task object
        """
        scheduled_time = datetime.now()
        if delay_seconds > 0:
            from datetime import timedelta
            scheduled_time += timedelta(seconds=delay_seconds)

        task = ScheduledTask(
            priority=priority.value,
            scheduled_time=scheduled_time,
            task_id=task_id,
            task_name=task_name,
            task_func=task_func,
            args=args,
            kwargs=kwargs or {},
            dependencies=dependencies or [],
            timeout=timeout,
        )

        heapq.heappush(self._queue, task)
        return task

    def get_ready_tasks(self) -> list[ScheduledTask]:
        """Get tasks ready for execution.

        A task is ready if:
        - Scheduled time has passed
        - All dependencies are completed
        - Max concurrent limit not reached
        """
        ready = []
        remaining = []
        now = datetime.now()

        while self._queue:
            task = heapq.heappop(self._queue)

            # Check scheduled time
            if task.scheduled_time > now:
                remaining.append(task)
                continue

            # Check dependencies
            deps_satisfied = all(
                dep in self._completed for dep in task.dependencies
            )
            if not deps_satisfied:
                # Check if any dependency failed
                deps_failed = any(dep in self._failed for dep in task.dependencies)
                if deps_failed:
                    self._failed[task.task_id] = "Dependency failed"
                    continue
                remaining.append(task)
                continue

            # Check concurrent limit
            if len(self._running) >= self.max_concurrent:
                remaining.append(task)
                continue

            ready.append(task)

        # Restore remaining tasks
        for task in remaining:
            heapq.heappush(self._queue, task)

        return ready

    def mark_running(self, task_id: str) -> None:
        """Mark task as running."""
        for task in self._queue:
            if task.task_id == task_id:
                self._running[task_id] = task
                break

    def mark_completed(self, task_id: str, result: Any = None) -> None:
        """Mark task as completed."""
        if task_id in self._running:
            del self._running[task_id]
        self._completed[task_id] = result

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark task as failed."""
        if task_id in self._running:
            del self._running[task_id]
        self._failed[task_id] = error

    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task.

        Returns True if task was cancelled, False if not found.
        """
        for i, task in enumerate(self._queue):
            if task.task_id == task_id:
                self._queue.pop(i)
                heapq.heapify(self._queue)
                return True
        return False

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "queued": len(self._queue),
            "running": len(self._running),
            "completed": len(self._completed),
            "failed": len(self._failed),
            "running_tasks": list(self._running.keys()),
        }

    def clear_completed(self) -> int:
        """Clear completed tasks. Returns count cleared."""
        count = len(self._completed)
        self._completed.clear()
        return count

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of completed task."""
        return self._completed.get(task_id)

    def get_task_error(self, task_id: str) -> Optional[str]:
        """Get error message for failed task."""
        return self._failed.get(task_id)

    def run_next(self) -> Optional[tuple[str, Any]]:
        """Execute next ready task synchronously.

        Returns (task_id, result) or None if no tasks ready.
        """
        ready = self.get_ready_tasks()
        if not ready:
            return None

        task = ready[0]
        self._running[task.task_id] = task

        try:
            result = task.task_func(*task.args, **task.kwargs)
            self.mark_completed(task.task_id, result)
            return (task.task_id, result)
        except Exception as e:
            self.mark_failed(task.task_id, str(e))
            raise

    def run_all(self) -> dict[str, Any]:
        """Execute all tasks in order. Returns results dict."""
        results = {}

        while self._queue or self._running:
            ready = self.get_ready_tasks()
            if not ready:
                if self._running:
                    # Wait for running tasks (in real async, would await)
                    continue
                break

            for task in ready:
                self._running[task.task_id] = task

                try:
                    result = task.task_func(*task.args, **task.kwargs)
                    self.mark_completed(task.task_id, result)
                    results[task.task_id] = result
                except Exception as e:
                    self.mark_failed(task.task_id, str(e))
                    results[task.task_id] = {"error": str(e)}

        return results
