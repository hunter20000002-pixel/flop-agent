from __future__ import annotations

from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.task import Task


class Planner:
    """Creates an execution plan for an autonomous-agent task."""

    def plan(self, task: Task) -> ExecutionPlan:
        """Create an execution plan for the supplied task."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        if not task.description.strip():
            raise ValueError("task description cannot be empty")

        step = ExecutionStep(
            description=task.description.strip(),
            order=1,
        )

        return ExecutionPlan(
            task_id=task.id,
            steps=(step,),
        )