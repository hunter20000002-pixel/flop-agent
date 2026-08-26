from __future__ import annotations

from collections.abc import Callable

from src.agent.plan import ExecutionStep
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.task import Task


StepExecutor = Callable[[ExecutionStep], None]


class AgentRuntime:
    """Orchestrates task planning and execution."""

    def __init__(
        self,
        planner: Planner | None = None,
        step_executor: StepExecutor | None = None,
    ) -> None:
        self.planner = planner or Planner()
        self.step_executor = step_executor or self._default_step_executor

    def run(self, task: Task) -> ExecutionResult:
        """Plan and execute a task."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        task.mark_planning()

        try:
            plan = self.planner.plan(task)

            if plan.is_empty:
                raise ValueError("execution plan cannot be empty")

            task.mark_ready()
            task.mark_running()

            executed_steps = 0

            for step in plan.steps:
                self.step_executor(step)
                executed_steps += 1

            task.mark_completed()

            return ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=executed_steps,
            )

        except Exception as exc:
            task.mark_failed()

            return ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=executed_steps if "executed_steps" in locals() else 0,
                error=str(exc),
            )

    @staticmethod
    def _default_step_executor(step: ExecutionStep) -> None:
        """Execute a step using the default deterministic executor."""

        # V0.2.5 intentionally performs no external work yet.
        _ = step