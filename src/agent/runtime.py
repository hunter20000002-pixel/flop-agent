from __future__ import annotations

from collections.abc import Callable

from src.agent.plan import ExecutionStep
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.task import Task
from src.inference.base import InferenceProvider, InferenceRequest


StepExecutor = Callable[[ExecutionStep], None]


class AgentRuntime:
    """Orchestrates task planning, inference, and execution."""

    def __init__(
        self,
        planner: Planner | None = None,
        step_executor: StepExecutor | None = None,
        inference_provider: InferenceProvider | None = None,
    ) -> None:
        self.planner = planner or Planner()
        self.step_executor = step_executor or self._default_step_executor
        self.inference_provider = inference_provider

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
            outputs: list[str] = []

            for step in plan.steps:
                if self.inference_provider is not None:
                    result = self.inference_provider.generate(
                        InferenceRequest(
                            prompt=step.description,
                        )
                    )

                    if result.failed:
                        raise RuntimeError(
                            result.error or "inference provider failed"
                        )

                    if result.output is not None:
                        outputs.append(result.output)
                else:
                    self.step_executor(step)

                executed_steps += 1

            task.mark_completed()

            return ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=executed_steps,
                output="\n".join(outputs) if outputs else None,
            )

        except Exception as exc:
            task.mark_failed()

            return ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=executed_steps
                if "executed_steps" in locals()
                else 0,
                output="\n".join(outputs) if "outputs" in locals() else None,
                error=str(exc),
            )

    @staticmethod
    def _default_step_executor(step: ExecutionStep) -> None:
        """Execute a step using the default deterministic executor."""

        _ = step