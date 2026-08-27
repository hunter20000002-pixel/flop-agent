from __future__ import annotations

from collections.abc import Callable

from src.agent.control import (
    ControlDecision,
    ExecutionController,
    StepOutcome,
)
from src.agent.history import ExecutionHistory
from src.agent.plan import ExecutionStep
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.task import Task
from src.inference.base import InferenceProvider, InferenceRequest
from src.tools.registry import ToolRegistry


StepExecutor = Callable[[ExecutionStep], None]


class AgentRuntime:
    """Orchestrates task planning, inference, tools, and execution."""

    def __init__(
        self,
        planner: Planner | None = None,
        step_executor: StepExecutor | None = None,
        inference_provider: InferenceProvider | None = None,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 100,
        controller: ExecutionController | None = None,
    ) -> None:
        self.planner = planner or Planner()
        self.step_executor = step_executor or self._default_step_executor
        self.inference_provider = inference_provider
        self.tool_registry = tool_registry

        if max_steps < 1:
            raise ValueError("max_steps must be greater than zero")

        self.max_steps = max_steps
        self.controller = controller or ExecutionController()

    def run(self, task: Task) -> ExecutionResult:
        """Plan and execute a task."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        task.mark_planning()

        history = ExecutionHistory(task_id=task.id)

        try:
            plan = self.planner.plan(task)

            if plan.is_empty:
                raise ValueError("execution plan cannot be empty")

            task.mark_ready()
            task.mark_running()

            executed_steps = 0
            outputs: list[str] = []

            for step in plan.steps:
                if executed_steps >= self.max_steps:
                    raise RuntimeError(
                        f"execution step limit exceeded: {self.max_steps}"
                    )

                outcome = self._execute_step(step)

                decision = self.controller.decide(outcome)

                history = history.record(
                    step,
                    success=outcome.success,
                    output=outcome.output,
                    error=outcome.error,
                    decision=decision,
                )

                if outcome.output is not None:
                    outputs.append(str(outcome.output))

                if decision == ControlDecision.FAIL:
                    raise RuntimeError(
                        outcome.error or "execution step failed"
                    )

                executed_steps += 1

                if decision == ControlDecision.STOP:
                    break

            task.mark_completed()

            return ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=executed_steps,
                output="\n".join(outputs) if outputs else None,
                history=history,
            )

        except Exception as exc:
            task.mark_failed()

            return ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=(
                    executed_steps
                    if "executed_steps" in locals()
                    else 0
                ),
                output=(
                    "\n".join(outputs)
                    if "outputs" in locals() and outputs
                    else None
                ),
                error=str(exc),
                history=history,
            )

    def _execute_step(self, step: ExecutionStep) -> StepOutcome:
        """Execute one step using the configured execution mechanism."""

        try:
            if step.tool_name is not None:
                return self._execute_tool_step(step)

            if self.inference_provider is not None:
                return self._execute_inference_step(step)

            self.step_executor(step)

            return StepOutcome(
                success=True,
            )

        except Exception as exc:
            return StepOutcome(
                success=False,
                error=str(exc),
            )

    def _execute_tool_step(self, step: ExecutionStep) -> StepOutcome:
        """Resolve and execute the tool requested by a plan step."""

        if self.tool_registry is None:
            return StepOutcome(
                success=False,
                error=(
                    f"step requires tool '{step.tool_name}', "
                    "but no tool registry is configured"
                ),
            )

        tool = self.tool_registry.get(step.tool_name)

        result = tool.execute(**step.tool_args)

        return StepOutcome(
            success=result.success,
            output=result.output,
            error=result.error,
        )

    def _execute_inference_step(
        self,
        step: ExecutionStep,
    ) -> StepOutcome:
        """Execute a plan step through the configured inference provider."""

        result = self.inference_provider.generate(
            InferenceRequest(
                prompt=step.description,
            )
        )

        return StepOutcome(
            success=result.success,
            output=result.output,
            error=result.error,
        )

    @staticmethod
    def _default_step_executor(step: ExecutionStep) -> None:
        """Execute a step using the default deterministic executor."""

        _ = step