from __future__ import annotations

from collections.abc import Callable

from src.agent.plan import ExecutionStep
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.task import Task
from src.inference.base import InferenceProvider, InferenceRequest
from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry
from src.agent.control import (
    ControlDecision,
    ExecutionController,
    StepOutcome,
)


StepExecutor = Callable[[ExecutionStep], None]


class AgentRuntime:
    """Orchestrates task planning, tools, inference, and execution."""

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

        executed_steps = 0
        outputs: list[str] = []

        try:
            plan = self.planner.plan(task)

            if plan.is_empty:
                raise ValueError("execution plan cannot be empty")

            task.mark_ready()
            task.mark_running()

            for step in plan.steps:
                if executed_steps >= self.max_steps:
                    raise RuntimeError(
                        f"execution step limit exceeded: {self.max_steps}"
                    )

                try:
                    output = self._execute_step(step)

                    outcome = StepOutcome(
                        success=True,
                        output=output,
                    )

                except Exception as exc:
                    outcome = StepOutcome(
                        success=False,
                        error=str(exc),
                    )

                    decision = self.controller.decide(outcome)

                    if decision == ControlDecision.FAIL:
                        raise RuntimeError(
                            outcome.error or "execution step failed"
                        )

                    if decision == ControlDecision.STOP:
                        break

                    continue

                decision = self.controller.decide(outcome)

                if outcome.output is not None:
                    outputs.append(str(outcome.output))

                executed_steps += 1

                if decision == ControlDecision.FAIL:
                    raise RuntimeError(
                        outcome.error or "execution step failed"
                    )

                if decision == ControlDecision.STOP:
                    break

                if decision == ControlDecision.FAIL:
                    raise RuntimeError(
                        outcome.error or "execution step failed"
                    )

                if decision == ControlDecision.STOP:
                    break

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
                executed_steps=executed_steps,
                output="\n".join(outputs) if outputs else None,
                error=str(exc),
            )

    def _execute_step(self, step: ExecutionStep) -> str | None:
        """Execute one plan step using the appropriate execution mechanism."""

        if step.uses_tool:
            return self._execute_tool_step(step)

        if self.inference_provider is not None:
            return self._execute_inference_step(step)

        self.step_executor(step)
        return None

    def _execute_tool_step(self, step: ExecutionStep) -> str | None:
        """Resolve and execute the tool requested by a plan step."""

        if self.tool_registry is None:
            raise RuntimeError(
                f"step requires tool '{step.tool_name}', "
                "but no tool registry is configured"
            )

        tool = self.tool_registry.get(step.tool_name)

        result = tool.execute(**step.tool_args)

        if not isinstance(result, ToolResult):
            raise TypeError(
                f"tool '{tool.name}' returned an invalid result"
            )

        if result.failed:
            raise RuntimeError(
                result.error or f"tool '{tool.name}' failed"
            )

        if result.output is None:
            return None

        return str(result.output)

    def _execute_inference_step(self, step: ExecutionStep) -> str | None:
        """Execute a plan step through the configured inference provider."""

        result = self.inference_provider.generate(
            InferenceRequest(
                prompt=step.description,
            )
        )

        if result.failed:
            raise RuntimeError(
                result.error or "inference provider failed"
            )

        return result.output

    @staticmethod
    def _default_step_executor(step: ExecutionStep) -> None:
        """Execute a step using the default deterministic executor."""

        _ = step