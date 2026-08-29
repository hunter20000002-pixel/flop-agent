from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from src.agent.context import AgentContext
from src.agent.control import (
    ControlDecision,
    ExecutionController,
    StepOutcome,
)
from src.agent.history import ExecutionHistory
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.task import Task
from src.inference.base import InferenceProvider, InferenceRequest
from src.tools.registry import ToolRegistry


StepExecutor = Callable[[ExecutionStep], None]


class AgentRuntime:
    """Orchestrates execution of agent plans."""

    def __init__(
        self,
        planner: Planner | None = None,
        *,
        step_executor: StepExecutor | None = None,
        inference_provider: InferenceProvider | None = None,
        tool_registry: ToolRegistry | None = None,
        controller: ExecutionController | None = None,
        max_steps: int = 100,
    ) -> None:
        self.planner = planner or Planner()

        self.step_executor = (
            step_executor or self._default_step_executor
        )

        self.inference_provider = inference_provider
        self.tool_registry = tool_registry

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero"
            )

        self.max_steps = max_steps
        self.controller = controller or ExecutionController()

    def run(
        self,
        task: Task,
        *,
        plan: ExecutionPlan | None = None,
    ) -> ExecutionResult:
        """
        Execute a task.

        If an execution plan is supplied, that exact plan is executed.
        Otherwise, the runtime creates a plan using the configured planner.

        This allows AgentLoop to separate planning from execution while
        preserving backward compatibility for callers that only provide
        a Task.
        """

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        if plan is not None:
            if not isinstance(plan, ExecutionPlan):
                raise TypeError(
                    "plan must be an ExecutionPlan or None"
                )

            if plan.task_id != task.id:
                raise ValueError(
                    "plan does not match the supplied task"
                )

        task.mark_planning()

        history = ExecutionHistory(task_id=task.id)

        executed_steps = 0
        outputs: list[str] = []

        try:
            context = AgentContext(
                task=task,
                history=history,
                state="planning",
            )

            if plan is None:
                plan = self._create_plan(
                    context=context,
                    task=task,
                )

            task.mark_ready()
            task.mark_running()

            context = context.with_plan(plan)
            context = context.with_state("running")

            for step in context.plan_steps:
                if executed_steps >= self.max_steps:
                    raise RuntimeError(
                        f"execution step limit exceeded: "
                        f"{self.max_steps}"
                    )

                started_at = datetime.now(timezone.utc)

                outcome = self._execute_step(step)

                completed_at = datetime.now(timezone.utc)

                decision = self.controller.decide(outcome)

                metadata = {
                    "step_order": step.order,
                    "execution_mode": (
                        "tool"
                        if step.tool_name is not None
                        else (
                            "inference"
                            if self.inference_provider is not None
                            else "executor"
                        )
                    ),
                }

                if step.tool_name is not None:
                    metadata["tool_name"] = step.tool_name

                if (
                    self.inference_provider is not None
                    and step.tool_name is None
                ):
                    metadata["provider"] = (
                        self.inference_provider.name
                    )

                history = history.record(
                    step,
                    success=outcome.success,
                    output=outcome.output,
                    error=outcome.error,
                    decision=decision,
                    started_at=started_at,
                    completed_at=completed_at,
                    metadata=metadata,
                )

                context = context.with_history(history)

                if outcome.output is not None:
                    outputs.append(str(outcome.output))

                if decision == ControlDecision.FAIL:
                    raise RuntimeError(
                        outcome.error
                        or "execution step failed"
                    )

                executed_steps += 1

                if decision == ControlDecision.STOP:
                    task.mark_completed()
                    context = context.with_state("completed")
                    break

            else:
                task.mark_completed()
                context = context.with_state("completed")

            return ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=executed_steps,
                output=(
                    "\n".join(outputs)
                    if outputs
                    else None
                ),
                history=context.history,
            )

        except Exception as exc:
            task.mark_failed()

            return ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=executed_steps,
                output=(
                    "\n".join(outputs)
                    if outputs
                    else None
                ),
                error=str(exc),
                history=history,
            )

    def _create_plan(
        self,
        context: AgentContext,
        task: Task,
    ) -> ExecutionPlan:
        """
        Create a plan while supporting both the built-in context-aware
        Planner and older custom planners.
        """

        if isinstance(self.planner, Planner):
            return self.planner.plan(context)

        return self.planner.plan(task)

    def _execute_step(
        self,
        step: ExecutionStep,
    ) -> StepOutcome:
        """Execute one step using the configured mechanism."""

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

    def _execute_tool_step(
        self,
        step: ExecutionStep,
    ) -> StepOutcome:
        """Resolve and execute the tool requested by a plan step."""

        if self.tool_registry is None:
            return StepOutcome(
                success=False,
                error=(
                    f"step requires tool '{step.tool_name}', "
                    "but no tool registry is configured"
                ),
            )

        tool = self.tool_registry.get(
            step.tool_name
        )

        result = tool.execute(
            **step.tool_args
        )

        return StepOutcome(
            success=result.success,
            output=result.output,
            error=result.error,
        )

    def _execute_inference_step(
        self,
        step: ExecutionStep,
    ) -> StepOutcome:
        """Execute one plan step through inference."""

        if self.inference_provider is None:
            return StepOutcome(
                success=False,
                error="no inference provider is configured",
            )

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
    def _default_step_executor(
        step: ExecutionStep,
    ) -> None:
        """Default executor for a step without an external mechanism."""

        return None