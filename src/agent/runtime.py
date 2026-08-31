from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from src.agent.decision import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
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
        autonomy_policy: AutonomyPolicy | None = None,
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

        # Autonomy is opt-in.
        #
        # Existing AgentRuntime behavior remains unchanged when
        # no autonomy policy is supplied.
        self.autonomy_policy = autonomy_policy
        self._autonomy_enabled = autonomy_policy is not None

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

        When an autonomy policy is explicitly supplied, it can request:
        EXECUTE, RETRY, REPLAN, STOP, or COMPLETE.

        Without an explicit autonomy policy, runtime preserves the
        original deterministic execution behavior.
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

            if not isinstance(plan, ExecutionPlan):
                raise TypeError(
                    "planner must return an ExecutionPlan"
                )

            task.mark_ready()
            task.mark_running()

            context = context.with_plan(plan)
            context = context.with_state("running")

            step_index = 0

            # When True, the next loop iteration executes the same step
            # directly because the autonomy policy already authorized RETRY.
            retry_pending = False

            while step_index < len(context.plan_steps):
                autonomy_decision: AutonomyDecision | None = None

                # --------------------------------------------------
                # AUTONOMY CONTROL
                # --------------------------------------------------
                if self._autonomy_enabled and not retry_pending:
                    autonomy_decision = (
                        self.autonomy_policy.decide(context)
                    )

                    if autonomy_decision.action == (
                        AutonomyAction.COMPLETE
                    ):
                        break

                    if autonomy_decision.action == (
                        AutonomyAction.STOP
                    ):
                        break

                    if autonomy_decision.action == (
                        AutonomyAction.REPLAN
                    ):
                        plan = self._create_plan(
                            context=context,
                            task=task,
                        )

                        if not isinstance(plan, ExecutionPlan):
                            raise TypeError(
                                "planner must return an ExecutionPlan"
                            )

                        context = context.with_plan(plan)
                        step_index = 0
                        retry_pending = False
                        continue

                    if autonomy_decision.action not in {
                        AutonomyAction.EXECUTE,
                        AutonomyAction.RETRY,
                    }:
                        raise RuntimeError(
                            "unsupported autonomy action: "
                            f"{autonomy_decision.action}"
                        )

                # --------------------------------------------------
                # STEP LIMIT
                # --------------------------------------------------
                if executed_steps >= self.max_steps:
                    raise RuntimeError(
                        f"execution step limit exceeded: "
                        f"{self.max_steps}"
                    )

                step = context.plan_steps[step_index]

                started_at = datetime.now(timezone.utc)

                outcome = self._execute_step(
                    step,
                    context,
                )

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

                if autonomy_decision is not None:
                    metadata["autonomy_action"] = (
                        autonomy_decision.action.value
                    )
                    metadata["autonomy_reason"] = (
                        autonomy_decision.reason
                    )

                if retry_pending:
                    metadata["retry"] = True

                if step.tool_name is not None:
                    metadata["tool_name"] = step.tool_name

                if (
                    self.inference_provider is not None
                    and step.tool_name is None
                ):
                    metadata["provider"] = (
                        self.inference_provider.name
                    )

                # --------------------------------------------------
                # RECORD EXECUTION
                # --------------------------------------------------
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

                # The retry has now actually been attempted.
                retry_pending = False

                # --------------------------------------------------
                # FAILURE
                # --------------------------------------------------
                if decision == ControlDecision.FAIL:

                    # Without an explicitly configured autonomy
                    # policy, preserve the original behavior:
                    # failure immediately fails the task.
                    if not self._autonomy_enabled:
                        raise RuntimeError(
                            outcome.error
                            or "execution step failed"
                        )

                    # Autonomy is enabled. Ask the policy what to do
                    # with the failed execution.
                    retry_decision = (
                        self.autonomy_policy.decide(context)
                    )

                    if retry_decision.action == (
                        AutonomyAction.RETRY
                    ):
                        # Retry the SAME step on the next iteration.
                        #
                        # Important: do not ask the autonomy policy again
                        # before that retry. The policy has already made
                        # the RETRY decision.
                        retry_pending = True
                        continue

                    if retry_decision.action == (
                        AutonomyAction.REPLAN
                    ):
                        plan = self._create_plan(
                            context=context,
                            task=task,
                        )

                        if not isinstance(plan, ExecutionPlan):
                            raise TypeError(
                                "planner must return an ExecutionPlan"
                            )

                        context = context.with_plan(plan)
                        step_index = 0
                        retry_pending = False
                        continue

                    if retry_decision.action == (
                        AutonomyAction.STOP
                    ):
                        break

                    if retry_decision.action == (
                        AutonomyAction.COMPLETE
                    ):
                        break

                    raise RuntimeError(
                        outcome.error
                        or "execution step failed"
                    )

                # --------------------------------------------------
                # SUCCESS / STOP
                # --------------------------------------------------
                executed_steps += 1

                if decision == ControlDecision.STOP:
                    break

                # Move to the next plan step.
                step_index += 1

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
        context: AgentContext,
    ) -> StepOutcome:
        """Execute one step using the configured mechanism."""

        try:
            if step.tool_name is not None:
                return self._execute_tool_step(step)

            if self.inference_provider is not None:
                return self._execute_inference_step(
                    step,
                    context,
                )

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
        context: AgentContext,
    ) -> StepOutcome:
        """Execute one plan step through inference."""

        if self.inference_provider is None:
            return StepOutcome(
                success=False,
                error="no inference provider is configured",
            )

        request_context = {
            "task": context.task,
            "task_id": context.task_id,
            "plan": context.plan,
            "history": context.history,
            "memories": context.memories,
            "agent_id": context.agent_id,
            "state": context.state,
        }

        result = self.inference_provider.generate(
            InferenceRequest(
                prompt=step.description,
                context=request_context,
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
