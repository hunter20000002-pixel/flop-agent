from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.agent.autonomy_context import AutonomyDecisionContext
from src.agent.context import AgentContext
from src.agent.control import ControlDecision
from src.agent.decision import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.history import ExecutionHistory
from src.agent.memory_integration import MemoryIntegration
from src.agent.observation import TechnocoreObservation
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.runtime import AgentRuntime
from src.agent.task import Task, TaskStatus
from src.tools.builtin import create_builtin_registry


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Result produced by an autonomous agent loop."""

    task_id: object
    result: ExecutionResult
    iterations: int
    action: AutonomyAction

    @property
    def completed(self) -> bool:
        """Return True when the loop completed the task."""

        return self.action == AutonomyAction.COMPLETE

    @property
    def stopped(self) -> bool:
        """Return True when the loop stopped."""

        return self.action == AutonomyAction.STOP


class AgentLoop:
    """Autonomous execution loop for agent tasks."""

    def __init__(
        self,
        *,
        planner: Planner | None = None,
        runtime: AgentRuntime | None = None,
        policy: AutonomyPolicy | None = None,
        memory: MemoryIntegration | None = None,
        max_iterations: int = 10,
        max_retries: int = 3,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than zero"
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries must be greater than or equal to zero"
            )

        self.planner = planner or Planner()

        self.runtime = runtime or AgentRuntime(
            planner=self.planner,
            tool_registry=create_builtin_registry(),
        )

        self.policy = policy or AutonomyPolicy()
        self.memory = memory
        self.max_iterations = max_iterations
        self.max_retries = max_retries

    def run(
        self,
        task: Task,
        *,
        allowed_capabilities: Iterable[str] | None = None,
    ) -> AgentLoopResult:
        """
        Run the autonomous loop for a task.

        ``allowed_capabilities`` is normalized once and carried inside
        the immutable AgentContext so planning and execution share
        the same authorization boundary.

        ``None`` preserves unrestricted local execution.
        """

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        capabilities = (
            None
            if allowed_capabilities is None
            else frozenset(allowed_capabilities)
        )

        context = AgentContext(
            task=task,
            agent_id=(
                self.memory.agent_id
                if self.memory is not None
                else None
            ),
            state="idle",
            allowed_capabilities=capabilities,
        )

        if self.memory is not None:
            context = self.memory.enrich_context(context)

        iterations = 0
        retry_count = 0
        failure_count = 0
        replan_count = 0

        last_result: ExecutionResult | None = None
        terminal_result: ExecutionResult | None = None
        last_action = AutonomyAction.REPLAN

        task.set_status(TaskStatus.PLANNING)

        while iterations < self.max_iterations:
            iterations += 1

            decision = self._decide(
                task=task,
                context=context,
                result=last_result,
                failure_count=failure_count,
                retry_count=retry_count,
                replan_count=replan_count,
                capabilities=capabilities,
            )
            last_action = decision.action

            if decision.action == AutonomyAction.COMPLETE:
                return self._finish(
                    task=task,
                    context=context,
                    result=last_result,
                    iterations=iterations,
                    action=AutonomyAction.COMPLETE,
                    allowed_capabilities=capabilities,
                )

            if decision.action == AutonomyAction.STOP:
                return self._finish(
                    task=task,
                    context=context,
                    result=last_result,
                    iterations=iterations,
                    action=AutonomyAction.STOP,
                    allowed_capabilities=capabilities,
                )

            if decision.action == AutonomyAction.REPLAN:
                task.mark_planning()

                if self.memory is not None:
                    context = self.memory.enrich_context(
                        context
                    )

                plan = self._create_plan(
                    context=context,
                    task=task,
                )

                context = context.with_plan(plan)
                context = context.with_state("ready")

                task.mark_ready()
                retry_count = 0

                # Initial planning is not counted as a replan.
                #
                # A replan_count represents an actual replacement of
                # an already-executed plan. This distinction matters
                # for goal verification: the first goal-verification
                # failure must be allowed to trigger the first real
                # replan.
                if last_result is not None:
                    replan_count += 1
                    terminal_result = last_result

                # The previous execution result belongs to the old plan.
                # Clear it so the next decision evaluates the new plan
                # instead of immediately reacting to stale verification
                # evidence.
                last_result = None

                continue

            if decision.action == AutonomyAction.RETRY:
                if retry_count >= self.max_retries:
                    task.mark_failed()

                    return self._finish(
                        task=task,
                        context=context,
                        result=last_result,
                        iterations=iterations,
                        action=AutonomyAction.STOP,
                        allowed_capabilities=capabilities,
                    )

                retry_count += 1

            elif decision.action == AutonomyAction.EXECUTE:
                retry_count = 0

            if decision.action not in (
                AutonomyAction.EXECUTE,
                AutonomyAction.RETRY,
            ):
                continue

            task.mark_running()
            context = context.with_state("running")

            if capabilities is None:
                last_result = self.runtime.run(
                    task,
                    plan=context.plan,
                )
            else:
                last_result = self.runtime.run(
                    task,
                    plan=context.plan,
                    allowed_capabilities=capabilities,
                )

            terminal_result = last_result

            context = self._update_context_after_execution(
                context=context,
                result=last_result,
            )

            if last_result.succeeded:
                if self._goal_satisfied(last_result):
                    task.mark_completed()
                    context = context.with_state("completed")

                    return AgentLoopResult(
                        task_id=task.id,
                        result=last_result,
                        iterations=iterations,
                        action=AutonomyAction.COMPLETE,
                    )

                task.mark_failed()
                context = context.with_state("failed")
                failure_count += 1
                continue

            task.mark_failed()
            context = context.with_state("failed")
            failure_count += 1

            continue

        # The loop exhausted its iteration budget. This is a terminal
        # condition owned by the loop itself, so do not ask the runtime
        # to execute another time just to manufacture a result.
        task.mark_failed()

        if terminal_result is None:
            return self._finish(
                task=task,
                context=context,
                result=None,
                iterations=iterations,
                action=AutonomyAction.STOP,
                allowed_capabilities=capabilities,
            )

        return AgentLoopResult(
            task_id=task.id,
            result=terminal_result,
            iterations=iterations,
            action=AutonomyAction.STOP,
        )

    def _decide(
        self,
        *,
        task: Task,
        context: AgentContext,
        result: ExecutionResult | None,
        failure_count: int,
        retry_count: int,
        replan_count: int,
        capabilities: frozenset[str] | None,
    ) -> AutonomyDecision:
        """Choose the next action using the richest available context."""

        use_autonomy_context = (
            replan_count > 0
            or (
                result is not None
                and result.goal_verification is not None
            )
        )

        if use_autonomy_context:
            autonomy_context = AutonomyDecisionContext(
                task=task,
                current_plan=context.plan,
                current_step=(
                    context.plan.steps[0]
                    if context.plan is not None
                    and context.plan.steps
                    else None
                ),
                execution_history=(
                    result.history
                    if result is not None
                    and result.history is not None
                    else (
                        context.history
                        if context.history is not None
                        else ExecutionHistory(task_id=task.id)
                    )
                ),
                last_result=result,
                failure_count=failure_count,
                retry_count=retry_count,
                replan_count=replan_count,
                allowed_capabilities=capabilities,
                remaining_step_budget=None,
                memories=context.memories,
            )

            return self.policy.decide(autonomy_context)

        return self.policy.decide(context)

    def _goal_satisfied(self, result: ExecutionResult) -> bool:
        """Return True when execution and semantic goal verification succeeded."""

        if result.goal_verification is None:
            return result.succeeded

        return (
            result.succeeded
            and result.goal_verification.satisfied
        )

    def _create_plan(
        self,
        *,
        context: AgentContext,
        task: Task,
    ):
        """Create a plan using the configured planner."""

        if isinstance(self.planner, Planner):
            return self.planner.plan(context)

        return self.planner.plan(task)

    def _update_context_after_execution(
        self,
        *,
        context: AgentContext,
        result: ExecutionResult,
    ) -> AgentContext:
        """Update agent context after runtime execution."""

        updated_context = context

        if result.history is not None:
            updated_context = updated_context.with_history(
                result.history
            )

        if (
            result.history is None
            and updated_context.plan is not None
            and updated_context.plan.steps
        ):
            executed_index = min(
                max(result.executed_steps, 1),
                len(updated_context.plan.steps),
            )

            executed_step = updated_context.plan.steps[
                executed_index - 1
            ]

            decision = (
                ControlDecision.CONTINUE
                if result.succeeded
                else ControlDecision.FAIL
            )

            history = ExecutionHistory.from_step(
                task_id=updated_context.task.id,
                step=executed_step,
                success=result.succeeded,
                output=result.output,
                error=result.error,
                decision=decision,
            )

            updated_context = updated_context.with_history(
                history
            )

        if (
            self.memory is not None
            and result.output is not None
        ):
            self.memory.store_execution_output(
                updated_context,
                result.output,
            )

        if (
            self.memory is not None
            and isinstance(
                result.data,
                TechnocoreObservation,
            )
        ):
            self.memory.store_observation(
                updated_context,
                result.data,
            )

        if self.memory is not None:
            updated_context = self.memory.enrich_context(
                updated_context
            )

        return updated_context

    def _finish(
        self,
        *,
        task: Task,
        context: AgentContext,
        result: ExecutionResult | None,
        iterations: int,
        action: AutonomyAction,
        allowed_capabilities: frozenset[str] | None,
    ) -> AgentLoopResult:
        """Finalize the loop with a valid execution result."""

        if result is None:
            task.mark_running()

            if allowed_capabilities is None:
                result = self.runtime.run(
                    task,
                    plan=context.plan,
                )
            else:
                result = self.runtime.run(
                    task,
                    plan=context.plan,
                    allowed_capabilities=allowed_capabilities,
                )

        if self._goal_satisfied(result):
            task.mark_completed()
        else:
            task.mark_failed()

        return AgentLoopResult(
            task_id=task.id,
            result=result,
            iterations=iterations,
            action=action,
        )
