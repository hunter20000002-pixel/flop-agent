from __future__ import annotations

from dataclasses import dataclass

from src.agent.context import AgentContext
from src.agent.control import ControlDecision
from src.agent.decision import (
    AutonomyAction,
    AutonomyPolicy,
)
from src.agent.history import ExecutionHistory
from src.agent.memory_integration import MemoryIntegration
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.runtime import AgentRuntime
from src.agent.task import Task, TaskStatus


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
        )

        self.policy = policy or AutonomyPolicy()
        self.memory = memory
        self.max_iterations = max_iterations
        self.max_retries = max_retries

    def run(self, task: Task) -> AgentLoopResult:
        """Run the autonomous loop for a task."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        context = AgentContext(
            task=task,
            state="idle",
        )

        if self.memory is not None:
            context = self.memory.enrich_context(context)

        iterations = 0
        retry_count = 0

        last_result: ExecutionResult | None = None
        last_action = AutonomyAction.REPLAN

        task.set_status(TaskStatus.PLANNING)

        while iterations < self.max_iterations:
            iterations += 1

            decision = self.policy.decide(context)
            last_action = decision.action

            if decision.action == AutonomyAction.COMPLETE:
                if last_result is None:
                    task.mark_running()

                    last_result = self.runtime.run(
                        task,
                        plan=context.plan,
                    )

                if last_result.succeeded:
                    task.mark_completed()
                else:
                    task.mark_failed()

                return AgentLoopResult(
                    task_id=task.id,
                    result=last_result,
                    iterations=iterations,
                    action=AutonomyAction.COMPLETE,
                )

            if decision.action == AutonomyAction.STOP:
                if last_result is None:
                    task.mark_running()

                    last_result = self.runtime.run(
                        task,
                        plan=context.plan,
                    )

                if last_result.succeeded:
                    task.mark_completed()
                else:
                    task.mark_failed()

                return AgentLoopResult(
                    task_id=task.id,
                    result=last_result,
                    iterations=iterations,
                    action=AutonomyAction.STOP,
                )

            if decision.action == AutonomyAction.REPLAN:
                task.mark_planning()

                if self.memory is not None:
                    context = self.memory.enrich_context(context)

                if isinstance(self.planner, Planner):
                    plan = self.planner.plan(context)
                else:
                    plan = self.planner.plan(task)

                context = context.with_plan(plan)
                context = context.with_state("ready")

                task.mark_ready()
                retry_count = 0

                continue

            if decision.action == AutonomyAction.RETRY:
                if retry_count >= self.max_retries:
                    task.mark_failed()

                    if last_result is None:
                        last_result = ExecutionResult(
                            task_id=task.id,
                            status=TaskStatus.FAILED,
                            executed_steps=0,
                            error="retry limit exceeded",
                        )

                    return AgentLoopResult(
                        task_id=task.id,
                        result=last_result,
                        iterations=iterations,
                        action=AutonomyAction.STOP,
                    )

                retry_count += 1

            elif decision.action == AutonomyAction.EXECUTE:
                retry_count = 0

            if decision.action in (
                AutonomyAction.EXECUTE,
                AutonomyAction.RETRY,
            ):
                task.mark_running()
                context = context.with_state("running")

                last_result = self.runtime.run(
                    task,
                    plan=context.plan,
                )

                if self.memory is not None:
                    if last_result.output is not None:
                        self.memory.store_execution_output(
                            context,
                            last_result.output,
                        )

                    if last_result.history is not None:
                        context = context.with_history(
                            last_result.history
                        )

                    context = self.memory.enrich_context(context)

                elif last_result.history is not None:
                    context = context.with_history(
                        last_result.history
                    )

                if (
                    last_result.history is None
                    and context.plan is not None
                    and context.plan.steps
                ):
                    executed_index = min(
                        max(last_result.executed_steps, 1),
                        len(context.plan.steps),
                    )

                    executed_step = context.plan.steps[
                        executed_index - 1
                    ]

                    decision_for_history = (
                        ControlDecision.CONTINUE
                        if last_result.succeeded
                        else ControlDecision.FAIL
                    )

                    history = ExecutionHistory.from_step(
                        task_id=task.id,
                        step=executed_step,
                        success=last_result.succeeded,
                        output=last_result.output,
                        error=last_result.error,
                        decision=decision_for_history,
                    )

                    context = context.with_history(history)

                if last_result.succeeded:
                    task.mark_completed()
                    context = context.with_state("completed")

                    return AgentLoopResult(
                        task_id=task.id,
                        result=last_result,
                        iterations=iterations,
                        action=AutonomyAction.COMPLETE,
                    )

                task.mark_failed()
                context = context.with_state("running")

                continue

        if last_result is None:
            task.mark_running()

            last_result = self.runtime.run(
                task,
                plan=context.plan,
            )

        if last_result.succeeded:
            task.mark_completed()
        else:
            task.mark_failed()

        return AgentLoopResult(
            task_id=task.id,
            result=last_result,
            iterations=iterations,
            action=last_action,
        )