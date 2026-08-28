from __future__ import annotations

from dataclasses import dataclass

from src.agent.context import AgentContext
from src.agent.decision import (
    AutonomyAction,
    AutonomyPolicy,
)
from src.agent.memory_integration import MemoryIntegration
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.runtime import AgentRuntime
from src.agent.task import Task


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
    ) -> None:
        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than zero"
            )

        self.planner = planner or Planner()
        self.runtime = runtime or AgentRuntime(
            planner=self.planner,
        )
        self.policy = policy or AutonomyPolicy()
        self.memory = memory
        self.max_iterations = max_iterations

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
        last_result: ExecutionResult | None = None
        last_action = AutonomyAction.REPLAN

        while iterations < self.max_iterations:
            iterations += 1

            decision = self.policy.decide(context)
            last_action = decision.action

            if decision.action == AutonomyAction.COMPLETE:
                if last_result is None:
                    last_result = self.runtime.run(task)

                return AgentLoopResult(
                    task_id=task.id,
                    result=last_result,
                    iterations=iterations,
                    action=decision.action,
                )

            if decision.action == AutonomyAction.STOP:
                if last_result is None:
                    last_result = self.runtime.run(task)

                return AgentLoopResult(
                    task_id=task.id,
                    result=last_result,
                    iterations=iterations,
                    action=decision.action,
                )

            if decision.action == AutonomyAction.REPLAN:
                plan = self.planner.plan(context)

                context = context.with_plan(plan)

                if self.memory is not None:
                    context = self.memory.enrich_context(context)

                continue

            if decision.action in (
                AutonomyAction.EXECUTE,
                AutonomyAction.RETRY,
            ):
                last_result = self.runtime.run(task)

                if self.memory is not None:
                    if last_result.output is not None:
                        self.memory.store_execution_output(
                            context,
                            last_result.output,
                        )

                    context = context.with_history(
                        last_result.history
                    )

                    context = self.memory.enrich_context(
                        context
                    )

                elif last_result.history is not None:
                    context = context.with_history(
                        last_result.history
                    )

                if last_result.succeeded:
                    context = context.with_state("completed")
                else:
                    context = context.with_state("running")

                continue

        if last_result is None:
            last_result = self.runtime.run(task)

        return AgentLoopResult(
            task_id=task.id,
            result=last_result,
            iterations=iterations,
            action=last_action,
        )