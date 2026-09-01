
from __future__ import annotations

from src.agent.decision import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.loop import AgentLoop
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.runtime import AgentRuntime
from src.agent.task import Task, TaskStatus
from src.tools.builtin import create_builtin_registry


class ReplanAfterFailurePolicy(AutonomyPolicy):
    """
    Force the sequence:

    REPLAN → EXECUTE → REPLAN → EXECUTE

    The second execution succeeds, so AgentLoop completes the task.
    """

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, context):
        self.calls += 1

        if self.calls in (1, 3):
            return AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="create or replace execution plan",
            )

        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="execute current execution plan",
        )


class RecordingPlanner(Planner):
    """Context-aware planner that records capability propagation."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.capabilities_seen: list[frozenset[str] | None] = []

    def plan(self, context):
        self.calls += 1
        self.capabilities_seen.append(
            context.allowed_capabilities
        )

        return super().plan(context)


class FailingThenSuccessfulRuntime(AgentRuntime):
    """Fail once, then succeed, while recording capabilities."""

    def __init__(self) -> None:
        super().__init__(
            tool_registry=create_builtin_registry(),
        )
        self.calls = 0
        self.capabilities_seen: list[frozenset[str] | None] = []

    def run(
        self,
        task: Task,
        *,
        plan=None,
        allowed_capabilities=None,
    ) -> ExecutionResult:
        self.calls += 1

        normalized_capabilities = (
            None
            if allowed_capabilities is None
            else frozenset(allowed_capabilities)
        )

        self.capabilities_seen.append(
            normalized_capabilities
        )

        if self.calls == 1:
            return ExecutionResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                executed_steps=0,
                error="forced first execution failure",
            )

        return ExecutionResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            executed_steps=1,
            output="5",
        )


def test_agent_loop_preserves_capabilities_through_replanning() -> None:
    """
    Capability restrictions must survive an execution failure and
    subsequent replanning.
    """

    planner = RecordingPlanner()
    runtime = FailingThenSuccessfulRuntime()
    policy = ReplanAfterFailurePolicy()

    task = Task(
        description="Calculate 2 + 3",
    )

    result = AgentLoop(
        planner=planner,
        runtime=runtime,
        policy=policy,
    ).run(
        task,
        allowed_capabilities={"calculator"},
    )

    expected = frozenset({"calculator"})

    assert result.result.succeeded
    assert result.result.output == "5"

    assert planner.calls == 2
    assert planner.capabilities_seen == [
        expected,
        expected,
    ]

    assert runtime.calls == 2
    assert runtime.capabilities_seen == [
        expected,
        expected,
    ]


def test_agent_loop_rejects_unauthorized_capability_during_planning() -> None:
    """
    A restricted task must fail during planning when its selected
    tool is not authorized.
    """

    task = Task(
        description="Calculate 2 + 3",
    )

    try:
        AgentLoop().run(
            task,
            allowed_capabilities=frozenset(),
        )
    except PermissionError as exc:
        assert str(exc) == (
            "tool 'calculator' requires capability "
            "'calculator', which is not authorized"
        )
    else:
        raise AssertionError(
            "expected unauthorized calculator planning to raise "
            "PermissionError"
        )
