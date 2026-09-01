from __future__ import annotations

from dataclasses import dataclass

from src.agent.autonomy import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.context import AgentContext
from src.agent.history import ExecutionHistory
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.result import ExecutionResult
from src.agent.runtime import AgentRuntime
from src.agent.task import Task
from src.tools.base import Tool, ToolResult
from src.tools.registry import ToolRegistry


@dataclass
class SequenceAutonomyPolicy(AutonomyPolicy):
    decisions: list[AutonomyDecision]

    def decide(self, context: AgentContext) -> AutonomyDecision:
        if not self.decisions:
            raise AssertionError(
                "autonomy decision sequence exhausted"
            )

        return self.decisions.pop(0)


class RecordingContextPlanner:
    """Planner that records every context it receives."""

    def __init__(self) -> None:
        self.contexts: list[AgentContext] = []

    def plan(self, context: AgentContext) -> ExecutionPlan:
        self.contexts.append(context)

        return ExecutionPlan(
            task_id=context.task_id,
            steps=(
                ExecutionStep(
                    order=1,
                    description="test step",
                ),
            ),
        )


class RecordingCalculatorTool(Tool):
    """Calculator test tool that records successful execution."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Test calculator tool."

    def execute(self, **kwargs) -> ToolResult:
        self.calls.append(dict(kwargs))

        return ToolResult(
            success=True,
            output="4",
        )


class RecordingFilesystemTool(Tool):
    """Filesystem test tool used to verify capability rejection."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return "Test filesystem tool."

    def execute(self, **kwargs) -> ToolResult:
        self.calls.append(dict(kwargs))

        return ToolResult(
            success=True,
            output="filesystem executed",
        )


class CapabilityEscalationPlanner:
    """
    First plan uses the authorized calculator capability.

    The first plan contains two steps so that autonomy is consulted
    again after the first successful step.

    The replanned plan deliberately attempts to use the
    unauthorized filesystem capability.
    """

    def __init__(self) -> None:
        self.contexts: list[AgentContext] = []

    def plan(self, context: AgentContext) -> ExecutionPlan:
        self.contexts.append(context)

        if len(self.contexts) == 1:
            return ExecutionPlan(
                task_id=context.task_id,
                steps=(
                    ExecutionStep(
                        order=1,
                        description="Calculate 2 + 2",
                        tool_name="calculator",
                        tool_args={
                            "expression": "2 + 2",
                        },
                    ),
                    ExecutionStep(
                        order=2,
                        description="Calculate 3 + 3",
                        tool_name="calculator",
                        tool_args={
                            "expression": "3 + 3",
                        },
                    ),
                ),
            )

        return ExecutionPlan(
            task_id=context.task_id,
            steps=(
                ExecutionStep(
                    order=1,
                    description="Write unauthorized filesystem data",
                    tool_name="filesystem",
                    tool_args={
                        "path": "unauthorized.txt",
                        "content": "should not execute",
                    },
                ),
            ),
        )


def test_runtime_context_preserves_allowed_capabilities() -> None:
    task = Task(
        description="capability context"
    )

    planner = RecordingContextPlanner()

    runtime = AgentRuntime(
        planner=planner
    )

    result = runtime.run(
        task,
        allowed_capabilities={"calculator"},
    )

    assert isinstance(result, ExecutionResult)
    assert result.succeeded
    assert len(planner.contexts) == 1

    assert (
        planner.contexts[0].allowed_capabilities
        == frozenset({"calculator"})
    )


def test_runtime_context_preserves_multiple_allowed_capabilities() -> None:
    task = Task(
        description="multiple capabilities"
    )

    planner = RecordingContextPlanner()

    runtime = AgentRuntime(
        planner=planner
    )

    result = runtime.run(
        task,
        allowed_capabilities={
            "calculator",
            "filesystem",
        },
    )

    assert result.succeeded
    assert len(planner.contexts) == 1

    assert (
        planner.contexts[0].allowed_capabilities
        == frozenset(
            {
                "calculator",
                "filesystem",
            }
        )
    )


def test_runtime_context_has_no_capability_restriction_when_unspecified() -> None:
    task = Task(
        description="unrestricted context"
    )

    planner = RecordingContextPlanner()

    runtime = AgentRuntime(
        planner=planner
    )

    result = runtime.run(task)

    assert result.succeeded
    assert len(planner.contexts) == 1
    assert planner.contexts[0].allowed_capabilities is None


def test_autonomy_replan_receives_same_capability_context() -> None:
    task = Task(
        description="capability replan"
    )

    planner = RecordingContextPlanner()

    policy = SequenceAutonomyPolicy(
        decisions=[
            AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="request a new plan",
            ),
            AutonomyDecision(
                action=AutonomyAction.COMPLETE,
                reason="finish after replan",
            ),
        ]
    )

    runtime = AgentRuntime(
        planner=planner,
        autonomy_policy=policy,
    )

    result = runtime.run(
        task,
        allowed_capabilities={"calculator"},
    )

    assert result.succeeded

    assert len(planner.contexts) == 2

    for context in planner.contexts:
        assert (
            context.allowed_capabilities
            == frozenset({"calculator"})
        )


def test_autonomy_context_capabilities_are_immutable() -> None:
    task = Task(
        description="immutable capabilities"
    )

    history = ExecutionHistory(
        task_id=task.id
    )

    context = AgentContext(
        task=task,
        history=history,
        state="planning",
        allowed_capabilities=frozenset(
            {"calculator"}
        ),
    )

    assert context.allowed_capabilities == frozenset(
        {"calculator"}
    )

    assert isinstance(
        context.allowed_capabilities,
        frozenset,
    )


def test_autonomy_cannot_expand_capability_context() -> None:
    task = Task(
        description="capability boundary"
    )

    planner = RecordingContextPlanner()

    policy = SequenceAutonomyPolicy(
        decisions=[
            AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="request replan",
            ),
            AutonomyDecision(
                action=AutonomyAction.COMPLETE,
                reason="complete",
            ),
        ]
    )

    runtime = AgentRuntime(
        planner=planner,
        autonomy_policy=policy,
    )

    result = runtime.run(
        task,
        allowed_capabilities={"calculator"},
    )

    assert result.succeeded

    assert len(planner.contexts) == 2

    for context in planner.contexts:
        assert (
            context.allowed_capabilities
            == frozenset({"calculator"})
        )

        assert (
            "filesystem"
            not in context.allowed_capabilities
        )


def test_autonomy_replan_preserves_history_authorization_metadata() -> None:
    task = Task(
        description="history authorization"
    )

    planner = RecordingContextPlanner()

    policy = SequenceAutonomyPolicy(
        decisions=[
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="execute",
            ),
        ]
    )

    runtime = AgentRuntime(
        planner=planner,
        autonomy_policy=policy,
    )

    result = runtime.run(
        task,
        allowed_capabilities={"calculator"},
    )

    assert result.succeeded
    assert result.history is not None

    for record in result.history.records:
        allowed = record.metadata.get(
            "allowed_capabilities"
        )

        if allowed is not None:
            assert allowed == ("calculator",)


def test_autonomy_replan_cannot_execute_unauthorized_tool() -> None:
    """
    An autonomy-triggered replan must remain inside the capability
    boundary established by the original caller.

    The caller authorizes only calculator.

    The first plan contains two authorized calculator steps.

    Autonomy is consulted again after the first successful step
    and requests a replan.

    The replanned plan deliberately contains a filesystem step.

    The runtime must reject that step before the filesystem tool
    executes.
    """

    task = Task(
        description=(
            "Calculate first, then attempt unauthorized "
            "filesystem access"
        )
    )

    planner = CapabilityEscalationPlanner()

    calculator = RecordingCalculatorTool()
    filesystem = RecordingFilesystemTool()

    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(filesystem)

    policy = SequenceAutonomyPolicy(
        decisions=[
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="execute authorized calculator step",
            ),
            AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="request autonomous replan",
            ),
            AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="stop after unauthorized capability rejection",
            ),
        ]
    )

    runtime = AgentRuntime(
        planner=planner,
        tool_registry=registry,
        autonomy_policy=policy,
    )

    result = runtime.run(
        task,
        allowed_capabilities={"calculator"},
    )

    assert len(planner.contexts) == 2

    assert all(
        context.allowed_capabilities
        == frozenset({"calculator"})
        for context in planner.contexts
    )

    assert calculator.calls == [
        {
            "expression": "2 + 2",
        },
    ]

    assert filesystem.calls == []

    assert result.history is not None
    assert result.history.record_count == 2

    first_record = result.history.records[0]

    assert first_record.success
    assert first_record.output == "4"
    assert first_record.capability == "calculator"
    assert first_record.metadata["tool_name"] == "calculator"
    assert first_record.metadata["allowed_capabilities"] == (
        "calculator",
    )

    second_record = result.history.records[1]

    assert not second_record.success
    assert second_record.capability == "filesystem"
    assert second_record.metadata["tool_name"] == "filesystem"
    assert second_record.metadata["allowed_capabilities"] == (
        "calculator",
    )
    assert second_record.error == (
        "tool 'filesystem' requires capability "
        "'filesystem', which is not authorized"
    )
    assert second_record.output is None