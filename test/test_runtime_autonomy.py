from __future__ import annotations

from src.agent.autonomy import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.control import (
    ControlDecision,
    ExecutionController,
    StepOutcome,
)
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.runtime import AgentRuntime
from src.agent.task import Task


class RecordingAutonomyPolicy(AutonomyPolicy):
    def __init__(
        self,
        decisions: list[AutonomyDecision],
    ) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def decide(self, context):
        self.calls += 1

        if self.decisions:
            return self.decisions.pop(0)

        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="default test decision",
        )


class FailingOnceController(ExecutionController):
    def __init__(self) -> None:
        self.calls = 0

    def decide(
        self,
        outcome: StepOutcome,
    ) -> ControlDecision:
        self.calls += 1

        if self.calls == 1:
            return ControlDecision.FAIL

        return ControlDecision.CONTINUE


class TwoStepPlanner:
    def plan(self, task):
        return ExecutionPlan(
            task_id=task.id,
            steps=(
                ExecutionStep(
                    description="First step",
                    order=1,
                ),
                ExecutionStep(
                    description="Second step",
                    order=2,
                ),
            ),
        )


def test_runtime_accepts_autonomy_policy() -> None:
    policy = RecordingAutonomyPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="execute test",
            ),
        ]
    )

    task = Task(
        description="Autonomy policy injection"
    )

    result = AgentRuntime(
        autonomy_policy=policy,
    ).run(task)

    assert result.status.value == "completed"
    assert policy.calls >= 1


def test_runtime_autonomy_policy_can_stop_execution() -> None:
    policy = RecordingAutonomyPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="test stop",
            ),
        ]
    )

    task = Task(
        description="Autonomy stop"
    )

    result = AgentRuntime(
        planner=TwoStepPlanner(),
        autonomy_policy=policy,
    ).run(task)

    assert result.status.value == "completed"
    assert result.executed_steps == 0
    assert policy.calls == 1


def test_runtime_autonomy_policy_can_complete_execution() -> None:
    policy = RecordingAutonomyPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.COMPLETE,
                reason="already complete",
            ),
        ]
    )

    task = Task(
        description="Autonomy complete"
    )

    result = AgentRuntime(
        planner=TwoStepPlanner(),
        autonomy_policy=policy,
    ).run(task)

    assert result.status.value == "completed"
    assert result.executed_steps == 0
    assert policy.calls == 1


def test_runtime_records_autonomy_metadata() -> None:
    policy = RecordingAutonomyPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="execute with metadata",
            ),
        ]
    )

    task = Task(
        description="Autonomy metadata"
    )

    result = AgentRuntime(
        autonomy_policy=policy,
    ).run(task)

    assert result.status.value == "completed"
    assert result.history is not None
    assert len(result.history.records) == 1

    record = result.history.records[0]

    assert record.metadata["autonomy_action"] == "execute"
    assert record.metadata["autonomy_reason"] == (
        "execute with metadata"
    )


def test_runtime_autonomy_policy_is_called_before_execution() -> None:
    policy = RecordingAutonomyPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="first execution",
            ),
            AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="stop after first step",
            ),
        ]
    )

    task = Task(
        description="Autonomy sequencing"
    )

    result = AgentRuntime(
        planner=TwoStepPlanner(),
        autonomy_policy=policy,
    ).run(task)

    assert result.status.value == "completed"
    assert result.executed_steps == 1
    assert policy.calls == 2


def test_runtime_uses_retry_decision_after_failure() -> None:
    policy = RecordingAutonomyPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="initial execution",
            ),
            AutonomyDecision(
                action=AutonomyAction.RETRY,
                reason="retry failed step",
            ),
        ]
    )

    attempts = 0

    def executor(step):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise RuntimeError("temporary failure")

    task = Task(
        description="Autonomy retry"
    )

    result = AgentRuntime(
        planner=TwoStepPlanner(),
        step_executor=executor,
        autonomy_policy=policy,
    ).run(task)

    assert result.status.value == "completed"
    assert attempts == 3
    assert result.executed_steps == 2