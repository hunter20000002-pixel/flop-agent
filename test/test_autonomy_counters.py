
from __future__ import annotations

from src.agent.decision import AutonomyAction, AutonomyDecision
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.runtime import AgentRuntime
from src.agent.task import Task


class RecordingPolicy:
    def __init__(self, decisions: list[AutonomyDecision]) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)

        if not self.decisions:
            raise AssertionError("policy ran out of decisions")

        return self.decisions.pop(0)


class TwoStepPlanner:
    def plan(self, context):
        return ExecutionPlan(
            task_id=context.task.id,
            steps=(
                ExecutionStep(
                    description="first step",
                    order=1,
                ),
                ExecutionStep(
                    description="second step",
                    order=2,
                ),
            ),
        )


class ReplanningPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, context):
        self.calls += 1

        return ExecutionPlan(
            task_id=context.task.id,
            steps=(
                ExecutionStep(
                    description=f"replanned step {self.calls}-1",
                    order=1,
                ),
                ExecutionStep(
                    description=f"replanned step {self.calls}-2",
                    order=2,
                ),
            ),
        )


def test_failure_count_tracks_actual_failed_execution() -> None:
    policy = RecordingPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="initial execution",
            ),
            AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="stop after failure",
            ),
        ]
    )

    def executor(step):
        raise RuntimeError("expected failure")

    task = Task(
        description="Failure counter"
    )

    result = AgentRuntime(
        planner=TwoStepPlanner(),
        step_executor=executor,
        autonomy_policy=policy,
    ).run(task)

    assert result.status.value == "completed"

    failure_context = policy.contexts[1]

    assert failure_context.failure_count == 1
    assert failure_context.retry_count == 0
    assert failure_context.replan_count == 0
    assert failure_context.last_result is not None
    assert failure_context.last_result.failed is True


def test_retry_count_tracks_retry_decisions() -> None:
    policy = RecordingPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="initial execution",
            ),
            AutonomyDecision(
                action=AutonomyAction.RETRY,
                reason="retry once",
            ),
            AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="stop after retry",
            ),
        ]
    )

    attempts = 0

    def executor(step):
        nonlocal attempts
        attempts += 1

        if attempts <= 2:
            raise RuntimeError("temporary failure")

    task = Task(
        description="Retry counter"
    )

    result = AgentRuntime(
        planner=TwoStepPlanner(),
        step_executor=executor,
        autonomy_policy=policy,
    ).run(task)

    assert result.status.value == "completed"
    assert attempts == 2

    retry_execution_context = policy.contexts[2]

    assert retry_execution_context.failure_count == 2
    assert retry_execution_context.retry_count == 1
    assert retry_execution_context.replan_count == 0


def test_replan_count_tracks_replan_decisions() -> None:
    policy = RecordingPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="initial execution",
            ),
            AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="replan after execution",
            ),
            AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="stop after replanned execution",
            ),
        ]
    )

    planner = ReplanningPlanner()
    executions = 0

    def executor(step):
        nonlocal executions
        executions += 1

    task = Task(
        description="Replan counter"
    )

    result = AgentRuntime(
        planner=planner,
        step_executor=executor,
        autonomy_policy=policy,
    ).run(task)

    assert result.status.value == "completed"
    assert planner.calls == 2
    assert executions == 2

    replan_execution_context = policy.contexts[2]

    assert replan_execution_context.failure_count == 0
    assert replan_execution_context.retry_count == 0
    assert replan_execution_context.replan_count == 1


def test_remaining_step_budget_decreases_per_execution_attempt() -> None:
    policy = RecordingPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="initial execution",
            ),
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="continue execution",
            ),
        ]
    )

    task = Task(
        description="Budget tracking"
    )

    result = AgentRuntime(
        planner=TwoStepPlanner(),
        step_executor=lambda step: None,
        autonomy_policy=policy,
        max_steps=10,
    ).run(task)

    assert result.status.value == "completed"

    assert policy.contexts[0].remaining_step_budget == 10
    assert policy.contexts[1].remaining_step_budget == 9


def test_remaining_budget_counts_failed_attempts() -> None:
    policy = RecordingPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="initial execution",
            ),
            AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="stop after failure",
            ),
        ]
    )

    def executor(step):
        raise RuntimeError("failure")

    task = Task(
        description="Failed budget tracking"
    )

    result = AgentRuntime(
        planner=TwoStepPlanner(),
        step_executor=executor,
        autonomy_policy=policy,
        max_steps=10,
    ).run(task)

    assert result.status.value == "completed"

    assert policy.contexts[0].remaining_step_budget == 10
    assert policy.contexts[1].remaining_step_budget == 9


def test_retry_attempts_consume_step_budget() -> None:
    policy = RecordingPolicy(
        [
            AutonomyDecision(
                action=AutonomyAction.EXECUTE,
                reason="initial execution",
            ),
            AutonomyDecision(
                action=AutonomyAction.RETRY,
                reason="retry",
            ),
            AutonomyDecision(
                action=AutonomyAction.STOP,
                reason="stop",
            ),
        ]
    )

    attempts = 0

    def executor(step):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("failure")

    task = Task(
        description="Retry budget tracking"
    )

    result = AgentRuntime(
        planner=TwoStepPlanner(),
        step_executor=executor,
        autonomy_policy=policy,
        max_steps=10,
    ).run(task)

    assert result.status.value == "completed"
    assert attempts == 2

    assert policy.contexts[0].remaining_step_budget == 10
    assert policy.contexts[1].remaining_step_budget == 9
    assert policy.contexts[2].remaining_step_budget == 8
