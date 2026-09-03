from __future__ import annotations

import pytest

from src.agent.decision import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.loop import AgentLoop
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus
from src.agent.runtime import AgentRuntime

from src.agent.decision import AutonomyAction
from src.agent.goal import GoalVerificationResult, GoalVerifier
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.runtime import AgentRuntime
from src.agent.task import Task
from src.tools.builtin import create_builtin_registry

class RecordingPlanner:
    def __init__(self) -> None:
        self.calls = 0
        self.plan_created: ExecutionPlan | None = None

    def plan(self, task: Task) -> ExecutionPlan:
        self.calls += 1

        self.plan_created = ExecutionPlan(
            task_id=task.id,
            steps=(
                ExecutionStep(
                    description="Test step",
                    order=1,
                ),
            ),
        )

        return self.plan_created


class RecordingRuntime:
    def __init__(
        self,
        *,
        results: list[ExecutionResult] | None = None,
    ) -> None:
        self.received_plan: ExecutionPlan | None = None
        self.received_capabilities = None
        self.calls = 0
        self.results = results or []

    def run(
        self,
        task: Task,
        *,
        plan: ExecutionPlan | None = None,
        allowed_capabilities=None,
    ) -> ExecutionResult:
        self.calls += 1
        self.received_plan = plan
        self.received_capabilities = allowed_capabilities

        if self.results:
            return self.results.pop(0)

        return ExecutionResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            executed_steps=1,
            output="test output",
        )


class ExecutePolicy(AutonomyPolicy):
    def decide(self, context):
        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="execute test plan",
        )


class ReplanThenExecutePolicy(AutonomyPolicy):
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, context):
        self.calls += 1

        if self.calls == 1:
            return AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="create test plan",
            )

        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="execute planned test plan",
        )


class RetryPolicy(AutonomyPolicy):
    def decide(self, context):
        if context.last_execution is not None:
            if context.last_execution.success:
                return AutonomyDecision(
                    action=AutonomyAction.COMPLETE,
                    reason="execution succeeded",
                )

            return AutonomyDecision(
                action=AutonomyAction.RETRY,
                reason="retry failed execution",
            )

        if context.plan is None:
            return AutonomyDecision(
                action=AutonomyAction.REPLAN,
                reason="create initial plan",
            )

        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="execute initial plan",
        )


class CompletePolicy(AutonomyPolicy):
    def decide(self, context):
        return AutonomyDecision(
            action=AutonomyAction.COMPLETE,
            reason="task complete",
        )


class StopPolicy(AutonomyPolicy):
    def decide(self, context):
        return AutonomyDecision(
            action=AutonomyAction.STOP,
            reason="stop requested",
        )


def test_agent_loop_rejects_invalid_task():
    with pytest.raises(TypeError, match="task must be a Task"):
        AgentLoop().run("not a task")


def test_agent_loop_rejects_invalid_max_iterations():
    with pytest.raises(
        ValueError,
        match="max_iterations must be greater than zero",
    ):
        AgentLoop(max_iterations=0)


def test_agent_loop_rejects_negative_max_retries():
    with pytest.raises(
        ValueError,
        match="max_retries must be greater than or equal to zero",
    ):
        AgentLoop(max_retries=-1)


def test_agent_loop_executes_task():
    task = Task(description="Execute task")

    result = AgentLoop(
        policy=ExecutePolicy(),
    ).run(task)

    assert isinstance(result.result, ExecutionResult)
    assert result.result.status == TaskStatus.COMPLETED
    assert result.result.succeeded
    assert result.result.task_id == task.id


def test_agent_loop_reports_iterations():
    task = Task(description="Count iterations")

    result = AgentLoop(
        policy=ExecutePolicy(),
    ).run(task)

    assert result.iterations >= 1


def test_agent_loop_can_complete_immediately():
    task = Task(description="Already complete")

    result = AgentLoop(
        policy=CompletePolicy(),
    ).run(task)

    assert isinstance(result.result, ExecutionResult)
    assert result.result.task_id == task.id


def test_agent_loop_can_stop_immediately():
    task = Task(description="Stop task")

    result = AgentLoop(
        policy=StopPolicy(),
    ).run(task)

    assert isinstance(result.result, ExecutionResult)
    assert result.result.task_id == task.id


def test_agent_loop_uses_planner():
    planner = RecordingPlanner()
    task = Task(description="Planned task")

    AgentLoop(
        planner=planner,
        policy=ReplanThenExecutePolicy(),
    ).run(task)

    assert planner.calls >= 1


def test_agent_loop_preserves_task_identity():
    task = Task(description="Identity test")

    result = AgentLoop(
        policy=ExecutePolicy(),
    ).run(task)

    assert result.result.task_id == task.id


def test_agent_loop_passes_planned_plan_to_runtime():
    planner = RecordingPlanner()
    runtime = RecordingRuntime()
    policy = ReplanThenExecutePolicy()
    task = Task(description="Plan contract test")

    AgentLoop(
        planner=planner,
        runtime=runtime,
        policy=policy,
    ).run(task)

    assert planner.plan_created is not None
    assert runtime.received_plan is planner.plan_created


def test_agent_loop_retries_failed_execution():
    task = Task(description="Retry test")

    failed_result = ExecutionResult(
        task_id=task.id,
        status=TaskStatus.FAILED,
        executed_steps=0,
        error="temporary failure",
    )

    successful_result = ExecutionResult(
        task_id=task.id,
        status=TaskStatus.COMPLETED,
        executed_steps=1,
        output="recovered",
    )

    runtime = RecordingRuntime(
        results=[
            failed_result,
            successful_result,
        ]
    )

    result = AgentLoop(
        runtime=runtime,
        policy=RetryPolicy(),
        max_retries=1,
    ).run(task)

    assert runtime.calls == 2
    assert result.result is successful_result
    assert result.result.succeeded


def test_agent_loop_stops_after_retry_limit():
    task = Task(description="Retry limit test")

    failed_result = ExecutionResult(
        task_id=task.id,
        status=TaskStatus.FAILED,
        executed_steps=0,
        error="persistent failure",
    )

    runtime = RecordingRuntime(
        results=[
            failed_result,
            failed_result,
        ]
    )

    result = AgentLoop(
        runtime=runtime,
        policy=RetryPolicy(),
        max_retries=1,
    ).run(task)

    assert runtime.calls == 2
    assert result.result is failed_result
    assert result.action == AutonomyAction.STOP
    assert result.result.failed


def test_agent_loop_executes_calculator_tool():
    task = Task(
        description="Calculate 12 * 8"
    )

    result = AgentLoop().run(task)

    assert result.result.succeeded
    assert result.result.output == "96"


def test_agent_loop_executes_filesystem_tool(tmp_path):
    task = Task(
        description=f"List the directory {tmp_path}"
    )

    result = AgentLoop().run(task)

    assert result.result.succeeded
    assert result.result.output is not None

def test_loop_does_not_complete_when_goal_verification_fails() -> None:
    from src.agent.goal import GoalVerificationResult, GoalVerifier

    class UnsatisfiedVerifier(GoalVerifier):
        def _verify(self, task, result):
            return GoalVerificationResult(
                satisfied=False,
                reason="goal was not achieved",
            )

    runtime = AgentRuntime(
        goal_verifier=UnsatisfiedVerifier(),
    )

    loop = AgentLoop(
        runtime=runtime,
        max_iterations=1,
    )

    task = Task(
        description="goal must not be treated as complete",
    )

    result = loop.run(task)

    assert result.completed is False
    assert result.result.goal_verification is not None
    assert result.result.goal_verification.satisfied is False
    assert task.status == TaskStatus.FAILED


def test_loop_reacts_to_goal_verification_failure_with_replan() -> None:
    from src.agent.goal import GoalVerificationResult, GoalVerifier

    class UnsatisfiedVerifier(GoalVerifier):
        def __init__(self):
            self.calls = 0

        def _verify(self, task, result):
            self.calls += 1

            return GoalVerificationResult(
                satisfied=False,
                reason="goal requires another attempt",
            )

    verifier = UnsatisfiedVerifier()

    loop = AgentLoop(
        runtime=AgentRuntime(
            goal_verifier=verifier,
        ),
        max_iterations=3,
    )

    task = Task(
        description="retry through replanning",
    )

    result = loop.run(task)

    assert verifier.calls >= 1
    assert result.action == AutonomyAction.STOP
    assert result.completed is False
    assert result.result.goal_verification is not None
    assert result.result.goal_verification.satisfied is False


def test_loop_executes_replanned_plan_after_goal_verification_failure() -> None:
    class EventuallySatisfiedVerifier(GoalVerifier):
        def __init__(self) -> None:
            self.calls = 0

        def _verify(
            self,
            task: Task,
            result: ExecutionResult,
        ) -> GoalVerificationResult:
            self.calls += 1

            return GoalVerificationResult(
                satisfied=self.calls >= 2,
                reason=(
                    "goal satisfied after replanning"
                    if self.calls >= 2
                    else "goal not satisfied yet"
                ),
            )

    verifier = EventuallySatisfiedVerifier()
    runtime = AgentRuntime(
        planner=Planner(),
        tool_registry=create_builtin_registry(),
        goal_verifier=verifier,
    )

    loop = AgentLoop(
        runtime=runtime,
        max_iterations=10,
    )

    task = Task(
        description="replan after semantic goal failure",
    )

    result = loop.run(task)

    assert result.completed
    assert result.action == AutonomyAction.COMPLETE
    assert result.iterations >= 3
    assert verifier.calls == 2
    assert result.result.goal_verification is not None
    assert result.result.goal_verification.satisfied
