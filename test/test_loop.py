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


class RecordingPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, task: Task) -> ExecutionPlan:
        self.calls += 1

        return ExecutionPlan(
            task_id=task.id,
            steps=(
                ExecutionStep(
                    description="Test step",
                    order=1,
                ),
            ),
        )


class ExecutePolicy(AutonomyPolicy):
    def decide(self, context):
        return AutonomyDecision(
            action=AutonomyAction.EXECUTE,
            reason="execute test plan",
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
        policy=ExecutePolicy(),
    ).run(task)

    assert planner.calls >= 1


def test_agent_loop_preserves_task_identity():
    task = Task(description="Identity test")

    result = AgentLoop(
        policy=ExecutePolicy(),
    ).run(task)

    assert result.result.task_id == task.id