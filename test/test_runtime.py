import pytest

from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.result import ExecutionResult
from src.agent.runtime import AgentRuntime
from src.agent.task import Task, TaskStatus


def test_runtime_completes_task():
    task = Task(description="Complete a test task")

    result = AgentRuntime().run(task)

    assert isinstance(result, ExecutionResult)
    assert result.task_id == task.id
    assert result.status == TaskStatus.COMPLETED
    assert result.succeeded
    assert not result.failed


def test_runtime_runs_all_steps():
    executed_steps = []

    def executor(step):
        executed_steps.append(step)

    task = Task(description="Execute steps")

    result = AgentRuntime(step_executor=executor).run(task)

    assert result.executed_steps == 1
    assert len(executed_steps) == 1
    assert executed_steps[0].description == "Execute steps"


def test_runtime_preserves_task_identity():
    task = Task(description="Identity test")

    result = AgentRuntime().run(task)

    assert result.task_id == task.id


def test_runtime_rejects_invalid_task():
    with pytest.raises(TypeError):
        AgentRuntime().run("not a task")


def test_runtime_returns_failed_result_when_execution_fails():
    def failing_executor(step):
        raise RuntimeError("execution failed")

    task = Task(description="Failing execution")

    result = AgentRuntime(step_executor=failing_executor).run(task)

    assert result.status == TaskStatus.FAILED
    assert result.failed
    assert not result.succeeded
    assert result.error == "execution failed"
    assert result.executed_steps == 0


def test_runtime_moves_task_through_lifecycle():
    observed_statuses = []

    class RecordingPlanner:
        def plan(self, task):
            observed_statuses.append(task.status)

            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Test step",
                        order=1,
                    ),
                ),
            )

    def executor(step):
        observed_statuses.append(TaskStatus.RUNNING)

    task = Task(description="Lifecycle test")

    result = AgentRuntime(
        planner=RecordingPlanner(),
        step_executor=executor,
    ).run(task)

    assert observed_statuses[0] == TaskStatus.PLANNING
    assert TaskStatus.RUNNING in observed_statuses
    assert result.status == TaskStatus.COMPLETED