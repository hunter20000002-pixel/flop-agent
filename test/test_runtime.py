import pytest

from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.runtime import AgentRuntime
from src.agent.task import Task, TaskStatus


def test_runtime_completes_task():
    task = Task(description="Complete a test task")

    result = AgentRuntime().run(task)

    assert result is task
    assert task.status == TaskStatus.COMPLETED


def test_runtime_runs_all_steps():
    executed_steps = []

    def executor(step):
        executed_steps.append(step)

    task = Task(description="Execute steps")

    AgentRuntime(step_executor=executor).run(task)

    assert len(executed_steps) == 1
    assert executed_steps[0].description == "Execute steps"


def test_runtime_preserves_task_identity():
    task = Task(description="Identity test")

    result = AgentRuntime().run(task)

    assert result.id == task.id


def test_runtime_rejects_invalid_task():
    with pytest.raises(TypeError):
        AgentRuntime().run("not a task")


def test_runtime_marks_task_failed_when_execution_fails():
    def failing_executor(step):
        raise RuntimeError("execution failed")

    task = Task(description="Failing execution")

    with pytest.raises(RuntimeError, match="execution failed"):
        AgentRuntime(step_executor=failing_executor).run(task)

    assert task.status == TaskStatus.FAILED


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

    AgentRuntime(
        planner=RecordingPlanner(),
        step_executor=executor,
    ).run(task)

    assert observed_statuses[0] == TaskStatus.PLANNING
    assert TaskStatus.RUNNING in observed_statuses
    assert task.status == TaskStatus.COMPLETED