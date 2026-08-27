from uuid import uuid4
from src.agent.history import ExecutionHistory, ExecutionRecord
from src.agent.result import ExecutionResult
from src.agent.task import TaskStatus


def test_successful_execution_result():
    task_id = uuid4()

    result = ExecutionResult(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        executed_steps=3,
        output="Task completed successfully.",
    )

    assert result.task_id == task_id
    assert result.status == TaskStatus.COMPLETED
    assert result.executed_steps == 3
    assert result.output == "Task completed successfully."
    assert result.error is None
    assert result.succeeded
    assert not result.failed


def test_failed_execution_result():
    task_id = uuid4()

    result = ExecutionResult(
        task_id=task_id,
        status=TaskStatus.FAILED,
        executed_steps=2,
        error="Tool execution failed.",
    )

    assert result.status == TaskStatus.FAILED
    assert result.executed_steps == 2
    assert result.error == "Tool execution failed."
    assert result.failed
    assert not result.succeeded


def test_execution_result_is_immutable():
    result = ExecutionResult(
        task_id=uuid4(),
        status=TaskStatus.COMPLETED,
        executed_steps=1,
    )

    try:
        result.executed_steps = 2
    except AttributeError:
        pass
    else:
        raise AssertionError("ExecutionResult should be immutable")

def test_execution_result_can_store_history():
    task_id = uuid4()

    history = ExecutionHistory(
        task_id=task_id,
        records=(
            ExecutionRecord(
                step_id=uuid4(),
                description="Step one",
                success=True,
                output="done",
            ),
        ),
    )

    result = ExecutionResult(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        executed_steps=1,
        history=history,
    )

    assert result.history == history
    assert result.history is not None
    assert result.history.record_count == 1


def test_execution_result_history_defaults_to_none():
    result = ExecutionResult(
        task_id=uuid4(),
        status=TaskStatus.COMPLETED,
    )

    assert result.history is None