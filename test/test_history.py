from uuid import uuid4

import pytest

from src.agent.control import ControlDecision
from src.agent.history import ExecutionHistory, ExecutionRecord
from src.agent.plan import ExecutionStep


def test_execution_record_stores_step_outcome():
    step_id = uuid4()

    record = ExecutionRecord(
        step_id=step_id,
        description="Run calculator",
        success=True,
        output="42",
        decision=ControlDecision.CONTINUE,
    )

    assert record.step_id == step_id
    assert record.description == "Run calculator"
    assert record.success
    assert record.output == "42"
    assert record.error is None
    assert record.decision == ControlDecision.CONTINUE
    assert not record.failed


def test_failed_execution_record_reports_failed():
    record = ExecutionRecord(
        step_id=uuid4(),
        description="Run failing tool",
        success=False,
        error="tool execution failed",
        decision=ControlDecision.FAIL,
    )

    assert record.failed
    assert record.error == "tool execution failed"
    assert record.decision == ControlDecision.FAIL


def test_empty_history():
    history = ExecutionHistory(task_id=uuid4())

    assert history.is_empty
    assert history.record_count == 0
    assert history.last is None


def test_history_add_returns_new_history():
    task_id = uuid4()
    history = ExecutionHistory(task_id=task_id)

    record = ExecutionRecord(
        step_id=uuid4(),
        description="Step one",
        success=True,
    )

    updated = history.add(record)

    assert history.is_empty
    assert updated.task_id == task_id
    assert updated.record_count == 1
    assert updated.last == record


def test_history_preserves_record_order():
    history = ExecutionHistory(task_id=uuid4())

    first = ExecutionRecord(
        step_id=uuid4(),
        description="Step one",
        success=True,
    )

    second = ExecutionRecord(
        step_id=uuid4(),
        description="Step two",
        success=True,
    )

    history = history.add(first)
    history = history.add(second)

    assert history.records == (first, second)
    assert history.last == second


def test_history_rejects_invalid_record():
    history = ExecutionHistory(task_id=uuid4())

    with pytest.raises(TypeError, match="record must be an ExecutionRecord"):
        history.add(None)


def test_from_step_creates_history_record():
    task_id = uuid4()

    step = ExecutionStep(
        description="Run calculator",
        order=1,
    )

    history = ExecutionHistory.from_step(
        task_id=task_id,
        step=step,
        success=True,
        output="42",
        decision=ControlDecision.CONTINUE,
    )

    assert history.task_id == task_id
    assert history.record_count == 1

    record = history.last
    assert record is not None
    assert record.step_id == step.id
    assert record.description == step.description
    assert record.success
    assert record.output == "42"


def test_from_step_rejects_invalid_step():
    with pytest.raises(TypeError, match="step must be an ExecutionStep"):
        ExecutionHistory.from_step(
            task_id=uuid4(),
            step=None,
            success=True,
        )