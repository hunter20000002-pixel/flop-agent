import pytest

from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4

from src.agent.control import ControlDecision
from src.agent.history import ExecutionHistory, ExecutionRecord
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.runtime import AgentRuntime
from src.agent.task import Task, TaskStatus

from src.inference.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
)

from src.tools.base import Tool, ToolResult
from src.tools.registry import ToolRegistry


class ExampleInferenceProvider(InferenceProvider):
    @property
    def name(self) -> str:
        return "example"

    def generate(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            success=True,
            output=f"Generated: {request.prompt}",
            provider=self.name,
            model="example-model",
        )


class ExampleTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Performs calculations."

    def execute(self, **kwargs):
        expression = kwargs["expression"]

        if expression == "2 + 2":
            return ToolResult(
                success=True,
                output="4",
            )

        return ToolResult(
            success=False,
            error="unsupported expression",
        )


def test_execution_record_stores_step_outcome():
    step_id = uuid4()

    record = ExecutionRecord(
        step_id=step_id,
        description="Run calculator",
        success=True,
        output="42",
    )

    assert record.step_id == step_id
    assert record.description == "Run calculator"
    assert record.success
    assert record.output == "42"
    assert record.error is None


def test_failed_execution_record_reports_failed():
    record = ExecutionRecord(
        step_id=uuid4(),
        description="Run failing step",
        success=False,
        error="execution failed",
    )

    assert record.failed
    assert not record.success
    assert record.error == "execution failed"


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
        description="Test step",
        success=True,
    )

    updated = history.add(record)

    assert updated is not history
    assert history.is_empty
    assert updated.record_count == 1
    assert updated.last == record
    assert updated.task_id == task_id


def test_history_preserves_record_order():
    history = ExecutionHistory(task_id=uuid4())

    first = ExecutionRecord(
        step_id=uuid4(),
        description="First step",
        success=True,
    )

    second = ExecutionRecord(
        step_id=uuid4(),
        description="Second step",
        success=True,
    )

    history = history.add(first)
    history = history.add(second)

    assert history.records == (first, second)
    assert history.last == second


def test_history_rejects_invalid_record():
    history = ExecutionHistory(task_id=uuid4())

    with pytest.raises(TypeError, match="record must be an ExecutionRecord"):
        history.add("not a record")


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
    assert record.description == "Run calculator"
    assert record.success
    assert record.output == "42"
    assert record.decision == ControlDecision.CONTINUE


def test_from_step_rejects_invalid_step():
    with pytest.raises(TypeError, match="step must be an ExecutionStep"):
        ExecutionHistory.from_step(
            task_id=uuid4(),
            step="not a step",
            success=True,
        )


def test_execution_record_rejects_invalid_step_id():
    with pytest.raises(TypeError, match="step_id must be a UUID"):
        ExecutionRecord(
            step_id="not a uuid",
            description="Test",
            success=True,
        )


def test_execution_record_rejects_invalid_description_type():
    with pytest.raises(TypeError, match="description must be a string"):
        ExecutionRecord(
            step_id=uuid4(),
            description=123,
            success=True,
        )


def test_execution_record_rejects_empty_description():
    with pytest.raises(ValueError, match="description must not be empty"):
        ExecutionRecord(
            step_id=uuid4(),
            description="   ",
            success=True,
        )


def test_execution_record_rejects_invalid_success_type():
    with pytest.raises(TypeError, match="success must be a boolean"):
        ExecutionRecord(
            step_id=uuid4(),
            description="Test",
            success="yes",
        )


def test_execution_record_rejects_invalid_decision():
    with pytest.raises(
        TypeError,
        match="decision must be a ControlDecision",
    ):
        ExecutionRecord(
            step_id=uuid4(),
            description="Test",
            success=True,
            decision="continue",
        )


def test_successful_execution_record_cannot_contain_error():
    with pytest.raises(
        ValueError,
        match="successful execution records cannot contain an error",
    ):
        ExecutionRecord(
            step_id=uuid4(),
            description="Test",
            success=True,
            error="unexpected error",
        )


def test_failed_execution_record_requires_error():
    with pytest.raises(
        ValueError,
        match="failed execution records must contain an error",
    ):
        ExecutionRecord(
            step_id=uuid4(),
            description="Test",
            success=False,
        )


def test_failed_execution_record_allows_error():
    record = ExecutionRecord(
        step_id=uuid4(),
        description="Test",
        success=False,
        error="failed",
    )

    assert record.failed
    assert record.error == "failed"


def test_execution_record_is_immutable():
    record = ExecutionRecord(
        step_id=uuid4(),
        description="Test",
        success=True,
    )

    with pytest.raises((AttributeError, TypeError)):
        record.success = False


def test_execution_record_has_execution_timestamps():
    started_at = datetime(
        2026,
        8,
        28,
        10,
        0,
        0,
        tzinfo=timezone.utc,
    )

    completed_at = started_at + timedelta(seconds=2.5)

    record = ExecutionRecord(
        step_id=uuid4(),
        description="Timed step",
        success=True,
        started_at=started_at,
        completed_at=completed_at,
    )

    assert record.started_at == started_at
    assert record.completed_at == completed_at
    assert record.duration_seconds == 2.5


def test_execution_record_defaults_to_current_utc_time():
    before = datetime.now(timezone.utc)

    record = ExecutionRecord(
        step_id=uuid4(),
        description="Timed step",
        success=True,
    )

    after = datetime.now(timezone.utc)

    assert before <= record.started_at <= after
    assert before <= record.completed_at <= after
    assert record.started_at.tzinfo == timezone.utc
    assert record.completed_at.tzinfo == timezone.utc


def test_execution_record_rejects_invalid_started_at():
    with pytest.raises(TypeError, match="started_at must be a datetime"):
        ExecutionRecord(
            step_id=uuid4(),
            description="Test",
            success=True,
            started_at="not datetime",
        )


def test_execution_record_rejects_invalid_completed_at():
    with pytest.raises(
        TypeError,
        match="completed_at must be a datetime",
    ):
        ExecutionRecord(
            step_id=uuid4(),
            description="Test",
            success=True,
            completed_at="not datetime",
        )


def test_execution_record_rejects_negative_duration():
    started_at = datetime(
        2026,
        8,
        28,
        10,
        0,
        2,
        tzinfo=timezone.utc,
    )

    completed_at = started_at - timedelta(seconds=1)

    with pytest.raises(
        ValueError,
        match="completed_at cannot be earlier than started_at",
    ):
        ExecutionRecord(
            step_id=uuid4(),
            description="Invalid timing",
            success=True,
            started_at=started_at,
            completed_at=completed_at,
        )


def test_history_record_preserves_execution_metadata():
    started_at = datetime(
        2026,
        8,
        28,
        10,
        0,
        0,
        tzinfo=timezone.utc,
    )

    completed_at = started_at + timedelta(seconds=1.25)

    step = ExecutionStep(
        description="Metadata step",
        order=1,
    )

    history = ExecutionHistory(
        task_id=uuid4(),
    ).record(
        step,
        success=True,
        output="result",
        decision=ControlDecision.CONTINUE,
        started_at=started_at,
        completed_at=completed_at,
        metadata={
            "executor": "test",
            "attempt": 1,
        },
    )

    record = history.last

    assert record is not None
    assert record.started_at == started_at
    assert record.completed_at == completed_at
    assert record.duration_seconds == 1.25
    assert record.metadata["executor"] == "test"
    assert record.metadata["attempt"] == 1


def test_execution_metadata_is_immutable():
    metadata = {
        "executor": "test",
        "attempt": 1,
    }

    record = ExecutionRecord(
        step_id=uuid4(),
        description="Metadata step",
        success=True,
        metadata=metadata,
    )

    metadata["attempt"] = 99

    assert record.metadata["attempt"] == 1

    with pytest.raises(TypeError):
        record.metadata["attempt"] = 2


def test_history_returns_successful_records():
    history = ExecutionHistory(task_id=uuid4())

    successful = ExecutionRecord(
        step_id=uuid4(),
        description="Successful",
        success=True,
    )

    failed = ExecutionRecord(
        step_id=uuid4(),
        description="Failed",
        success=False,
        error="failed",
    )

    history = history.add(successful)
    history = history.add(failed)

    assert history.successful_records == (successful,)


def test_history_returns_failed_records():
    history = ExecutionHistory(task_id=uuid4())

    successful = ExecutionRecord(
        step_id=uuid4(),
        description="Successful",
        success=True,
    )

    failed = ExecutionRecord(
        step_id=uuid4(),
        description="Failed",
        success=False,
        error="failed",
    )

    history = history.add(successful)
    history = history.add(failed)

    assert history.failed_records == (failed,)


def test_history_reports_whether_failures_exist():
    history = ExecutionHistory(task_id=uuid4())

    assert not history.has_failures

    failed = ExecutionRecord(
        step_id=uuid4(),
        description="Failed",
        success=False,
        error="failed",
    )

    history = history.add(failed)

    assert history.has_failures


def test_history_returns_records_for_step():
    history = ExecutionHistory(task_id=uuid4())

    step_id = uuid4()

    first = ExecutionRecord(
        step_id=step_id,
        description="First attempt",
        success=False,
        error="failed",
    )

    second = ExecutionRecord(
        step_id=uuid4(),
        description="Other step",
        success=True,
    )

    third = ExecutionRecord(
        step_id=step_id,
        description="Second attempt",
        success=True,
    )

    history = history.add(first)
    history = history.add(second)
    history = history.add(third)

    assert history.records_for_step(step_id) == (first, third)


def test_history_rejects_invalid_step_id_for_query():
    history = ExecutionHistory(task_id=uuid4())

    with pytest.raises(TypeError, match="step_id must be a UUID"):
        history.records_for_step("not a uuid")


def test_history_total_duration_seconds():
    started_at = datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    first_completed_at = started_at + timedelta(seconds=2)

    second_started_at = first_completed_at
    second_completed_at = second_started_at + timedelta(seconds=3.5)

    first = ExecutionRecord(
        step_id=uuid4(),
        description="First",
        success=True,
        started_at=started_at,
        completed_at=first_completed_at,
    )

    second = ExecutionRecord(
        step_id=uuid4(),
        description="Second",
        success=True,
        started_at=second_started_at,
        completed_at=second_completed_at,
    )

    history = ExecutionHistory(
        task_id=uuid4(),
        records=(first, second),
    )

    assert history.total_duration_seconds == 5.5


def test_history_query_helpers_preserve_execution_order():
    history = ExecutionHistory(task_id=uuid4())

    first = ExecutionRecord(
        step_id=uuid4(),
        description="First",
        success=True,
    )

    second = ExecutionRecord(
        step_id=uuid4(),
        description="Second",
        success=False,
        error="failed",
    )

    third = ExecutionRecord(
        step_id=uuid4(),
        description="Third",
        success=True,
    )

    history = history.add(first)
    history = history.add(second)
    history = history.add(third)

    assert history.successful_records == (first, third)
    assert history.failed_records == (second,)
    assert history.records == (first, second, third)


def test_execution_record_preserves_metadata():
    step_id = UUID("12345678-1234-5678-1234-567812345678")

    record = ExecutionRecord(
        step_id=step_id,
        description="Metadata test",
        success=True,
        metadata={
            "executor": "runtime",
            "attempt": 1,
            "source": "test",
        },
    )

    assert record.metadata == {
        "executor": "runtime",
        "attempt": 1,
        "source": "test",
    }


def test_execution_record_metadata_is_immutable():
    step_id = UUID("12345678-1234-5678-1234-567812345678")

    metadata = {
        "executor": "runtime",
        "attempt": 1,
    }

    record = ExecutionRecord(
        step_id=step_id,
        description="Metadata test",
        success=True,
        metadata=metadata,
    )

    metadata["attempt"] = 999

    assert record.metadata["attempt"] == 1

    with pytest.raises(TypeError):
        record.metadata["attempt"] = 2


def test_runtime_records_executor_metadata():
    task = Task(description="Metadata test")

    result = AgentRuntime().run(task)

    assert result.history is not None
    record = result.history.last

    assert record is not None
    assert record.metadata["execution_mode"] == "executor"


def test_runtime_records_tool_metadata():
    registry = ToolRegistry()
    registry.register(ExampleTool())

    class ToolPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Calculate two plus two",
                        order=1,
                        tool_name="calculator",
                        tool_args={"expression": "2 + 2"},
                    ),
                ),
            )

    task = Task(description="Calculate two plus two")

    result = AgentRuntime(
        planner=ToolPlanner(),
        tool_registry=registry,
    ).run(task)

    assert result.history is not None
    record = result.history.last

    assert record is not None
    assert record.metadata["execution_mode"] == "tool"
    assert record.metadata["tool_name"] == "calculator"


def test_runtime_records_inference_metadata():
    task = Task(description="Explain something")

    result = AgentRuntime(
        inference_provider=ExampleInferenceProvider(),
    ).run(task)

    assert result.history is not None
    record = result.history.last

    assert record is not None
    assert record.metadata["execution_mode"] == "inference"
    assert record.metadata["provider"] == "example"