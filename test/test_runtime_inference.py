import pytest

from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.result import ExecutionResult
from src.agent.runtime import AgentRuntime
from src.agent.task import Task, TaskStatus
from src.inference.base import InferenceProvider, InferenceRequest
from src.inference.mock import MockInferenceProvider


class RecordingInferenceProvider(InferenceProvider):
    def __init__(
        self,
        response: str = "inference response",
    ) -> None:
        self.response = response
        self.requests: list[InferenceRequest] = []

    @property
    def name(self) -> str:
        return "recording"

    def generate(
        self,
        request: InferenceRequest,
    ):
        self.requests.append(request)

        return self._result()

    def _result(self):
        from src.inference.base import InferenceResult

        return InferenceResult(
            success=True,
            output=self.response,
            provider=self.name,
            model="recording-model",
        )


def test_runtime_executes_unassigned_step_through_inference_provider():
    task = Task(
        description="Explain autonomous agents",
    )

    step = ExecutionStep(
        description="Explain autonomous agents",
        order=1,
    )

    plan = ExecutionPlan(
        task_id=task.id,
        steps=(step,),
    )

    provider = RecordingInferenceProvider(
        response="Autonomous agents plan and execute tasks.",
    )

    runtime = AgentRuntime(
        inference_provider=provider,
    )

    result = runtime.run(
        task,
        plan=plan,
    )

    assert result.succeeded
    assert result.status == TaskStatus.COMPLETED
    assert result.executed_steps == 1
    assert result.output == (
        "Autonomous agents plan and execute tasks."
    )

    assert len(provider.requests) == 1
    assert provider.requests[0].prompt == (
        "Explain autonomous agents"
    )


def test_runtime_records_inference_metadata():
    task = Task(
        description="Explain planning",
    )

    step = ExecutionStep(
        description="Explain planning",
        order=1,
    )

    plan = ExecutionPlan(
        task_id=task.id,
        steps=(step,),
    )

    provider = MockInferenceProvider(
        response="Planning response",
    )

    runtime = AgentRuntime(
        inference_provider=provider,
    )

    result = runtime.run(
        task,
        plan=plan,
    )

    assert result.succeeded
    assert result.history is not None
    assert result.history.record_count == 1

    record = result.history.last

    assert record is not None
    assert record.success
    assert record.output == "Planning response"
    assert record.metadata["execution_mode"] == "inference"
    assert record.metadata["provider"] == "mock"


def test_runtime_prefers_tool_over_inference():
    task = Task(
        description="Calculate 2 + 2",
    )

    step = ExecutionStep(
        description="Calculate 2 + 2",
        order=1,
        tool_name="calculator",
        tool_args={
            "expression": "2 + 2",
        },
    )

    plan = ExecutionPlan(
        task_id=task.id,
        steps=(step,),
    )

    provider = RecordingInferenceProvider()

    runtime = AgentRuntime(
        inference_provider=provider,
    )

    result = runtime.run(
        task,
        plan=plan,
    )

    assert result.failed
    assert result.error is not None
    assert "no tool registry" in result.error.lower()

    assert provider.requests == []


def test_runtime_without_inference_provider_uses_step_executor():
    task = Task(
        description="Perform custom action",
    )

    step = ExecutionStep(
        description="Perform custom action",
        order=1,
    )

    plan = ExecutionPlan(
        task_id=task.id,
        steps=(step,),
    )

    executed: list[str] = []

    def executor(execution_step: ExecutionStep) -> None:
        executed.append(execution_step.description)

    runtime = AgentRuntime(
        step_executor=executor,
    )

    result = runtime.run(
        task,
        plan=plan,
    )

    assert result.succeeded
    assert result.executed_steps == 1
    assert executed == ["Perform custom action"]


def test_runtime_rejects_plan_for_different_task():
    task = Task(
        description="Original task",
    )

    other_task = Task(
        description="Different task",
    )

    step = ExecutionStep(
        description="Different task",
        order=1,
    )

    plan = ExecutionPlan(
        task_id=other_task.id,
        steps=(step,),
    )

    runtime = AgentRuntime(
        inference_provider=MockInferenceProvider(),
    )

    with pytest.raises(ValueError):
        runtime.run(
            task,
            plan=plan,
        )