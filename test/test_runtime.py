import pytest

from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.result import ExecutionResult
from src.agent.runtime import AgentRuntime
from src.agent.task import Task, TaskStatus
from src.inference.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
)


class RecordingPlanner:
    def __init__(self, steps):
        self.steps = steps

    def plan(self, task):
        return ExecutionPlan(
            task_id=task.id,
            steps=tuple(self.steps),
        )


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


class FailingInferenceProvider(InferenceProvider):
    @property
    def name(self) -> str:
        return "failing"

    def generate(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            success=False,
            error="Inference provider failed.",
            provider=self.name,
        )


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

    class LifecyclePlanner:
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
        planner=LifecyclePlanner(),
        step_executor=executor,
    ).run(task)

    assert observed_statuses[0] == TaskStatus.PLANNING
    assert TaskStatus.RUNNING in observed_statuses
    assert result.status == TaskStatus.COMPLETED


def test_runtime_can_use_inference_provider():
    task = Task(description="Explain autonomous agents")

    result = AgentRuntime(
        inference_provider=ExampleInferenceProvider(),
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.succeeded
    assert result.executed_steps == 1
    assert result.output == "Generated: Explain autonomous agents"
    assert result.error is None


def test_runtime_passes_step_description_to_inference_provider():
    requests = []

    class RecordingProvider(InferenceProvider):
        @property
        def name(self) -> str:
            return "recording"

        def generate(self, request: InferenceRequest) -> InferenceResult:
            requests.append(request)

            return InferenceResult(
                success=True,
                output="done",
                provider=self.name,
            )

    task = Task(description="Analyze this task")

    AgentRuntime(
        inference_provider=RecordingProvider(),
    ).run(task)

    assert len(requests) == 1
    assert requests[0].prompt == "Analyze this task"


def test_runtime_collects_multiple_inference_outputs():
    steps = [
        ExecutionStep(
            description="First step",
            order=1,
        ),
        ExecutionStep(
            description="Second step",
            order=2,
        ),
    ]

    task = Task(description="Multi-step task")

    result = AgentRuntime(
        planner=RecordingPlanner(steps),
        inference_provider=ExampleInferenceProvider(),
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.executed_steps == 2
    assert result.output == (
        "Generated: First step\n"
        "Generated: Second step"
    )


def test_runtime_fails_when_inference_provider_fails():
    task = Task(description="Inference failure")

    result = AgentRuntime(
        inference_provider=FailingInferenceProvider(),
    ).run(task)

    assert result.status == TaskStatus.FAILED
    assert result.failed
    assert result.executed_steps == 0
    assert result.error == "Inference provider failed."


def test_runtime_uses_step_executor_without_inference_provider():
    executed = []

    def executor(step):
        executed.append(step.description)

    task = Task(description="Use executor")

    result = AgentRuntime(
        step_executor=executor,
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert executed == ["Use executor"]
    assert result.output is None