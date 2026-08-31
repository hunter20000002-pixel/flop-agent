from __future__ import annotations

from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.runtime import AgentRuntime
from src.agent.task import Task, TaskStatus
from src.inference.base import InferenceRequest, InferenceResult


class ContextRecordingProvider:
    name = "context-recorder"

    def __init__(self) -> None:
        self.requests: list[InferenceRequest] = []

    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        self.requests.append(request)

        return InferenceResult(
            success=True,
            output=f"output-{len(self.requests)}",
            provider=self.name,
        )


class ContextPlanner:
    def plan(
        self,
        task: Task,
    ) -> ExecutionPlan:
        return ExecutionPlan(
            task_id=task.id,
            steps=(
                ExecutionStep(
                    description="First inference step",
                    order=1,
                ),
                ExecutionStep(
                    description="Second inference step",
                    order=2,
                ),
            ),
        )


def test_runtime_passes_context_to_inference_provider() -> None:
    task = Task(
        description="Context-aware execution"
    )

    provider = ContextRecordingProvider()

    result = AgentRuntime(
        planner=ContextPlanner(),
        inference_provider=provider,
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert len(provider.requests) == 2

    first_context = provider.requests[0].context

    assert first_context is not None
    assert first_context["task"] is task
    assert first_context["task_id"] == task.id
    assert first_context["plan"] is not None
    assert first_context["history"] is not None
    assert first_context["memories"] == ()
    assert first_context["agent_id"] is None
    assert first_context["state"] == "running"


def test_runtime_updates_context_history_between_steps() -> None:
    task = Task(
        description="Track execution history"
    )

    provider = ContextRecordingProvider()

    result = AgentRuntime(
        planner=ContextPlanner(),
        inference_provider=provider,
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert len(provider.requests) == 2

    first_history = provider.requests[0].context["history"]
    second_history = provider.requests[1].context["history"]

    assert first_history is not None
    assert second_history is not None

    assert len(first_history.records) == 0
    assert len(second_history.records) == 1

    assert second_history.records[0].output == "output-1"


def test_runtime_context_contains_previous_execution_result() -> None:
    task = Task(
        description="Use previous execution context"
    )

    provider = ContextRecordingProvider()

    result = AgentRuntime(
        planner=ContextPlanner(),
        inference_provider=provider,
    ).run(task)

    assert result.status == TaskStatus.COMPLETED

    second_request = provider.requests[1]
    history = second_request.context["history"]

    assert history is not None
    assert history.last is not None
    assert history.last.success is True
    assert history.last.output == "output-1"


def test_runtime_returns_updated_execution_history() -> None:
    task = Task(
        description="Return execution history"
    )

    provider = ContextRecordingProvider()

    result = AgentRuntime(
        planner=ContextPlanner(),
        inference_provider=provider,
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.history is not None
    assert len(result.history.records) == 2

    assert result.history.records[0].output == "output-1"
    assert result.history.records[1].output == "output-2"