import pytest

from src.agent.control import ControlDecision, ExecutionController
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

from src.tools.base import Tool, ToolResult
from src.tools.registry import ToolRegistry


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


class FailingTool(Tool):
    @property
    def name(self) -> str:
        return "failing"

    @property
    def description(self) -> str:
        return "Always fails."

    def execute(self, **kwargs):
        return ToolResult(
            success=False,
            error="tool execution failed",
        )


def test_runtime_executes_tool_from_registry():
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

    assert result.status == TaskStatus.COMPLETED
    assert result.succeeded
    assert result.executed_steps == 1
    assert result.output == "4"
    assert result.error is None


def test_runtime_fails_when_required_tool_registry_is_missing():
    class ToolPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Use calculator",
                        order=1,
                        tool_name="calculator",
                    ),
                ),
            )

    task = Task(description="Use calculator")

    result = AgentRuntime(
        planner=ToolPlanner(),
    ).run(task)

    assert result.status == TaskStatus.FAILED
    assert result.failed
    assert result.error == (
        "step requires tool 'calculator', "
        "but no tool registry is configured"
    )


def test_runtime_fails_when_tool_is_not_registered():
    registry = ToolRegistry()

    class ToolPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Use calculator",
                        order=1,
                        tool_name="calculator",
                    ),
                ),
            )

    task = Task(description="Use calculator")

    result = AgentRuntime(
        planner=ToolPlanner(),
        tool_registry=registry,
    ).run(task)

    assert result.status == TaskStatus.FAILED
    assert result.failed
    assert result.error == "'tool not found: calculator'"


def test_runtime_fails_when_tool_execution_fails():
    registry = ToolRegistry()
    registry.register(FailingTool())

    class ToolPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Run failing tool",
                        order=1,
                        tool_name="failing",
                    ),
                ),
            )

    task = Task(description="Run failing tool")

    result = AgentRuntime(
        planner=ToolPlanner(),
        tool_registry=registry,
    ).run(task)

    assert result.status == TaskStatus.FAILED
    assert result.failed
    assert result.error == "tool execution failed"
    assert result.executed_steps == 0


def test_runtime_can_execute_tools_and_inference_in_same_plan():
    registry = ToolRegistry()
    registry.register(ExampleTool())

    class MixedPlanner:
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
                    ExecutionStep(
                        description="Explain the result",
                        order=2,
                    ),
                ),
            )

    task = Task(description="Calculate and explain")

    result = AgentRuntime(
        planner=MixedPlanner(),
        tool_registry=registry,
        inference_provider=ExampleInferenceProvider(),
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.succeeded
    assert result.executed_steps == 2
    assert result.output == (
        "4\n"
        "Generated: Explain the result"
    )

def test_runtime_rejects_invalid_max_steps():
    with pytest.raises(ValueError, match="max_steps must be greater than zero"):
        AgentRuntime(max_steps=0)


def test_runtime_respects_max_steps():
    class MultiStepPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Step one",
                        order=1,
                    ),
                    ExecutionStep(
                        description="Step two",
                        order=2,
                    ),
                ),
            )

    task = Task(description="Run multiple steps")

    result = AgentRuntime(
        planner=MultiStepPlanner(),
        max_steps=1,
    ).run(task)

    assert result.status == TaskStatus.FAILED
    assert result.failed
    assert result.executed_steps == 1
    assert result.error == "execution step limit exceeded: 1"


def test_runtime_allows_execution_within_max_steps():
    class MultiStepPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Step one",
                        order=1,
                    ),
                    ExecutionStep(
                        description="Step two",
                        order=2,
                    ),
                ),
            )

    task = Task(description="Run multiple steps")

    result = AgentRuntime(
        planner=MultiStepPlanner(),
        max_steps=2,
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.succeeded
    assert result.executed_steps == 2

def test_runtime_uses_execution_controller():
    decisions = []

    class RecordingController(ExecutionController):
        def decide(self, outcome):
            decision = super().decide(outcome)
            decisions.append(decision)
            return decision

    class SingleStepPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Step one",
                        order=1,
                    ),
                ),
            )

    task = Task(description="Controller test")

    result = AgentRuntime(
        planner=SingleStepPlanner(),
        controller=RecordingController(),
    ).run(task)

    assert result.succeeded
    assert decisions == [ControlDecision.CONTINUE]

def test_runtime_stops_when_controller_returns_stop():
    class StopController(ExecutionController):
        def decide(self, outcome):
            return ControlDecision.STOP

    class MultiStepPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Step one",
                        order=1,
                    ),
                    ExecutionStep(
                        description="Step two",
                        order=2,
                    ),
                ),
            )

    task = Task(description="Stop test")

    result = AgentRuntime(
        planner=MultiStepPlanner(),
        controller=StopController(),
    ).run(task)

    assert result.succeeded
    assert result.executed_steps == 1

def test_runtime_records_successful_step_in_history():
    task = Task(description="Run a successful step")

    result = AgentRuntime().run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.history is not None
    assert result.history.record_count == 1

    record = result.history.last
    assert record is not None
    assert record.description == "Run a successful step"
    assert record.success
    assert record.error is None
    assert record.decision == ControlDecision.CONTINUE


def test_runtime_records_tool_execution_in_history():
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

    assert result.status == TaskStatus.COMPLETED
    assert result.history is not None
    assert result.history.record_count == 1

    record = result.history.last
    assert record is not None
    assert record.description == "Calculate two plus two"
    assert record.success
    assert record.output == "4"
    assert record.error is None
    assert record.decision == ControlDecision.CONTINUE


def test_runtime_records_inference_execution_in_history():
    class InferencePlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Explain the result",
                        order=1,
                    ),
                ),
            )

    task = Task(description="Explain something")

    result = AgentRuntime(
        planner=InferencePlanner(),
        inference_provider=ExampleInferenceProvider(),
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.history is not None
    assert result.history.record_count == 1

    record = result.history.last
    assert record is not None
    assert record.description == "Explain the result"
    assert record.success
    assert record.output == "Generated: Explain the result"
    assert record.error is None
    assert record.decision == ControlDecision.CONTINUE


def test_runtime_records_failed_step_in_history():
    def failing_executor(step):
        raise RuntimeError("execution failed")

    task = Task(description="Failing execution")

    result = AgentRuntime(
        step_executor=failing_executor,
    ).run(task)

    assert result.status == TaskStatus.FAILED
    assert result.failed
    assert result.history is not None
    assert result.history.record_count == 1

    record = result.history.last
    assert record is not None
    assert record.description == "Failing execution"
    assert not record.success
    assert record.failed
    assert record.output is None
    assert record.error == "execution failed"
    assert record.decision == ControlDecision.FAIL


def test_runtime_history_preserves_execution_order():
    class MultiStepPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="First step",
                        order=1,
                    ),
                    ExecutionStep(
                        description="Second step",
                        order=2,
                    ),
                    ExecutionStep(
                        description="Third step",
                        order=3,
                    ),
                ),
            )

    task = Task(description="Run three steps")

    result = AgentRuntime(
        planner=MultiStepPlanner(),
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.history is not None
    assert result.history.record_count == 3

    assert [
        record.description
        for record in result.history.records
    ] == [
        "First step",
        "Second step",
        "Third step",
    ]

    assert all(record.success for record in result.history.records)
    assert all(
        record.decision == ControlDecision.CONTINUE
        for record in result.history.records
    )

def test_runtime_executes_multi_step_plan_from_planner():
    class MultiStepPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="First step",
                        order=1,
                    ),
                    ExecutionStep(
                        description="Second step",
                        order=2,
                    ),
                ),
            )

    task = Task(description="Execute a multi-step task")

    result = AgentRuntime(
        planner=MultiStepPlanner(),
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.succeeded
    assert result.executed_steps == 2

    assert result.history is not None
    assert result.history.record_count == 2

    assert [
        record.description
        for record in result.history.records
    ] == [
        "First step",
        "Second step",
    ]

def test_runtime_executes_multi_step_tool_and_inference_plan():
    registry = ToolRegistry()
    registry.register(ExampleTool())

    class MultiStepPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Calculate 2 + 2",
                        order=1,
                        tool_name="calculator",
                        tool_args={"expression": "2 + 2"},
                    ),
                    ExecutionStep(
                        description="Explain the result",
                        order=2,
                    ),
                ),
            )

    task = Task(
        description="Calculate 2 + 2 and explain the result"
    )

    result = AgentRuntime(
        planner=MultiStepPlanner(),
        tool_registry=registry,
        inference_provider=ExampleInferenceProvider(),
    ).run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.succeeded
    assert result.executed_steps == 2

    assert result.output == (
        "4\n"
        "Generated: Explain the result"
    )

    assert result.history is not None
    assert result.history.record_count == 2

    assert result.history.records[0].output == "4"
    assert result.history.records[1].output == (
        "Generated: Explain the result"
    )