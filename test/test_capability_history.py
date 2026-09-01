from __future__ import annotations

import pytest

from src.agent.control import ControlDecision
from src.agent.history import ExecutionHistory, ExecutionRecord
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.runtime import AgentRuntime
from src.agent.task import Task
from src.tools.base import Tool, ToolResult
from src.tools.registry import ToolRegistry


class CalculatorTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Performs calculations."

    def execute(self, **kwargs):
        expression = kwargs["expression"]

        if expression == "2 + 3":
            return ToolResult(
                success=True,
                output="5",
            )

        return ToolResult(
            success=False,
            error="unsupported expression",
        )


def test_execution_record_stores_capability() -> None:
    step = ExecutionStep(
        description="Calculate",
        order=1,
    )

    record = ExecutionRecord(
        step_id=step.id,
        description=step.description,
        success=True,
        capability="calculator",
    )

    assert record.capability == "calculator"


def test_execution_record_rejects_invalid_capability_type() -> None:
    step = ExecutionStep(
        description="Calculate",
        order=1,
    )

    with pytest.raises(
        TypeError,
        match="capability must be a string or None",
    ):
        ExecutionRecord(
            step_id=step.id,
            description=step.description,
            success=True,
            capability=123,
        )


def test_execution_record_rejects_empty_capability() -> None:
    step = ExecutionStep(
        description="Calculate",
        order=1,
    )

    with pytest.raises(
        ValueError,
        match="capability must not be empty",
    ):
        ExecutionRecord(
            step_id=step.id,
            description=step.description,
            success=True,
            capability="   ",
        )


def test_history_queries_records_by_capability() -> None:
    first_step = ExecutionStep(
        description="Calculate",
        order=1,
    )

    second_step = ExecutionStep(
        description="Explain",
        order=2,
    )

    first = ExecutionRecord(
        step_id=first_step.id,
        description=first_step.description,
        success=True,
        capability="calculator",
    )

    second = ExecutionRecord(
        step_id=second_step.id,
        description=second_step.description,
        success=True,
    )

    history = ExecutionHistory(
        task_id=Task(
            description="Capability query"
        ).id,
        records=(first, second),
    )

    assert history.records_for_capability(
        "calculator"
    ) == (first,)

    assert history.capabilities_used == frozenset(
        {"calculator"}
    )


def test_history_queries_records_by_tool() -> None:
    step = ExecutionStep(
        description="Calculate",
        order=1,
    )

    record = ExecutionRecord(
        step_id=step.id,
        description=step.description,
        success=True,
        capability="calculator",
        metadata={
            "tool_name": "calculator",
        },
    )

    history = ExecutionHistory(
        task_id=Task(
            description="Tool query"
        ).id,
        records=(record,),
    )

    assert history.records_for_tool(
        "calculator"
    ) == (record,)

    assert history.tool_names_used == frozenset(
        {"calculator"}
    )


def test_history_rejects_invalid_tool_query() -> None:
    history = ExecutionHistory(
        task_id=Task(
            description="Invalid query"
        ).id,
    )

    with pytest.raises(
        TypeError,
        match="tool_name must be a string",
    ):
        history.records_for_tool(123)


def test_history_rejects_empty_tool_query() -> None:
    history = ExecutionHistory(
        task_id=Task(
            description="Invalid query"
        ).id,
    )

    with pytest.raises(
        ValueError,
        match="tool_name must not be empty",
    ):
        history.records_for_tool("   ")


def test_history_rejects_invalid_capability_query() -> None:
    history = ExecutionHistory(
        task_id=Task(
            description="Invalid query"
        ).id,
    )

    with pytest.raises(
        TypeError,
        match="capability must be a string",
    ):
        history.records_for_capability(123)


def test_history_rejects_empty_capability_query() -> None:
    history = ExecutionHistory(
        task_id=Task(
            description="Invalid query"
        ).id,
    )

    with pytest.raises(
        ValueError,
        match="capability must not be empty",
    ):
        history.records_for_capability("   ")


def test_runtime_records_calculator_capability() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    class CalculatorPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Calculate 2 + 3",
                        order=1,
                        tool_name="calculator",
                        tool_args={
                            "expression": "2 + 3",
                        },
                    ),
                ),
            )

    task = Task(
        description="Calculate 2 + 3"
    )

    result = AgentRuntime(
        planner=CalculatorPlanner(),
        tool_registry=registry,
    ).run(
        task,
        allowed_capabilities={"calculator"},
    )

    assert result.succeeded
    assert result.history is not None

    record = result.history.last

    assert record is not None
    assert record.capability == "calculator"
    assert record.metadata["tool_name"] == "calculator"
    assert record.metadata[
        "allowed_capabilities"
    ] == ("calculator",)

    assert result.history.capabilities_used == frozenset(
        {"calculator"}
    )


def test_runtime_records_failed_tool_capability() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    class FailingPlanner:
        def plan(self, task):
            return ExecutionPlan(
                task_id=task.id,
                steps=(
                    ExecutionStep(
                        description="Calculate unsupported",
                        order=1,
                        tool_name="calculator",
                        tool_args={
                            "expression": "9 + 9",
                        },
                    ),
                ),
            )

    task = Task(
        description="Fail calculator"
    )

    result = AgentRuntime(
        planner=FailingPlanner(),
        tool_registry=registry,
    ).run(
        task,
        allowed_capabilities={"calculator"},
    )

    assert result.failed
    assert result.history is not None

    record = result.history.last

    assert record is not None
    assert record.capability == "calculator"
    assert record.success is False
    assert record.error == "unsupported expression"
    assert record.decision == ControlDecision.FAIL


def test_runtime_records_restriction_metadata() -> None:
    task = Task(
        description="Restricted executor"
    )

    result = AgentRuntime().run(
        task,
        allowed_capabilities={"calculator"},
    )

    assert result.succeeded
    assert result.history is not None

    record = result.history.last

    assert record is not None
    assert record.capability is None
    assert record.metadata[
        "allowed_capabilities"
    ] == ("calculator",)