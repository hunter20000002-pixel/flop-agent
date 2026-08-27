from uuid import UUID, uuid4

import pytest

from src.agent.plan import ExecutionPlan, ExecutionStep


def test_execution_step_has_valid_defaults():
    step = ExecutionStep(
        description="Research the requested topic",
        order=1,
    )

    assert isinstance(step.id, UUID)
    assert step.description == "Research the requested topic"
    assert step.order == 1


def test_execution_step_rejects_empty_description():
    with pytest.raises(ValueError):
        ExecutionStep(description="   ", order=1)


def test_execution_step_rejects_invalid_order():
    with pytest.raises(ValueError):
        ExecutionStep(description="Do something", order=0)


def test_execution_plan_can_contain_steps():
    task_id = uuid4()

    step_one = ExecutionStep(
        description="Collect information",
        order=1,
    )

    step_two = ExecutionStep(
        description="Analyze information",
        order=2,
    )

    plan = ExecutionPlan(
        task_id=task_id,
        steps=(step_one, step_two),
    )

    assert plan.task_id == task_id
    assert plan.steps == (step_one, step_two)
    assert plan.step_count == 2
    assert not plan.is_empty


def test_execution_plan_can_be_empty():
    plan = ExecutionPlan(task_id=uuid4())

    assert plan.is_empty
    assert plan.step_count == 0


def test_execution_plan_rejects_duplicate_orders():
    step_one = ExecutionStep(
        description="First step",
        order=1,
    )

    step_two = ExecutionStep(
        description="Another first step",
        order=1,
    )

    with pytest.raises(ValueError):
        ExecutionPlan(
            task_id=uuid4(),
            steps=(step_one, step_two),
        )


def test_execution_plan_rejects_unsorted_steps():
    step_one = ExecutionStep(
        description="First step",
        order=1,
    )

    step_two = ExecutionStep(
        description="Second step",
        order=2,
    )

    with pytest.raises(ValueError):
        ExecutionPlan(
            task_id=uuid4(),
            steps=(step_two, step_one),
        )


def test_execution_plan_is_immutable():
    step = ExecutionStep(
        description="Immutable step",
        order=1,
    )

    plan = ExecutionPlan(
        task_id=uuid4(),
        steps=(step,),
    )

    with pytest.raises(AttributeError):
        plan.task_id = uuid4()

def test_execution_step_can_reference_tool():
    step = ExecutionStep(
        description="Calculate a value",
        order=1,
        tool_name="calculator",
        tool_args={"expression": "2 + 2"},
    )

    assert step.tool_name == "calculator"
    assert step.tool_args == {"expression": "2 + 2"}
    assert step.uses_tool


def test_execution_step_without_tool_does_not_use_tool():
    step = ExecutionStep(
        description="Think about the task",
        order=1,
    )

    assert step.tool_name is None
    assert step.tool_args == {}
    assert not step.uses_tool


def test_execution_step_rejects_empty_tool_name():
    with pytest.raises(ValueError, match="tool name cannot be empty"):
        ExecutionStep(
            description="Use a tool",
            order=1,
            tool_name="   ",
        )


def test_execution_step_copies_tool_arguments():
    arguments = {"value": 42}

    step = ExecutionStep(
        description="Process value",
        order=1,
        tool_name="processor",
        tool_args=arguments,
    )

    arguments["value"] = 99

    assert step.tool_args == {"value": 42}