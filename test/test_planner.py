from uuid import uuid4

import pytest

from src.agent.context import AgentContext
from src.agent.memory import MemoryEntry
from src.agent.plan import ExecutionPlan
from src.agent.planner import Planner
from src.agent.task import Task


def test_planner_accepts_context():
    task = Task(description="Research decentralized AI")

    context = AgentContext(
        task=task,
        agent_id="test-agent",
    )

    plan = Planner().plan(context)

    assert isinstance(plan, ExecutionPlan)
    assert plan.task_id == task.id
    assert plan.step_count == 1
    assert plan.steps[0].description == "Research decentralized AI"


def test_planner_rejects_context_for_different_task():
    task = Task(description="Research decentralized AI")
    other_task = Task(description="Research blockchain")

    context = AgentContext(
        task=task,
        agent_id="test-agent",
    )

    with pytest.raises(ValueError):
        Planner().plan(
            context,
            task=other_task,
        )


def test_planner_uses_memory_context():
    task = Task(description="Research decentralized AI")

    memory = MemoryEntry(
        content="Previous research found decentralized inference networks.",
        task_id=task.id,
    )

    context = AgentContext(
        task=task,
        memories=(memory,),
        agent_id="test-agent",
    )

    plan = Planner().plan(context)

    assert isinstance(plan, ExecutionPlan)
    assert plan.task_id == task.id
    assert plan.step_count == 1
    assert "Previous research" in plan.steps[0].description

def test_planner_creates_multiple_steps():
    task = Task(
        description="Calculate 2 + 2 and then explain the result"
    )

    context = AgentContext(task=task)

    plan = Planner().plan(context)

    assert plan.step_count == 2

    assert plan.steps[0].description == "Calculate 2 + 2"
    assert plan.steps[0].order == 1
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].tool_args == {
        "expression": "2 + 2"
    }

    assert plan.steps[1].description == "explain the result"
    assert plan.steps[1].order == 2
    assert plan.steps[1].tool_name is None
    assert plan.steps[1].tool_args == {}


def test_planner_preserves_single_step_tasks():
    task = Task(
        description="Research decentralized AI"
    )

    context = AgentContext(task=task)

    plan = Planner().plan(context)

    assert plan.step_count == 1
    assert plan.steps[0].description == "Research decentralized AI"


def test_planner_supports_then_separator():
    task = Task(
        description="Calculate 10 * 5 then explain the result"
    )

    context = AgentContext(task=task)

    plan = Planner().plan(context)

    assert plan.step_count == 2
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].tool_args == {
        "expression": "10 * 5"
    }
    assert plan.steps[1].tool_name is None


def test_planner_assigns_sequential_step_orders():
    task = Task(
        description=(
            "Calculate 2 + 2 "
            "and then explain the result "
            "and then summarize the answer"
        )
    )

    context = AgentContext(task=task)

    plan = Planner().plan(context)

    assert plan.step_count == 3
    assert [step.order for step in plan.steps] == [1, 2, 3]

def test_planner_creates_multiple_steps_for_compound_task():
    task = Task(
        description=(
            "calculate 10 + 20 and list C:\\temp"
        )
    )

    context = AgentContext(task=task)

    plan = Planner().plan(context)

    assert isinstance(plan, ExecutionPlan)
    assert plan.task_id == task.id
    assert plan.step_count == 2

    assert plan.steps[0].order == 1
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].tool_args == {
        "expression": "10 + 20",
    }

    assert plan.steps[1].order == 2
    assert plan.steps[1].tool_name == "filesystem"
    assert plan.steps[1].tool_args == {
        "operation": "list",
        "path": "C:\\temp",
    }


def test_planner_preserves_single_step_behavior():
    task = Task(
        description="calculate 10 + 20"
    )

    context = AgentContext(task=task)

    plan = Planner().plan(context)

    assert plan.step_count == 1
    assert plan.steps[0].order == 1
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].tool_args == {
        "expression": "10 + 20",
    }


def test_planner_creates_multiple_steps_for_two_calculations():
    task = Task(
        description=(
            "calculate 10 + 20 and calculate 5 * 5"
        )
    )

    context = AgentContext(task=task)

    plan = Planner().plan(context)

    assert plan.step_count == 2

    assert plan.steps[0].order == 1
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].tool_args == {
        "expression": "10 + 20",
    }

    assert plan.steps[1].order == 2
    assert plan.steps[1].tool_name == "calculator"
    assert plan.steps[1].tool_args == {
        "expression": "5 * 5",
    }