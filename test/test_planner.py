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