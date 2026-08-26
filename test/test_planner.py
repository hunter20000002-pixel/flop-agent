from uuid import uuid4

import pytest

from src.agent.plan import ExecutionPlan
from src.agent.planner import Planner
from src.agent.task import Task


def test_planner_creates_execution_plan():
    task = Task(description="Research decentralized AI")

    planner = Planner()
    plan = planner.plan(task)

    assert isinstance(plan, ExecutionPlan)
    assert plan.task_id == task.id
    assert plan.step_count == 1
    assert plan.steps[0].description == "Research decentralized AI"
    assert plan.steps[0].order == 1


def test_planner_strips_task_description():
    task = Task(description="  Research decentralized AI  ")

    plan = Planner().plan(task)

    assert plan.steps[0].description == "Research decentralized AI"


def test_planner_rejects_invalid_task():
    planner = Planner()

    with pytest.raises(TypeError):
        planner.plan("Research decentralized AI")


def test_planner_rejects_empty_description():
    task = Task(description="   ")

    with pytest.raises(ValueError):
        Planner().plan(task)


def test_planner_preserves_task_identity():
    task_id = uuid4()
    task = Task(
        id=task_id,
        description="Test identity preservation",
    )

    plan = Planner().plan(task)

    assert plan.task_id == task_id