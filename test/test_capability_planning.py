from __future__ import annotations

from uuid import uuid4

import pytest

from src.agent.context import AgentContext
from src.agent.plan import ExecutionPlan
from src.agent.planner import Planner
from src.agent.task import Task, TaskStatus
from src.config import DEFAULT_CONFIG


def make_task(description: str) -> Task:
    return Task(
        id=uuid4(),
        description=description,
        status=TaskStatus.PENDING,
    )


def make_context(
    description: str,
    *,
    allowed_capabilities: frozenset[str] | None = None,
) -> AgentContext:
    return AgentContext(
        task=make_task(description),
        memories=(),
        allowed_capabilities=allowed_capabilities,
    )


def test_context_preserves_capabilities_through_immutable_updates() -> None:
    """Context updates must retain the authorized capability set."""

    context = make_context(
        "Calculate 10 + 5",
        allowed_capabilities=frozenset({"calculator"}),
    )

    plan = ExecutionPlan(
        task_id=context.task.id,
        steps=(),
    )

    updated = (
        context
        .with_plan(plan)
        .with_history(None)
        .with_memories(())
        .with_state("running")
    )

    assert updated is not context
    assert updated.allowed_capabilities == frozenset({"calculator"})


def test_planner_accepts_authorized_calculator() -> None:
    """An authorized calculator capability should permit calculator planning."""

    context = make_context(
        "Calculate 10 + 5",
        allowed_capabilities=frozenset({"calculator"}),
    )

    plan = Planner().plan(context)

    assert plan.step_count == 1
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].tool_args == {
        "expression": "10 + 5",
    }


def test_planner_rejects_unauthorized_calculator() -> None:
    """Calculator planning must fail when calculator is not authorized."""

    context = make_context(
        "Calculate 10 + 5",
        allowed_capabilities=frozenset(),
    )

    with pytest.raises(
        PermissionError,
        match=(
            r"tool 'calculator' requires capability "
            r"'calculator', which is not authorized"
        ),
    ):
        Planner().plan(context)


def test_planner_rejects_unauthorized_technocore_observer() -> None:
    """Technocore observation must fail without observer authorization."""

    context = make_context(
        "Observe recent Technocore activity",
        allowed_capabilities=frozenset(),
    )

    with pytest.raises(
        PermissionError,
        match=(
            r"tool 'technocore_observer' requires capability "
            r"'technocore_observer', which is not authorized"
        ),
    ):
        Planner().plan(context)


def test_planner_remains_unrestricted_when_capabilities_are_none() -> None:
    """None must preserve the existing unrestricted planner behavior."""

    context = make_context(
        "Observe recent Technocore activity",
        allowed_capabilities=None,
    )

    plan = Planner().plan(context)

    assert plan.step_count == 1

    step = plan.steps[0]

    assert step.tool_name == "technocore_observer"
    assert step.tool_args == {
        "room": DEFAULT_CONFIG.room,
        "since": 0,
    }


def test_context_with_allowed_capabilities_returns_new_context() -> None:
    """Capability updates must preserve immutability."""

    context = make_context(
        "Calculate 2 + 2",
        allowed_capabilities=None,
    )

    updated = context.with_allowed_capabilities(
        frozenset({"calculator"})
    )

    assert updated is not context
    assert context.allowed_capabilities is None
    assert updated.allowed_capabilities == frozenset(
        {"calculator"}
    )