from __future__ import annotations

from uuid import uuid4

import pytest

from src.agent.context import AgentContext
from src.agent.planner import Planner
from src.agent.task import Task, TaskStatus
from src.config import DEFAULT_CONFIG


def make_context(description: str) -> AgentContext:
    """Create a minimal context for planner tests."""

    task = Task(
        id=uuid4(),
        description=description,
        status=TaskStatus.PENDING,
    )

    return AgentContext(
        task=task,
        memories=(),
    )


def test_planner_selects_technocore_observer() -> None:
    """Technocore observation tasks should use the observer tool."""

    context = make_context(
        "Observe recent Technocore activity"
    )

    plan = Planner().plan(context)

    assert plan.step_count == 1

    step = plan.steps[0]

    assert step.tool_name == "technocore_observer"
    assert step.tool_args == {
        "room": DEFAULT_CONFIG.room,
        "since": 0,
    }


def test_planner_recognizes_technocore_analysis() -> None:
    """Analysis requests involving Technocore should use the observer."""

    context = make_context(
        "Analyze recent Technocore messages"
    )

    plan = Planner().plan(context)

    assert plan.step_count == 1
    assert plan.steps[0].tool_name == "technocore_observer"


def test_planner_does_not_select_technocore_for_unrelated_tasks() -> None:
    """Mentioning unrelated content must not trigger the observer."""

    context = make_context(
        "Calculate the value of a Technocore token"
    )

    plan = Planner().plan(context)

    assert plan.step_count == 1
    assert plan.steps[0].tool_name == "calculator"


def test_planner_splits_technocore_observation_from_calculation() -> None:
    """Compound tasks should create separate execution steps."""

    context = make_context(
        "Observe recent Technocore activity and then "
        "calculate 10 + 5"
    )

    plan = Planner().plan(context)

    assert plan.step_count == 2

    first_step = plan.steps[0]
    second_step = plan.steps[1]

    assert first_step.order == 1
    assert first_step.tool_name == "technocore_observer"
    assert first_step.tool_args == {
        "room": DEFAULT_CONFIG.room,
        "since": 0,
    }

    assert second_step.order == 2
    assert second_step.tool_name == "calculator"
    assert second_step.tool_args == {
        "expression": "10 + 5",
    }


def test_planner_creates_observe_then_analyze_pipeline() -> None:
    """
    An explicit observation followed by analysis should create a
    tool step followed by an inference step.
    """

    context = make_context(
        "Observe recent Technocore activity and then "
        "analyze the messages for important agent activity"
    )

    plan = Planner().plan(context)

    assert plan.step_count == 2

    observation_step = plan.steps[0]
    analysis_step = plan.steps[1]

    assert observation_step.order == 1
    assert observation_step.tool_name == "technocore_observer"
    assert observation_step.tool_args == {
        "room": DEFAULT_CONFIG.room,
        "since": 0,
    }

    assert analysis_step.order == 2
    assert analysis_step.tool_name is None
    assert analysis_step.tool_args == {}
    assert "analyze the messages" in (
        analysis_step.description.lower()
    )


def test_planner_creates_observe_then_summarize_pipeline() -> None:
    """Observation followed by summarization should use inference."""

    context = make_context(
        "Inspect recent Technocore messages and then "
        "summarize the important activity"
    )

    plan = Planner().plan(context)

    assert plan.step_count == 2

    observation_step = plan.steps[0]
    summary_step = plan.steps[1]

    assert observation_step.tool_name == "technocore_observer"
    assert summary_step.tool_name is None
    assert "summarize the important activity" in (
        summary_step.description.lower()
    )


@pytest.mark.parametrize(
    "description",
    (
        "Inspect Technocore activity",
        "Monitor Technocore messages",
        "Check Technocore for updates",
        "Read Technocore messages",
        "Review Technocore activity",
        "Scan Technocore for relevant messages",
    ),
)
def test_planner_recognizes_technocore_observation_verbs(
    description: str,
) -> None:
    """Supported observation verbs should select the observer."""

    context = make_context(description)

    plan = Planner().plan(context)

    assert plan.step_count == 1
    assert plan.steps[0].tool_name == "technocore_observer"