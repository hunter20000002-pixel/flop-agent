from __future__ import annotations

from uuid import uuid4

import pytest

from src.agent.context import AgentContext
from src.agent.memory import MemoryEntry
from src.agent.plan import ExecutionPlan
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


def test_planner_uses_highest_priority_memory() -> None:
    """
    Planner should preserve memory priority order when injecting
    multiple memories into a planning step.
    """

    context = make_context(
        "Explain the memory-aware planning architecture"
    )

    highest_priority = MemoryEntry(
        content=(
            "The planner should use the most relevant historical "
            "memory when forming an execution step."
        ),
        task_id=uuid4(),
    )

    lower_priority = MemoryEntry(
        content=(
            "This memory is less relevant and should not be selected "
            "before a higher-priority memory."
        ),
        task_id=uuid4(),
    )

    context = context.with_memories(
        (
            highest_priority,
            lower_priority,
        )
    )

    plan = Planner().plan(context)

    assert plan.step_count == 1

    step = plan.steps[0]

    assert (
        "The planner should use the most relevant historical memory"
        in step.description
    )

    assert (
        "This memory is less relevant"
        in step.description
    )

    assert step.description.index(
        "The planner should use the most relevant historical memory"
    ) < step.description.index(
        "This memory is less relevant"
    )


def test_planner_includes_multiple_prioritized_memories() -> None:
    """Planner should include multiple memories in priority order."""

    context = make_context(
        "Explain the memory-aware planning architecture"
    )

    first = MemoryEntry(
        content="First high-priority planning memory.",
        task_id=uuid4(),
    )

    second = MemoryEntry(
        content="Second high-priority planning memory.",
        task_id=uuid4(),
    )

    third = MemoryEntry(
        content="Third high-priority planning memory.",
        task_id=uuid4(),
    )

    context = context.with_memories(
        (
            first,
            second,
            third,
        )
    )

    plan = Planner().plan(context)

    description = plan.steps[0].description

    assert "First high-priority planning memory." in description
    assert "Second high-priority planning memory." in description
    assert "Third high-priority planning memory." in description

    assert description.index(
        "First high-priority planning memory."
    ) < description.index(
        "Second high-priority planning memory."
    )

    assert description.index(
        "Second high-priority planning memory."
    ) < description.index(
        "Third high-priority planning memory."
    )


def test_planner_limits_planning_memories_to_three() -> None:
    """Planner should bound memory injected into a planning step."""

    context = make_context(
        "Explain the memory-aware planning architecture"
    )

    memories = tuple(
        MemoryEntry(
            content=f"Planning memory {index}.",
            task_id=uuid4(),
        )
        for index in range(1, 6)
    )

    context = context.with_memories(memories)

    plan = Planner().plan(context)

    description = plan.steps[0].description

    assert "Planning memory 1." in description
    assert "Planning memory 2." in description
    assert "Planning memory 3." in description

    assert "Planning memory 4." not in description
    assert "Planning memory 5." not in description


def test_planner_deduplicates_memory_content() -> None:
    """Duplicate memory content should only be injected once."""

    context = make_context(
        "Explain the memory-aware planning architecture"
    )

    duplicate_one = MemoryEntry(
        content="The same planning fact.",
        task_id=uuid4(),
    )

    duplicate_two = MemoryEntry(
        content="The same planning fact.",
        task_id=uuid4(),
    )

    distinct = MemoryEntry(
        content="A different planning fact.",
        task_id=uuid4(),
    )

    context = context.with_memories(
        (
            duplicate_one,
            duplicate_two,
            distinct,
        )
    )

    plan = Planner().plan(context)

    description = plan.steps[0].description

    assert description.count(
        "The same planning fact."
    ) == 1

    assert "A different planning fact." in description


def test_planner_deduplication_preserves_priority_order() -> None:
    """
    Deduplication should preserve the first occurrence of each memory
    and therefore preserve retrieval priority.
    """

    context = make_context(
        "Explain the memory-aware planning architecture"
    )

    first = MemoryEntry(
        content="Priority A.",
        task_id=uuid4(),
    )

    duplicate = MemoryEntry(
        content="Priority A.",
        task_id=uuid4(),
    )

    second = MemoryEntry(
        content="Priority B.",
        task_id=uuid4(),
    )

    third = MemoryEntry(
        content="Priority C.",
        task_id=uuid4(),
    )

    context = context.with_memories(
        (
            first,
            duplicate,
            second,
            third,
        )
    )

    plan = Planner().plan(context)

    description = plan.steps[0].description

    assert description.count("Priority A.") == 1

    assert description.index(
        "Priority A."
    ) < description.index(
        "Priority B."
    )

    assert description.index(
        "Priority B."
    ) < description.index(
        "Priority C."
    )


def test_planner_memory_budget_counts_unique_non_empty_memories() -> None:
    """
    Duplicate memories should not consume the three-memory planning
    budget.
    """

    context = make_context(
        "Explain the memory-aware planning architecture"
    )

    memories = (
        MemoryEntry(
            content="Memory A.",
            task_id=uuid4(),
        ),
        MemoryEntry(
            content="Memory A.",
            task_id=uuid4(),
        ),
        MemoryEntry(
            content="Memory B.",
            task_id=uuid4(),
        ),
        MemoryEntry(
            content="Memory C.",
            task_id=uuid4(),
        ),
        MemoryEntry(
            content="Memory D.",
            task_id=uuid4(),
        ),
    )

    context = context.with_memories(memories)

    plan = Planner().plan(context)

    description = plan.steps[0].description

    assert "Memory A." in description
    assert "Memory B." in description
    assert "Memory C." in description
    assert "Memory D." not in description


def test_planner_memory_enrichment_does_not_change_tool_selection() -> None:
    """Memory text must not affect conservative tool selection."""

    context = make_context(
        "Calculate 10 + 5"
    )

    memory = MemoryEntry(
        content=(
            "This memory mentions filesystem, directory, file, "
            "read, list, and folder."
        ),
        task_id=uuid4(),
    )

    context = context.with_memories(
        (memory,)
    )

    plan = Planner().plan(context)

    assert plan.step_count == 1
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].tool_args == {
        "expression": "10 + 5",
    }


def test_planner_memory_enrichment_applies_to_each_step() -> None:
    """Each execution step should receive the same prioritized memory set."""

    context = make_context(
        "Observe recent Technocore activity and then "
        "calculate 10 + 5"
    )

    first = MemoryEntry(
        content="The agent should preserve observation context.",
        task_id=uuid4(),
    )

    second = MemoryEntry(
        content="The agent should preserve calculation context.",
        task_id=uuid4(),
    )

    context = context.with_memories(
        (
            first,
            second,
        )
    )

    plan = Planner().plan(context)

    assert plan.step_count == 2

    for step in plan.steps:
        assert (
            "The agent should preserve observation context."
            in step.description
        )

        assert (
            "The agent should preserve calculation context."
            in step.description
        )
