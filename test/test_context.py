from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from src.agent.context import AgentContext
from src.agent.history import ExecutionHistory, ExecutionRecord
from src.agent.memory import MemoryEntry
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.task import Task
from src.agent.control import ControlDecision


def make_task(description: str = "Test task") -> Task:
    return Task(description=description)


def make_step(
    description: str = "Run test",
    order: int = 1,
) -> ExecutionStep:
    return ExecutionStep(
        description=description,
        order=order,
    )


def make_plan(task: Task) -> ExecutionPlan:
    return ExecutionPlan(
        task_id=task.id,
        steps=(
            make_step(),
        ),
    )


def make_history(task: Task) -> ExecutionHistory:
    started_at = datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    completed_at = datetime(
        2026,
        1,
        1,
        12,
        0,
        1,
        tzinfo=timezone.utc,
    )

    record = ExecutionRecord(
        step_id=uuid4(),
        description="Run test",
        success=True,
        output="done",
        decision=ControlDecision.CONTINUE,
        started_at=started_at,
        completed_at=completed_at,
    )

    return ExecutionHistory(
        task_id=task.id,
        records=(record,),
    )


def make_memory(
    content: str = "Previous execution succeeded",
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
    )


def test_context_stores_task():
    task = make_task()

    context = AgentContext(task=task)

    assert context.task is task


def test_context_defaults_to_no_plan():
    task = make_task()

    context = AgentContext(task=task)

    assert context.plan is None
    assert context.plan_steps == ()


def test_context_accepts_execution_plan():
    task = make_task()
    plan = make_plan(task)

    context = AgentContext(
        task=task,
        plan=plan,
    )

    assert context.plan is plan
    assert context.plan_steps == plan.steps


def test_context_accepts_execution_history():
    task = make_task()
    history = make_history(task)

    context = AgentContext(
        task=task,
        history=history,
    )

    assert context.history is history
    assert context.history_records == history.records
    assert context.last_execution is history.last


def test_context_defaults_to_empty_history_view():
    task = make_task()

    context = AgentContext(task=task)

    assert context.history is None
    assert context.history_records == ()
    assert context.last_execution is None
    assert context.successful_executions == ()
    assert context.failed_executions == ()
    assert context.has_execution_failures is False


def test_context_accepts_memories():
    task = make_task()

    first = make_memory("First memory")
    second = make_memory("Second memory")

    context = AgentContext(
        task=task,
        memories=(first, second),
    )

    assert context.memories == (first, second)
    assert context.memory_count == 2
    assert context.has_memories is True


def test_context_defaults_to_no_memories():
    task = make_task()

    context = AgentContext(task=task)

    assert context.memories == ()
    assert context.memory_count == 0
    assert context.has_memories is False


def test_context_preserves_memory_order():
    task = make_task()

    first = make_memory("First")
    second = make_memory("Second")
    third = make_memory("Third")

    context = AgentContext(
        task=task,
        memories=(first, second, third),
    )

    assert context.memories == (
        first,
        second,
        third,
    )


def test_context_accepts_agent_id():
    task = make_task()

    context = AgentContext(
        task=task,
        agent_id="flop-agent",
    )

    assert context.agent_id == "flop-agent"


def test_context_defaults_to_idle_state():
    task = make_task()

    context = AgentContext(task=task)

    assert context.state == "idle"


def test_context_accepts_custom_state():
    task = make_task()

    context = AgentContext(
        task=task,
        state="planning",
    )

    assert context.state == "planning"


def test_context_exposes_task_id():
    task = make_task()

    context = AgentContext(task=task)

    assert context.task_id == task.id


def test_context_rejects_invalid_task():
    with pytest.raises(TypeError, match="task must be a Task"):
        AgentContext(task="not a task")  # type: ignore[arg-type]


def test_context_rejects_invalid_plan():
    task = make_task()

    with pytest.raises(
        TypeError,
        match="plan must be an ExecutionPlan or None",
    ):
        AgentContext(
            task=task,
            plan="not a plan",  # type: ignore[arg-type]
        )


def test_context_rejects_invalid_history():
    task = make_task()

    with pytest.raises(
        TypeError,
        match="history must be an ExecutionHistory or None",
    ):
        AgentContext(
            task=task,
            history="not history",  # type: ignore[arg-type]
        )


def test_context_rejects_non_tuple_memories():
    task = make_task()

    memory = make_memory()

    with pytest.raises(
        TypeError,
        match="memories must be a tuple",
    ):
        AgentContext(
            task=task,
            memories=[memory],  # type: ignore[arg-type]
        )


def test_context_rejects_invalid_memory_entries():
    task = make_task()

    with pytest.raises(
        TypeError,
        match="memories must contain only MemoryEntry objects",
    ):
        AgentContext(
            task=task,
            memories=("invalid",),  # type: ignore[arg-type]
        )


def test_context_rejects_invalid_agent_id_type():
    task = make_task()

    with pytest.raises(
        TypeError,
        match="agent_id must be a string or None",
    ):
        AgentContext(
            task=task,
            agent_id=123,  # type: ignore[arg-type]
        )


def test_context_rejects_empty_agent_id():
    task = make_task()

    with pytest.raises(
        ValueError,
        match="agent_id must not be empty",
    ):
        AgentContext(
            task=task,
            agent_id="   ",
        )


def test_context_rejects_invalid_state_type():
    task = make_task()

    with pytest.raises(
        TypeError,
        match="state must be a string",
    ):
        AgentContext(
            task=task,
            state=123,  # type: ignore[arg-type]
        )


def test_context_rejects_empty_state():
    task = make_task()

    with pytest.raises(
        ValueError,
        match="state must not be empty",
    ):
        AgentContext(
            task=task,
            state="   ",
        )


def test_context_is_immutable():
    task = make_task()

    context = AgentContext(
        task=task,
        agent_id="agent",
        state="running",
    )

    with pytest.raises(AttributeError):
        context.state = "stopped"  # type: ignore[misc]


def test_context_memories_are_immutable_collection():
    task = make_task()
    memory = make_memory()

    context = AgentContext(
        task=task,
        memories=(memory,),
    )

    assert isinstance(context.memories, tuple)


def test_context_returns_memory_by_id():
    task = make_task()
    first = make_memory("First")
    second = make_memory("Second")

    context = AgentContext(
        task=task,
        memories=(first, second),
    )

    assert context.memory_for_id(first.id) is first
    assert context.memory_for_id(second.id) is second


def test_context_returns_none_for_missing_memory():
    task = make_task()
    memory = make_memory()

    context = AgentContext(
        task=task,
        memories=(memory,),
    )

    assert context.memory_for_id(uuid4()) is None


def test_context_rejects_invalid_memory_id():
    task = make_task()

    context = AgentContext(task=task)

    with pytest.raises(
        TypeError,
        match="memory_id must be a UUID",
    ):
        context.memory_for_id("not-a-uuid")  # type: ignore[arg-type]


def test_context_exposes_successful_executions():
    task = make_task()
    history = make_history(task)

    context = AgentContext(
        task=task,
        history=history,
    )

    assert context.successful_executions == history.successful_records


def test_context_exposes_failed_executions():
    task = make_task()

    started_at = datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    completed_at = datetime(
        2026,
        1,
        1,
        12,
        0,
        1,
        tzinfo=timezone.utc,
    )

    record = ExecutionRecord(
        step_id=uuid4(),
        description="Failed step",
        success=False,
        error="failure",
        started_at=started_at,
        completed_at=completed_at,
    )

    history = ExecutionHistory(
        task_id=task.id,
        records=(record,),
    )

    context = AgentContext(
        task=task,
        history=history,
    )

    assert context.failed_executions == history.failed_records
    assert context.has_execution_failures is True


def test_with_plan_returns_new_context():
    task = make_task()
    original = AgentContext(task=task)
    plan = make_plan(task)

    updated = original.with_plan(plan)

    assert updated is not original
    assert original.plan is None
    assert updated.plan is plan
    assert updated.task is task


def test_with_history_returns_new_context():
    task = make_task()
    original = AgentContext(task=task)
    history = make_history(task)

    updated = original.with_history(history)

    assert updated is not original
    assert original.history is None
    assert updated.history is history


def test_with_memories_returns_new_context():
    task = make_task()
    original = AgentContext(task=task)

    memory = make_memory()

    updated = original.with_memories((memory,))

    assert updated is not original
    assert original.memories == ()
    assert updated.memories == (memory,)


def test_with_state_returns_new_context():
    task = make_task()

    original = AgentContext(
        task=task,
        state="idle",
    )

    updated = original.with_state("planning")

    assert updated is not original
    assert original.state == "idle"
    assert updated.state == "planning"


def test_context_updates_preserve_other_fields():
    task = make_task()
    plan = make_plan(task)
    history = make_history(task)
    memory = make_memory()

    original = AgentContext(
        task=task,
        plan=plan,
        history=history,
        memories=(memory,),
        agent_id="agent-1",
        state="running",
    )

    updated = original.with_state("completed")

    assert updated.task is original.task
    assert updated.plan is original.plan
    assert updated.history is original.history
    assert updated.memories == original.memories
    assert updated.agent_id == original.agent_id
    assert updated.state == "completed"


def test_context_can_be_created_with_all_components():
    task = make_task()
    plan = make_plan(task)
    history = make_history(task)
    memories = (
        make_memory("Memory one"),
        make_memory("Memory two"),
    )

    context = AgentContext(
        task=task,
        plan=plan,
        history=history,
        memories=memories,
        agent_id="flop-agent",
        state="running",
    )

    assert context.task is task
    assert context.plan is plan
    assert context.history is history
    assert context.memories == memories
    assert context.agent_id == "flop-agent"
    assert context.state == "running"


def test_context_preserves_identity_when_updated():
    task = make_task()
    memory = make_memory()

    context = AgentContext(
        task=task,
        memories=(memory,),
        agent_id="agent-1",
    )

    updated = context.with_state("planning")

    assert updated.task_id == context.task_id
    assert updated.agent_id == context.agent_id
    assert updated.memories == context.memories