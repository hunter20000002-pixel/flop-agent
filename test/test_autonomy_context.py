from __future__ import annotations

from uuid import uuid4

import pytest

from src.agent.autonomy_context import AutonomyDecisionContext
from src.agent.history import ExecutionHistory
from src.agent.memory import MemoryEntry
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus


def make_context(
    *,
    allowed_capabilities: frozenset[str] | None = None,
    failure_count: int = 0,
    retry_count: int = 0,
    replan_count: int = 0,
    remaining_step_budget: int | None = None,
    memories: tuple[MemoryEntry, ...] = (),
) -> AutonomyDecisionContext:
    task = Task(
        description="Autonomy decision context"
    )

    history = ExecutionHistory(
        task_id=task.id
    )

    step = ExecutionStep(
        order=1,
        description="Test step",
    )

    plan = ExecutionPlan(
        task_id=task.id,
        steps=(step,),
    )

    return AutonomyDecisionContext(
        task=task,
        current_plan=plan,
        current_step=step,
        execution_history=history,
        last_result=None,
        failure_count=failure_count,
        retry_count=retry_count,
        replan_count=replan_count,
        allowed_capabilities=allowed_capabilities,
        remaining_step_budget=remaining_step_budget,
        memories=memories,
    )


def test_context_stores_decision_evidence() -> None:
    context = make_context(
        allowed_capabilities=frozenset(
            {"calculator"}
        ),
        failure_count=2,
        retry_count=1,
        replan_count=3,
        remaining_step_budget=7,
    )

    assert context.task_id == context.task.id
    assert context.has_plan
    assert context.has_current_step
    assert not context.has_last_result

    assert context.failure_count == 2
    assert context.retry_count == 1
    assert context.replan_count == 3
    assert context.remaining_step_budget == 7

    assert context.allowed_capabilities == frozenset(
        {"calculator"}
    )


def test_context_stores_memories() -> None:
    memory = MemoryEntry(
        content="Observed Technocore activity.",
        task_id=uuid4(),
    )

    context = make_context(
        memories=(memory,),
    )

    assert context.memories == (memory,)
    assert context.memory_count == 1
    assert context.has_memories


def test_context_without_memories_has_no_memory_evidence() -> None:
    context = make_context()

    assert context.memories == ()
    assert context.memory_count == 0
    assert not context.has_memories


def test_context_rejects_non_tuple_memories() -> None:
    with pytest.raises(
        TypeError,
        match="memories must be a tuple",
    ):
        make_context(
            memories=["invalid"],
        )


def test_context_rejects_invalid_memory_entry() -> None:
    with pytest.raises(
        TypeError,
        match="memories must contain only MemoryEntry objects",
    ):
        make_context(
            memories=("invalid",),
        )


def test_context_exposes_plan_steps() -> None:
    context = make_context()

    assert len(context.plan_steps) == 1
    assert context.plan_steps[0] == context.current_step


def test_context_without_plan_has_no_steps() -> None:
    task = Task(
        description="No plan"
    )

    history = ExecutionHistory(
        task_id=task.id
    )

    context = AutonomyDecisionContext(
        task=task,
        current_plan=None,
        current_step=None,
        execution_history=history,
        last_result=None,
    )

    assert context.current_plan is None
    assert context.current_step is None
    assert context.plan_steps == ()
    assert not context.has_plan
    assert not context.has_current_step


def test_context_is_immutable() -> None:
    context = make_context()

    with pytest.raises(AttributeError):
        context.failure_count = 10


def test_capabilities_are_normalized_to_frozenset() -> None:
    context = make_context(
        allowed_capabilities={"calculator"}
    )

    assert isinstance(
        context.allowed_capabilities,
        frozenset,
    )

    assert context.allowed_capabilities == frozenset(
        {"calculator"}
    )


def test_context_rejects_invalid_task() -> None:
    task = Task(
        description="Invalid task test"
    )

    history = ExecutionHistory(
        task_id=task.id
    )

    with pytest.raises(
        TypeError,
        match="task must be a Task",
    ):
        AutonomyDecisionContext(
            task="invalid",
            current_plan=None,
            current_step=None,
            execution_history=history,
            last_result=None,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "failure_count",
        "retry_count",
        "replan_count",
    ],
)
def test_context_rejects_negative_counters(
    field_name: str,
) -> None:
    task = Task(
        description="Negative counter"
    )

    history = ExecutionHistory(
        task_id=task.id
    )

    kwargs = {
        field_name: -1,
    }

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be a non-negative integer",
    ):
        AutonomyDecisionContext(
            task=task,
            current_plan=None,
            current_step=None,
            execution_history=history,
            last_result=None,
            **kwargs,
        )


def test_context_rejects_negative_step_budget() -> None:
    task = Task(
        description="Negative budget"
    )

    history = ExecutionHistory(
        task_id=task.id
    )

    with pytest.raises(
        ValueError,
        match=(
            "remaining_step_budget must be "
            "a non-negative integer"
        ),
    ):
        AutonomyDecisionContext(
            task=task,
            current_plan=None,
            current_step=None,
            execution_history=history,
            last_result=None,
            remaining_step_budget=-1,
        )


def test_context_rejects_boolean_counter() -> None:
    task = Task(
        description="Boolean counter"
    )

    history = ExecutionHistory(
        task_id=task.id
    )

    with pytest.raises(
        TypeError,
        match="failure_count must be a non-negative integer",
    ):
        AutonomyDecisionContext(
            task=task,
            current_plan=None,
            current_step=None,
            execution_history=history,
            last_result=None,
            failure_count=True,
        )


def test_context_rejects_invalid_capability() -> None:
    task = Task(
        description="Invalid capability"
    )

    history = ExecutionHistory(
        task_id=task.id
    )

    with pytest.raises(
        TypeError,
        match="allowed_capabilities must contain strings",
    ):
        AutonomyDecisionContext(
            task=task,
            current_plan=None,
            current_step=None,
            execution_history=history,
            last_result=None,
            allowed_capabilities=frozenset(
                {"calculator", 123}
            ),
        )


def test_context_rejects_empty_capability() -> None:
    task = Task(
        description="Empty capability"
    )

    history = ExecutionHistory(
        task_id=task.id
    )

    with pytest.raises(
        ValueError,
        match="allowed_capabilities must not contain empty strings",
    ):
        AutonomyDecisionContext(
            task=task,
            current_plan=None,
            current_step=None,
            execution_history=history,
            last_result=None,
            allowed_capabilities=frozenset(
                {"calculator", ""}
            ),
        )


def test_with_counters_returns_new_context() -> None:
    context = make_context(
        failure_count=1,
        retry_count=2,
        replan_count=3,
    )

    updated = context.with_counters(
        failure_count=4,
        retry_count=5,
        replan_count=6,
    )

    assert updated is not context

    assert context.failure_count == 1
    assert context.retry_count == 2
    assert context.replan_count == 3

    assert updated.failure_count == 4
    assert updated.retry_count == 5
    assert updated.replan_count == 6

    assert updated.task is context.task
    assert updated.execution_history is context.execution_history
    assert (
        updated.allowed_capabilities
        == context.allowed_capabilities
    )
    assert updated.memories is context.memories


def test_with_plan_returns_new_context() -> None:
    context = make_context()

    new_plan = ExecutionPlan(
        task_id=context.task.id,
        steps=(
            ExecutionStep(
                order=1,
                description="Replanned step",
            ),
        ),
    )

    updated = context.with_plan(new_plan)

    assert updated is not context
    assert updated.current_plan is new_plan
    assert updated.current_step is context.current_step
    assert updated.memories is context.memories


def test_with_step_returns_new_context() -> None:
    context = make_context()

    new_step = ExecutionStep(
        order=2,
        description="New current step",
    )

    updated = context.with_step(new_step)

    assert updated is not context
    assert updated.current_step is new_step
    assert updated.current_plan is context.current_plan
    assert updated.memories is context.memories


def test_with_result_returns_new_context() -> None:
    context = make_context()

    result = ExecutionResult(
        task_id=context.task.id,
        status="completed",
        executed_steps=1,
        history=context.execution_history,
    )

    updated = context.with_result(result)

    assert updated is not context
    assert updated.last_result is result
    assert context.last_result is None
    assert updated.has_last_result
    assert updated.memories is context.memories


def test_with_remaining_step_budget_returns_new_context() -> None:
    context = make_context(
        remaining_step_budget=10
    )

    updated = context.with_remaining_step_budget(4)

    assert updated is not context
    assert context.remaining_step_budget == 10
    assert updated.remaining_step_budget == 4
    assert updated.memories is context.memories


def test_with_memories_returns_new_context() -> None:
    context = make_context()

    memory = MemoryEntry(
        content="New autonomy memory.",
        task_id=uuid4(),
    )

    updated = context.with_memories(
        (memory,),
    )

    assert updated is not context
    assert context.memories == ()
    assert updated.memories == (memory,)
    assert updated.memory_count == 1


def test_context_evidence_properties() -> None:
    context = make_context(
        failure_count=1,
        retry_count=1,
        replan_count=1,
        remaining_step_budget=0,
    )

    assert context.has_failures
    assert context.has_retries
    assert context.has_replans
    assert context.budget_exhausted


def test_positive_step_budget_is_not_exhausted() -> None:
    context = make_context(
        remaining_step_budget=3
    )

    assert not context.budget_exhausted


def test_goal_verification_failed_property() -> None:
    from src.agent.goal import GoalVerificationResult

    task = Task(
        description="goal verification context",
    )

    result = ExecutionResult(
        task_id=task.id,
        status=TaskStatus.COMPLETED,
        goal_verification=GoalVerificationResult(
            satisfied=False,
            reason="goal not satisfied",
        ),
    )

    context = AutonomyDecisionContext(
        task=task,
        current_plan=None,
        current_step=None,
        execution_history=ExecutionHistory(
            task_id=task.id,
        ),
        last_result=result,
    )

    assert context.goal_verification_failed is True
    assert context.goal_verification_succeeded is False


def test_goal_verification_succeeded_property() -> None:
    from src.agent.goal import GoalVerificationResult

    task = Task(
        description="successful goal verification context",
    )

    result = ExecutionResult(
        task_id=task.id,
        status=TaskStatus.COMPLETED,
        goal_verification=GoalVerificationResult(
            satisfied=True,
            reason="goal satisfied",
        ),
    )

    context = AutonomyDecisionContext(
        task=task,
        current_plan=None,
        current_step=None,
        execution_history=ExecutionHistory(
            task_id=task.id,
        ),
        last_result=result,
    )

    assert context.goal_verification_failed is False
    assert context.goal_verification_succeeded is True
