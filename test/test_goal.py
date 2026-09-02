from __future__ import annotations

from uuid import uuid4

import pytest

from src.agent.goal import (
    GoalVerificationResult,
    GoalVerifier,
)
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus


def make_task() -> Task:
    return Task(
        description="verify the task goal",
    )


def make_result(
    task: Task,
    *,
    status: TaskStatus = TaskStatus.COMPLETED,
    executed_steps: int = 1,
) -> ExecutionResult:
    return ExecutionResult(
        task_id=task.id,
        status=status,
        executed_steps=executed_steps,
        output="done",
    )


def test_goal_verification_result_accepts_valid_values() -> None:
    result = GoalVerificationResult(
        satisfied=True,
        reason="goal was satisfied",
        evidence={"value": 42},
    )

    assert result.satisfied is True
    assert result.reason == "goal was satisfied"
    assert result.evidence == {"value": 42}


def test_goal_verification_result_is_immutable() -> None:
    result = GoalVerificationResult(
        satisfied=True,
        reason="goal was satisfied",
    )

    with pytest.raises(AttributeError):
        result.satisfied = False  # type: ignore[misc]


def test_goal_verification_result_rejects_non_boolean_satisfied() -> None:
    with pytest.raises(TypeError, match="satisfied must be a boolean"):
        GoalVerificationResult(
            satisfied="yes",  # type: ignore[arg-type]
            reason="goal was satisfied",
        )


def test_goal_verification_result_rejects_non_string_reason() -> None:
    with pytest.raises(TypeError, match="reason must be a string"):
        GoalVerificationResult(
            satisfied=True,
            reason=123,  # type: ignore[arg-type]
        )


def test_goal_verification_result_rejects_empty_reason() -> None:
    with pytest.raises(ValueError, match="reason must not be empty"):
        GoalVerificationResult(
            satisfied=True,
            reason="   ",
        )


def test_goal_verifier_returns_satisfied_for_successful_execution() -> None:
    task = make_task()
    result = make_result(task)

    verification = GoalVerifier().verify(task, result)

    assert verification.satisfied is True
    assert verification.reason == (
        "execution result indicates success"
    )
    assert verification.evidence["task_id"] == str(task.id)
    assert verification.evidence["status"] == "completed"
    assert verification.evidence["executed_steps"] == 1


def test_goal_verifier_returns_unsatisfied_for_failed_execution() -> None:
    task = make_task()
    result = make_result(
        task,
        status=TaskStatus.FAILED,
    )

    verification = GoalVerifier().verify(task, result)

    assert verification.satisfied is False
    assert verification.reason == (
        "execution result indicates failure"
    )


def test_goal_verifier_rejects_invalid_task() -> None:
    task = make_task()
    result = make_result(task)

    with pytest.raises(TypeError, match="task must be a Task"):
        GoalVerifier().verify("invalid", result)  # type: ignore[arg-type]


def test_goal_verifier_rejects_invalid_result() -> None:
    task = make_task()

    with pytest.raises(
        TypeError,
        match="result must be an ExecutionResult",
    ):
        GoalVerifier().verify(task, "invalid")  # type: ignore[arg-type]


def test_goal_verifier_rejects_result_for_different_task() -> None:
    task = make_task()
    different_task = make_task()

    result = make_result(different_task)

    assert result.task_id != task.id

    with pytest.raises(
        ValueError,
        match="result.task_id must match task.id",
    ):
        GoalVerifier().verify(task, result)


def test_goal_verifier_can_be_subclassed() -> None:
    class AlwaysSatisfiedVerifier(GoalVerifier):
        def _verify(
            self,
            task: Task,
            result: ExecutionResult,
        ) -> GoalVerificationResult:
            return GoalVerificationResult(
                satisfied=True,
                reason="custom verifier accepted the goal",
                evidence={"task": str(task.id)},
            )

    task = make_task()
    result = make_result(
        task,
        status=TaskStatus.FAILED,
    )

    verification = AlwaysSatisfiedVerifier().verify(
        task,
        result,
    )

    assert verification.satisfied is True
    assert verification.reason == (
        "custom verifier accepted the goal"
    )


def test_goal_verifier_preserves_task_identity() -> None:
    task = make_task()
    result = make_result(task)

    verification = GoalVerifier().verify(task, result)

    assert verification.evidence["task_id"] == str(task.id)


def test_goal_verifier_accepts_zero_executed_steps() -> None:
    task = make_task()
    result = make_result(
        task,
        executed_steps=0,
    )

    verification = GoalVerifier().verify(task, result)

    assert verification.satisfied is True
    assert verification.evidence["executed_steps"] == 0


def test_goal_verifier_does_not_depend_on_random_external_identity() -> None:
    task = make_task()
    result = make_result(task)

    assert task.id != uuid4()

    verification = GoalVerifier().verify(task, result)

    assert verification.evidence["task_id"] == str(task.id)