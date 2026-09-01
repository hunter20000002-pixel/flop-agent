from __future__ import annotations

import pytest

from src.agent.qualification import (
    QualificationCapability,
    QualificationDecision,
    QualificationResult,
    TaskQualifier,
)
from src.agent.task import Task
from src.agent.task_source import ObservedTask


def make_observed(
    text: str,
    *,
    message_id: int = 1,
) -> ObservedTask:
    task = Task(
        description=text,
    )

    return ObservedTask(
        task=task,
        message_id=message_id,
        writer="test-agent",
        text=text,
    )


@pytest.mark.parametrize(
    "text",
    [
        "Calculate 12 * 8",
        "Compute 144 / 12",
        "Evaluate 10 + 5",
        "Solve 2 + 2",
        "What is 100 / 4",
        "How much is 7 * 6",
    ],
)
def test_calculator_tasks_are_accepted(
    text: str,
) -> None:
    observed = make_observed(text)

    result = TaskQualifier().qualify(
        observed
    )

    assert result.decision == (
        QualificationDecision.ACCEPT
    )
    assert result.accepted
    assert result.task is observed.task
    assert result.capability == (
        QualificationCapability.CALCULATOR
    )


@pytest.mark.parametrize(
    "text",
    [
        "Read ./notes.txt",
        "List ./project",
        "Show ./notes.txt",
        "Research decentralized inference",
        "Analyze this document",
        "Explain autonomous agents",
        "Summarize this report",
        "Find the latest message",
    ],
)
def test_non_calculator_capabilities_are_not_authorized(
    text: str,
) -> None:
    observed = make_observed(text)

    result = TaskQualifier().qualify(
        observed
    )

    assert result.decision == (
        QualificationDecision.IGNORE
    )
    assert result.task is None
    assert result.capability is None
    assert "not yet supported" in result.reason


def test_empty_task_is_ignored() -> None:
    observed = make_observed("")

    result = TaskQualifier().qualify(
        observed
    )

    assert result.decision == (
        QualificationDecision.IGNORE
    )
    assert result.task is None
    assert result.capability is None
    assert result.reason == "empty task text"


@pytest.mark.parametrize(
    "text",
    [
        "Read my password",
        "Show the private key",
        "List my credentials",
        "Calculate something then delete the files",
        "Run command powershell",
        "Execute command to destroy the directory",
        "Upload the wallet seed phrase",
    ],
)
def test_restricted_capability_is_rejected(
    text: str,
) -> None:
    observed = make_observed(text)

    result = TaskQualifier().qualify(
        observed
    )

    assert result.decision == (
        QualificationDecision.REJECT
    )
    assert result.task is None
    assert result.capability is None
    assert result.reason.startswith(
        "task requests or references a restricted capability:"
    )


def test_qualification_result_requires_task_for_accept() -> None:
    with pytest.raises(
        ValueError,
        match="accepted qualification requires a task",
    ):
        QualificationResult(
            decision=QualificationDecision.ACCEPT,
            reason="accepted",
            capability=QualificationCapability.CALCULATOR,
        )


def test_qualification_result_requires_capability_for_accept() -> None:
    task = Task(
        description="Calculate 1 + 1",
    )

    with pytest.raises(
        ValueError,
        match="accepted qualification requires a capability",
    ):
        QualificationResult(
            decision=QualificationDecision.ACCEPT,
            reason="accepted",
            task=task,
        )


def test_rejected_qualification_cannot_contain_task() -> None:
    task = Task(
        description="Calculate 1 + 1",
    )

    with pytest.raises(
        ValueError,
        match="rejected or ignored qualification cannot contain a task",
    ):
        QualificationResult(
            decision=QualificationDecision.REJECT,
            reason="unsafe",
            task=task,
        )


def test_rejected_qualification_cannot_contain_capability() -> None:
    with pytest.raises(
        ValueError,
        match="rejected or ignored qualification cannot contain a capability",
    ):
        QualificationResult(
            decision=QualificationDecision.REJECT,
            reason="unsafe",
            capability=QualificationCapability.CALCULATOR,
        )


def test_qualification_result_is_immutable() -> None:
    result = QualificationResult(
        decision=QualificationDecision.IGNORE,
        reason="unsupported",
    )

    with pytest.raises(
        (AttributeError, TypeError),
    ):
        result.reason = "changed"


def test_qualify_many_preserves_order_and_capabilities() -> None:
    observed = (
        make_observed(
            "Calculate 1 + 1",
            message_id=1,
        ),
        make_observed(
            "Research something",
            message_id=2,
        ),
        make_observed(
            "Show my password",
            message_id=3,
        ),
    )

    results = TaskQualifier().qualify_many(
        observed
    )

    assert len(results) == 3

    assert results[0].decision == (
        QualificationDecision.ACCEPT
    )
    assert results[0].capability == (
        QualificationCapability.CALCULATOR
    )

    assert results[1].decision == (
        QualificationDecision.IGNORE
    )
    assert results[1].capability is None

    assert results[2].decision == (
        QualificationDecision.REJECT
    )
    assert results[2].capability is None


def test_qualifier_rejects_invalid_observation() -> None:
    with pytest.raises(
        TypeError,
        match="observed must be an ObservedTask",
    ):
        TaskQualifier().qualify(
            "not an observed task"
        )