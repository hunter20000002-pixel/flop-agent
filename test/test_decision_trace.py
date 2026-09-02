
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.agent.decision import AutonomyAction
from src.agent.decision_trace import AutonomyDecisionEvent


def test_decision_event_stores_required_fields() -> None:
    step_id = uuid4()
    evidence = {
        "failure_count": 1,
        "progress_made": False,
    }
    timestamp = datetime.now(timezone.utc)

    event = AutonomyDecisionEvent(
        sequence=3,
        action=AutonomyAction.REPLAN,
        reason="execution made no progress",
        evidence=evidence,
        step_id=step_id,
        trigger="failure",
        timestamp=timestamp,
    )

    assert event.sequence == 3
    assert event.action == AutonomyAction.REPLAN
    assert event.reason == "execution made no progress"
    assert event.evidence == evidence
    assert event.step_id == step_id
    assert event.trigger == "failure"
    assert event.timestamp == timestamp


def test_decision_event_defaults_evidence() -> None:
    event = AutonomyDecisionEvent(
        sequence=0,
        action=AutonomyAction.EXECUTE,
        reason="execute available plan",
    )

    assert event.evidence == {}


def test_decision_event_defaults_step_id() -> None:
    event = AutonomyDecisionEvent(
        sequence=0,
        action=AutonomyAction.EXECUTE,
        reason="execute available plan",
    )

    assert event.step_id is None


def test_decision_event_defaults_trigger() -> None:
    event = AutonomyDecisionEvent(
        sequence=0,
        action=AutonomyAction.EXECUTE,
        reason="execute available plan",
    )

    assert event.trigger == "policy"


def test_decision_event_generates_id() -> None:
    first = AutonomyDecisionEvent(
        sequence=0,
        action=AutonomyAction.EXECUTE,
        reason="first",
    )

    second = AutonomyDecisionEvent(
        sequence=1,
        action=AutonomyAction.EXECUTE,
        reason="second",
    )

    assert first.id != second.id


def test_decision_event_is_immutable() -> None:
    event = AutonomyDecisionEvent(
        sequence=0,
        action=AutonomyAction.EXECUTE,
        reason="immutable",
    )

    with pytest.raises(AttributeError):
        event.reason = "changed"


def test_decision_event_rejects_negative_sequence() -> None:
    with pytest.raises(
        ValueError,
        match="sequence must not be negative",
    ):
        AutonomyDecisionEvent(
            sequence=-1,
            action=AutonomyAction.EXECUTE,
            reason="invalid",
        )


def test_decision_event_rejects_invalid_action() -> None:
    with pytest.raises(
        TypeError,
        match="action must be an AutonomyAction",
    ):
        AutonomyDecisionEvent(
            sequence=0,
            action="execute",
            reason="invalid",
        )


def test_decision_event_rejects_empty_reason() -> None:
    with pytest.raises(
        ValueError,
        match="reason must not be empty",
    ):
        AutonomyDecisionEvent(
            sequence=0,
            action=AutonomyAction.EXECUTE,
            reason="   ",
        )


def test_decision_event_rejects_invalid_evidence() -> None:
    with pytest.raises(
        TypeError,
        match="evidence must be a mapping",
    ):
        AutonomyDecisionEvent(
            sequence=0,
            action=AutonomyAction.EXECUTE,
            reason="invalid evidence",
            evidence=[],
        )


def test_decision_event_copies_evidence() -> None:
    evidence = {
        "failure_count": 1,
    }

    event = AutonomyDecisionEvent(
        sequence=0,
        action=AutonomyAction.RETRY,
        reason="retry failed step",
        evidence=evidence,
    )

    evidence["failure_count"] = 99

    assert event.evidence["failure_count"] == 1


def test_decision_event_rejects_invalid_step_id() -> None:
    with pytest.raises(
        TypeError,
        match="step_id must be a UUID or None",
    ):
        AutonomyDecisionEvent(
            sequence=0,
            action=AutonomyAction.EXECUTE,
            reason="invalid step",
            step_id="step",
        )


def test_decision_event_rejects_empty_trigger() -> None:
    with pytest.raises(
        ValueError,
        match="trigger must not be empty",
    ):
        AutonomyDecisionEvent(
            sequence=0,
            action=AutonomyAction.EXECUTE,
            reason="invalid trigger",
            trigger="   ",
        )