import pytest

from src.agent.control import (
    ControlDecision,
    ExecutionController,
    StepOutcome,
)


def test_successful_outcome_continues():
    controller = ExecutionController()

    outcome = StepOutcome(
        success=True,
        output="step completed",
    )

    assert controller.decide(outcome) == ControlDecision.CONTINUE


def test_failed_outcome_fails():
    controller = ExecutionController()

    outcome = StepOutcome(
        success=False,
        error="tool execution failed",
    )

    assert controller.decide(outcome) == ControlDecision.FAIL


def test_step_outcome_failed_property():
    assert StepOutcome(success=True).failed is False
    assert StepOutcome(success=False).failed is True


def test_controller_rejects_invalid_outcome():
    controller = ExecutionController()

    with pytest.raises(TypeError, match="outcome must be a StepOutcome"):
        controller.decide(None)


def test_control_decision_values():
    assert ControlDecision.CONTINUE.value == "continue"
    assert ControlDecision.STOP.value == "stop"
    assert ControlDecision.FAIL.value == "fail"