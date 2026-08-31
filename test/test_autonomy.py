from __future__ import annotations

from src.agent.autonomy import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.context import AgentContext
from src.agent.control import ControlDecision
from src.agent.history import ExecutionHistory
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.task import Task


def make_context(
    *,
    state: str = "running",
    with_plan: bool = True,
    with_history: bool = False,
    decision: ControlDecision | None = None,
) -> AgentContext:
    task = Task(
        description="Test autonomy"
    )

    plan = None

    if with_plan:
        plan = ExecutionPlan(
            task_id=task.id,
            steps=(
                ExecutionStep(
                    description="Test step",
                    order=1,
                ),
            ),
        )

    history = None

    if with_history:
        history = ExecutionHistory(
            task_id=task.id,
        )

        if decision is not None:
            history = history.record(
                plan.steps[0],
                success=decision != ControlDecision.FAIL,
                error=(
                    "test failure"
                    if decision == ControlDecision.FAIL
                    else None
                ),
                decision=decision,
            )

    return AgentContext(
        task=task,
        plan=plan,
        history=history,
        state=state,
    )


def test_autonomy_decision_properties() -> None:
    execute = AutonomyDecision(
        action=AutonomyAction.EXECUTE,
        reason="execute",
    )

    assert execute.should_execute
    assert not execute.should_retry
    assert not execute.should_replan
    assert not execute.should_stop
    assert not execute.should_complete

    retry = AutonomyDecision(
        action=AutonomyAction.RETRY,
        reason="retry",
    )

    assert retry.should_retry

    replan = AutonomyDecision(
        action=AutonomyAction.REPLAN,
        reason="replan",
    )

    assert replan.should_replan

    stop = AutonomyDecision(
        action=AutonomyAction.STOP,
        reason="stop",
    )

    assert stop.should_stop

    complete = AutonomyDecision(
        action=AutonomyAction.COMPLETE,
        reason="complete",
    )

    assert complete.should_complete


def test_autonomy_policy_rejects_invalid_context() -> None:
    policy = AutonomyPolicy()

    try:
        policy.decide("invalid")  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == "context must be an AgentContext"
    else:
        raise AssertionError("TypeError was not raised")


def test_autonomy_policy_completes_completed_context() -> None:
    policy = AutonomyPolicy()

    context = make_context(
        state="completed"
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.COMPLETE
    assert decision.should_complete
    assert decision.reason == "task is already completed"


def test_autonomy_policy_stops_stopped_context() -> None:
    policy = AutonomyPolicy()

    context = make_context(
        state="stopped"
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.STOP
    assert decision.should_stop
    assert decision.reason == "task is already stopped"


def test_autonomy_policy_stops_cancelled_context() -> None:
    policy = AutonomyPolicy()

    context = make_context(
        state="cancelled"
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.STOP
    assert decision.should_stop
    assert decision.reason == "task is already cancelled"


def test_autonomy_policy_requests_replan_without_plan() -> None:
    policy = AutonomyPolicy()

    context = make_context(
        with_plan=False
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.REPLAN
    assert decision.should_replan
    assert decision.reason == "no execution plan is available"


def test_autonomy_policy_completes_empty_plan() -> None:
    policy = AutonomyPolicy()

    task = Task(
        description="Empty plan"
    )

    plan = ExecutionPlan(
        task_id=task.id,
        steps=(),
    )

    context = AgentContext(
        task=task,
        plan=plan,
        state="running",
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.COMPLETE
    assert decision.should_complete
    assert decision.reason == "execution plan contains no steps"


def test_autonomy_policy_executes_available_plan() -> None:
    policy = AutonomyPolicy()

    context = make_context()

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.EXECUTE
    assert decision.should_execute
    assert decision.reason == "executable plan is available"


def test_autonomy_policy_retries_after_failed_execution() -> None:
    policy = AutonomyPolicy()

    context = make_context(
        with_history=True,
        decision=ControlDecision.FAIL,
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.RETRY
    assert decision.should_retry
    assert decision.reason == "most recent execution failed"


def test_autonomy_policy_stops_after_controller_stop() -> None:
    policy = AutonomyPolicy()

    context = make_context(
        with_history=True,
        decision=ControlDecision.STOP,
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.STOP
    assert decision.should_stop
    assert decision.reason == "controller requested a stop"


def test_autonomy_policy_executes_after_successful_execution() -> None:
    policy = AutonomyPolicy()

    context = make_context(
        with_history=True,
        decision=ControlDecision.CONTINUE,
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.EXECUTE
    assert decision.should_execute
    assert decision.reason == "executable plan is available"


def test_autonomy_policy_state_matching_is_case_insensitive() -> None:
    policy = AutonomyPolicy()

    context = make_context(
        state="  COMPLETED  "
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.COMPLETE


def test_autonomy_policy_does_not_stop_for_unknown_state() -> None:
    policy = AutonomyPolicy()

    context = make_context(
        state="custom-running-state"
    )

    decision = policy.decide(context)

    assert decision.action == AutonomyAction.EXECUTE