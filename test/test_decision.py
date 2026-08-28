import pytest

from src.agent.context import AgentContext
from src.agent.control import ControlDecision
from src.agent.decision import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.history import ExecutionHistory
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.task import Task


def make_task() -> Task:
    return Task(description="Test autonomous decision making")


def make_plan(task: Task) -> ExecutionPlan:
    return ExecutionPlan(
        task_id=task.id,
        steps=(
            ExecutionStep(
                description="Perform test action",
                order=1,
            ),
        ),
    )


def make_empty_plan(task: Task) -> ExecutionPlan:
    return ExecutionPlan(
        task_id=task.id,
        steps=(),
    )


def make_history_with_decision(
    task: Task,
    decision: ControlDecision,
) -> ExecutionHistory:
    step = ExecutionStep(
        description="Previous execution",
        order=1,
    )

    return ExecutionHistory(task_id=task.id).record(
        step,
        success=decision != ControlDecision.FAIL,
        output=(
            "success"
            if decision != ControlDecision.FAIL
            else None
        ),
        error=(
            None
            if decision != ControlDecision.FAIL
            else "execution failed"
        ),
        decision=decision,
    )


def test_autonomy_action_values():
    assert AutonomyAction.EXECUTE.value == "execute"
    assert AutonomyAction.RETRY.value == "retry"
    assert AutonomyAction.REPLAN.value == "replan"
    assert AutonomyAction.STOP.value == "stop"
    assert AutonomyAction.COMPLETE.value == "complete"


def test_autonomy_decision_stores_action_and_reason():
    decision = AutonomyDecision(
        action=AutonomyAction.EXECUTE,
        reason="ready to execute",
    )

    assert decision.action == AutonomyAction.EXECUTE
    assert decision.reason == "ready to execute"


def test_autonomy_decision_is_immutable():
    decision = AutonomyDecision(
        action=AutonomyAction.EXECUTE,
        reason="ready",
    )

    with pytest.raises(AttributeError):
        decision.action = AutonomyAction.STOP


def test_execute_decision_should_execute():
    decision = AutonomyDecision(
        action=AutonomyAction.EXECUTE,
        reason="execute",
    )

    assert decision.should_execute
    assert not decision.should_retry
    assert not decision.should_replan
    assert not decision.should_stop
    assert not decision.should_complete


def test_retry_decision_should_retry():
    decision = AutonomyDecision(
        action=AutonomyAction.RETRY,
        reason="retry",
    )

    assert decision.should_retry
    assert not decision.should_execute
    assert not decision.should_replan
    assert not decision.should_stop
    assert not decision.should_complete


def test_replan_decision_should_replan():
    decision = AutonomyDecision(
        action=AutonomyAction.REPLAN,
        reason="replan",
    )

    assert decision.should_replan
    assert not decision.should_execute
    assert not decision.should_retry
    assert not decision.should_stop
    assert not decision.should_complete


def test_stop_decision_should_stop():
    decision = AutonomyDecision(
        action=AutonomyAction.STOP,
        reason="stop",
    )

    assert decision.should_stop
    assert not decision.should_execute
    assert not decision.should_retry
    assert not decision.should_replan
    assert not decision.should_complete


def test_complete_decision_should_complete():
    decision = AutonomyDecision(
        action=AutonomyAction.COMPLETE,
        reason="complete",
    )

    assert decision.should_complete
    assert not decision.should_execute
    assert not decision.should_retry
    assert not decision.should_replan
    assert not decision.should_stop


def test_policy_rejects_invalid_context():
    policy = AutonomyPolicy()

    with pytest.raises(TypeError, match="context must be an AgentContext"):
        policy.decide("not a context")


def test_policy_replans_when_no_plan_exists():
    task = make_task()

    context = AgentContext(
        task=task,
        state="running",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.REPLAN
    assert decision.should_replan
    assert "no execution plan" in decision.reason


def test_policy_completes_empty_plan():
    task = make_task()

    context = AgentContext(
        task=task,
        plan=make_empty_plan(task),
        state="running",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.COMPLETE
    assert decision.should_complete
    assert "no steps" in decision.reason


def test_policy_executes_when_plan_is_available():
    task = make_task()

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        state="running",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.EXECUTE
    assert decision.should_execute
    assert "executable plan" in decision.reason


def test_policy_completes_completed_context():
    task = make_task()

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        state="completed",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.COMPLETE
    assert decision.should_complete


def test_policy_stops_stopped_context():
    task = make_task()

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        state="stopped",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.STOP
    assert decision.should_stop


def test_policy_stops_cancelled_context():
    task = make_task()

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        state="cancelled",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.STOP
    assert decision.should_stop


def test_policy_retries_after_failed_execution():
    task = make_task()

    history = make_history_with_decision(
        task,
        ControlDecision.FAIL,
    )

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        history=history,
        state="running",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.RETRY
    assert decision.should_retry
    assert "most recent execution failed" in decision.reason


def test_policy_stops_when_controller_requested_stop():
    task = make_task()

    history = make_history_with_decision(
        task,
        ControlDecision.STOP,
    )

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        history=history,
        state="running",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.STOP
    assert decision.should_stop
    assert "controller requested a stop" in decision.reason


def test_policy_executes_after_successful_execution():
    task = make_task()

    history = make_history_with_decision(
        task,
        ControlDecision.CONTINUE,
    )

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        history=history,
        state="running",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.EXECUTE
    assert decision.should_execute


def test_policy_prioritizes_completed_state():
    task = make_task()

    history = make_history_with_decision(
        task,
        ControlDecision.FAIL,
    )

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        history=history,
        state="completed",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.COMPLETE


def test_policy_prioritizes_stopped_state():
    task = make_task()

    history = make_history_with_decision(
        task,
        ControlDecision.FAIL,
    )

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        history=history,
        state="stopped",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.STOP


def test_policy_prioritizes_missing_plan_over_execution_history():
    task = make_task()

    context = AgentContext(
        task=task,
        history=ExecutionHistory(task_id=task.id),
        state="running",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.REPLAN


def test_policy_returns_structured_decision():
    task = make_task()

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        state="running",
    )

    decision = AutonomyPolicy().decide(context)

    assert isinstance(decision, AutonomyDecision)
    assert isinstance(decision.action, AutonomyAction)
    assert isinstance(decision.reason, str)
    assert decision.reason


def test_policy_does_not_mutate_context():
    task = make_task()

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        state="running",
    )

    original_plan = context.plan
    original_history = context.history
    original_state = context.state

    AutonomyPolicy().decide(context)

    assert context.plan is original_plan
    assert context.history is original_history
    assert context.state == original_state