from __future__ import annotations

import pytest

from src.agent.autonomy_context import AutonomyDecisionContext
from src.agent.context import AgentContext
from src.agent.control import ControlDecision
from src.agent.decision import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.history import ExecutionHistory
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus

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


def make_autonomy_context(
    *,
    failure_count: int = 0,
    retry_count: int = 0,
    replan_count: int = 0,
    remaining_step_budget: int | None = None,
    last_result: ExecutionResult | None = None,
    plan: ExecutionPlan | None = None,
) -> AutonomyDecisionContext:
    task = make_task()

    history = ExecutionHistory(task_id=task.id)

    if plan is None:
        plan = make_plan(task)

    return AutonomyDecisionContext(
        task=task,
        current_plan=plan,
        current_step=plan.steps[0] if plan.steps else None,
        execution_history=history,
        last_result=last_result,
        failure_count=failure_count,
        retry_count=retry_count,
        replan_count=replan_count,
        remaining_step_budget=remaining_step_budget,
    )


def make_failed_result(
    context: AutonomyDecisionContext,
) -> ExecutionResult:
    return ExecutionResult(
        task_id=context.task.id,
        status="failed",
        executed_steps=1,
        history=context.execution_history,
        error="execution failed",
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


def test_autonomy_decision_defaults_to_empty_evidence():
    decision = AutonomyDecision(
        action=AutonomyAction.EXECUTE,
        reason="ready to execute",
    )

    assert decision.evidence == {}
    assert len(decision.evidence) == 0


def test_autonomy_decision_stores_evidence():
    evidence = {
        "failure_count": 2,
        "retry_count": 1,
        "remaining_step_budget": 7,
    }

    decision = AutonomyDecision(
        action=AutonomyAction.REPLAN,
        reason="repeated failure",
        evidence=evidence,
    )

    assert decision.evidence["failure_count"] == 2
    assert decision.evidence["retry_count"] == 1
    assert decision.evidence["remaining_step_budget"] == 7


def test_autonomy_decision_evidence_is_immutable():
    decision = AutonomyDecision(
        action=AutonomyAction.EXECUTE,
        reason="ready",
        evidence={
            "failure_count": 0,
        },
    )

    with pytest.raises(TypeError):
        decision.evidence["failure_count"] = 10


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

    with pytest.raises(
        TypeError,
        match="context must be an AgentContext",
    ):
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


def test_autonomy_context_retries_first_failure():
    context = make_autonomy_context(
        failure_count=1,
    )

    context = context.with_result(
        make_failed_result(context)
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.RETRY
    assert decision.should_retry
    assert "most recent execution failed" in decision.reason


def test_autonomy_context_replans_after_repeated_failures():
    context = make_autonomy_context(
        failure_count=2,
    )

    context = context.with_result(
        make_failed_result(context)
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.REPLAN
    assert decision.should_replan
    assert "repeated execution failures" in decision.reason


def test_autonomy_context_replans_when_failure_count_exceeds_two():
    context = make_autonomy_context(
        failure_count=5,
    )

    context = context.with_result(
        make_failed_result(context)
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.REPLAN
    assert decision.should_replan


def test_autonomy_context_stops_when_budget_is_exhausted():
    context = make_autonomy_context(
        failure_count=1,
        remaining_step_budget=0,
    )

    context = context.with_result(
        make_failed_result(context)
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.STOP
    assert decision.should_stop
    assert "budget" in decision.reason


def test_autonomy_context_replans_before_executing_without_a_plan():
    context = make_autonomy_context(
        plan=None,
    )

    decision = AutonomyPolicy().decide(
        context.with_plan(None)
    )

    assert decision.action == AutonomyAction.REPLAN
    assert decision.should_replan
    assert "no execution plan" in decision.reason


def test_autonomy_context_completes_empty_plan():
    task = make_task()

    empty_plan = make_empty_plan(task)

    context = make_autonomy_context(
        plan=empty_plan,
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.COMPLETE
    assert decision.should_complete
    assert "no steps" in decision.reason


def test_autonomy_context_replans_when_execution_made_no_progress():
    context = make_autonomy_context()

    result = ExecutionResult(
        task_id=context.task.id,
        status="completed",
        executed_steps=1,
        history=context.execution_history,
        progress_made=False,
    )

    context = context.with_result(result)

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.REPLAN
    assert decision.should_replan
    assert "no progress" in decision.reason


def test_autonomy_context_executes_when_execution_made_progress():
    context = make_autonomy_context()

    result = ExecutionResult(
        task_id=context.task.id,
        status="completed",
        executed_steps=1,
        history=context.execution_history,
        progress_made=True,
    )

    context = context.with_result(result)

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.EXECUTE
    assert decision.should_execute


def test_autonomy_context_executes_when_progress_is_unknown():
    context = make_autonomy_context()

    result = ExecutionResult(
        task_id=context.task.id,
        status="completed",
        executed_steps=1,
        history=context.execution_history,
    )

    context = context.with_result(result)

    decision = AutonomyPolicy().decide(context)

    assert result.progress_made is None
    assert decision.action == AutonomyAction.EXECUTE
    assert decision.should_execute


def test_autonomy_context_failure_takes_priority_over_no_progress():
    context = make_autonomy_context(
        failure_count=1,
    )

    result = ExecutionResult(
        task_id=context.task.id,
        status="failed",
        executed_steps=1,
        history=context.execution_history,
        error="execution failed",
        progress_made=False,
    )

    context = context.with_result(result)

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.RETRY
    assert decision.should_retry


def test_policy_evidence_contains_runtime_counters():
    context = make_autonomy_context(
        failure_count=2,
        retry_count=3,
        replan_count=4,
        remaining_step_budget=7,
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.evidence["failure_count"] == 2
    assert decision.evidence["retry_count"] == 3
    assert decision.evidence["replan_count"] == 4
    assert decision.evidence["remaining_step_budget"] == 7


def test_policy_evidence_contains_task_and_current_step():
    context = make_autonomy_context()

    decision = AutonomyPolicy().decide(context)

    assert decision.evidence["task_id"] == context.task.id
    assert (
        decision.evidence["current_step"]
        == context.current_step.id
    )


def test_policy_evidence_contains_progress():
    context = make_autonomy_context()

    result = ExecutionResult(
        task_id=context.task.id,
        status="completed",
        executed_steps=1,
        history=context.execution_history,
        progress_made=True,
    )

    context = context.with_result(result)

    decision = AutonomyPolicy().decide(context)

    assert decision.evidence["progress_made"] is True


def test_policy_evidence_preserves_unknown_progress():
    context = make_autonomy_context()

    decision = AutonomyPolicy().decide(context)

    assert decision.evidence["progress_made"] is None


def test_policy_legacy_context_includes_evidence():
    task = make_task()

    context = AgentContext(
        task=task,
        plan=make_plan(task),
        state="running",
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.evidence["task_id"] == task.id
    assert decision.evidence["failure_count"] == 0
    assert decision.evidence["retry_count"] == 0
    assert decision.evidence["replan_count"] == 0
    assert decision.evidence["remaining_step_budget"] is None
    assert decision.evidence["current_step"] == context.plan.steps[0].id

def test_autonomy_policy_replans_when_goal_verification_fails() -> None:
    from src.agent.goal import GoalVerificationResult

    task = Task(
        description="goal verification failure",
    )

    result = ExecutionResult(
        task_id=task.id,
        status=TaskStatus.COMPLETED,
        goal_verification=GoalVerificationResult(
            satisfied=False,
            reason="required outcome was not achieved",
        ),
    )

    context = AutonomyDecisionContext(
        task=task,
        current_plan=ExecutionPlan(
            task_id=task.id,
            steps=(),
        ),
        current_step=None,
        execution_history=ExecutionHistory(
            task_id=task.id,
        ),
        last_result=result,
    )

    # A plan with zero steps is normally an immediate COMPLETE case,
    # so construct a context with a real step.
    step = ExecutionStep(
        order=1,
        description="attempt goal",
    )

    context = AutonomyDecisionContext(
        task=task,
        current_plan=ExecutionPlan(
            task_id=task.id,
            steps=(step,),
        ),
        current_step=step,
        execution_history=ExecutionHistory(
            task_id=task.id,
        ),
        last_result=result,
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.REPLAN
    assert "goal verification" in decision.reason


def test_autonomy_policy_stops_after_goal_verification_failure_and_replan() -> None:
    from src.agent.goal import GoalVerificationResult

    task = Task(
        description="repeated goal verification failure",
    )

    step = ExecutionStep(
        order=1,
        description="attempt goal",
    )

    result = ExecutionResult(
        task_id=task.id,
        status=TaskStatus.COMPLETED,
        goal_verification=GoalVerificationResult(
            satisfied=False,
            reason="goal still not achieved",
        ),
    )

    context = AutonomyDecisionContext(
        task=task,
        current_plan=ExecutionPlan(
            task_id=task.id,
            steps=(step,),
        ),
        current_step=step,
        execution_history=ExecutionHistory(
            task_id=task.id,
        ),
        last_result=result,
        replan_count=1,
    )

    decision = AutonomyPolicy().decide(context)

    assert decision.action == AutonomyAction.STOP
    assert "replanning" in decision.reason