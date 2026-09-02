from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone

from src.agent.autonomy import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.autonomy_context import AutonomyDecisionContext
from src.agent.context import AgentContext
from src.agent.control import (
    ControlDecision,
    ExecutionController,
    StepOutcome,
)
from src.agent.decision_trace import AutonomyDecisionEvent
from src.agent.goal import GoalVerifier
from src.agent.history import ExecutionHistory
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.planner import Planner
from src.agent.result import ExecutionResult
from src.agent.task import Task, TaskStatus
from src.inference.base import InferenceProvider, InferenceRequest
from src.tools.registry import ToolRegistry


StepExecutor = Callable[[ExecutionStep], None]


class AgentRuntime:
    """Orchestrates execution of agent plans."""

    TOOL_CAPABILITIES = {
        "calculator": "calculator",
        "filesystem": "filesystem",
    }

    def __init__(
        self,
        planner: Planner | None = None,
        *,
        step_executor: StepExecutor | None = None,
        inference_provider: InferenceProvider | None = None,
        tool_registry: ToolRegistry | None = None,
        controller: ExecutionController | None = None,
        autonomy_policy: AutonomyPolicy | None = None,
        goal_verifier: GoalVerifier | None = None,
        max_steps: int = 100,
    ) -> None:
        self.planner = planner or Planner()

        self.step_executor = (
            step_executor or self._default_step_executor
        )

        self.inference_provider = inference_provider
        self.tool_registry = tool_registry

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero"
            )

        self.max_steps = max_steps
        self.controller = controller or ExecutionController()

        self.autonomy_policy = autonomy_policy
        self._autonomy_enabled = autonomy_policy is not None

        if goal_verifier is not None and not isinstance(
            goal_verifier,
            GoalVerifier,
        ):
            raise TypeError(
                "goal_verifier must be a GoalVerifier or None"
            )

        self.goal_verifier = goal_verifier

    def run(
        self,
        task: Task,
        *,
        plan: ExecutionPlan | None = None,
        allowed_capabilities: Iterable[str] | None = None,
    ) -> ExecutionResult:
        """
        Execute a task.

        If an execution plan is supplied, that exact plan is executed.
        Otherwise, the runtime creates a plan using the configured planner.

        ``allowed_capabilities`` optionally restricts which tool-backed
        capabilities may be used by the execution plan.

        When ``allowed_capabilities`` is None, direct runtime execution
        remains unrestricted for backward compatibility.

        When an autonomy policy is explicitly supplied, it can request:
        EXECUTE, RETRY, REPLAN, STOP, or COMPLETE.

        Without an explicit autonomy policy, runtime preserves the
        original deterministic execution behavior.

        When a goal verifier is configured, the final execution result
        is passed through that verifier before being returned.
        """

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        if plan is not None:
            if not isinstance(plan, ExecutionPlan):
                raise TypeError(
                    "plan must be an ExecutionPlan or None"
                )

            if plan.task_id != task.id:
                raise ValueError(
                    "plan does not match the supplied task"
                )

        capabilities = (
            None
            if allowed_capabilities is None
            else frozenset(allowed_capabilities)
        )

        task.mark_planning()

        history = ExecutionHistory(task_id=task.id)

        executed_steps = 0
        execution_attempts = 0

        failure_count = 0
        retry_count = 0
        replan_count = 0
        decision_sequence = 0

        outputs: list[str] = []

        last_result: ExecutionResult | None = None

        try:
            context = AgentContext(
                task=task,
                history=history,
                state="planning",
                allowed_capabilities=capabilities,
            )

            if plan is None:
                plan = self._create_plan(
                    context=context,
                    task=task,
                )

            if not isinstance(plan, ExecutionPlan):
                raise TypeError(
                    "planner must return an ExecutionPlan"
                )

            task.mark_ready()
            task.mark_running()

            context = context.with_plan(plan)
            context = context.with_state("running")

            step_index = 0

            pending_autonomy_decision: (
                AutonomyDecision | None
            ) = None

            while step_index < len(context.plan_steps):

                current_step = context.plan_steps[step_index]

                decision_context = self._build_decision_context(
                    task=task,
                    context=context,
                    current_step=current_step,
                    history=history,
                    last_result=last_result,
                    failure_count=failure_count,
                    retry_count=retry_count,
                    replan_count=replan_count,
                    capabilities=capabilities,
                    execution_attempts=execution_attempts,
                )

                autonomy_decision: AutonomyDecision | None = (
                    pending_autonomy_decision
                )

                was_pending_decision = (
                    autonomy_decision is not None
                )

                pending_autonomy_decision = None

                # --------------------------------------------------
                # AUTONOMY CONTROL
                # --------------------------------------------------
                if (
                    self._autonomy_enabled
                    and autonomy_decision is None
                ):
                    autonomy_decision = (
                        self.autonomy_policy.decide(
                            decision_context
                        )
                    )

                if autonomy_decision is not None:
                    if not was_pending_decision:
                        history = self._record_autonomy_decision(
                            history=history,
                            decision=autonomy_decision,
                            sequence=decision_sequence,
                            step=current_step,
                            trigger="policy",
                        )

                        decision_sequence += 1
                        context = context.with_history(history)

                    if autonomy_decision.action == (
                        AutonomyAction.COMPLETE
                    ):
                        break

                    if autonomy_decision.action == (
                        AutonomyAction.STOP
                    ):
                        break

                    if autonomy_decision.action == (
                        AutonomyAction.REPLAN
                    ):
                        replan_count += 1

                        context = self._replan(
                            context=context,
                            task=task,
                        )

                        history = context.history

                        step_index = 0

                        pending_autonomy_decision = (
                            AutonomyDecision(
                                action=AutonomyAction.EXECUTE,
                                reason=(
                                    "execute replanned "
                                    "execution plan"
                                ),
                            )
                        )
                        continue

                    if autonomy_decision.action not in {
                        AutonomyAction.EXECUTE,
                        AutonomyAction.RETRY,
                    }:
                        raise RuntimeError(
                            "unsupported autonomy action: "
                            f"{autonomy_decision.action}"
                        )

                # --------------------------------------------------
                # STEP LIMIT
                # --------------------------------------------------
                if execution_attempts >= self.max_steps:
                    raise RuntimeError(
                        f"execution step limit exceeded: "
                        f"{self.max_steps}"
                    )

                step = context.plan_steps[step_index]

                started_at = datetime.now(timezone.utc)

                execution_attempts += 1

                outcome = self._execute_step(
                    step,
                    context,
                    allowed_capabilities=capabilities,
                )

                completed_at = datetime.now(timezone.utc)

                decision = self.controller.decide(outcome)

                metadata = {
                    "step_order": step.order,
                    "execution_mode": self._execution_mode(
                        step,
                        has_inference_provider=(
                            self.inference_provider is not None
                        ),
                    ),
                }

                if autonomy_decision is not None:
                    metadata["autonomy_action"] = (
                        autonomy_decision.action.value
                    )
                    metadata["autonomy_reason"] = (
                        autonomy_decision.reason
                    )
                    metadata["autonomy_evidence"] = (
                        autonomy_decision.evidence
                    )

                if step.tool_name is not None:
                    metadata["tool_name"] = step.tool_name

                if capabilities is not None:
                    metadata["allowed_capabilities"] = tuple(
                        sorted(capabilities)
                    )

                if (
                    self.inference_provider is not None
                    and step.tool_name is None
                ):
                    metadata["provider"] = (
                        self.inference_provider.name
                    )

                capability = self._capability_for_step(step)

                history = history.record(
                    step,
                    success=outcome.success,
                    output=outcome.output,
                    error=outcome.error,
                    decision=decision,
                    started_at=started_at,
                    completed_at=completed_at,
                    metadata=metadata,
                    capability=capability,
                )

                context = context.with_history(history)

                if outcome.output is not None:
                    outputs.append(str(outcome.output))

                # --------------------------------------------------
                # FAILURE
                # --------------------------------------------------
                if decision == ControlDecision.FAIL:
                    failure_count += 1

                    last_result = ExecutionResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        executed_steps=executed_steps,
                        output=(
                            "\n".join(outputs)
                            if outputs
                            else None
                        ),
                        error=(
                            outcome.error
                            or "execution step failed"
                        ),
                        history=history,
                        progress_made=outcome.progress_made,
                    )

                    if not self._autonomy_enabled:
                        raise RuntimeError(
                            outcome.error
                            or "execution step failed"
                        )

                    failure_context = (
                        self._build_decision_context(
                            task=task,
                            context=context,
                            current_step=step,
                            history=history,
                            last_result=last_result,
                            failure_count=failure_count,
                            retry_count=retry_count,
                            replan_count=replan_count,
                            capabilities=capabilities,
                            execution_attempts=execution_attempts,
                        )
                    )

                    failure_decision = (
                        self.autonomy_policy.decide(
                            failure_context
                        )
                    )

                    history = self._record_autonomy_decision(
                        history=history,
                        decision=failure_decision,
                        sequence=decision_sequence,
                        step=step,
                        trigger="failure",
                    )

                    decision_sequence += 1
                    context = context.with_history(history)

                    if failure_decision.action == (
                        AutonomyAction.RETRY
                    ):
                        retry_count += 1

                        pending_autonomy_decision = (
                            failure_decision
                        )
                        continue

                    if failure_decision.action == (
                        AutonomyAction.REPLAN
                    ):
                        replan_count += 1

                        context = self._replan(
                            context=context,
                            task=task,
                        )

                        history = context.history

                        step_index = 0

                        pending_autonomy_decision = (
                            AutonomyDecision(
                                action=AutonomyAction.EXECUTE,
                                reason=(
                                    "execute replanned "
                                    "execution plan"
                                ),
                            )
                        )
                        continue

                    if failure_decision.action == (
                        AutonomyAction.STOP
                    ):
                        break

                    if failure_decision.action == (
                        AutonomyAction.COMPLETE
                    ):
                        break

                    raise RuntimeError(
                        outcome.error
                        or "execution step failed"
                    )

                # --------------------------------------------------
                # SUCCESS / STOP
                # --------------------------------------------------
                executed_steps += 1

                last_result = ExecutionResult(
                    task_id=task.id,
                    status=task.status,
                    executed_steps=executed_steps,
                    output=(
                        "\n".join(outputs)
                        if outputs
                        else None
                    ),
                    history=history,
                    progress_made=outcome.progress_made,
                )

                if decision == ControlDecision.STOP:
                    break

                step_index += 1

                # Retry count represents the current retry chain.
                # A successful execution terminates that chain.
                retry_count = 0

            task.mark_completed()

            result = ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=executed_steps,
                output=(
                    "\n".join(outputs)
                    if outputs
                    else None
                ),
                history=context.history,
                progress_made=(
                    last_result.progress_made
                    if last_result is not None
                    else None
                ),
            )

            return self._verify_goal(
                task=task,
                result=result,
            )

        except Exception as exc:
            task.mark_failed()

            return ExecutionResult(
                task_id=task.id,
                status=task.status,
                executed_steps=executed_steps,
                output=(
                    "\n".join(outputs)
                    if outputs
                    else None
                ),
                error=str(exc),
                history=history,
            )

    def _verify_goal(
        self,
        *,
        task: Task,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Verify the final execution result when a verifier is configured."""

        if self.goal_verifier is None:
            return result

        verification = self.goal_verifier.verify(
            task,
            result,
        )

        return ExecutionResult(
            task_id=result.task_id,
            status=result.status,
            executed_steps=result.executed_steps,
            output=result.output,
            error=result.error,
            history=result.history,
            progress_made=result.progress_made,
            goal_verification=verification,
        )

    @staticmethod
    def _record_autonomy_decision(
        *,
        history: ExecutionHistory,
        decision: AutonomyDecision,
        sequence: int,
        step: ExecutionStep | None,
        trigger: str,
    ) -> ExecutionHistory:
        """Record one autonomy decision in the execution trace."""

        event = AutonomyDecisionEvent(
            sequence=sequence,
            action=decision.action,
            reason=decision.reason,
            evidence=decision.evidence,
            step_id=(
                step.id
                if step is not None
                else None
            ),
            trigger=trigger,
        )

        return history.add_autonomy_event(event)

    def _build_decision_context(
        self,
        *,
        task: Task,
        context: AgentContext,
        current_step: ExecutionStep | None,
        history: ExecutionHistory,
        last_result: ExecutionResult | None,
        failure_count: int,
        retry_count: int,
        replan_count: int,
        capabilities: frozenset[str] | None,
        execution_attempts: int,
    ) -> AutonomyDecisionContext:
        """Build runtime-owned autonomy evidence."""

        return AutonomyDecisionContext(
            task=task,
            current_plan=context.plan,
            current_step=current_step,
            execution_history=history,
            last_result=last_result,
            failure_count=failure_count,
            retry_count=retry_count,
            replan_count=replan_count,
            allowed_capabilities=capabilities,
            remaining_step_budget=max(
                self.max_steps - execution_attempts,
                0,
            ),
        )

    def _replan(
        self,
        *,
        context: AgentContext,
        task: Task,
    ) -> AgentContext:
        """Create and install a replacement execution plan."""

        task.mark_planning()

        replanned = self._create_plan(
            context=context,
            task=task,
        )

        if not isinstance(replanned, ExecutionPlan):
            raise TypeError(
                "planner must return an ExecutionPlan"
            )

        task.mark_ready()
        task.mark_running()

        return context.with_plan(replanned).with_state(
            "running"
        )

    def _create_plan(
        self,
        context: AgentContext,
        task: Task,
    ) -> ExecutionPlan:
        """
        Create a plan while supporting both context-aware and legacy
        custom planners.
        """

        if isinstance(self.planner, Planner):
            return self.planner.plan(context)

        plan_method = self.planner.plan

        code = getattr(plan_method, "__code__", None)

        if code is not None:
            positional_names = code.co_varnames[
                : code.co_argcount
            ]

            if (
                len(positional_names) >= 2
                and positional_names[1] == "context"
            ):
                return plan_method(context)

            if (
                len(positional_names) >= 1
                and positional_names[0] == "context"
            ):
                return plan_method(context)

        return plan_method(task)

    def _execute_step(
        self,
        step: ExecutionStep,
        context: AgentContext,
        *,
        allowed_capabilities: frozenset[str] | None,
    ) -> StepOutcome:
        """Execute one step using the configured mechanism."""

        try:
            if step.tool_name is not None:
                self._require_tool_capability(
                    step.tool_name,
                    allowed_capabilities,
                )
                return self._execute_tool_step(step)

            if self.inference_provider is not None:
                return self._execute_inference_step(
                    step,
                    context,
                )

            self.step_executor(step)

            return StepOutcome(
                success=True,
            )

        except Exception as exc:
            return StepOutcome(
                success=False,
                error=str(exc),
            )

    @classmethod
    def _capability_for_step(
        cls,
        step: ExecutionStep,
    ) -> str | None:
        """Return the capability required by a step, if any."""

        if step.tool_name is None:
            return None

        return cls.TOOL_CAPABILITIES.get(
            step.tool_name
        )

    @staticmethod
    def _execution_mode(
        step: ExecutionStep,
        *,
        has_inference_provider: bool,
    ) -> str:
        """Return the mechanism used to execute a step."""

        if step.tool_name is not None:
            return "tool"

        if has_inference_provider:
            return "inference"

        return "executor"

    @classmethod
    def _require_tool_capability(
        cls,
        tool_name: str,
        allowed_capabilities: frozenset[str] | None,
    ) -> None:
        """
        Enforce the capability boundary for tool-backed execution.
        """

        if allowed_capabilities is None:
            return

        capability = cls.TOOL_CAPABILITIES.get(
            tool_name
        )

        if capability is None:
            raise RuntimeError(
                f"tool '{tool_name}' has no registered capability"
            )

        if capability not in allowed_capabilities:
            raise RuntimeError(
                f"tool '{tool_name}' requires capability "
                f"'{capability}', which is not authorized"
            )

    def _execute_tool_step(
        self,
        step: ExecutionStep,
    ) -> StepOutcome:
        """Resolve and execute the tool requested by a plan step."""

        if self.tool_registry is None:
            return StepOutcome(
                success=False,
                error=(
                    f"step requires tool '{step.tool_name}', "
                    "but no tool registry is configured"
                ),
            )

        tool = self.tool_registry.get(
            step.tool_name
        )

        result = tool.execute(
            **step.tool_args
        )

        return StepOutcome(
            success=result.success,
            output=result.output,
            error=result.error,
            progress_made=result.progress_made,
        )

    def _execute_inference_step(
        self,
        step: ExecutionStep,
        context: AgentContext,
    ) -> StepOutcome:
        """Execute one plan step through inference."""

        if self.inference_provider is None:
            return StepOutcome(
                success=False,
                error="no inference provider is configured",
            )

        request_context = {
            "task": context.task,
            "task_id": context.task_id,
            "plan": context.plan,
            "history": context.history,
            "memories": context.memories,
            "agent_id": context.agent_id,
            "state": context.state,
        }

        result = self.inference_provider.generate(
            InferenceRequest(
                prompt=step.description,
                context=request_context,
            )
        )

        return StepOutcome(
            success=result.success,
            output=result.output,
            error=result.error,
        )

    @staticmethod
    def _default_step_executor(
        step: ExecutionStep,
    ) -> None:
        """Default executor for a step without an external mechanism."""
        return None