from datetime import datetime, timezone
from unittest.mock import Mock

from src.agent.context import AgentContext
from src.agent.decision import (
    AutonomyAction,
    AutonomyDecision,
    AutonomyPolicy,
)
from src.agent.loop import AgentLoop
from src.agent.memory import MemoryEntry
from src.agent.memory_integration import MemoryIntegration
from src.agent.observation import TechnocoreObservation
from src.agent.result import ExecutionResult
from src.agent.task import Task
from src.client import Message


def make_observation() -> TechnocoreObservation:
    return TechnocoreObservation(
        room="lobby",
        since=100,
        messages=(
            Message(
                seq=101,
                timestamp="2026-09-03T10:00:00Z",
                short_did="z6Mk...test1",
                text="Observed activity.",
                raw_line=(
                    "[101] 2026-09-03T10:00:00Z "
                    "<z6Mk...test1> Observed activity."
                ),
            ),
        ),
        observed_at=datetime(
            2026,
            9,
            3,
            10,
            1,
            tzinfo=timezone.utc,
        ),
    )


def make_context(task: Task) -> AgentContext:
    return AgentContext(
        task=task,
        state="running",
    )


def test_loop_stores_structured_technocore_observation():
    task = Task(description="Store Technocore observation")
    context = make_context(task)
    observation = make_observation()

    memory = Mock(spec=MemoryIntegration)
    memory.agent_id = "agent-1"
    memory.enrich_context.side_effect = lambda value: value

    result = ExecutionResult(
        task_id=task.id,
        status=task.status,
        output=observation.to_untrusted_text(),
        data=observation,
    )

    loop = AgentLoop(
        memory=memory,
    )

    updated_context = loop._update_context_after_execution(
        context=context,
        result=result,
    )

    memory.store_observation.assert_called_once_with(
        updated_context,
        observation,
    )


def test_loop_does_not_store_non_observation_data():
    task = Task(description="Ignore unrelated structured data")
    context = make_context(task)

    memory = Mock(spec=MemoryIntegration)
    memory.agent_id = "agent-1"
    memory.enrich_context.side_effect = lambda value: value

    result = ExecutionResult(
        task_id=task.id,
        status=task.status,
        data={
            "kind": "unrelated",
            "value": 42,
        },
    )

    loop = AgentLoop(
        memory=memory,
    )

    loop._update_context_after_execution(
        context=context,
        result=result,
    )

    memory.store_observation.assert_not_called()


def test_loop_passes_context_memories_into_autonomy_decision():
    task = Task(
        description="Use historical memory",
    )

    memory_entry = MemoryEntry(
        content="Historical Technocore observation.",
        task_id=task.id,
    )

    context = AgentContext(
        task=task,
        state="ready",
        memories=(memory_entry,),
    )

    planner = Mock()
    runtime = Mock()
    policy = Mock(spec=AutonomyPolicy)

    expected_decision = AutonomyDecision(
        action=AutonomyAction.REPLAN,
        reason="test decision",
    )

    policy.decide.return_value = expected_decision

    loop = AgentLoop(
        planner=planner,
        runtime=runtime,
        policy=policy,
    )

    decision = loop._decide(
        task=task,
        context=context,
        result=None,
        failure_count=0,
        retry_count=0,
        replan_count=1,
        capabilities=None,
    )

    assert decision is expected_decision

    policy.decide.assert_called_once()

    autonomy_context = policy.decide.call_args.args[0]

    assert autonomy_context.memories == (
        memory_entry,
    )
    assert autonomy_context.memory_count == 1
    assert autonomy_context.has_memories
