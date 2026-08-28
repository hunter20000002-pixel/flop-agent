from __future__ import annotations

from typing import Any
from uuid import UUID

from src.agent.context import AgentContext
from src.agent.memory import InMemoryStore, MemoryEntry, MemoryStore
from src.agent.result import ExecutionResult
from src.agent.task import Task


class MemoryIntegration:
    """Integrates agent memory with tasks and execution context."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        if store is not None and not isinstance(store, MemoryStore):
            raise TypeError("store must be a MemoryStore or None")

        if agent_id is not None:
            if not isinstance(agent_id, str):
                raise TypeError("agent_id must be a string or None")

            if not agent_id.strip():
                raise ValueError("agent_id must not be empty")

        self.store = store or InMemoryStore()
        self.agent_id = agent_id

    def retrieve(
        self,
        context: AgentContext,
    ) -> tuple[MemoryEntry, ...]:
        """Retrieve memories relevant to an agent context."""

        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        agent_id = (
            self.agent_id
            if self.agent_id is not None
            else context.agent_id
        )

        return self.store.query(
            task_id=context.task.id,
            agent_id=agent_id,
        )

    def retrieve_for_task(
        self,
        task_id: UUID,
    ) -> tuple[MemoryEntry, ...]:
        """Retrieve memories belonging to a specific task."""

        if not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID")

        return self.store.query(
            task_id=task_id,
            agent_id=self.agent_id,
        )

    def remember(
        self,
        context: AgentContext,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a memory associated with the current task and agent."""

        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        return self.store_for_task(
            context.task.id,
            content,
            metadata=metadata,
        )

    def store_for_task(
        self,
        task_id: UUID,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a memory for a task under this integration's agent."""

        if not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID")

        entry = MemoryEntry(
            content=content,
            task_id=task_id,
            agent_id=self.agent_id,
            metadata=metadata or {},
        )

        return self.store.store(entry)

    def store_execution_output(
        self,
        context: AgentContext,
        output: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Persist execution output as task memory."""

        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        if not isinstance(output, str):
            raise TypeError("output must be a string")

        combined_metadata = {
            "source": "runtime",
        }

        if metadata is not None:
            combined_metadata.update(metadata)

        return self.store_for_task(
            context.task.id,
            output,
            metadata=combined_metadata,
        )

    def enrich_context(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """Return a new context enriched with relevant memories."""

        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        memories = self.retrieve(context)

        return context.with_memories(memories)