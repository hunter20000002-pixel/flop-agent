from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from src.agent.context import AgentContext
from src.agent.memory import InMemoryStore, MemoryEntry, MemoryStore
from src.agent.observation import TechnocoreObservation
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
        """Retrieve current-task and relevant historical memories."""

        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        agent_id = (
            self.agent_id
            if self.agent_id is not None
            else context.agent_id
        )

        current_task_memories = self.store.query(
            task_id=context.task.id,
            agent_id=agent_id,
        )

        historical_memories = self.retrieve_relevant(
            context.task,
            agent_id=agent_id,
        )

        return current_task_memories + historical_memories

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

    def retrieve_relevant(
        self,
        task: Task,
        *,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """Retrieve historical memories relevant to a task.

        Relevance is determined by deterministic lexical overlap between
        the task description and memory content. Memories are restricted
        to the requested agent and are ordered by descending relevance,
        with creation order used as the deterministic tie-breaker.

        The current task itself is excluded from the historical search.
        """

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")

        if agent_id is None:
            agent_id = self.agent_id

        if agent_id is not None:
            if not isinstance(agent_id, str):
                raise TypeError(
                    "agent_id must be a string or None"
                )

            if not agent_id.strip():
                raise ValueError(
                    "agent_id must not be empty"
                )

        if limit is not None:
            if not isinstance(limit, int):
                raise TypeError(
                    "limit must be an integer or None"
                )

            if limit < 0:
                raise ValueError(
                    "limit must not be negative"
                )

            if limit == 0:
                return ()

        task_tokens = self._tokens(
            task.description,
        )

        if not task_tokens:
            return ()

        candidates = self.store.query(
            agent_id=agent_id,
        )

        scored: list[
            tuple[
                int,
                int,
                MemoryEntry,
            ]
        ] = []

        for position, entry in enumerate(candidates):
            if entry.task_id == task.id:
                continue

            memory_tokens = self._tokens(
                entry.content,
            )

            if not memory_tokens:
                continue

            overlap = task_tokens & memory_tokens

            if not overlap:
                continue

            scored.append(
                (
                    len(overlap),
                    -position,
                    entry,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                -item[1],
            )
        )

        results = tuple(
            entry
            for _, _, entry in scored
        )

        if limit is not None:
            return results[:limit]

        return results

    def remember(
        self,
        context: AgentContext,
        content: str,
        *,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a memory associated with the current task and agent."""

        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        return self.store_for_task(
            context.task.id,
            content,
            source=source,
            metadata=metadata,
        )

    def store_for_task(
        self,
        task_id: UUID,
        content: str,
        *,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a memory for a task under this integration's agent."""

        if not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID")

        self._validate_source(source)

        combined_metadata = dict(metadata or {})

        if "source" not in combined_metadata and source is not None:
            combined_metadata["source"] = source

        entry = MemoryEntry(
            content=content,
            task_id=task_id,
            agent_id=self.agent_id,
            metadata=combined_metadata,
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

    def store_observation(
        self,
        context: AgentContext,
        observation: TechnocoreObservation,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Persist a Technocore observation as task memory."""

        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        if not isinstance(
            observation,
            TechnocoreObservation,
        ):
            raise TypeError(
                "observation must be a TechnocoreObservation"
            )

        combined_metadata = {
            "source": "observation",
            "room": observation.room,
            "since": observation.since,
            "message_count": observation.message_count,
            "first_sequence": observation.first_sequence,
            "last_sequence": observation.last_sequence,
            "observed_at": observation.observed_at.isoformat(),
        }

        if metadata is not None:
            combined_metadata.update(metadata)

        return self.store_for_task(
            context.task.id,
            observation.to_untrusted_text(),
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

    @staticmethod
    def _validate_source(
        source: str | None,
    ) -> None:
        """Validate an explicitly supplied memory source."""

        if source is None:
            return

        if not isinstance(source, str):
            raise TypeError("source must be a string or None")

        if not source.strip():
            raise ValueError("source must not be empty")

    @staticmethod
    def _tokens(
        text: str,
    ) -> frozenset[str]:
        """Return normalized lexical tokens for relevance matching."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return frozenset(
            token
            for token in re.findall(
                r"[a-z0-9]+",
                text.lower(),
            )
            if len(token) >= 2
        )