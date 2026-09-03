from __future__ import annotations

import re

from src.agent.context import AgentContext
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.task import Task
from src.config import DEFAULT_CONFIG


class Planner:
    """Creates context-aware execution plans for agent tasks."""

    MAX_PLANNING_MEMORIES = 3

    TOOL_CAPABILITIES = {
        "calculator": "calculator",
        "filesystem": "filesystem",
        "technocore_observer": "technocore_observer",
    }

    def plan(
        self,
        context: AgentContext,
        *,
        task: Task | None = None,
    ) -> ExecutionPlan:
        """Create an execution plan using the supplied agent context."""

        if not isinstance(context, AgentContext):
            raise TypeError("context must be an AgentContext")

        if task is not None and not isinstance(task, Task):
            raise TypeError("task must be a Task or None")

        if task is not None and task.id != context.task.id:
            raise ValueError(
                "task does not match the task in the context"
            )

        current_task = context.task
        description = current_task.description.strip()

        if not description:
            raise ValueError(
                "task description cannot be empty"
            )

        step_descriptions = self._split_into_steps(
            description
        )

        steps: list[ExecutionStep] = []

        for order, step_description in enumerate(
            step_descriptions,
            start=1,
        ):
            enriched_description = self._build_step_description(
                context,
                step_description,
            )

            tool_name, tool_args = self._select_tool(
                step_description
            )

            self._authorize_tool(
                context,
                tool_name,
            )

            steps.append(
                ExecutionStep(
                    description=enriched_description,
                    order=order,
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
            )

        return ExecutionPlan(
            task_id=current_task.id,
            steps=tuple(steps),
        )

    @staticmethod
    def _split_into_steps(
        description: str,
    ) -> tuple[str, ...]:
        """
        Split a compound task into ordered execution steps.

        Supported natural-language separators include:

        - "and then"
        - "then"
        - "and"

        The splitter is intentionally conservative and keeps common
        arithmetic expressions such as "10 + 20" intact.
        """

        parts = re.split(
            r"\s+(?:and\s+then|then)\s+",
            description,
            flags=re.IGNORECASE,
        )

        expanded: list[str] = []

        for part in parts:
            subparts = re.split(
                r"\s+and\s+(?="
                r"(?:calculate|compute|evaluate|solve|explain|"
                r"summarize|list|read|show|open|display)\b"
                r")",
                part,
                flags=re.IGNORECASE,
            )

            expanded.extend(subparts)

        cleaned = [
            part.strip(" \t\r\n.,;")
            for part in expanded
            if part.strip(" \t\r\n.,;")
        ]

        if not cleaned:
            return (description,)

        return tuple(cleaned)

    @classmethod
    def _build_step_description(
        cls,
        context: AgentContext,
        description: str,
    ) -> str:
        """
        Build the execution description using bounded, prioritized memory.

        Memories are consumed in the order supplied by AgentContext.
        MemoryIntegration is responsible for retrieval and ranking, so
        the planner preserves that priority order.

        Blank memories are ignored. Duplicate memory content is included
        only once. At most MAX_PLANNING_MEMORIES memories are injected
        into a single execution step.
        """

        if not context.memories:
            return description

        selected_memories: list[str] = []
        seen: set[str] = set()

        for memory in context.memories:
            content = memory.content.strip()

            if not content:
                continue

            if content in seen:
                continue

            seen.add(content)
            selected_memories.append(content)

            if len(selected_memories) >= cls.MAX_PLANNING_MEMORIES:
                break

        if not selected_memories:
            return description

        memory_context = "\n\n".join(
            selected_memories
        )

        return (
            f"{description}\n\n"
            f"Relevant memory:\n"
            f"{memory_context}"
        )

    @classmethod
    def _authorize_tool(
        cls,
        context: AgentContext,
        tool_name: str | None,
    ) -> None:
        """
        Enforce the capability boundary during planning.

        A None capability set means unrestricted planning for backward
        compatibility. When capabilities are explicitly provided, every
        selected tool must have its required capability authorized.
        """

        if tool_name is None:
            return

        allowed_capabilities = context.allowed_capabilities

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
            raise PermissionError(
                f"tool '{tool_name}' requires capability "
                f"'{capability}', which is not authorized"
            )

    @staticmethod
    def _select_tool(
        description: str,
    ) -> tuple[str | None, dict[str, object]]:
        """
        Select an appropriate built-in tool from the task description.

        Tool selection is intentionally conservative. The planner should
        only select a tool when the task clearly indicates that one is
        required.
        """

        normalized = description.lower()

        if Planner._looks_like_technocore_task(normalized):
            return (
                "technocore_observer",
                {
                    "room": DEFAULT_CONFIG.room,
                    "since": 0,
                },
            )

        if Planner._looks_like_calculation(normalized):
            expression = Planner._extract_expression(
                description
            )

            if expression is not None:
                return (
                    "calculator",
                    {
                        "expression": expression,
                    },
                )

        if Planner._looks_like_filesystem_task(normalized):
            operation = Planner._detect_filesystem_operation(
                normalized
            )

            path = Planner._extract_path(
                description
            )

            if operation is not None and path is not None:
                return (
                    "filesystem",
                    {
                        "operation": operation,
                        "path": path,
                    },
                )

        return None, {}

    @staticmethod
    def _looks_like_technocore_task(
        description: str,
    ) -> bool:
        """Return True when the task requests Technocore observation."""

        if "technocore" not in description:
            return False

        observation_keywords = (
            "observe",
            "inspect",
            "monitor",
            "check",
            "read",
            "review",
            "scan",
            "analyze",
        )

        return any(
            keyword in description
            for keyword in observation_keywords
        )

    @staticmethod
    def _looks_like_calculation(
        description: str,
    ) -> bool:
        """Return True when the task appears to require arithmetic."""

        calculation_keywords = (
            "calculate",
            "compute",
            "evaluate",
            "solve",
            "what is",
            "how much is",
        )

        return any(
            keyword in description
            for keyword in calculation_keywords
        )

    @staticmethod
    def _extract_expression(
        description: str,
    ) -> str | None:
        """
        Extract a simple mathematical expression from a task.

        This intentionally supports the basic expressions handled by
        CalculatorTool without attempting to parse natural-language
        mathematics.
        """

        markers = (
            "calculate",
            "compute",
            "evaluate",
            "solve",
            "what is",
            "how much is",
        )

        lowered = description.lower()

        for marker in markers:
            index = lowered.find(marker)

            if index == -1:
                continue

            expression = description[
                index + len(marker):
            ].strip()

            expression = expression.rstrip("?. ")

            if expression:
                return expression

        return None

    @staticmethod
    def _looks_like_filesystem_task(
        description: str,
    ) -> bool:
        """Return True when the task appears to require filesystem access."""

        filesystem_keywords = (
            "file",
            "directory",
            "folder",
            "filesystem",
            "read",
            "list",
        )

        return any(
            keyword in description
            for keyword in filesystem_keywords
        )

    @staticmethod
    def _detect_filesystem_operation(
        description: str,
    ) -> str | None:
        """Determine the requested filesystem operation."""

        if any(
            keyword in description
            for keyword in (
                "list",
                "show files",
                "show directory",
                "contents of",
                "directory contents",
            )
        ):
            return "list"

        if any(
            keyword in description
            for keyword in (
                "read",
                "open",
                "display contents",
                "show contents",
            )
        ):
            return "read"

        return None

    @staticmethod
    def _extract_path(
        description: str,
    ) -> str | None:
        """
        Extract a filesystem path from a task description.

        Supports common Windows and Unix-style paths. This is deliberately
        conservative; sophisticated path extraction belongs in the
        inference/planning layer later.
        """

        patterns = (
            r"[A-Za-z]:\\[^\s]+",
            r"/[^\s]+",
            r"\./[^\s]+",
            r"\.\./[^\s]+",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                description,
            )

            if match:
                return match.group(0).rstrip(
                    ".,;:!?)]}"
                )

        return None