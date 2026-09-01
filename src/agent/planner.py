from __future__ import annotations

import re

from src.agent.context import AgentContext
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.task import Task
from src.config import DEFAULT_CONFIG


class Planner:
    """Creates context-aware execution plans for agent tasks."""

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

        step_descriptions = self._split_into_steps(description)

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
                tool_name,
                context.allowed_capabilities,
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
    def _authorize_tool(
        tool_name: str | None,
        allowed_capabilities: frozenset[str] | None,
    ) -> None:
        """
        Ensure a selected tool is authorized by the planning context.

        ``None`` means unrestricted planning.

        When a capability set is supplied, tool names use the same
        string capability identifiers enforced by AgentRuntime.
        """

        if tool_name is None or allowed_capabilities is None:
            return

        if tool_name not in allowed_capabilities:
            raise PermissionError(
                f"tool '{tool_name}' requires capability "
                f"'{tool_name}', which is not authorized"
            )

    @staticmethod
    def _split_into_steps(
        description: str,
    ) -> tuple[str, ...]:
        """
        Split a compound task into ordered execution steps.

        Supported separators include:

        - "and then"
        - "then"
        - "and" when followed by a recognized action

        The splitter is intentionally conservative so ordinary uses of
        "and" inside a single task are preserved.
        """

        text = description.strip()

        if not text:
            return (description,)

        parts = re.split(
            r"\s+(?:and\s+then|then)\s+",
            text,
            flags=re.IGNORECASE,
        )

        expanded: list[str] = []

        for part in parts:
            part = part.strip()

            if not part:
                continue

            subparts = re.split(
                r"\s+and\s+(?="
                r"(?:calculate|compute|evaluate|solve|"
                r"explain|summarize|list|read|show|open|display|"
                r"research|analyze|analyse|find|search|"
                r"inspect|monitor|check|review|scan)\b"
                r")",
                part,
                flags=re.IGNORECASE,
            )

            expanded.extend(subparts)

        cleaned: list[str] = []

        for part in expanded:
            cleaned_part = part.strip(
                " \t\r\n.,;:"
            )

            if cleaned_part:
                cleaned.append(cleaned_part)

        if not cleaned:
            return (text,)

        return tuple(cleaned)

    @staticmethod
    def _build_step_description(
        context: AgentContext,
        description: str,
    ) -> str:
        """Build the execution description, including relevant memory."""

        if not context.memories:
            return description

        relevant_memory = context.memories[-1]

        if not relevant_memory.content.strip():
            return description

        return (
            f"{description}\n\n"
            f"Relevant memory:\n"
            f"{relevant_memory.content.strip()}"
        )

    @staticmethod
    def _select_tool(
        description: str,
    ) -> tuple[str | None, dict[str, object]]:
        """
        Select an appropriate tool from the task description.

        Tool selection is intentionally conservative. A tool is selected
        only when the step clearly indicates that the corresponding
        capability is required.
        """

        normalized = description.lower()

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

        if Planner._looks_like_technocore_observation(
            normalized
        ):
            return (
                "technocore_observer",
                {
                    "room": DEFAULT_CONFIG.room,
                    "since": 0,
                },
            )

        return None, {}

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
        Extract a simple mathematical expression from a task step.

        The input should already represent a single execution step.
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

            expression = re.split(
                r"\s+(?:and\s+then|then)\s+",
                expression,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]

            expression = re.split(
                r"\s+and\s+(?="
                r"(?:calculate|compute|evaluate|solve)\b"
                r")",
                expression,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]

            expression = expression.rstrip(
                "?. "
            ).strip()

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

        Supports common Windows and Unix-style paths.
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

    @staticmethod
    def _looks_like_technocore_observation(
        description: str,
    ) -> bool:
        """
        Return True when a step clearly requests Technocore observation.

        The planner deliberately requires both a Technocore reference
        and an observation-oriented action. This prevents unrelated tasks
        that merely mention Technocore from invoking the network observer.
        """

        if "technocore" not in description:
            return False

        observation_verbs = (
            "observe",
            "inspect",
            "monitor",
            "check",
            "read",
            "review",
            "scan",
        )

        observation_nouns = (
            "activity",
            "messages",
            "message",
            "updates",
            "room",
        )

        has_observation_verb = any(
            verb in description
            for verb in observation_verbs
        )

        has_observation_noun = any(
            noun in description
            for noun in observation_nouns
        )

        return (
            has_observation_verb
            or has_observation_noun
        )