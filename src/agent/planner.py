from __future__ import annotations

from src.agent.context import AgentContext
from src.agent.plan import ExecutionPlan, ExecutionStep
from src.agent.task import Task


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

        step_description = description

        if context.memories:
            relevant_memory = context.memories[-1]

            if relevant_memory.content.strip():
                step_description = (
                    f"{description}\n\n"
                    f"Relevant memory:\n"
                    f"{relevant_memory.content.strip()}"
                )

        step = ExecutionStep(
            description=step_description,
            order=1,
        )

        return ExecutionPlan(
            task_id=current_task.id,
            steps=(step,),
        )