from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.agent.task import Task
from src.agent.task_source import ObservedTask


class QualificationDecision(str, Enum):
    """Decision made before an externally sourced task can execute."""

    ACCEPT = "accept"
    REJECT = "reject"
    IGNORE = "ignore"


class QualificationCapability(str, Enum):
    """Explicit capabilities that remote tasks may be authorized to use."""

    CALCULATOR = "calculator"


@dataclass(frozen=True, slots=True)
class QualificationResult:
    """Structured result of task qualification."""

    decision: QualificationDecision
    reason: str
    task: Task | None = None
    capability: QualificationCapability | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.decision,
            QualificationDecision,
        ):
            raise TypeError(
                "decision must be a QualificationDecision"
            )

        if not isinstance(self.reason, str):
            raise TypeError(
                "reason must be a string"
            )

        if not self.reason.strip():
            raise ValueError(
                "reason must not be empty"
            )

        if self.task is not None and not isinstance(
            self.task,
            Task,
        ):
            raise TypeError(
                "task must be a Task or None"
            )

        if self.capability is not None and not isinstance(
            self.capability,
            QualificationCapability,
        ):
            raise TypeError(
                "capability must be a QualificationCapability or None"
            )

        if (
            self.decision == QualificationDecision.ACCEPT
            and self.task is None
        ):
            raise ValueError(
                "accepted qualification requires a task"
            )

        if (
            self.decision == QualificationDecision.ACCEPT
            and self.capability is None
        ):
            raise ValueError(
                "accepted qualification requires a capability"
            )

        if (
            self.decision != QualificationDecision.ACCEPT
            and self.task is not None
        ):
            raise ValueError(
                "rejected or ignored qualification cannot contain a task"
            )

        if (
            self.decision != QualificationDecision.ACCEPT
            and self.capability is not None
        ):
            raise ValueError(
                "rejected or ignored qualification cannot contain a capability"
            )

    @property
    def accepted(self) -> bool:
        """Return True when the task is approved for execution."""

        return self.decision == QualificationDecision.ACCEPT

    @property
    def rejected(self) -> bool:
        """Return True when the task is explicitly rejected."""

        return self.decision == QualificationDecision.REJECT

    @property
    def ignored(self) -> bool:
        """Return True when the observation should be ignored."""

        return self.decision == QualificationDecision.IGNORE


class TaskQualifier:
    """
    Deterministic safety boundary for externally sourced tasks.

    Qualification happens before planning and execution.

    Only explicitly authorized capabilities can cross the boundary.
    The current remote capability set intentionally contains only
    calculator operations.
    """

    _SUPPORTED_PREFIXES = (
        "calculate ",
        "compute ",
        "evaluate ",
        "solve ",
        "what is ",
        "how much is ",
    )

    _UNSAFE_TERMS = (
        "password",
        "private key",
        "secret key",
        "seed phrase",
        "mnemonic",
        "credential",
        "credentials",
        "api key",
        "token",
        "cookie",
        "session",
        "auth token",
        "ssh key",
        "wallet",
        "delete",
        "remove",
        "destroy",
        "wipe",
        "format",
        "shutdown",
        "restart",
        "kill",
        "terminate",
        "execute command",
        "run command",
        "shell",
        "powershell",
        "cmd.exe",
        "subprocess",
        "os.system",
        "download and run",
        "install software",
        "upload",
        "exfiltrate",
    )

    _UNSUPPORTED_PREFIXES = (
        "research ",
        "analyze ",
        "analyse ",
        "inspect ",
        "find ",
        "explain ",
        "summarize ",
        "read ",
        "list ",
        "show ",
        "please ",
        "can you ",
        "could you ",
    )

    def qualify(
        self,
        observed: ObservedTask,
    ) -> QualificationResult:
        """Qualify one externally observed task."""

        if not isinstance(observed, ObservedTask):
            raise TypeError(
                "observed must be an ObservedTask"
            )

        text = observed.text.strip()
        normalized = text.lower()

        if not normalized:
            return QualificationResult(
                decision=QualificationDecision.IGNORE,
                reason="empty task text",
            )

        unsafe_term = self._find_unsafe_term(
            normalized
        )

        if unsafe_term is not None:
            return QualificationResult(
                decision=QualificationDecision.REJECT,
                reason=(
                    "task requests or references a "
                    f"restricted capability: {unsafe_term}"
                ),
            )

        if self._is_calculator_task(normalized):
            return QualificationResult(
                decision=QualificationDecision.ACCEPT,
                reason="task matches the authorized calculator capability",
                task=observed.task,
                capability=QualificationCapability.CALCULATOR,
            )

        if self._looks_like_unsupported_request(normalized):
            return QualificationResult(
                decision=QualificationDecision.IGNORE,
                reason=(
                    "task is not yet supported by the current "
                    "execution capability set"
                ),
            )

        return QualificationResult(
            decision=QualificationDecision.IGNORE,
            reason=(
                "message does not contain a recognized executable "
                "task form"
            ),
        )

    def qualify_many(
        self,
        observed_tasks: tuple[ObservedTask, ...],
    ) -> tuple[QualificationResult, ...]:
        """Qualify multiple observed tasks in order."""

        if not isinstance(observed_tasks, tuple):
            raise TypeError(
                "observed_tasks must be a tuple"
            )

        return tuple(
            self.qualify(observed)
            for observed in observed_tasks
        )

    @classmethod
    def _find_unsafe_term(
        cls,
        normalized: str,
    ) -> str | None:
        for term in cls._UNSAFE_TERMS:
            if term in normalized:
                return term

        return None

    @classmethod
    def _is_calculator_task(
        cls,
        normalized: str,
    ) -> bool:
        return normalized.startswith(
            cls._SUPPORTED_PREFIXES
        )

    @classmethod
    def _looks_like_unsupported_request(
        cls,
        normalized: str,
    ) -> bool:
        return normalized.startswith(
            cls._UNSUPPORTED_PREFIXES
        )