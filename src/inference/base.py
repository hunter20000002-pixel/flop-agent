from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Request sent to an inference provider."""

    prompt: str
    context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")

        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")

        if self.context is not None and not isinstance(
            self.context,
            dict,
        ):
            raise TypeError("context must be a dictionary or None")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Structured response returned by an inference provider."""

    success: bool
    output: str | None = None
    provider: str = ""
    model: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")

        if self.output is not None and not isinstance(
            self.output,
            str,
        ):
            raise TypeError("output must be a string or None")

        if not isinstance(self.provider, str):
            raise TypeError("provider must be a string")

        if not self.provider.strip():
            raise ValueError("provider must not be empty")

        if self.model is not None and not isinstance(
            self.model,
            str,
        ):
            raise TypeError("model must be a string or None")

        if self.error is not None and not isinstance(
            self.error,
            str,
        ):
            raise TypeError("error must be a string or None")


class InferenceProvider(ABC):
    """Interface implemented by all agent inference providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Generate an inference result for a request."""

        raise NotImplementedError