from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Input supplied to an inference provider."""

    prompt: str
    system_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Structured response returned by an inference provider."""

    success: bool
    output: str | None = None
    error: str | None = None
    provider: str | None = None
    model: str | None = None

    @property
    def failed(self) -> bool:
        """Return True when inference failed."""

        return not self.success


class InferenceProvider(ABC):
    """Provider-independent interface for agent inference."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider's unique name."""

        raise NotImplementedError

    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResult:
        """Generate an inference result from a request."""

        raise NotImplementedError