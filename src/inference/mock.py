from __future__ import annotations

from src.inference.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
)


class MockInferenceProvider(InferenceProvider):
    """Deterministic inference provider used for testing."""

    @property
    def name(self) -> str:
        return "mock"

    def __init__(
        self,
        response: str = "Mock inference response.",
    ) -> None:
        if not isinstance(response, str):
            raise TypeError("response must be a string")

        if not response.strip():
            raise ValueError("response must not be empty")

        self.response = response

    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Return the configured deterministic response."""

        if not isinstance(request, InferenceRequest):
            raise TypeError(
                "request must be an InferenceRequest"
            )

        return InferenceResult(
            success=True,
            output=self.response,
            provider=self.name,
            model="mock-model",
        )