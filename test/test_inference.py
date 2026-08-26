import pytest

from src.inference.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
)


class ExampleProvider(InferenceProvider):
    @property
    def name(self) -> str:
        return "example"

    def generate(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            success=True,
            output=f"Generated: {request.prompt}",
            provider=self.name,
            model="example-model",
        )


class FailingProvider(InferenceProvider):
    @property
    def name(self) -> str:
        return "failing"

    def generate(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            success=False,
            error="Inference failed.",
            provider=self.name,
        )


def test_inference_request_stores_prompt():
    request = InferenceRequest(
        prompt="Explain autonomous agents.",
    )

    assert request.prompt == "Explain autonomous agents."
    assert request.system_prompt is None


def test_inference_request_supports_system_prompt():
    request = InferenceRequest(
        prompt="Explain autonomous agents.",
        system_prompt="You are a technical assistant.",
    )

    assert request.prompt == "Explain autonomous agents."
    assert request.system_prompt == "You are a technical assistant."


def test_provider_exposes_name():
    provider = ExampleProvider()

    assert provider.name == "example"


def test_provider_returns_structured_result():
    provider = ExampleProvider()

    result = provider.generate(
        InferenceRequest(prompt="Hello"),
    )

    assert isinstance(result, InferenceResult)
    assert result.success
    assert not result.failed
    assert result.output == "Generated: Hello"
    assert result.provider == "example"
    assert result.model == "example-model"
    assert result.error is None


def test_failed_inference_result():
    provider = FailingProvider()

    result = provider.generate(
        InferenceRequest(prompt="Hello"),
    )

    assert isinstance(result, InferenceResult)
    assert not result.success
    assert result.failed
    assert result.output is None
    assert result.error == "Inference failed."
    assert result.provider == "failing"


def test_provider_requires_implementation():
    with pytest.raises(TypeError):
        InferenceProvider()