import pytest

from src.inference.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
)
from src.inference.mock import MockInferenceProvider


def test_inference_request_accepts_valid_prompt():
    request = InferenceRequest(
        prompt="Explain autonomous agents",
    )

    assert request.prompt == "Explain autonomous agents"
    assert request.context is None


def test_inference_request_accepts_context():
    request = InferenceRequest(
        prompt="Use the context",
        context={"key": "value"},
    )

    assert request.context == {"key": "value"}


def test_inference_request_rejects_non_string_prompt():
    with pytest.raises(TypeError):
        InferenceRequest(prompt=123)  # type: ignore[arg-type]


def test_inference_request_rejects_empty_prompt():
    with pytest.raises(ValueError):
        InferenceRequest(prompt="   ")


def test_inference_request_rejects_invalid_context():
    with pytest.raises(TypeError):
        InferenceRequest(
            prompt="test",
            context=[],  # type: ignore[arg-type]
        )


def test_inference_result_accepts_successful_result():
    result = InferenceResult(
        success=True,
        output="42",
        provider="example",
        model="example-model",
    )

    assert result.success
    assert result.output == "42"
    assert result.provider == "example"
    assert result.model == "example-model"
    assert result.error is None


def test_inference_result_accepts_failed_result():
    result = InferenceResult(
        success=False,
        provider="example",
        error="inference failed",
    )

    assert not result.success
    assert result.output is None
    assert result.error == "inference failed"


def test_inference_result_rejects_non_boolean_success():
    with pytest.raises(TypeError):
        InferenceResult(
            success="yes",  # type: ignore[arg-type]
            provider="example",
        )


def test_inference_result_rejects_empty_provider():
    with pytest.raises(ValueError):
        InferenceResult(
            success=True,
            provider="   ",
        )


def test_mock_provider_implements_inference_provider():
    provider = MockInferenceProvider()

    assert isinstance(provider, InferenceProvider)
    assert provider.name == "mock"


def test_mock_provider_generates_response():
    provider = MockInferenceProvider(
        response="42",
    )

    request = InferenceRequest(
        prompt="Calculate something",
    )

    result = provider.generate(request)

    assert result.success
    assert result.output == "42"
    assert result.provider == "mock"
    assert result.model == "mock-model"


def test_mock_provider_rejects_invalid_request():
    provider = MockInferenceProvider()

    with pytest.raises(TypeError):
        provider.generate("invalid")  # type: ignore[arg-type]


def test_mock_provider_rejects_empty_response():
    with pytest.raises(ValueError):
        MockInferenceProvider(response="   ")