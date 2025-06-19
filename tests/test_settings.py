import os
from unittest.mock import patch

import pytest

from legisdefagent.settings import GroqModelName, OpenAIModelName, Settings


@pytest.fixture
def clean_env():
    old_env = os.environ.copy()
    keys_to_clear = [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "TOGETHER_API_KEY",
        "VLLM_ENABLED",
        "DEFAULT_MODEL",
    ]
    for key in keys_to_clear:
        if key in os.environ:
            del os.environ[key]

    yield

    os.environ.clear()
    os.environ.update(old_env)


def test_validation_error_no_keys(clean_env):
    """Ensure initialization fails if no API keys and no VLLM are provided."""
    with pytest.raises(ValueError) as excinfo:
        Settings()
    assert "At least one LLM API key must be provided" in str(excinfo.value)


def test_single_provider_openai(clean_env, monkeypatch):
    """Test standard initialization with a single provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings()

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert settings.default_model == OpenAIModelName.GPT_4O_MINI
    assert OpenAIModelName.GPT_4O in settings.available_models


def test_multiple_providers_priority(clean_env, monkeypatch):
    """
    Test that if multiple keys are provided:
    1. The first defined in the 'api_keys' dict (OpenAI) becomes the default.
    2. Models from BOTH providers are added to available_models.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "gtest-key")

    settings = Settings()

    # openai is first in the api_keys dict, so it should be the default
    assert settings.default_model == OpenAIModelName.GPT_4O_MINI

    # but both sets of models should be available still
    assert OpenAIModelName.GPT_4O in settings.available_models
    assert GroqModelName.LLAMA_33_70B in settings.available_models


def test_explicit_default_model_override(clean_env, monkeypatch):
    """Test that setting DEFAULT_MODEL env var overrides the auto-detection."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DEFAULT_MODEL", "custom-model-v1")

    settings = Settings()

    assert settings.default_model == "custom-model-v1"


@patch("legisdefagent.settings.get_available_vllm_models")
def test_vllm_integration(mock_get_models, clean_env, monkeypatch):
    """
    Test VLLM integration mocking the network call.
    Also tests that VLLM works without any API keys.
    """
    monkeypatch.setenv("VLLM", '{"enabled": true, "host": "localhost"}')

    mock_models = ["vllm-llama-3", "vllm-mistral-7b"]
    mock_get_models.return_value = mock_models

    settings = Settings()

    assert settings.vllm.enabled is True
    assert "vllm-llama-3" in settings.available_models
    assert settings.default_model == "vllm-llama-3"

    mock_get_models.assert_called_once_with("localhost", 8001)
