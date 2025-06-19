import pytest
from unittest.mock import patch, MagicMock

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_community.chat_models import FakeListChatModel

from legisdefagent.core.providers import get_model
from legisdefagent.core.schema.models import (
    AnthropicModelName,
    FakeModelName,
    GroqModelName,
    OpenAIModelName,
)


class TestProviders:
    def test_get_openai_model(self):
        model = get_model(OpenAIModelName.GPT_4O_MINI)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4o-mini"

    def test_get_anthropic_model(self):
        model = get_model(AnthropicModelName.HAIKU_35)
        assert isinstance(model, ChatAnthropic)

    def test_get_groq_model(self):
        model = get_model(GroqModelName.LLAMA_33_70B)
        assert isinstance(model, ChatGroq)

    def test_get_fake_model(self):
        model = get_model(FakeModelName.FAKE)
        assert isinstance(model, FakeListChatModel)

    def test_get_unsupported_model(self):
        with pytest.raises(ValueError, match="Model unsupported-model-name is not supported."):
            get_model("unsupported-model-name")

    @patch("legisdefagent.core.providers.settings")
    def test_get_vllm_model(self, mock_settings):
        mock_settings.vllm.enabled = True
        mock_settings.vllm.host = "localhost"
        mock_settings.vllm.port = 8000

        model = get_model("custom-vllm-model")

        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "custom-vllm-model"
        assert model.openai_api_base == "http://localhost:8000/v1"
